import json
import pandas as pd
import numpy as np
import re
from hilbertcurve.hilbertcurve import HilbertCurve

# === CONFIGURATION ===
BITS = 8
DIMENSIONS = 3  # RTT + CPU + RAM
MAX_CPU = 30
MAX_RAM = 30
COORD_FILE = r"C:\Users\malik\Desktop\pythonscripts\rttsfor80nodes.txt"

# === LOAD DATA ===
with open(COORD_FILE, "r") as f:
    content = f.read()
    serf_data = json.loads(content)

# === ASSIGN CPU/RAM IN BLOCKS ===
tiers = [(30,30),(25,25),(20,20),(15,15),(10,10),(5,5),(3,3),(1,1)]

def extract_serf_number(name):
    m = re.search(r"serf(\d+)", name)
    return int(m.group(1)) if m else float('inf')

serf_data.sort(key=lambda x: extract_serf_number(x["name"]))
for i, node in enumerate(serf_data):
    node["cpu"], node["ram"] = tiers[i // 10]

# === PREPARE DATAFRAME WITH avg RTT ===
records = []
for node in serf_data:
    rtts = list(node.get("rtts", {}).values())
    avg_rtt = np.mean(rtts) if rtts else 0
    records.append({
        "name": node["name"],
        "avg_rtt": avg_rtt,
        "cpu": node["cpu"],
        "ram": node["ram"]
    })
df = pd.DataFrame(records)

# === NORMALIZATION ===
def normalize(val, min_val, max_val):
    return int(((val - min_val) / (max_val - min_val)) * ((2**BITS) - 1)) if max_val > min_val else 0

rtt_bounds = (df["avg_rtt"].min(), df["avg_rtt"].max())
hilbert = HilbertCurve(p=BITS, n=DIMENSIONS)

def encode_node(avg_rtt, cpu, ram):
    return hilbert.distance_from_point([
        normalize(avg_rtt, *rtt_bounds),
        normalize(cpu, 1, MAX_CPU),
        normalize(ram, 1, MAX_RAM)
    ])

df["hilbert"] = df.apply(lambda r: encode_node(r["avg_rtt"], r["cpu"], r["ram"]), axis=1)

# === RTT THRESHOLD BINS based on real data distribution ===
all_rtts = [v for node in serf_data for v in node.get("rtts", {}).values()]
quantiles = np.percentile(all_rtts, [10, 25, 50, 75, 90])
latency_bins = [round(q) for q in quantiles]

# === HILBERT DISTANCE THRESHOLD MAPPING ===
def build_latency_hilbert_mapping(df, query_node_name, serf_data, percent=95):
    query_row = df[df["name"] == query_node_name].iloc[0]
    q_index = encode_node(query_row["avg_rtt"], query_row["cpu"], query_row["ram"])

    latencies, dists = [], []
    for node in df.itertuples():
        if node.name == query_node_name:
            continue
        rtt = next((n["rtts"].get(query_node_name) for n in serf_data if n["name"] == node.name), None)
        if rtt is None:
            continue
        hd = abs(node.hilbert - q_index)
        latencies.append(rtt)
        dists.append(hd)

    mapping = {}
    for max_rtt in latency_bins:
        d_subset = [d for r, d in zip(latencies, dists) if r <= max_rtt]
        mapping[max_rtt] = int(np.percentile(d_subset, percent)) if d_subset else None
    return mapping

# === RANGE QUERY ===
def range_query(query_node_name, rtt_threshold_ms):
    threshold_map = build_latency_hilbert_mapping(df, query_node_name, serf_data)
    hilbert_thr = threshold_map.get(rtt_threshold_ms)
    if hilbert_thr is None:
        print(f"No mapping for RTT≤{rtt_threshold_ms}")
        return

    query_row = df[df["name"] == query_node_name].iloc[0]
    q_index = encode_node(query_row["avg_rtt"], query_row["cpu"], query_row["ram"])
    df["hilbert_distance"] = np.abs(df["hilbert"] - q_index)
    df_filtered = df[df["hilbert_distance"] <= hilbert_thr]

    print(f"\n=== Hilbert-Only Range Query from {query_node_name} ===")
    print(f"RTT ≤ {rtt_threshold_ms}ms — Hilbert threshold: {hilbert_thr}")
    print(f"Nodes returned: {len(df_filtered)}")
    print(df_filtered.sort_values("hilbert_distance")[["name", "cpu", "ram", "avg_rtt", "hilbert_distance"]])

# === ACCURACY TEST ===
def test_range_query_accuracy(query_node_name, rtt_threshold_ms):
    query_idx = next(i for i, n in enumerate(serf_data) if n["name"] == query_node_name)
    query_rtts = serf_data[query_idx]["rtts"]
    true_set = {k for k, v in query_rtts.items() if v <= rtt_threshold_ms}

    threshold_map = build_latency_hilbert_mapping(df, query_node_name, serf_data)
    hilbert_thr = threshold_map.get(rtt_threshold_ms)
    if hilbert_thr is None:
        print(f"No Hilbert threshold for RTT ≤ {rtt_threshold_ms}ms")
        return

    query_row = df[df["name"] == query_node_name].iloc[0]
    q_index = encode_node(query_row["avg_rtt"], query_row["cpu"], query_row["ram"])
    df["hilbert_distance"] = np.abs(df["hilbert"] - q_index)
    hilbert_set = set(df[df["hilbert_distance"] <= hilbert_thr]["name"])
    hilbert_set.discard(query_node_name)

    intersection = hilbert_set & true_set
    union = hilbert_set | true_set

    precision = len(intersection) / len(hilbert_set) if hilbert_set else 0
    recall = len(intersection) / len(true_set) if true_set else 0
    jaccard = len(intersection) / len(union) if union else 0

    print(f"\n--- Accuracy Test for {query_node_name} (RTT ≤ {rtt_threshold_ms}ms) ---")
    print(f"Hilbert: {len(hilbert_set)} nodes | RTT True: {len(true_set)} nodes")
    print(f"Match: {len(intersection)} | Precision: {precision:.2f} | Recall: {recall:.2f} | Jaccard: {jaccard:.2f}")
    return {"threshold": rtt_threshold_ms, "precision": precision, "recall": recall, "jaccard": jaccard}

# === RUN TEST ===
if __name__ == "__main__":
    query_node = "clab-century-serf1"
    print(f"\n📌 Using RTT bins based on data: {latency_bins}")
    for rtt in latency_bins:
        range_query(query_node, rtt)
        test_range_query_accuracy(query_node, rtt)
