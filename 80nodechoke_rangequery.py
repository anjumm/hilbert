import json
import pandas as pd
import numpy as np
import re
from hilbertcurve.hilbertcurve import HilbertCurve

# === CONFIGURATION ===
BITS = 6
DIMENSIONS = 4
MAX_CPU = 15
MAX_RAM = 15
COORD_FILE = r"C:\Users\malik\Desktop\pythonscripts\rttsfor80nodes.txt"

# === LOAD DATA (JSON inside .txt) ===
with open(COORD_FILE, "r") as f:
    content = f.read()
    serf_data = json.loads(content)

# === CPU/RAM ASSIGNMENT IN 8 BLOCKS OF 10 ===
tiers = [
    (15, 15), (13, 13), (11, 11), (9, 9),
    (7, 7), (5, 5), (3, 3), (1, 1)
]

# Extract serf number and sort the list accordingly
def extract_serf_number(node_name):
    match = re.search(r"serf(\d+)", node_name)
    return int(match.group(1)) if match else float('inf')

# Sort by serf number before assignment
serf_data.sort(key=lambda x: extract_serf_number(x["name"]))

# Assign CPU/RAM based on block
for i, node in enumerate(serf_data):
    cpu, ram = tiers[i // 10]
    node["cpu"] = cpu
    node["ram"] = ram

# === VERIFY CPU/RAM ASSIGNMENT ===
print("\n=== CPU/RAM Assignment for All Nodes ===")
for i, node in enumerate(serf_data):
    print(f"{i:2d}  {node['name']:25}  CPU={node['cpu']:2d}  RAM={node['ram']:2d}")

# === PREPARE DataFrame ===
records = []
for node in serf_data:
    x, y = node["coordinate"]["Vec"][:2]
    records.append({
        "name": node["name"],
        "x": x,
        "y": y,
        "cpu": node["cpu"],
        "ram": node["ram"]
    })
df = pd.DataFrame(records)

# === NORMALIZATION ===
def normalize(val, min_val, max_val):
    return int(((val - min_val) / (max_val - min_val)) * (2**BITS - 1))

x_bounds = (df["x"].min(), df["x"].max())
y_bounds = (df["y"].min(), df["y"].max())
hilbert = HilbertCurve(p=BITS, n=DIMENSIONS)

def encode_node(x, y, cpu, ram):
    x_n = normalize(x, *x_bounds)
    y_n = normalize(y, *y_bounds)
    cpu_n = normalize(cpu, 1, MAX_CPU)
    ram_n = normalize(ram, 1, MAX_RAM)
    return hilbert.distance_from_point([x_n, y_n, cpu_n, ram_n])

df["hilbert"] = df.apply(lambda row: encode_node(row["x"], row["y"], row["cpu"], row["ram"]), axis=1)

# === RANGE QUERY SUPPORT ===
def build_latency_hilbert_mapping(df, query_node_name, serf_data, percent=95):
    query_row = df[df["name"] == query_node_name].iloc[0]
    q_index = encode_node(query_row["x"], query_row["y"], query_row["cpu"], query_row["ram"])

    latencies, dists = [], []
    for node in df.itertuples():
        if node.name == query_node_name:
            continue
        data_idx = next(i for i, n in enumerate(serf_data) if n["name"] == node.name)
        rtt = serf_data[data_idx]["rtts"].get(query_node_name)
        if rtt is None:
            continue
        hd = abs(node.hilbert - q_index)
        latencies.append(rtt)
        dists.append(hd)

    latency_bins = [5, 10, 20, 40, 50, 100, 150, 200]
    mapping = {}
    for max_rtt in latency_bins:
        bucket = [d for r, d in zip(latencies, dists) if r <= max_rtt]
        mapping[max_rtt] = int(np.percentile(bucket, percent)) if bucket else None
    return mapping

def range_query(query_node_name, cpu_min, ram_min, rtt_threshold_ms):
    thr_map = build_latency_hilbert_mapping(df, query_node_name, serf_data)
    hilbert_thr = thr_map.get(rtt_threshold_ms)
    if hilbert_thr is None:
        raise ValueError(f"No mapping for RTT≤{rtt_threshold_ms} ms")

    query_node = df[df["name"] == query_node_name].iloc[0]
    q_index = encode_node(query_node["x"], query_node["y"], query_node["cpu"], query_node["ram"])

    df["hilbert_distance"] = np.abs(df["hilbert"] - q_index)
    df_filtered = df[
        (df["cpu"] <= cpu_min) &
        (df["ram"] <= ram_min) &
        (df["hilbert_distance"] <= hilbert_thr)
    ]

    print(f"\n=== Range Query from {query_node_name} ===")
    print(f"RTT ≤ {rtt_threshold_ms}ms, CPU ≤ {cpu_min}, RAM ≤ {ram_min}")
    print(f"Hilbert threshold: {hilbert_thr}")
    print(f"Results: {len(df_filtered)}")
    print(df_filtered.sort_values("hilbert_distance")[["name", "cpu", "ram", "hilbert", "hilbert_distance"]])

# === EXAMPLE USAGE ===
if __name__ == "__main__":
    range_query("clab-century-serf1", cpu_min=10, ram_min=10, rtt_threshold_ms=40)
    range_query("clab-century-serf1", cpu_min=5, ram_min=5, rtt_threshold_ms=150)
    range_query("clab-century-serf25", cpu_min=1, ram_min=1, rtt_threshold_ms=200)
