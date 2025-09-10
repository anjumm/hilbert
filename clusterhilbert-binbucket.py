import json
import pandas as pd
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve

# === CONFIGURATION ===
BITS = 6             # 8 bits per dimension: 0–255 range
DIMENSIONS = 4       # (x, y, cpu, ram)
MAX_CPU = 15
MAX_RAM = 15
COORD_FILE = r"C:\Users\malik\Desktop\pythonscripts\coordinates_fully_updated_newzone_15_may.txt"  # Path to the file with coordinates

# === LOAD DATA ===
with open(COORD_FILE, "r") as f:
    serf_data = json.load(f)

# Assign deterministic CPU and RAM (can be randomized or real)
for i, node in enumerate(serf_data):
    node["cpu"] = MAX_CPU
    node["ram"] = MAX_RAM

# === PREPARE DataFrame ===
records = []
for node in serf_data:
    x, y = node["coordinate"]["Vec"]
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
    """Convert real value to 0–255 range"""
    return int(((val - min_val) / (max_val - min_val)) * (2**BITS - 1))

x_bounds = (df["x"].min(), df["x"].max())
y_bounds = (df["y"].min(), df["y"].max())

# === HILBERT CURVE SETUP ===
hilbert = HilbertCurve(p=BITS, n=DIMENSIONS)

def encode_node(x, y, cpu, ram):
    x_n = normalize(x, *x_bounds)
    y_n = normalize(y, *y_bounds)
    cpu_n = normalize(cpu, 1, MAX_CPU)
    ram_n = normalize(ram, 1, MAX_RAM)
    return hilbert.distance_from_point([x_n, y_n, cpu_n, ram_n])

# Compute Hilbert index once
df["hilbert"] = df.apply(lambda row: encode_node(row["x"], row["y"], row["cpu"], row["ram"]), axis=1)

# === BUILD LATENCY→HILBERT MAPPING ===
def build_latency_hilbert_mapping(df, query_node_name, serf_data, percent=95):
    """
    Returns a dict: { rtt_bin_ms: hilbert_distance_threshold, ... }
    Uses RTTs from serf_data and df["hilbert"].
    """
    # find query node index in df and serf_data
    query_row = df[df["name"] == query_node_name].iloc[0]
    q_index = encode_node(query_row["x"], query_row["y"], query_row["cpu"], query_row["ram"])

    # gather (rtt, hilbert_dist) pairs
    latencies, dists = [], []
    for idx, node in enumerate(df.itertuples()):
        if node.name == query_node_name:
            continue
        # find corresponding RTT in serf_data
        data_idx = next(i for i, n in enumerate(serf_data) if n["name"] == node.name)
        rtt = serf_data[data_idx]["rtts"][query_node_name]
        hd = abs(node.hilbert - q_index)
        latencies.append(rtt)
        dists.append(hd)

    # define RTT bins (ms)
    latency_bins = [5, 10, 20, 50, 100]
    mapping = {}
    for max_rtt in latency_bins:
        bucket = [d for r, d in zip(latencies, dists) if r <= max_rtt]
        if bucket:
            mapping[max_rtt] = int(np.percentile(bucket, percent))
        else:
            mapping[max_rtt] = None
    return mapping

# === UPDATED RANGE QUERY ===
def range_query(query_node_name, cpu_min, ram_min, rtt_threshold_ms):
    # build or cache latency→Hilbert mapping
    thr_map = build_latency_hilbert_mapping(df, query_node_name, serf_data)

    hilbert_thr = thr_map.get(rtt_threshold_ms)
    if hilbert_thr is None:
        raise ValueError(f"No mapping for RTT≤{rtt_threshold_ms} ms; try a larger bin.")

    # perform the same filtering logic
    query_node = df[df["name"] == query_node_name].iloc[0]
    q_index = encode_node(query_node["x"], query_node["y"], query_node["cpu"], query_node["ram"])

    df["hilbert_distance"] = np.abs(df["hilbert"] - q_index)

    df_filtered = df[
        (df["cpu"] >= cpu_min) &
        (df["ram"] >= ram_min) &
        (df["hilbert_distance"] <= hilbert_thr)
    ]

    print(f"\n=== Range Query from {query_node_name} ===")
    print(f"Target: (x={query_node['x']:.4f}, y={query_node['y']:.4f}, CPU≥{cpu_min}, RAM≥{ram_min})")
    print(f"RTT threshold: {rtt_threshold_ms} ms → Hilbert dist ≤ {hilbert_thr}")
    print(f"Results found: {len(df_filtered)}")
    #print(df_filtered.sort_values("hilbert_distance")[["name","x","y","cpu","ram","hilbert","hilbert_distance"]])
    print(df_filtered.sort_values("hilbert_distance")[["name","x","y","cpu","ram","hilbert"]])


# === EXAMPLES ===
if __name__ == "__main__":
    range_query("clab-century-serf1", cpu_min=5, ram_min=5, rtt_threshold_ms=5)
    range_query("clab-century-serf1", cpu_min=5, ram_min=5, rtt_threshold_ms=100)
    range_query("clab-century-serf25", cpu_min=5, ram_min=5, rtt_threshold_ms=5)
    range_query("clab-century-serf25", cpu_min=5, ram_min=5, rtt_threshold_ms=100)
