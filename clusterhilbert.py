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
#    node["cpu"] = (i % MAX_CPU) + 1
#    node["ram"] = ((MAX_RAM - i) % MAX_RAM) + 1

    node["cpu"] = MAX_CPU
    node["ram"] = MAX_RAM
# === PREPARE DATA ===
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

# === COMPUTE HILBERT INDEX FOR EACH NODE ===
df["hilbert"] = df.apply(lambda row: encode_node(row["x"], row["y"], row["cpu"], row["ram"]), axis=1)

# === RANGE QUERY FUNCTION ===
def range_query(query_node_name, cpu_min, ram_min, threshold):
    query_node = df[df["name"] == query_node_name].iloc[0]
    q_index = encode_node(query_node["x"], query_node["y"], cpu_min, ram_min)

    df["hilbert_distance"] = np.abs(df["hilbert"] - q_index)

    df_filtered = df[
        (df["cpu"] >= cpu_min) &
        (df["ram"] >= ram_min) &
        (df["hilbert_distance"] <= threshold)
    ]

    print(f"\n=== Range Query from {query_node_name} ===")
    print(f"Target: (x={query_node['x']:.4f}, y={query_node['y']:.4f}, CPU≥{cpu_min}, RAM≥{ram_min})")
    print(f"Distance threshold: {threshold}")
    print(f"Results found: {len(df_filtered)}")
    print(df_filtered.sort_values("hilbert_distance")[["name", "x", "y", "cpu", "ram", "hilbert", "hilbert_distance"]])

# === PERFORM RANGE QUERIES ===
range_query("clab-century-serf1", cpu_min=5, ram_min=5, threshold= 3000000)
range_query("clab-century-serf25", cpu_min=5, ram_min=5, threshold=2000000)
