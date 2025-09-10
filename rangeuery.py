import re
from hilbertcurve.hilbertcurve import HilbertCurve
import matplotlib.pyplot as plt

# 1. Load and parse the coordinates file
nodes = []

# Full path to your file
file_path = 'C:/Users/malik/Desktop/pythonscripts/coordinates.log'

with open(file_path, 'r') as f:
    for line in f:
        match = re.search(r'Node: (.*?) \| Vec: \[(.*?) (.*?)\]', line)
        if match:
            name = match.group(1)
            x = float(match.group(2))
            y = float(match.group(3))
            nodes.append({
                'name': name,
                'coordinate': {'Vec': [x, y]}
            })

# 2. Normalize coordinates to 0..65535
coords = [node['coordinate']['Vec'] for node in nodes]
x_vals = [c[0] for c in coords]
y_vals = [c[1] for c in coords]

min_x, max_x = min(x_vals), max(x_vals)
min_y, max_y = min(y_vals), max(y_vals)

def normalize(val, min_val, max_val, bits=16):
    scale = (2**bits - 1)
    return int((val - min_val) / (max_val - min_val) * scale)

# 3. Create Hilbert curve object
p = 16  # bits per dimension
n = 2   # 2D
hc = HilbertCurve(p, n)

# 4. Assign Hilbert index to each node
for node in nodes:
    x, y = node['coordinate']['Vec']
    norm_x = normalize(x, min_x, max_x)
    norm_y = normalize(y, min_y, max_y)
    hilbert_index = hc.distance_from_point([norm_x, norm_y])
    node['hilbert_index'] = hilbert_index

# 5. Sort nodes by Hilbert index
nodes.sort(key=lambda n: n['hilbert_index'])

# 6. Function: Find neighbors by sorted Hilbert index
def find_neighbors(target_node, count=10):
    target_index = target_node['hilbert_index']
    idx = next((i for i, node in enumerate(nodes) if node['name'] == target_node['name']), None)
    
    if idx is None:
        return []
    
    start = max(idx - count, 0)
    end = min(idx + count + 1, len(nodes))
    
    return nodes[start:end]

# 7. Interactive search
def interactive_search():
    print("\n--- Interactive Neighbor Search ---")
    while True:
        search_name = input("\nEnter a node name (or type 'exit' to quit): ").strip()
        
        if search_name.lower() == 'exit':
            print("Exiting search.")
            break
        
        target_node = next((node for node in nodes if node['name'] == search_name), None)
        
        if target_node is None:
            print(f"Node '{search_name}' not found. Try again.")
            continue
        
        neighbors = find_neighbors(target_node, count=5)
        
        print(f"\nClosest nodes to {search_name}:")
        for n in neighbors:
            if n['name'] != search_name:  # Skip itself
                print(f" - {n['name']} (hilbert index: {n['hilbert_index']})")

# 8. Optional: Visualization
def plot_nodes():
    plt.figure(figsize=(8,8))
    colors = [node['hilbert_index'] for node in nodes]
    plt.scatter(
        [node['coordinate']['Vec'][0] for node in nodes],
        [node['coordinate']['Vec'][1] for node in nodes],
        c=colors,
        cmap='plasma',
        s=20
    )
    plt.colorbar(label='Hilbert Index')
    plt.title('Nodes mapped on (x,y) with Hilbert Curve Order')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

# 9. Main execution
if __name__ == "__main__":
    # First, plot the map (optional)
    plot_nodes()
    
    # Then run the interactive search
    interactive_search()
