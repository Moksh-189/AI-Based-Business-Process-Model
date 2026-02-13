"""
Process Flow Visualization
Generates a hierarchical flowchart of the business process.
Focuses on the "Happy Path" (most frequent edges) to show clear flow.
"""
import json
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def load_data():
    print("[INFO] Loading process data...")
    try:
        with open('dfg_data.json') as f:
            dfg = json.load(f)
        with open('bottleneck_report.json') as f:
            report = json.load(f)
        
        # Create bottleneck map for coloring
        bn_map = {b['activity']: b['bottleneck_score'] for b in report['bottlenecks']}
        return dfg['edges'], bn_map
    except FileNotFoundError:
        print("[ERROR] dfg_data.json or bottleneck_report.json not found.")
        return [], {}

def build_graph(edges, max_edges=30):
    """
    Builds a directed graph with only the most frequent edges 
    to properly show the main process flow.
    """
    G = nx.DiGraph()
    
    # Sort edges by frequency
    sorted_edges = sorted(edges, key=lambda x: x['frequency'], reverse=True)
    
    # Add top edges
    # We filter out self-loops for the layout to avoid clutter
    # but might add them back for drawing if needed.
    count = 0
    for edge in sorted_edges:
        if count >= max_edges: break
        
        src = edge['source']
        tgt = edge['target']
        freq = edge['frequency']
        duration = edge['avg_duration_hours']
        
        # Include if it's a significant path
        G.add_edge(src, tgt, weight=freq, duration=duration)
        count += 1
        
    return G

def get_hierarchical_layout(G):
    """
    Custom layout to place nodes in layers (Time/Sequence).
    Returns pos dict {node: (x, y)}
    """
    # 1. Identify Start Nodes (in-degree 0)
    start_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
    
    # Fallback if no clear start (due to cycles)
    if not start_nodes:
        # Heuristic: 'Create Purchase Requisition' or 'Create Purchase Order' often start
        candidates = [n for n in G.nodes() if 'Create' in n]
        if candidates:
            start_nodes = [candidates[0]]
        else:
            start_nodes = [list(G.nodes())[0]] # Random pick
            
    # 2. Assign Layers (BFS)
    layers = {node: 0 for node in G.nodes()} 
    visited = set()
    queue = [(n, 0) for n in start_nodes]
    
    # To handle cycles: if we see a node again, we only push it deeper if it makes sense?
    # Actually for simple DAG layout, we just do BFS. 
    # For cycles, we ignore back-edges in layer assignment.
    
    while queue:
        node, layer = queue.pop(0)
        
        if node in visited:
            # If we found a longer path to this node, update layer?
            # Simple approach: max layer
            if layer > layers[node]:
                layers[node] = layer
            continue

        visited.add(node)
        layers[node] = layer
        
        # Add neighbors
        for neighbor in G.successors(node):
            if neighbor not in visited: 
                 queue.append((neighbor, layer + 1))
            else:
                 # Check if this is a forward edge we missed
                 if layers[neighbor] < layer + 1:
                     # Potential cycle or just multi-path.
                     # We push it forward only if not a direct cycle back to current path
                     pass

    # 3. Group by Layer
    layer_nodes = {}
    for node, layer in layers.items():
        if layer not in layer_nodes: layer_nodes[layer] = []
        layer_nodes[layer].append(node)
        
    # 4. Assign (x, y) coordinates
    # x = layer (horizontal flow)
    # y = position in layer
    
    pos = {}
    
    # Sort layers
    sorted_layers = sorted(layer_nodes.keys())
    
    for layer in sorted_layers:
        nodes = layer_nodes[layer]
        # Center the nodes vertically
        # y = 0 is center. 
        # range: -len/2 to +len/2
        
        n_count = len(nodes)
        for i, node in enumerate(nodes):
            x = layer * 3.0 # Spacing horizontal
            y = (i - n_count / 2.0) * 1.5 # Spacing vertical
            pos[node] = (x, -y) # Negative y so first item is top
            
    return pos

def draw_chart(G, pos, bn_map):
    plt.figure(figsize=(20, 10))
    ax = plt.gca()
    
    # Draw Edges (curved)
    for u, v in G.edges():
        duration = G[u][v].get('duration', 0)
        label = f"{duration:.1f}h" if duration < 48 else f"{duration/24:.1f}d"
        
        # Curve edges to avoid overlap
        rad = 0.1
        
        # Color based on frequency? 
        # Or just grey.
        
        # Use simple networkx draw for edges first
        nx.draw_networkx_edges(G, pos, ax=ax, 
                               width=1.5, 
                               edge_color='#555555', 
                               arrowstyle='-|>', 
                               arrowsize=20,
                               connectionstyle=f"arc3,rad={rad}",
                               min_source_margin=15,
                               min_target_margin=15)
        
        # Edge Labels
        # Calculate mid-point for label
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        edge_x = (x1 + x2) / 2
        edge_y = (y1 + y2) / 2
        
        # Lift label slightly
        edge_y += 0.1
        
        plt.text(edge_x, edge_y, label, 
                 fontsize=8, color='#333333', 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # Draw Nodes
    # Color by bottleneck score
    # 0.0 (Green) to 1.0 (Red)
    
    for node in G.nodes():
        score = bn_map.get(node, 0)
        # Color interpolation: Green -> Yellow -> Red
        if score < 0.5:
            # Green (0,1,0) to Yellow (1,1,0)
            ratio = score * 2
            color = (ratio, 1.0, 0.0)
        else:
            # Yellow (1,1,0) to Red (1,0,0)
            ratio = (score - 0.5) * 2
            color = (1.0, 1.0 - ratio, 0.0)
            
        x, y = pos[node]
        
        # Draw Box
        box = mpatches.FancyBboxPatch((x - 0.6, y - 0.3), 1.2, 0.6,
                                      boxstyle="round,pad=0.1",
                                      ec="black", fc=color, zorder=10)
        ax.add_patch(box)
        
        # Draw Text
        # Wrap text
        label_text = node.replace("Purchase Order", "PO").replace("Requisition", "Req")
        if len(label_text) > 15:
            parts = label_text.split(' ')
            mid = len(parts)//2
            label_text = " ".join(parts[:mid]) + "\n" + " ".join(parts[mid:])
            
        plt.text(x, y, label_text, 
                 ha='center', va='center', 
                 fontsize=9, fontweight='bold', zorder=11)

    plt.axis('off')
    plt.tight_layout()
    plt.title("Business Process Flow (Hierarchical View)\nRed = High Bottleneck Risk", fontsize=16)
    
    plt.savefig('process_flow_chart.png', dpi=150, bbox_inches='tight')
    print("[SUCCESS] Flow chart saved to 'process_flow_chart.png'")

if __name__ == "__main__":
    edges, bn_map = load_data()
    if edges:
        # Configure how many edges to show for clarity
        G = build_graph(edges, max_edges=25)
        pos = get_hierarchical_layout(G)
        draw_chart(G, pos, bn_map)
