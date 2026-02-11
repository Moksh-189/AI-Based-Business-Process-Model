"""Quick script to visualize the process graph."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

with open('dfg_data.json') as f:
    dfg = json.load(f)
with open('bottleneck_report.json') as f:
    report = json.load(f)
bn = {b['activity']: b['bottleneck_score'] for b in report['bottlenecks']}

G = nx.DiGraph()
for edge in dfg['edges'][:20]:
    G.add_edge(edge['source'], edge['target'], w=edge['frequency'], d=edge['avg_duration_hours'])

pos = nx.kamada_kawai_layout(G)
fig, ax = plt.subplots(figsize=(18, 12))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

ew = [G[u][v]['w'] for u, v in G.edges()]
mx = max(ew)
widths = [0.5 + 4 * (w / mx) for w in ew]
nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#8b949e', width=widths,
                       arrows=True, arrowsize=18, alpha=0.5,
                       connectionstyle='arc3,rad=0.08')

colors = []
sizes = []
ms = max(bn.values())
for n in G.nodes():
    s = bn.get(n, 0)
    r = min(1, s / ms * 2)
    g = min(1, max(0, 1 - s / ms * 1.5)) * 0.85
    colors.append((r, g, 0.15, 0.95))
    sizes.append(1200 + 2500 * (s / ms))

nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes,
                       edgecolors='#58a6ff', linewidths=2)

labels = {}
for n in G.nodes():
    l = n.replace('Record ', '').replace('Purchase Order', 'PO')
    l = l.replace('Service Entry Sheet', 'SES')
    l = l.replace('creates invoice', 'Invoice')
    l = l.replace('creates debit memo', 'Debit Memo')
    labels[n] = l

nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8,
                        font_color='white', font_weight='bold')

el = {}
for u, v in G.edges():
    d = G[u][v]['d']
    if d > 0:
        el[(u, v)] = f"{d:.0f}h"
nx.draw_networkx_edge_labels(G, pos, el, ax=ax, font_size=6, font_color='#8b949e')

ax.set_title('SAP Process Graph - Top 20 DFG Transitions\n'
             'Node color: bottleneck severity (green=low, red=high) | Edge: avg duration',
             fontsize=14, color='white', pad=20, fontweight='bold')
ax.legend(handles=[
    mpatches.Patch(color=(0, 0.85, 0.15), label='Low bottleneck'),
    mpatches.Patch(color=(1, 0, 0.15), label='High bottleneck')
], loc='lower left', fontsize=10, facecolor='#161b22', edgecolor='#30363d',
    labelcolor='white')
ax.axis('off')
plt.tight_layout()
plt.savefig('process_graph_viz.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
print('[OK] Saved process_graph_viz.png')
