"""
Phase 2A — Process Graph Builder
Converts process mining outputs into a PyTorch Geometric graph for GNN training.

Node types:
  - Activity nodes (~42): features from bottleneck report
  - Resource nodes (~628): features from resource utilization

Edge types:
  - Activity → Activity: directly-follows (from DFG)
  - Resource → Activity: resource performs activity (from event log)

Output: process_graph.pt (PyG Data object)
"""

import json
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data


def load_mining_outputs():
    """Load Phase 1 outputs."""
    print("[1/4] Loading process mining outputs...")

    with open('bottleneck_report.json', 'r') as f:
        report = json.load(f)

    with open('dfg_data.json', 'r') as f:
        dfg = json.load(f)

    with open('process_stats.json', 'r') as f:
        stats = json.load(f)

    bottlenecks = report['bottlenecks']
    resources = report['resources']

    print(f"   {len(bottlenecks)} activities, {len(resources)} resources, "
          f"{len(dfg['edges'])} DFG edges")

    return bottlenecks, resources, dfg, stats


def build_node_features(bottlenecks, resources):
    """Build node feature tensors for activities and resources."""
    print("[2/4] Building node features...")

    # === ACTIVITY NODES (indices 0 to N_act-1) ===
    activity_names = []
    activity_features = []

    for b in bottlenecks:
        activity_names.append(b['activity'])
        activity_features.append([
            b.get('frequency', 0),
            b.get('avg_duration_hours', 0),
            b.get('median_duration_hours', 0),
            b.get('max_duration_hours', 0),
            b.get('total_duration_hours', 0),
            b.get('bottleneck_score', 0),
            b.get('avg_value_eur', 0),
            b.get('total_value_eur', 0),
        ])

    n_activities = len(activity_names)

    # === RESOURCE NODES (indices N_act to N_act+N_res-1) ===
    resource_names = []
    resource_features = []

    for r in resources:
        resource_names.append(r['resource'])
        resource_features.append([
            r.get('events_handled', 0),
            r.get('unique_activities', 0),
            r.get('unique_cases', 0),
            r.get('activity_diversity', 0),
            r.get('utilization', 0),
            r.get('total_value_handled', 0),
            0,  # padding
            0,  # padding
        ])

    n_resources = len(resource_names)

    # Combine into unified feature matrix
    all_features = activity_features + resource_features
    feature_tensor = torch.tensor(all_features, dtype=torch.float32)

    # Normalize each feature column to [0, 1]
    for col in range(feature_tensor.shape[1]):
        col_max = feature_tensor[:, col].max()
        if col_max > 0:
            feature_tensor[:, col] = feature_tensor[:, col] / col_max

    # Node type labels: 0 = activity, 1 = resource
    node_types = torch.cat([
        torch.zeros(n_activities, dtype=torch.long),
        torch.ones(n_resources, dtype=torch.long)
    ])

    print(f"   {n_activities} activity nodes + {n_resources} resource nodes "
          f"= {n_activities + n_resources} total")
    print(f"   Feature dimension: {feature_tensor.shape[1]}")

    return feature_tensor, node_types, activity_names, resource_names


def build_edges(dfg, activity_names, resource_names, sap_csv='sap_event_log.csv'):
    """Build edge index and edge features."""
    print("[3/4] Building edges...")

    n_act = len(activity_names)
    act_to_idx = {name: i for i, name in enumerate(activity_names)}
    res_to_idx = {name: n_act + i for i, name in enumerate(resource_names)}

    edge_src = []
    edge_dst = []
    edge_features = []
    edge_types = []  # 0 = activity->activity, 1 = resource->activity

    # === DFG EDGES (Activity → Activity) ===
    dfg_count = 0
    for edge in dfg['edges']:
        src = edge['source']
        tgt = edge['target']
        if src in act_to_idx and tgt in act_to_idx:
            edge_src.append(act_to_idx[src])
            edge_dst.append(act_to_idx[tgt])
            edge_features.append([
                edge['frequency'],
                edge['avg_duration_hours'],
            ])
            edge_types.append(0)
            dfg_count += 1

    # === RESOURCE → ACTIVITY EDGES ===
    # Load SAP event log to get resource-activity relationships
    print("   Loading event log for resource-activity edges...")
    df = pd.read_csv(sap_csv, usecols=['Activity', 'Resource'])

    ra_stats = df.groupby(['Resource', 'Activity']).size().reset_index(name='frequency')

    ra_count = 0
    for _, row in ra_stats.iterrows():
        res = row['Resource']
        act = row['Activity']
        if res in res_to_idx and act in act_to_idx:
            edge_src.append(res_to_idx[res])
            edge_dst.append(act_to_idx[act])
            edge_features.append([
                row['frequency'],
                0,  # no duration for this edge type
            ])
            edge_types.append(1)
            ra_count += 1

    # Build tensors
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

    # Normalize edge features
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    for col in range(edge_attr.shape[1]):
        col_max = edge_attr[:, col].max()
        if col_max > 0:
            edge_attr[:, col] = edge_attr[:, col] / col_max

    edge_type_tensor = torch.tensor(edge_types, dtype=torch.long)

    print(f"   {dfg_count} activity->activity edges (DFG)")
    print(f"   {ra_count} resource->activity edges")
    print(f"   {dfg_count + ra_count} total edges")

    return edge_index, edge_attr, edge_type_tensor


def build_graph():
    """Main function: build and save the process graph."""
    print("=" * 55)
    print("  Phase 2A: Process Graph Construction")
    print("=" * 55 + "\n")

    # Load Phase 1 outputs
    bottlenecks, resources, dfg, stats = load_mining_outputs()

    # Build nodes
    x, node_types, activity_names, resource_names = build_node_features(
        bottlenecks, resources
    )

    # Build edges
    edge_index, edge_attr, edge_types = build_edges(
        dfg, activity_names, resource_names
    )

    # Build training labels for activity nodes
    # Label: bottleneck_score (regression target)
    activity_labels = torch.tensor(
        [b.get('bottleneck_score', 0) for b in bottlenecks],
        dtype=torch.float32
    )

    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_types=node_types,
        edge_types=edge_types,
        y=activity_labels,
        num_activity_nodes=len(activity_names),
        num_resource_nodes=len(resource_names),
    )

    # Save metadata for later use
    metadata = {
        'activity_names': activity_names,
        'resource_names': resource_names,
        'n_activities': len(activity_names),
        'n_resources': len(resource_names),
        'n_features': x.shape[1],
        'n_edge_features': edge_attr.shape[1],
        'optimization_score': stats.get('optimization_score', 0),
    }

    # Save
    torch.save({'graph': data, 'metadata': metadata}, 'process_graph.pt')

    print(f"\n[4/4] Graph built and saved!")
    print(f"   Nodes: {data.num_nodes}")
    print(f"   Edges: {data.num_edges}")
    print(f"   Node features: {data.x.shape}")
    print(f"   Edge features: {data.edge_attr.shape}")
    print(f"\n[OK] Saved 'process_graph.pt'")

    return data, metadata


if __name__ == '__main__':
    build_graph()
