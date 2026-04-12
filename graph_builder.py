"""
Phase 2A — Process Graph Builder
Converts process mining outputs into PyTorch Geometric graphs for GNN training.

Supports:
  - Homogeneous Data (legacy): build_graph()
  - HeteroData (v2):           build_hetero_graph()
  - Temporal Event Stream:     build_temporal_events()

Node types:
  - Activity nodes (~42): features from bottleneck report
  - Resource nodes (~628): features from resource utilization

Edge types:
  - Activity → Activity: directly-follows (from DFG)
  - Resource → Activity: resource performs activity (from event log)

Output:
  - process_graph.pt       (PyG Data object — legacy)
  - process_graph_hetero.pt (PyG HeteroData object — v2)

Note: bottleneck_score is EXCLUDED from input features to prevent
feature-label leakage. Labels are log-transformed to reduce skew.
"""

import json
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, HeteroData


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
    """Build node feature tensors for activities and resources.
    
    NOTE: bottleneck_score is EXCLUDED from features to prevent label leakage.
    Engineered features (log_freq, dur/freq ratio, value/event) are added instead.
    Activity and resource features are normalized SEPARATELY to preserve scale alignment.
    """
    print("[2/4] Building node features (leakage-free)...")

    # === ACTIVITY NODES (indices 0 to N_act-1) ===
    activity_names = []
    activity_features = []

    # Pre-compute activity-scope max values for ratio features
    all_freqs = [b.get('frequency', 0) for b in bottlenecks]
    all_durs = [b.get('avg_duration_hours', 0) for b in bottlenecks]
    max_act_freq = max(all_freqs) if all_freqs else 1
    max_act_dur = max(all_durs) if all_durs else 1

    for b in bottlenecks:
        activity_names.append(b['activity'])
        freq = b.get('frequency', 0)
        avg_dur = b.get('avg_duration_hours', 0)
        avg_val = b.get('avg_value_eur', 0)
        activity_features.append([
            freq,                                                      # raw frequency
            avg_dur,                                                   # avg duration
            b.get('median_duration_hours', 0),                         # median duration
            b.get('max_duration_hours', 0),                            # max duration
            b.get('total_duration_hours', 0),                          # total duration
            np.log1p(freq),                                            # log-scaled frequency (engineered)
            freq / max(max_act_freq, 1),                               # activity-scope freq ratio
            avg_dur / max(max_act_dur, 1),                             # activity-scope dur ratio
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

    # Normalize SEPARATELY per node type to preserve scale alignment
    act_tensor = torch.tensor(activity_features, dtype=torch.float32)
    for col in range(act_tensor.shape[1]):
        col_max = act_tensor[:, col].max()
        if col_max > 0:
            act_tensor[:, col] = act_tensor[:, col] / col_max

    res_tensor = torch.tensor(resource_features, dtype=torch.float32)
    for col in range(res_tensor.shape[1]):
        col_max = res_tensor[:, col].max()
        if col_max > 0:
            res_tensor[:, col] = res_tensor[:, col] / col_max

    # Combine into unified feature matrix
    feature_tensor = torch.cat([act_tensor, res_tensor], dim=0)

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
    # Label: raw bottleneck_score — features now have aligned activity-scope ratios
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


# =====================================================================
#  V2: Heterogeneous Graph Builder (HeteroData)
# =====================================================================

def build_hetero_graph():
    """Build a HeteroData graph with separate node types and edge types.
    
    Node types:
        'activity': 8 features (frequency, durations, bottleneck_score, values)
        'resource': 6 features (events_handled, unique_activities/cases, diversity, utilization, value)
    
    Edge types:
        ('activity', 'follows', 'activity'): DFG edges with [frequency, avg_duration]
        ('resource', 'performs', 'activity'): resource-activity edges with [frequency]
    """
    print("=" * 55)
    print("  Phase 2A-v2: Heterogeneous Graph Construction")
    print("=" * 55 + "\n")

    bottlenecks, resources, dfg, stats = load_mining_outputs()

    # === ACTIVITY NODE FEATURES (8-dim, NO bottleneck_score to prevent leakage) ===
    activity_names = []
    activity_features = []

    # Pre-compute activity-scope max for ratio features
    all_freqs = [b.get('frequency', 0) for b in bottlenecks]
    all_durs = [b.get('avg_duration_hours', 0) for b in bottlenecks]
    max_act_freq = max(all_freqs) if all_freqs else 1
    max_act_dur = max(all_durs) if all_durs else 1

    for b in bottlenecks:
        activity_names.append(b['activity'])
        freq = b.get('frequency', 0)
        avg_dur = b.get('avg_duration_hours', 0)
        avg_val = b.get('avg_value_eur', 0)
        activity_features.append([
            freq,                                                      # raw frequency
            avg_dur,                                                   # avg duration
            b.get('median_duration_hours', 0),                         # median duration
            b.get('max_duration_hours', 0),                            # max duration
            b.get('total_duration_hours', 0),                          # total duration
            np.log1p(freq),                                            # log-scaled frequency (engineered)
            freq / max(max_act_freq, 1),                               # activity-scope freq ratio
            avg_dur / max(max_act_dur, 1),                             # activity-scope dur ratio
        ])
    n_activities = len(activity_names)

    # === RESOURCE NODE FEATURES (6-dim, no padding) ===
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
        ])
    n_resources = len(resource_names)

    # Normalize features per-column
    act_tensor = torch.tensor(activity_features, dtype=torch.float32)
    for col in range(act_tensor.shape[1]):
        col_max = act_tensor[:, col].max()
        if col_max > 0:
            act_tensor[:, col] /= col_max

    res_tensor = torch.tensor(resource_features, dtype=torch.float32)
    for col in range(res_tensor.shape[1]):
        col_max = res_tensor[:, col].max()
        if col_max > 0:
            res_tensor[:, col] /= col_max

    print(f"[2/4] Nodes: {n_activities} activity ({act_tensor.shape[1]}-dim), "
          f"{n_resources} resource ({res_tensor.shape[1]}-dim)")

    # === EDGES ===
    act_to_idx = {name: i for i, name in enumerate(activity_names)}
    res_to_idx = {name: i for i, name in enumerate(resource_names)}

    # DFG edges: activity → activity
    dfg_src, dfg_dst, dfg_attr = [], [], []
    for edge in dfg['edges']:
        src, tgt = edge['source'], edge['target']
        if src in act_to_idx and tgt in act_to_idx:
            dfg_src.append(act_to_idx[src])
            dfg_dst.append(act_to_idx[tgt])
            dfg_attr.append([edge['frequency'], edge['avg_duration_hours']])

    # Resource → activity edges
    print("   Loading event log for resource-activity edges...")
    df = pd.read_csv('sap_event_log.csv', usecols=['Activity', 'Resource'])
    ra_stats = df.groupby(['Resource', 'Activity']).size().reset_index(name='frequency')

    ra_src, ra_dst, ra_attr = [], [], []
    for _, row in ra_stats.iterrows():
        res, act = row['Resource'], row['Activity']
        if res in res_to_idx and act in act_to_idx:
            ra_src.append(res_to_idx[res])
            ra_dst.append(act_to_idx[act])
            ra_attr.append([row['frequency']])

    # Normalize edge features
    dfg_edge_attr = torch.tensor(dfg_attr, dtype=torch.float32) if dfg_attr else torch.zeros((0, 2))
    for col in range(dfg_edge_attr.shape[1]):
        col_max = dfg_edge_attr[:, col].max()
        if col_max > 0:
            dfg_edge_attr[:, col] /= col_max

    ra_edge_attr = torch.tensor(ra_attr, dtype=torch.float32) if ra_attr else torch.zeros((0, 1))
    if ra_edge_attr.numel() > 0:
        col_max = ra_edge_attr[:, 0].max()
        if col_max > 0:
            ra_edge_attr[:, 0] /= col_max

    print(f"[3/4] Edges: {len(dfg_src)} follows, {len(ra_src)} performs")

    # === BUILD HETERODATA ===
    data = HeteroData()
    data['activity'].x = act_tensor
    data['resource'].x = res_tensor

    data['activity', 'follows', 'activity'].edge_index = torch.tensor(
        [dfg_src, dfg_dst], dtype=torch.long
    ) if dfg_src else torch.zeros((2, 0), dtype=torch.long)
    data['activity', 'follows', 'activity'].edge_attr = dfg_edge_attr

    data['resource', 'performs', 'activity'].edge_index = torch.tensor(
        [ra_src, ra_dst], dtype=torch.long
    ) if ra_src else torch.zeros((2, 0), dtype=torch.long)
    data['resource', 'performs', 'activity'].edge_attr = ra_edge_attr

    # Training labels (raw bottleneck score — features aligned with activity-scope ratios)
    data['activity'].y = torch.tensor(
        [b.get('bottleneck_score', 0) for b in bottlenecks],
        dtype=torch.float32
    )

    metadata = {
        'activity_names': activity_names,
        'resource_names': resource_names,
        'n_activities': n_activities,
        'n_resources': n_resources,
        'activity_features': act_tensor.shape[1],
        'resource_features': res_tensor.shape[1],
        'dfg_edge_features': dfg_edge_attr.shape[1] if len(dfg_src) > 0 else 2,
        'ra_edge_features': ra_edge_attr.shape[1] if len(ra_src) > 0 else 1,
        'optimization_score': stats.get('optimization_score', 0),
    }

    torch.save({'graph': data, 'metadata': metadata}, 'process_graph_hetero.pt')

    print(f"\n[4/4] HeteroData graph built and saved!")
    print(f"   Activity nodes: {n_activities} × {act_tensor.shape[1]}")
    print(f"   Resource nodes: {n_resources} × {res_tensor.shape[1]}")
    print(f"   DFG edges: {len(dfg_src)}")
    print(f"   R→A edges: {len(ra_src)}")
    print(f"\n[OK] Saved 'process_graph_hetero.pt'")

    return data, metadata


# =====================================================================
#  Temporal Event Stream Builder (for TGN)
# =====================================================================

def build_temporal_events(sap_csv='sap_event_log.csv', max_events=50000):
    """Build a time-ordered event stream for Temporal Graph Network training.
    
    Each event: (timestamp_seconds, source_node_idx, target_node_idx,
                 event_type, features)
    
    Returns:
        events: list of dicts with keys:
            ts, src_type, src_idx, dst_type, dst_idx, features
        metadata: activity/resource name mappings
    """
    print("[TGN] Building temporal event stream...")

    # Load bottleneck report for activity names
    with open('bottleneck_report.json', 'r') as f:
        report = json.load(f)
    activity_names = [b['activity'] for b in report['bottlenecks']]
    act_to_idx = {name: i for i, name in enumerate(activity_names)}

    with open('bottleneck_report.json', 'r') as f:
        report = json.load(f)
    resource_names = [r['resource'] for r in report['resources']]
    res_to_idx = {name: i for i, name in enumerate(resource_names)}

    # Load event log
    df = pd.read_csv(sap_csv, usecols=['Case_ID', 'Activity', 'Resource', 'Timestamp', 'Value'],
                     nrows=max_events)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp']).sort_values('Timestamp')

    # Convert timestamps to relative seconds from first event
    t0 = df['Timestamp'].min()
    df['ts'] = (df['Timestamp'] - t0).dt.total_seconds()

    events = []
    for _, row in df.iterrows():
        act = row['Activity']
        res = row['Resource']
        if act not in act_to_idx or res not in res_to_idx:
            continue

        events.append({
            'ts': row['ts'],
            'src_type': 'resource',
            'src_idx': res_to_idx[res],
            'dst_type': 'activity',
            'dst_idx': act_to_idx[act],
            'features': [row.get('Value', 0) / 1e6],  # normalized value
        })

    print(f"[TGN] Built {len(events)} temporal events from {len(df)} rows")
    return events, {
        'activity_names': activity_names,
        'resource_names': resource_names,
        'n_activities': len(activity_names),
        'n_resources': len(resource_names),
    }


if __name__ == '__main__':
    # Build both graph types
    print("\n" + "#" * 55)
    print("  Building Legacy Homogeneous Graph")
    print("#" * 55)
    build_graph()

    print("\n" + "#" * 55)
    print("  Building Heterogeneous Graph")
    print("#" * 55)
    build_hetero_graph()
