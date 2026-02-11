"""
Phase 1 — Object-Oriented Process Mining
Uses PM4Py to discover process models, detect bottlenecks, and measure conformance
from the SAP procurement event log.

Outputs:
  - bottleneck_report.json   (activity-level bottleneck analysis)
  - dfg_data.json            (directly-follows graph with frequencies & durations)
  - process_stats.json       (overall process statistics & resource utilization)
"""

import pandas as pd
import numpy as np
import json
import time
import warnings
warnings.filterwarnings('ignore')

# PM4Py imports
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.statistics.start_activities.log import get as start_act_get
from pm4py.statistics.end_activities.log import get as end_act_get


SAP_CSV = 'sap_event_log.csv'


def load_event_log(csv_path=SAP_CSV, sample_size=None):
    """Load SAP CSV and convert to PM4Py event log."""
    print(f"[1/6] Loading {csv_path}...")
    start = time.time()

    df = pd.read_csv(csv_path)

    if sample_size and len(df) > sample_size:
        # Sample by complete cases (not random rows)
        cases = df['Case_ID'].unique()
        sampled_cases = np.random.choice(cases, size=min(sample_size, len(cases)), replace=False)
        df = df[df['Case_ID'].isin(sampled_cases)]
        print(f"   Sampled {len(sampled_cases)} cases ({len(df)} events)")

    # Rename for PM4Py standard naming
    df = df.rename(columns={
        'Case_ID': 'case:concept:name',
        'Activity': 'concept:name',
        'Timestamp': 'time:timestamp',
        'Resource': 'org:resource',
    })

    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], errors='coerce', utc=True)
    df = df.dropna(subset=['time:timestamp'])
    df = df.sort_values(['case:concept:name', 'time:timestamp'])

    # Convert to PM4Py event log
    log = log_converter.apply(df, variant=log_converter.Variants.TO_EVENT_LOG)

    elapsed = time.time() - start
    print(f"   Loaded {len(df)} events, {df['case:concept:name'].nunique()} cases ({elapsed:.1f}s)")
    return log, df


def discover_dfg(log, df):
    """Discover Directly-Follows Graph with frequencies and performance."""
    print("[2/6] Discovering Directly-Follows Graph...")

    # Frequency DFG
    dfg_freq = dfg_discovery.apply(log, variant=dfg_discovery.Variants.FREQUENCY)

    # Performance DFG (durations between activities)
    dfg_perf = dfg_discovery.apply(log, variant=dfg_discovery.Variants.PERFORMANCE)

    # Start and end activities
    start_activities = start_act_get.get_start_activities(log)
    end_activities = end_act_get.get_end_activities(log)

    # Build serialisable DFG data
    dfg_data = {
        "edges": [],
        "start_activities": dict(start_activities),
        "end_activities": dict(end_activities),
    }

    for (src, tgt), freq in dfg_freq.items():
        duration = dfg_perf.get((src, tgt), 0)
        if isinstance(duration, (int, float)):
            duration_hours = round(duration / 3600, 2)
        else:
            duration_hours = 0
        dfg_data["edges"].append({
            "source": src,
            "target": tgt,
            "frequency": int(freq),
            "avg_duration_hours": duration_hours
        })

    # Sort by frequency descending
    dfg_data["edges"].sort(key=lambda x: x["frequency"], reverse=True)

    print(f"   Found {len(dfg_data['edges'])} edges in DFG")
    print(f"   Top 5 transitions:")
    for edge in dfg_data["edges"][:5]:
        print(f"      {edge['source']} -> {edge['target']}: "
              f"{edge['frequency']}x, avg {edge['avg_duration_hours']}h")

    return dfg_data, dfg_freq


def detect_bottlenecks(log, df):
    """Identify activities with the longest processing/wait times."""
    print("[3/6] Detecting bottlenecks...")

    # Calculate activity durations manually from event timestamps
    df_sorted = df.sort_values(['case:concept:name', 'time:timestamp'])
    df_sorted['next_timestamp'] = df_sorted.groupby('case:concept:name')['time:timestamp'].shift(-1)
    df_sorted['duration_seconds'] = (
        df_sorted['next_timestamp'] - df_sorted['time:timestamp']
    ).dt.total_seconds()

    # Per-activity stats
    activity_stats = df_sorted.groupby('concept:name').agg(
        frequency=('concept:name', 'count'),
        avg_duration_hours=('duration_seconds', lambda x: round(x.mean() / 3600, 2)),
        median_duration_hours=('duration_seconds', lambda x: round(x.median() / 3600, 2)),
        max_duration_hours=('duration_seconds', lambda x: round(x.max() / 3600, 2) if pd.notna(x.max()) else 0),
        total_duration_hours=('duration_seconds', lambda x: round(x.sum() / 3600, 2)),
    ).reset_index()
    activity_stats = activity_stats.rename(columns={'concept:name': 'activity'})

    # Add value info
    if 'Value_EUR' in df.columns:
        value_stats = df.groupby('concept:name')['Value_EUR'].agg(['mean', 'sum']).reset_index()
        value_stats.columns = ['activity', 'avg_value_eur', 'total_value_eur']
        value_stats['avg_value_eur'] = value_stats['avg_value_eur'].round(2)
        value_stats['total_value_eur'] = value_stats['total_value_eur'].round(2)
        activity_stats = activity_stats.merge(value_stats, on='activity', how='left')

    # Bottleneck score: weighted combination of duration and frequency
    max_dur = activity_stats['avg_duration_hours'].max()
    max_freq = activity_stats['frequency'].max()

    if max_dur > 0 and max_freq > 0:
        activity_stats['bottleneck_score'] = (
            0.6 * (activity_stats['avg_duration_hours'] / max_dur) +
            0.4 * (activity_stats['frequency'] / max_freq)
        ).round(3)
    else:
        activity_stats['bottleneck_score'] = 0

    activity_stats = activity_stats.sort_values('bottleneck_score', ascending=False)

    # Build report
    bottlenecks = activity_stats.to_dict(orient='records')

    # Clean NaN values for JSON
    for b in bottlenecks:
        for k, v in b.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                b[k] = 0

    print(f"   Top 5 bottlenecks:")
    for b in bottlenecks[:5]:
        score = b['bottleneck_score']
        dur = b['avg_duration_hours']
        freq = b['frequency']
        print(f"      [{score:.3f}] {b['activity']}: "
              f"avg {dur}h, {freq} occurrences")

    return bottlenecks


def check_conformance(log, df):
    """Discover a process model and check conformance via token replay."""
    print("[4/6] Checking conformance...")

    try:
        # Discover process model using Inductive Miner
        # PM4Py 2.7+ returns a ProcessTree, so we convert to Petri net
        process_tree = inductive_miner.apply(log)
        net, initial_marking, final_marking = pm4py.convert_to_petri_net(process_tree)

        # Token-based replay for conformance
        replayed = token_replay.apply(
            log, net, initial_marking, final_marking
        )

        # Calculate fitness
        fit_values = [t['trace_fitness'] for t in replayed if 'trace_fitness' in t]
        fitness = round(np.mean(fit_values), 4) if fit_values else 0

        # Detailed conformance
        fully_fitting = sum(1 for v in fit_values if v >= 0.999)
        partially_fitting = sum(1 for v in fit_values if 0.5 <= v < 0.999)
        non_fitting = sum(1 for v in fit_values if v < 0.5)

        conformance = {
            "fitness": fitness,
            "fully_fitting_traces": fully_fitting,
            "partially_fitting_traces": partially_fitting,
            "non_fitting_traces": non_fitting,
            "total_traces": len(fit_values),
            "fitness_percentage": round(fitness * 100, 2)
        }

        print(f"   Conformance fitness: {fitness:.4f} ({fitness*100:.1f}%)")
        print(f"   Fully fitting: {fully_fitting}/{len(fit_values)} traces")

    except Exception as e:
        print(f"   [WARN] Conformance check failed: {e}")
        conformance = {
            "fitness": 0,
            "error": str(e),
            "total_traces": 0
        }

    return conformance


def analyze_resources(df):
    """Analyze resource utilization and performance."""
    print("[5/6] Analyzing resource utilization...")

    resource_stats = df.groupby('org:resource').agg(
        events_handled=('org:resource', 'count'),
        unique_activities=('concept:name', 'nunique'),
        unique_cases=('case:concept:name', 'nunique'),
    ).reset_index()
    resource_stats = resource_stats.rename(columns={'org:resource': 'resource'})

    # Activity diversity (how many different activities each resource does)
    resource_stats['activity_diversity'] = (
        resource_stats['unique_activities'] / df['concept:name'].nunique()
    ).round(3)

    # Utilization proxy: events relative to the busiest resource
    max_events = resource_stats['events_handled'].max()
    resource_stats['utilization'] = (
        resource_stats['events_handled'] / max_events
    ).round(3)

    # Add value handled
    if 'Value_EUR' in df.columns:
        val = df.groupby('org:resource')['Value_EUR'].sum().reset_index()
        val.columns = ['resource', 'total_value_handled']
        val['total_value_handled'] = val['total_value_handled'].round(2)
        resource_stats = resource_stats.merge(val, on='resource', how='left')

    resource_stats = resource_stats.sort_values('events_handled', ascending=False)

    resources = resource_stats.to_dict(orient='records')

    # Clean NaN
    for r in resources:
        for k, v in r.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                r[k] = 0

    print(f"   {len(resources)} resources analyzed")
    print(f"   Top 5 by events handled:")
    for r in resources[:5]:
        print(f"      {r['resource']}: {r['events_handled']} events, "
              f"util={r['utilization']:.2f}, "
              f"{r['unique_activities']} activities")

    return resources


def build_process_stats(df, bottlenecks, conformance, resources, dfg_data):
    """Compile overall process statistics."""
    print("[6/6] Compiling process statistics...")

    total_cases = df['case:concept:name'].nunique()
    total_events = len(df)
    total_activities = df['concept:name'].nunique()
    total_resources = df['org:resource'].nunique()

    # Case duration stats
    case_times = df.groupby('case:concept:name')['time:timestamp'].agg(['min', 'max'])
    case_times['duration_hours'] = (case_times['max'] - case_times['min']).dt.total_seconds() / 3600
    case_durations = case_times['duration_hours'].dropna()

    # Top bottleneck
    top_bottleneck = bottlenecks[0] if bottlenecks else {}

    # Optimization score (0-100)
    # Based on: conformance, bottleneck severity, resource balance
    conformance_score = conformance.get('fitness', 0) * 40  # 0-40 points

    # Bottleneck severity (lower is better) — inverse of max bottleneck score
    top_bn_score = top_bottleneck.get('bottleneck_score', 1)
    bottleneck_points = (1 - top_bn_score) * 30  # 0-30 points

    # Resource balance (lower std dev of utilization is better)
    utils = [r['utilization'] for r in resources if r['resource'] != 'NONE']
    if utils:
        util_std = np.std(utils)
        resource_points = max(0, (1 - util_std * 2)) * 30  # 0-30 points
    else:
        resource_points = 0

    optimization_score = round(conformance_score + bottleneck_points + resource_points)
    optimization_score = max(0, min(100, optimization_score))

    stats = {
        "overview": {
            "total_cases": int(total_cases),
            "total_events": int(total_events),
            "total_activities": int(total_activities),
            "total_resources": int(total_resources),
            "date_range": {
                "start": str(df['time:timestamp'].min()),
                "end": str(df['time:timestamp'].max())
            }
        },
        "case_duration": {
            "avg_hours": round(float(case_durations.mean()), 2),
            "median_hours": round(float(case_durations.median()), 2),
            "min_hours": round(float(case_durations.min()), 2),
            "max_hours": round(float(case_durations.max()), 2),
        },
        "optimization_score": optimization_score,
        "score_breakdown": {
            "conformance_points": round(conformance_score, 1),
            "bottleneck_points": round(bottleneck_points, 1),
            "resource_balance_points": round(resource_points, 1),
        },
        "conformance": conformance,
        "top_bottleneck": {
            "activity": top_bottleneck.get('activity', 'N/A'),
            "avg_duration_hours": top_bottleneck.get('avg_duration_hours', 0),
            "frequency": top_bottleneck.get('frequency', 0),
            "bottleneck_score": top_bottleneck.get('bottleneck_score', 0),
        },
        "dfg_summary": {
            "total_edges": len(dfg_data['edges']),
            "start_activities": list(dfg_data['start_activities'].keys()),
            "end_activities": list(dfg_data['end_activities'].keys()),
        }
    }

    print(f"\n{'='*55}")
    print(f"  PROCESS MINING RESULTS")
    print(f"{'='*55}")
    print(f"  Cases: {total_cases:,}  |  Events: {total_events:,}")
    print(f"  Activities: {total_activities}  |  Resources: {total_resources}")
    print(f"  Avg case duration: {stats['case_duration']['avg_hours']:.1f} hours")
    print(f"  Conformance fitness: {conformance.get('fitness', 0)*100:.1f}%")
    print(f"  Top bottleneck: {stats['top_bottleneck']['activity']}")
    print(f"  OPTIMIZATION SCORE: {optimization_score}/100")
    print(f"    Conformance:      {conformance_score:.1f}/40")
    print(f"    Bottleneck:       {bottleneck_points:.1f}/30")
    print(f"    Resource balance: {resource_points:.1f}/30")
    print(f"{'='*55}")

    return stats


def main():
    print("="*55)
    print("  Phase 1: Object-Oriented Process Mining")
    print("="*55 + "\n")

    # Load full dataset (12,868 cases, ~1.5M events)
    log, df = load_event_log(SAP_CSV, sample_size=None)

    # Discover DFG
    dfg_data, dfg_freq = discover_dfg(log, df)

    # Detect bottlenecks
    bottlenecks = detect_bottlenecks(log, df)

    # Conformance checking
    conformance = check_conformance(log, df)

    # Resource analysis
    resources = analyze_resources(df)

    # Overall stats
    stats = build_process_stats(df, bottlenecks, conformance, resources, dfg_data)

    # Save outputs
    with open('bottleneck_report.json', 'w') as f:
        json.dump({"bottlenecks": bottlenecks, "resources": resources}, f, indent=2)
    print("\n[OK] Saved 'bottleneck_report.json'")

    with open('dfg_data.json', 'w') as f:
        json.dump(dfg_data, f, indent=2)
    print("[OK] Saved 'dfg_data.json'")

    with open('process_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("[OK] Saved 'process_stats.json'")

    print(f"\n[SUCCESS] Phase 1 complete. All outputs saved.")
    return stats


if __name__ == '__main__':
    main()
