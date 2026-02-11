"""
Unify SAP + Jira + Teams datasets into a single master CSV.
Input:  sap_event_log.csv, synthetic_jira_data.csv, synthetic_teams_data.csv
Output: unified_master.csv
"""
import pandas as pd
import sys


def unify():
    print("=" * 55)
    print("  Dataset Unification Pipeline")
    print("=" * 55)

    # ─── 1. Load SAP ───────────────────────────────────────
    try:
        sap = pd.read_csv('sap_event_log.csv')
        sap['Source'] = 'SAP'
        print(f"[OK] SAP events:   {len(sap):>8,} rows")
    except FileNotFoundError:
        print("[ERROR] sap_event_log.csv not found!")
        return

    # ─── 2. Load Jira ──────────────────────────────────────
    try:
        jira = pd.read_csv('synthetic_jira_data.csv')
        jira['Source'] = 'Jira'
        print(f"[OK] Jira tickets: {len(jira):>8,} rows")
    except FileNotFoundError:
        print("[ERROR] synthetic_jira_data.csv not found!")
        return

    # ─── 3. Load Teams ─────────────────────────────────────
    try:
        teams = pd.read_csv('synthetic_teams_data.csv')
        teams['Source'] = 'Teams'
        print(f"[OK] Teams msgs:   {len(teams):>8,} rows")
    except FileNotFoundError:
        print("[ERROR] synthetic_teams_data.csv not found!")
        return

    # ─── 4. Normalize columns ──────────────────────────────
    # Ensure all DataFrames have common columns for stacking
    common_cols = ['Case_ID', 'Activity', 'Timestamp', 'Resource', 'Source']

    # SAP: rename columns if needed
    if 'Case_ID' not in sap.columns and 'Case ID' in sap.columns:
        sap = sap.rename(columns={'Case ID': 'Case_ID'})

    # Ensure Timestamp columns are datetime
    for name, df in [('SAP', sap), ('Jira', jira), ('Teams', teams)]:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', utc=True)

    # ─── 5. Build unified event log ────────────────────────
    # SAP events
    sap_events = sap[['Case_ID', 'Activity', 'Timestamp', 'Resource', 'Source']].copy()
    if 'Value_EUR' in sap.columns:
        sap_events['Value'] = sap['Value_EUR']
    else:
        sap_events['Value'] = 0

    # For SAP, also carry SAP_PO_ID = Case_ID
    sap_events['SAP_PO_ID'] = sap_events['Case_ID']
    sap_events['Sentiment_Score'] = None

    # Jira events (ticket lifecycle = one "event" per ticket)
    jira_events = jira[['Case_ID', 'Timestamp', 'Source']].copy()
    jira_events['Activity'] = 'Jira: ' + jira['Status'].astype(str)
    jira_events['Resource'] = jira['Assignee']
    jira_events['Value'] = jira['Value']
    jira_events['SAP_PO_ID'] = jira['SAP_PO_ID']
    jira_events['Sentiment_Score'] = None

    # Teams events
    teams_events = teams[['Case_ID', 'Activity', 'Timestamp', 'Resource', 'Source']].copy()
    teams_events['Value'] = 0
    teams_events['SAP_PO_ID'] = teams.get('SAP_PO_ID', '')
    teams_events['Sentiment_Score'] = teams['Sentiment_Score']

    # ─── 6. Stack & sort ───────────────────────────────────
    unified = pd.concat([sap_events, jira_events, teams_events], ignore_index=True)
    unified = unified.sort_values('Timestamp').reset_index(drop=True)

    # ─── 7. Save ───────────────────────────────────────────
    unified.to_csv('unified_master.csv', index=False)

    print(f"\n[SUCCESS] Unified master dataset: {len(unified):,} events -> 'unified_master.csv'")
    print(f"   By source:")
    for src, count in unified['Source'].value_counts().items():
        print(f"      {src}: {count:,}")
    print(f"   Unique SAP POs: {unified['SAP_PO_ID'].nunique():,}")
    print(f"   Unique Jira tickets: {unified[unified['Source']=='Jira']['Case_ID'].nunique():,}")
    print(f"   Date range: {unified['Timestamp'].min()} to {unified['Timestamp'].max()}")

    # ─── 8. Also save Jira-specific CSV for training ───────
    # The PPO agent needs a simple (Case ID, Value, Priority) CSV
    if 'Priority' in jira.columns:
        training = jira[['Case_ID', 'Value', 'Priority']].copy()
        training = training.rename(columns={'Case_ID': 'Case ID'})
        training.to_csv('training_data.csv', index=False)
        print(f"\n[OK] Training data: {len(training):,} rows -> 'training_data.csv'")


if __name__ == '__main__':
    unify()
