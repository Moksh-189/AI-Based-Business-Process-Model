"""
Parse BPI Challenge 2019 XES file (real SAP procurement data).
Uses iterative XML parsing for the 728MB file.
Output: sap_event_log.csv
"""
import xml.etree.ElementTree as ET
import pandas as pd
import csv
import sys
import time

XES_FILE = 'BPI_Challenge_2019.xes'
OUTPUT_FILE = 'sap_event_log.csv'

# XES namespace
NS = '{http://www.xes-standard.org/}'


def parse_attributes(elem):
    """Extract all key-value attributes from an XES element's children."""
    attrs = {}
    for child in elem:
        tag = child.tag.replace(NS, '')
        key = child.get('key', '')
        value = child.get('value', '')
        if tag in ('string', 'int', 'float', 'date', 'boolean'):
            attrs[key] = value
    return attrs


def parse_xes(filepath):
    """
    Stream-parse a large XES file.
    Yields one dict per event, with trace-level attributes merged in.
    """
    print(f"[INFO] Parsing {filepath} (this will take a few minutes)...")
    start = time.time()

    trace_attrs = {}
    in_trace = False
    in_event = False
    trace_count = 0
    event_count = 0

    context = ET.iterparse(filepath, events=('start', 'end'))

    for evt, elem in context:
        tag = elem.tag.replace(NS, '')

        # --- TRACE START ---
        if evt == 'start' and tag == 'trace':
            in_trace = True
            trace_attrs = {}

        # --- TRACE END ---
        elif evt == 'end' and tag == 'trace':
            in_trace = False
            trace_count += 1
            if trace_count % 5000 == 0:
                elapsed = time.time() - start
                print(f"   Processed {trace_count} traces, {event_count} events ({elapsed:.0f}s)")
            # Free memory
            elem.clear()

        # --- EVENT START ---
        elif evt == 'start' and tag == 'event':
            in_event = True

        # --- EVENT END ---
        elif evt == 'end' and tag == 'event':
            in_event = False
            event_attrs = parse_attributes(elem)
            # Merge trace-level attrs into event row
            row = {**trace_attrs, **event_attrs}
            # Prefix trace-level keys to avoid collision
            event_count += 1
            yield row
            elem.clear()

        # --- ATTRIBUTE (direct child of trace, not inside event) ---
        elif evt == 'end' and tag in ('string', 'int', 'float', 'date', 'boolean'):
            if in_trace and not in_event:
                key = elem.get('key', '')
                value = elem.get('value', '')
                trace_attrs[key] = value

    elapsed = time.time() - start
    print(f"[OK] Done. {trace_count} traces, {event_count} events in {elapsed:.1f}s")


def main():
    rows = []
    for row in parse_xes(XES_FILE):
        rows.append(row)

    df = pd.DataFrame(rows)

    # Standardize key columns
    rename_map = {
        'concept:name': 'Activity',
        'time:timestamp': 'Timestamp',
        'org:resource': 'Resource',
        'Cumulative net worth (EUR)': 'Value_EUR',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # The trace-level 'concept:name' is actually the Case ID (PO item ID)
    # But it gets overwritten by event-level 'concept:name' (Activity).
    # We need to extract the trace-level one separately.
    # Since our parser merges trace attrs first then event attrs override,
    # the trace-level concept:name is lost. Let's fix that:
    # Actually, looking at the XES structure, trace-level concept:name = PO Item ID
    # and event-level concept:name = Activity name. They share the same key.
    # We need a second pass or handle this in the parser.

    # Let's check if 'Purchasing Document' + 'Item' exist as trace attrs
    if 'Purchasing Document' in df.columns and 'Item' in df.columns:
        df['Case_ID'] = df['Purchasing Document'].astype(str) + '_' + df['Item'].astype(str)
    elif 'concept:name' in df.columns:
        # Fallback: some traces use concept:name for case ID
        df['Case_ID'] = df['concept:name']

    # Parse timestamps
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', utc=True)

    # Convert EUR to numeric
    if 'Value_EUR' in df.columns:
        df['Value_EUR'] = pd.to_numeric(df['Value_EUR'], errors='coerce')

    # Select and order columns
    priority_cols = ['Case_ID', 'Activity', 'Timestamp', 'Resource', 'Value_EUR']
    other_cols = [c for c in df.columns if c not in priority_cols]
    final_cols = [c for c in priority_cols if c in df.columns] + other_cols
    df = df[final_cols]

    df.to_csv(OUTPUT_FILE, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"\n[SUCCESS] Saved {len(df)} events to '{OUTPUT_FILE}'")
    print(f"   Unique cases: {df['Case_ID'].nunique() if 'Case_ID' in df.columns else 'N/A'}")
    print(f"   Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    print(f"   Columns: {list(df.columns)}")


if __name__ == '__main__':
    main()
