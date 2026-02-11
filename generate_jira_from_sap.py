"""
Generate Synthetic Jira Tickets from SAP Event Log.
Each SAP Purchase Order becomes a group of 1-5 engineering Jira tickets.
Input:  sap_event_log.csv
Output: synthetic_jira_data.csv
"""
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import timedelta

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# ─── Mapping SAP → Jira ───────────────────────────────────────────────────────

# SAP activities → Jira-like ticket types
SAP_TO_JIRA_TYPE = {
    'SRM: Created': 'Task',
    'SRM: Complete': 'Task',
    'SRM: Awaiting Approval': 'Task',
    'SRM: Document Completed': 'Task',
    'SRM: Ordered': 'Story',
    'Create Purchase Order Item': 'Story',
    'Record Goods Receipt': 'Story',
    'Record Invoice Receipt': 'Task',
    'Record Service Entry Sheet': 'Task',
    'Vendor creates invoice': 'Task',
    'Clear Invoice': 'Task',
    'Cancel Goods Receipt': 'Bug',
    'Cancel Invoice Receipt': 'Bug',
    'SRM: Transfer Failed (E.Sys.)': 'Bug',
    'SRM: Change was Transmitted': 'Task',
    'SRM: In Transfer to Execution Syst.': 'Task',
    'Change Quantity': 'Change Request',
    'Change Price': 'Change Request',
    'Change Delivery Indicator': 'Change Request',
    'Change Approval for Purchase Order': 'Change Request',
    'Delete Purchase Order Item': 'Bug',
    'Set Payment Block': 'Bug',
    'Remove Payment Block': 'Task',
}

# Spend area → Engineering domain
SPEND_TO_DOMAIN = {
    'Marketing': 'frontend',
    'Digital Marketing': 'frontend',
    'IT': 'backend',
    'Logistics': 'devops',
    'Professional Services': 'backend',
    'MRO': 'devops',
    'Travel': 'frontend',
    'Contingent Workforce': 'backend',
}

PRIORITIES = ['Critical', 'High', 'Medium', 'Low', 'Trivial']
STATUSES = ['Open', 'In Progress', 'In Review', 'Resolved', 'Closed']
ENGINEERS = [
    'Dev_Rahul', 'Dev_Sarah', 'Dev_Mike', 'Dev_Priya',
    'Dev_Alex', 'Dev_Chen', 'Dev_Emma', 'Dev_Omar'
]


def derive_priority(value_eur):
    """Map EUR value to ticket priority."""
    if value_eur >= 100000:
        return 'Critical'
    elif value_eur >= 50000:
        return 'High'
    elif value_eur >= 10000:
        return 'Medium'
    elif value_eur >= 1000:
        return 'Low'
    return 'Trivial'


def generate_jira_tickets(sap_csv='sap_event_log.csv'):
    print(f"[INFO] Reading {sap_csv}...")
    try:
        df = pd.read_csv(sap_csv)
    except FileNotFoundError:
        print(f"[ERROR] '{sap_csv}' not found. Run parse_sap_xes.py first!")
        return

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', utc=True)
    df = df.dropna(subset=['Timestamp'])

    # Group by Case_ID (each SAP PO item)
    groups = df.groupby('Case_ID')
    print(f"[INFO] Found {len(groups)} unique SAP purchase orders")

    jira_tickets = []
    ticket_counter = 1

    for case_id, group in groups:
        group = group.sort_values('Timestamp')
        first_event = group.iloc[0]
        last_event = group.iloc[-1]

        # Get trace-level attributes
        value_eur = first_event.get('Value_EUR', 0)
        if pd.isna(value_eur):
            value_eur = random.uniform(500, 50000)
        value_eur = float(value_eur)

        spend_area = str(first_event.get('Spend area text', ''))
        vendor = str(first_event.get('Vendor', ''))
        company = str(first_event.get('Company', ''))
        item_type = str(first_event.get('Item Type', ''))
        domain = SPEND_TO_DOMAIN.get(spend_area, random.choice(['frontend', 'backend', 'devops']))

        # Determine priority from value
        priority = derive_priority(value_eur)

        # Generate 1-5 Jira tickets per PO
        num_tickets = random.choices([1, 2, 3, 4, 5], weights=[0.3, 0.3, 0.2, 0.1, 0.1])[0]

        # Timeline: spread tickets across the PO lifecycle
        po_start = first_event['Timestamp']
        po_end = last_event['Timestamp']
        po_duration = (po_end - po_start).total_seconds()

        for i in range(num_tickets):
            ticket_id = f"JIRA-{ticket_counter:05d}"
            ticket_counter += 1

            # Stagger creation times across the PO lifecycle
            offset_seconds = random.uniform(0, max(po_duration, 3600))
            created = po_start + timedelta(seconds=offset_seconds)

            # Resolution time: 1 hour to 30 days
            resolution_hours = random.uniform(1, 720) / (PRIORITIES.index(priority) + 1)
            resolved = created + timedelta(hours=resolution_hours)

            # Pick a relevant SAP activity for this ticket
            activities_in_po = group['Activity'].tolist()
            sap_activity = random.choice(activities_in_po) if activities_in_po else 'SRM: Created'
            ticket_type = SAP_TO_JIRA_TYPE.get(sap_activity, 'Task')

            # Assign engineer (skill-match bias)
            assignee = random.choice(ENGINEERS)

            # Determine status
            if resolved <= pd.Timestamp.now(tz='UTC'):
                status = random.choice(['Resolved', 'Closed'])
            else:
                status = random.choice(['Open', 'In Progress', 'In Review'])

            # Ticket summary
            summaries = [
                f"Implement {domain} changes for {item_type} order",
                f"Review {spend_area} procurement workflow",
                f"Fix integration issue with vendor {vendor[-8:]}",
                f"Update {domain} module for PO processing",
                f"QA validation for {item_type} delivery",
                f"Deploy {domain} service update",
                f"Resolve {domain} pipeline bottleneck",
                f"Configure {item_type} automation rule",
            ]

            jira_tickets.append({
                'Case_ID': ticket_id,
                'SAP_PO_ID': case_id,
                'Summary': random.choice(summaries),
                'Type': ticket_type,
                'Priority': priority,
                'Status': status,
                'Assignee': assignee,
                'Domain': domain,
                'Timestamp': created,
                'Resolved': resolved if status in ('Resolved', 'Closed') else pd.NaT,
                'Value': round(value_eur / num_tickets, 2),  # Split PO value
                'Spend_Area': spend_area,
                'Vendor': vendor,
                'Company': company,
            })

    result_df = pd.DataFrame(jira_tickets)
    result_df = result_df.sort_values('Timestamp')
    result_df.to_csv('synthetic_jira_data.csv', index=False)

    print(f"\n[SUCCESS] Generated {len(result_df)} Jira tickets -> 'synthetic_jira_data.csv'")
    print(f"   Linked to {result_df['SAP_PO_ID'].nunique()} SAP purchase orders")
    print(f"   Priority distribution:")
    for p, count in result_df['Priority'].value_counts().items():
        print(f"      {p}: {count}")


if __name__ == '__main__':
    generate_jira_tickets()
