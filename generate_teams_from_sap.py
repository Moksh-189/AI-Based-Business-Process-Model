"""
Generate Synthetic Microsoft Teams Chat Data from SAP Event Log.
Teams data is linked directly to SAP Purchase Orders (the anchor).
High-value / complex POs produce more chatter with lower sentiment.
Input:  sap_event_log.csv
Output: synthetic_teams_data.csv
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

# Resources (the people chatting in Teams)
DEVS = ['Dev_Rahul', 'Dev_Sarah', 'Dev_Mike', 'Dev_Priya',
        'Dev_Alex', 'Dev_Chen', 'Dev_Emma', 'Dev_Omar']
QAS = ['QA_Lead', 'QA_Tester_1', 'QA_Tester_2']
PMS = ['Product_Manager', 'Scrum_Master', 'Tech_Lead']
PROCUREMENT = ['Proc_Manager', 'Proc_Lead', 'Finance_Controller']
ALL_STAFF = DEVS + QAS + PMS + PROCUREMENT

ACTIVITY_TYPES = ['Message Sent', 'Mentioned', 'File Shared', 'Huddle Started',
                  'Thread Reply', 'Reaction Added']
ACTIVITY_WEIGHTS = [0.55, 0.15, 0.10, 0.05, 0.10, 0.05]


def generate_teams_data(sap_csv='sap_event_log.csv'):
    print(f"[INFO] Reading {sap_csv}...")
    try:
        df = pd.read_csv(sap_csv)
    except FileNotFoundError:
        print(f"[ERROR] '{sap_csv}' not found. Run parse_sap_xes.py first!")
        return

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', utc=True)
    df = df.dropna(subset=['Timestamp'])

    # Group by SAP Case_ID (purchase order items)
    groups = df.groupby('Case_ID')
    print(f"[INFO] Generating Teams chatter for {len(groups)} SAP purchase orders...")

    teams_data = []
    po_count = 0

    for case_id, group in groups:
        group = group.sort_values('Timestamp')
        first_event = group.iloc[0]
        last_event = group.iloc[-1]

        po_start = first_event['Timestamp']
        po_end = last_event['Timestamp']
        lifespan = po_end - po_start

        if lifespan.total_seconds() <= 0:
            lifespan = timedelta(hours=random.randint(1, 48))
            po_end = po_start + lifespan

        # Get PO attributes
        value_eur = float(first_event.get('Value_EUR', 0)) if pd.notna(first_event.get('Value_EUR', None)) else random.uniform(500, 50000)
        num_events = len(group)
        resources_involved = group['Resource'].nunique()

        # Determine chatter level based on PO complexity
        # High-value, many-event, many-resource POs are "noisy"
        is_noisy = (value_eur > 20000 or num_events > 10 or
                    resources_involved > 3 or random.random() > 0.8)

        if is_noisy:
            num_messages = random.randint(10, 40)
        else:
            num_messages = random.randint(2, 8)

        # Get SAP resource involved for sender bias
        sap_resources = group['Resource'].unique().tolist()

        for _ in range(num_messages):
            random_seconds = random.randint(0, int(lifespan.total_seconds()))
            msg_time = po_start + timedelta(seconds=random_seconds)

            # Sender — bias towards SAP resource or procurement staff
            if random.random() < 0.3 and sap_resources:
                sender = random.choice(sap_resources)
            elif random.random() < 0.5:
                sender = random.choice(PROCUREMENT)
            else:
                sender = random.choice(ALL_STAFF)

            activity_type = np.random.choice(ACTIVITY_TYPES, p=ACTIVITY_WEIGHTS)

            # Sentiment: noisy/high-value POs trend negative
            if is_noisy:
                sentiment = round(random.uniform(0.1, 0.55), 2)
            else:
                sentiment = round(random.uniform(0.5, 0.95), 2)

            teams_data.append({
                'Case_ID': case_id,       # Links directly to SAP PO
                'Activity': activity_type,
                'Timestamp': msg_time,
                'Resource': sender,
                'Sentiment_Score': sentiment,
                'Platform': 'Microsoft Teams',
            })

        po_count += 1
        if po_count % 50000 == 0:
            print(f"   Processed {po_count} POs, {len(teams_data)} messages so far...")

    teams_df = pd.DataFrame(teams_data)
    teams_df = teams_df.sort_values('Timestamp')
    teams_df.to_csv('synthetic_teams_data.csv', index=False)

    print(f"\n[SUCCESS] Generated {len(teams_df)} Teams interactions -> 'synthetic_teams_data.csv'")
    print(f"   Linked to {teams_df['Case_ID'].nunique()} SAP purchase orders")
    avg_sentiment = teams_df['Sentiment_Score'].mean()
    print(f"   Average sentiment: {avg_sentiment:.2f}")


if __name__ == '__main__':
    generate_teams_data()
