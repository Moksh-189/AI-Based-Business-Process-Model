"""
Worker Performance Data Generator
Generates synthetic worker profiles and assigns workers to existing Jira tickets.
Outputs: worker_profiles.csv, worker_assignments.csv
"""
import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# ─── Worker Profiles ───────────────────────────────────────────────────────────

SKILLS = ['frontend', 'backend', 'database', 'devops']
DESKS = ['Desk_Alpha', 'Desk_Beta', 'Desk_Gamma']

def generate_worker_profiles(num_workers=10):
    """Create synthetic worker profiles with skills, speed, and desk assignments."""
    workers = []
    for i in range(1, num_workers + 1):
        primary_skill = random.choice(SKILLS)
        secondary_skill = random.choice([s for s in SKILLS if s != primary_skill])

        workers.append({
            'Worker_ID': f'W{i:03d}',
            'Worker_Name': fake.first_name(),
            'Primary_Skill': primary_skill,
            'Secondary_Skill': secondary_skill,
            'Speed_Rating': round(random.uniform(0.4, 1.0), 2),   # 0.4=slow, 1.0=fast
            'Quality_Rating': round(random.uniform(0.5, 1.0), 2), # 0.5=low, 1.0=high
            'Experience_Years': random.randint(1, 15),
            'Desk': random.choice(DESKS),
            'Max_Concurrent_Tickets': random.randint(2, 5),
        })

    df = pd.DataFrame(workers)
    df.to_csv('worker_profiles.csv', index=False)
    print(f"[OK] Generated {len(df)} worker profiles → 'worker_profiles.csv'")
    return df


# ─── Worker–Ticket Assignments ─────────────────────────────────────────────────

def generate_worker_assignments(workers_df, jira_csv='synthetic_jira_data.csv'):
    """
    Assign each Jira ticket to a worker.  
    Workers who are faster finish sooner; workers whose skill matches the ticket
    domain also finish faster.  This creates learnable patterns for the GNN.
    """
    print(f"[INFO] Reading {jira_csv}...")
    try:
        jira = pd.read_csv(jira_csv)
    except FileNotFoundError:
        print(f"[ERROR] '{jira_csv}' not found. Run generate_jira_from_sap.py first!")
        return None

    jira['Timestamp'] = pd.to_datetime(jira['Timestamp'], errors='coerce')
    jira['Resolved']  = pd.to_datetime(jira['Resolved'], errors='coerce')
    jira = jira.dropna(subset=['Timestamp'])

    worker_ids = workers_df['Worker_ID'].tolist()
    speed_map   = dict(zip(workers_df['Worker_ID'], workers_df['Speed_Rating']))
    skill_map   = dict(zip(workers_df['Worker_ID'], workers_df['Primary_Skill']))
    quality_map = dict(zip(workers_df['Worker_ID'], workers_df['Quality_Rating']))

    # Simulate a rough "ticket domain" based on project name hash
    def ticket_domain(row):
        project = str(row.get('Project', ''))
        h = hash(project) % len(SKILLS)
        return SKILLS[h]

    assignments = []
    for idx, row in jira.iterrows():
        ticket_id   = row['Case ID']
        domain      = ticket_domain(row)

        # Pick a worker — weighted random, favouring skill match
        weights = []
        for wid in worker_ids:
            w = 1.0
            if skill_map[wid] == domain:
                w += 3.0        # strong preference for skill match
            w += speed_map[wid] # faster workers get picked slightly more
            weights.append(w)

        weights = np.array(weights)
        weights /= weights.sum()
        chosen_worker = np.random.choice(worker_ids, p=weights)

        # Simulate completion time (hours)
        base_hours = random.uniform(2, 120)               # raw effort
        speed_factor  = speed_map[chosen_worker]           # 0.4-1.0
        skill_bonus   = 0.7 if skill_map[chosen_worker] == domain else 1.0
        completion_hours = round(base_hours / speed_factor * skill_bonus, 1)

        assignments.append({
            'Case ID': ticket_id,
            'Worker_ID': chosen_worker,
            'Ticket_Domain': domain,
            'Completion_Hours': completion_hours,
            'Worker_Speed': speed_map[chosen_worker],
            'Worker_Quality': quality_map[chosen_worker],
            'Skill_Match': skill_map[chosen_worker] == domain,
        })

    df = pd.DataFrame(assignments)
    df.to_csv('worker_assignments.csv', index=False)
    print(f"[OK] Generated {len(df)} assignments → 'worker_assignments.csv'")
    return df


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("  Worker Performance Data Generator")
    print("=" * 55)

    workers = generate_worker_profiles(num_workers=10)
    assignments = generate_worker_assignments(workers)

    if assignments is not None:
        print("\n[SUMMARY]")
        print(f"  Workers : {len(workers)}")
        print(f"  Assignments: {len(assignments)}")
        print(f"  Skill-match rate: {assignments['Skill_Match'].mean()*100:.1f}%")
        print(f"  Avg completion: {assignments['Completion_Hours'].mean():.1f} hrs")
