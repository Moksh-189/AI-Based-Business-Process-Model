import pandas as pd
import random

def generate_dependencies(input_csv='synthetic_jira_data.csv'):
    print(f"[INFO] Reading {input_csv}...")
    
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print("[ERROR] Jira data not found.")
        return

    # Only look at tickets that have valid IDs
    tickets = df['Case ID'].unique().tolist()
    
    dependencies = []
    
    print(f"[INFO] Weaving dependency web for {len(tickets)} tickets...")

    # We will make ~15% of tickets have a dependency
    num_dependencies = int(len(tickets) * 0.15)

    for _ in range(num_dependencies):
        # Pick two random tickets
        # Logic: blocker (A) usually comes before blocked (B), but not always.
        ticket_A, ticket_B = random.sample(tickets, 2)
        
        # Prevent self-loops
        if ticket_A == ticket_B: continue
        
        # Structure: Ticket A BLOCKS Ticket B
        # This means B cannot finish until A is done.
        dependencies.append({
            'Blocker_ID': ticket_A,
            'Blocked_ID': ticket_B,
            'Type': 'Blocks'
        })

    # Save
    dep_df = pd.DataFrame(dependencies)
    dep_df.to_csv('synthetic_dependencies.csv', index=False)

    print("\n[SUCCESS]")
    print(f"Generated {len(dep_df)} dependency links.")
    print("Output saved to 'synthetic_dependencies.csv'")
    print("Your AI can now learn 'Critical Path' logic!")

if __name__ == "__main__":
    generate_dependencies()