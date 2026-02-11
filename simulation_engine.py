import simpy
import pandas as pd
import numpy as np
from datetime import datetime

class CompanySimulation:
    def __init__(self, jira_file, num_developers=5):
        self.env = simpy.Environment()
        self.jira_data = pd.read_csv(jira_file)
        
        # Resource Constraint: limited developer pool
        self.developer_team = simpy.Resource(self.env, capacity=num_developers)
        
        # Metrics Tracking
        self.total_revenue_processed = 0
        self.total_revenue_lost_in_queue = 0
        self.ticket_log = []

        self._prepare_data()
        
    def _prepare_data(self):
        print("[INFO] Preprocessing data...")
        
        # Strip any extra quotes that might be present from CSV
        self.jira_data['Timestamp'] = self.jira_data['Timestamp'].astype(str).str.strip().str.strip('"').str.strip("'")
        
        # Handle Resolved column if present
        if 'Resolved' in self.jira_data.columns:
            self.jira_data['Resolved'] = self.jira_data['Resolved'].astype(str).str.strip().str.strip('"').str.strip("'")
            self.jira_data['Resolved'] = pd.to_datetime(self.jira_data['Resolved'], errors='coerce')
        
        # Parse dates
        self.jira_data['Timestamp'] = pd.to_datetime(self.jira_data['Timestamp'], errors='coerce')
        self.jira_data = self.jira_data.dropna(subset=['Timestamp'])
        self.jira_data = self.jira_data.sort_values(by='Timestamp')
        
        self.start_time = self.jira_data['Timestamp'].iloc[0]
        
        # Calculate arrival relative to start
        self.jira_data['arrival_tick'] = (self.jira_data['Timestamp'] - self.start_time).dt.total_seconds()
        
        # Calculate actual work duration (Work Effort)
        # Estimate 'Actual Work' is ~30% of total lifespan
        if 'Resolved' in self.jira_data.columns:
            total_duration = (self.jira_data['Resolved'] - self.jira_data['Timestamp']).dt.total_seconds()
            self.jira_data['work_effort_seconds'] = total_duration.fillna(172800) * 0.3
        else:
            self.jira_data['work_effort_seconds'] = 172800 * 0.3  # Default 2 days * 30%

    def run(self):
        print(f"[INFO] Starting Simulation with {self.developer_team.capacity} Developers...")
        print("------------------------------------------------")
        
        # Get case ID column name
        case_col = 'Case_ID' if 'Case_ID' in self.jira_data.columns else 'Case ID'
        
        for _, row in self.jira_data.iterrows():
            arrival = row['arrival_tick']
            ticket_id = row[case_col]
            effort = row['work_effort_seconds']
            
            # Get value directly from the Value column
            value = row.get('Value', 0)
            if pd.isna(value):
                value = 0

            self.env.process(self.lifecycle_of_a_ticket(arrival, ticket_id, effort, value))
            
        self.env.run()
        self._print_summary()

    def lifecycle_of_a_ticket(self, arrival_time, ticket_id, work_effort, value):
        # 1. Wait for Ticket Arrival
        yield self.env.timeout(arrival_time - self.env.now)
        
        arrival_ts = self.env.now
        
        # 2. REQUEST A DEVELOPER (The Queue Phase)
        with self.developer_team.request() as request:
            yield request  # Wait until a developer is free
            
            # 3. Work Phase (The Processing)
            wait_time = self.env.now - arrival_ts
            
            # Log bottlenecks
            if wait_time > 86400:  # If waited > 1 day
                 print(f"[{self.format_time()}] BOTTLENECK: Ticket {ticket_id} "
                       f"waited {wait_time/3600:.1f} hours! (Value: ${value:,.0f})")
            
            yield self.env.timeout(work_effort)
            
            # 4. Done
            self.total_revenue_processed += value
            self.ticket_log.append({'id': ticket_id, 'wait_time': wait_time, 'value': value})

    def format_time(self):
        current_sim_time = self.start_time + pd.Timedelta(seconds=self.env.now)
        return current_sim_time.strftime('%Y-%m-%d')

    def _print_summary(self):
        print("\n[SIMULATION SUMMARY]")
        print("---------------------")
        processed = len(self.ticket_log)
        total_wait = sum(t['wait_time'] for t in self.ticket_log)
        avg_wait = total_wait / processed / 3600 if processed else 0
        
        print(f"Total Tickets Processed: {processed}")
        print(f"Total Revenue Delivered: ${self.total_revenue_processed:,.2f}")
        print(f"Average Wait Time (Queue): {avg_wait:.1f} hours")
        print("---------------------")

if __name__ == "__main__":
    sim = CompanySimulation(
        jira_file='sap_event_log.csv', 
        num_developers=3 
    )
    sim.run()