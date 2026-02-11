import simpy
import pandas as pd
import numpy as np
import json
from collections import defaultdict

class DigitalTwin:
    def __init__(self, event_log_path='sap_event_log.csv', stats_path='process_stats.json'):
        self.env = simpy.Environment()
        self.log_path = event_log_path
        self.stats_path = stats_path
        self.data = None
        self.resource_pools = {}  # {activity: simpy.FilterStore or Resource}
        self.resources = {}       # {resource_id: simpy.Resource}
        self.case_traces = {}     # {case_id: [(activity, duration, resource_id), ...]}
        self.start_times = {}     # {case_id: start_timestamp}
        self.pools = {}           # {activity: [resource_ids]}
        
        # Simulation Metrics
        self.results = {
            "total_cases": 0,
            "total_cycle_time": 0,
            "blocked_time": 0,
            "resource_utilization": defaultdict(float)
        }
        
    def load_data(self):
        print(f"[INFO] Loading Digital Twin data from {self.log_path}...")
        try:
            df = pd.read_csv(self.log_path, parse_dates=['Timestamp'], dtype={'Case_ID': str, 'Activity': str, 'Resource': str})
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}")
            return
        # Sort by Case then Time
        df = df.sort_values(['Case_ID', 'Timestamp'])
        
        # Vectorized Duration Calculation
        print("[INFO] Calculating durations...")
        df['next_ts'] = df.groupby('Case_ID')['Timestamp'].shift(-1)
        df['duration'] = (df['next_ts'] - df['Timestamp']).dt.total_seconds()
        df['duration'] = df['duration'].fillna(1800) # Default 30 mins for last event
        df['duration'] = df['duration'].clip(upper=28800) # Cap at 8 hours
        
        self.data = df
        
        # Build traces
        print("[INFO] Building simulation traces (Vectorized)...")
        
        # Group to Dict
        # This is much faster than iterating
        self.start_times = df.groupby('Case_ID')['Timestamp'].first().to_dict()
        
        # Create traces dictionary: Case_ID -> List of dicts
        # Only keep necessary columns to save memory
        traces_df = df[['Case_ID', 'Activity', 'Resource', 'duration']]
        
        # Groupby apply is still slowish for 12k groups, but faster than loop.
        # Alternative: Iterate fast.
        self.case_traces = {}
        for case_id, group in traces_df.groupby('Case_ID'):
            self.case_traces[case_id] = group.to_dict('records')
            
        print(f"[INFO] Prepared {len(self.case_traces)} traces.")
        
        # Identify Resource-Activity Mapping (Who CAN do what)
        self.activity_resources = defaultdict(set)
        for _, row in df.iterrows():
            self.activity_resources[row['Activity']].add(row['Resource'])
            
    def configure_resources(self, override_mapping=None):
        """
        Setup SimPy resources.
        override_mapping: { 'Activity_Name': ['user_001', 'user_002'] } 
        """
        self.env = simpy.Environment() # Reset environment
        self.resources = {}
        self.results = {
            "total_cases": 0,
            "total_cycle_time": 0,
            "blocked_time": 0,
            "resource_utilization": defaultdict(float)
        }
        
        # 1. Create all unique resources as SimPy resources (Capacity=1)
        if self.data is not None:
            unique_users = set(self.data['Resource'].unique())
            # Add any new users from overrides if they don't exist in data
            if override_mapping:
                for users in override_mapping.values():
                    unique_users.update(users)
                    
            for u in unique_users:
                self.resources[u] = simpy.Resource(self.env, capacity=1)
            
            # 2. Define Activity Pools (Who is eligible)
            self.pools = {}
            for act in self.activity_resources:
                allowed = self.activity_resources[act]
                
                # Apply Overrides (The "Switching" Logic)
                if override_mapping and act in override_mapping:
                    allowed = set(override_mapping[act])
                    
                self.pools[act] = list(allowed)

    def run_simulation(self, max_cases=1000):
        print(f"[INFO] Running Simulation (Max {max_cases} cases)...")
        
        # Sort cases by start time
        sorted_cases = sorted(self.start_times.items(), key=lambda x: x[1])
        if not sorted_cases:
            print("[WARN] No cases to simulate.")
            return {}
            
        base_time = sorted_cases[0][1]
        
        count = 0
        for case_id, start_ts in sorted_cases:
            if count >= max_cases: break
            
            # Arrival delay relative to first case
            arrival_delay = (start_ts - base_time).total_seconds()
            self.env.process(self.process_case(case_id, arrival_delay))
            count += 1
            
        self.env.run()
        
        # Calculate summary
        avg_cycle = self.results['total_cycle_time'] / count if count else 0
        avg_block = self.results['blocked_time'] / count if count else 0
        
        throughput = count / (self.env.now / 3600) if self.env.now > 0 else 0
        
        return {
            "cases": count,
            "avg_cycle_time_hours": avg_cycle / 3600,
            "avg_blocked_hours": avg_block / 3600,
            "throughput_cases_per_hour": throughput,
            "total_duration_simulated_hours": self.env.now / 3600
        }
        
    def process_case(self, case_id, arrival_delay):
        yield self.env.timeout(arrival_delay)
        
        trace = self.case_traces[case_id]
        case_start = self.env.now
        
        for step in trace:
            act = step['Activity']
            duration = step['duration']
            
            # Request Resource
            allowed_users = self.pools.get(act, [])
            
            if not allowed_users:
                # No one can do it? Skip or assume default time
                yield self.env.timeout(duration)
                continue
            
            # Resource Selection Strategy:
            # 1. Try to find a free user
            chosen_user = None
            
            # Randomize order to prevent bias towards first user in list
            candidates = list(allowed_users)
            np.random.shuffle(candidates)
            
            for u in candidates:
                if u in self.resources and self.resources[u].count < self.resources[u].capacity:
                    chosen_user = u
                    break
            
            # 2. If all busy, pick random (Load Balancing) 
            if not chosen_user:
                 chosen_user = candidates[0] # Just pick first
                 
            if chosen_user in self.resources:
                res_obj = self.resources[chosen_user]
                
                req_start = self.env.now
                with res_obj.request() as req:
                    yield req # Wait for resource
                    wait = self.env.now - req_start
                    self.results['blocked_time'] += wait
                    
                    # Work
                    yield self.env.timeout(duration)
                    self.results['resource_utilization'][chosen_user] += duration
            else:
                yield self.env.timeout(duration)
                
        case_end = self.env.now
        self.results['total_cycle_time'] += (case_end - case_start)

if __name__ == "__main__":
    # Test Run
    twin = DigitalTwin()
    twin.load_data()
    
    print("\n--- BASELINE SIMULATION ---")
    twin.configure_resources() # Default config
    stats_base = twin.run_simulation(max_cases=500)
    print(json.dumps(stats_base, indent=2))
    
    print("\n--- WHAT-IF: Optimization ---")
    # Identify a bottleneck activity and add resources
    # E.g. "Create Purchase Order Item" often has bottleneck
    # We find who does it, and add 'user_002' to it
    
    # Get current users for an activity
    activity = 'Create Purchase Order Item'
    if activity in twin.activity_resources:
        current_users = list(twin.activity_resources[activity])
        print(f"Current users for {activity}: {len(current_users)}")
        
        # Add a "Super User" (or just reuse another user)
        # Let's verify if user_002 is in it. If not, add them.
        new_users = current_users + ['user_TEST_OPTIMIZED']
        
        override = {activity: new_users}
        twin.configure_resources(override_mapping=override)
        stats_new = twin.run_simulation(max_cases=500)
        print(json.dumps(stats_new, indent=2))
        
        # Improvement?
        base_wait = stats_base.get('avg_blocked_hours', 0)
        new_wait = stats_new.get('avg_blocked_hours', 0)
        print(f"\nImprovement in Wait Time: {base_wait - new_wait:.2f} hours")
