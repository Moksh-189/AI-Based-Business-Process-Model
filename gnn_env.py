"""
Phase 3A — GNN-Enhanced RL Environment
Gymnasium environment that uses GNN node embeddings as additional features
for smarter ticket prioritization.

Differences from custom_env.py:
  - Observation includes GNN embeddings per ticket (activity + resource info)
  - Reward accounts for bottleneck avoidance (from GNN predictions)
  - Richer state representation (8 features per ticket vs 3)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import random


class GNNEnhancedEnv(gym.Env):
    """
    GNN-Enhanced ticket prioritization environment.
    Uses node embeddings from Phase 2 to enrich the observation space.
    """

    def __init__(self, df, embeddings_path='node_embeddings.pt',
                 stats_path='process_stats.json'):
        super().__init__()

        self.df = df
        self.current_step = 0
        self.backlog = []

        # Load GNN embeddings
        emb_data = torch.load(embeddings_path, weights_only=False)
        self.embeddings = emb_data['embeddings'].numpy()  # (670, 32)
        self.activity_names = emb_data['activity_names']
        self.n_activities = emb_data['n_activities']

        # Build activity name -> embedding index mapping
        self.act_to_idx = {name: i for i, name in enumerate(self.activity_names)}

        # Calculate stats for normalization
        self.max_value = df['Value'].max() if df['Value'].max() > 0 else 1.0

        # Load bottleneck scores from embeddings for reward shaping
        import json
        with open('bottleneck_report.json', 'r') as f:
            report = json.load(f)
        self.activity_bottleneck = {
            b['activity']: b.get('bottleneck_score', 0)
            for b in report['bottlenecks']
        }

        # Observation: 5 tickets × 8 features each
        # Features: [value_norm, priority_norm, wait_norm, bottleneck_score,
        #            emb_mean, emb_std, emb_max, domain_encoded]
        self.n_tickets = 5
        self.n_features = 8
        self.action_space = spaces.Discrete(self.n_tickets)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.n_tickets, self.n_features),
            dtype=np.float32
        )

        # Metrics
        self.total_value_processed = 0
        self.bottlenecks_avoided = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.total_value_processed = 0
        self.bottlenecks_avoided = 0
        self.active_df = self.df.sample(frac=1).reset_index(drop=True)
        self.backlog = [self._get_next_ticket() for _ in range(self.n_tickets)]
        self.backlog = [t for t in self.backlog if t is not None]
        return self._get_observation(), {}

    def step(self, action):
        if len(self.backlog) == 0:
            return self._get_observation(), 0, True, False, {}

        valid_action = min(action, len(self.backlog) - 1)
        selected_ticket = self.backlog.pop(valid_action)

        selected_value = selected_ticket['Value']
        selected_bottleneck = selected_ticket['bottleneck_score']

        # === GNN-INFORMED REWARD SHAPING ===
        # Rank by value (higher = better)
        all_values = [selected_value] + [t['Value'] for t in self.backlog]
        sorted_values = sorted(all_values, reverse=True)
        rank = sorted_values.index(selected_value)

        # Base reward (same as v4 for compatibility)
        if rank == 0:
            reward = 10.0 + (selected_value / self.max_value) * 5.0
        elif rank == 1:
            reward = 2.0
        elif rank == 2:
            reward = 0.0
        else:
            reward = -5.0 * rank

        # BONUS: Reward for processing high-bottleneck activities
        # (clearing bottleneck activities improves overall throughput)
        if selected_bottleneck > 0.4:
            reward += 3.0 * selected_bottleneck  # Up to +1.8 bonus

        # BONUS: Penalty for letting high-value tickets age in queue
        max_wait = max((t['Wait_Time_Simulated'] for t in self.backlog), default=0)
        if max_wait > 36:  # If any ticket waited > 36 hours
            reward -= 1.0

        self.total_value_processed += selected_value
        if selected_bottleneck > 0.4:
            self.bottlenecks_avoided += 1

        # Refill backlog
        if self.current_step < len(self.active_df):
            new_ticket = self._get_next_ticket()
            if new_ticket:
                self.backlog.append(new_ticket)

        terminated = len(self.backlog) == 0
        truncated = False

        info = {
            'value_processed': self.total_value_processed,
            'bottlenecks_cleared': self.bottlenecks_avoided,
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        """Build GNN-enriched observation."""
        obs = np.zeros((self.n_tickets, self.n_features), dtype=np.float32)

        # Sort backlog by value (descending)
        sorted_backlog = sorted(self.backlog, key=lambda x: x['Value'], reverse=True)

        for i, ticket in enumerate(sorted_backlog):
            if i >= self.n_tickets:
                break

            # Get GNN embedding for this ticket's activity
            emb = ticket.get('embedding', np.zeros(32))
            emb_mean = float(np.mean(emb)) if len(emb) > 0 else 0
            emb_std = float(np.std(emb)) if len(emb) > 0 else 0
            emb_max = float(np.max(emb)) if len(emb) > 0 else 0

            obs[i] = [
                ticket['Value'] / self.max_value,                          # 0: value
                ticket['Priority_Score'] / 5.0,                            # 1: priority
                min(ticket['Wait_Time_Simulated'] / 48.0, 1.0),           # 2: wait time
                ticket['bottleneck_score'],                                # 3: bottleneck
                min(max(emb_mean, 0), 1),                                 # 4: emb mean
                min(max(emb_std, 0), 1),                                  # 5: emb std
                min(max(emb_max, 0), 1),                                  # 6: emb max
                ticket.get('domain_encoded', 0),                          # 7: domain
            ]

        self.backlog = sorted_backlog
        return obs

    def _get_next_ticket(self):
        """Get next ticket with GNN-enriched features."""
        if self.current_step >= len(self.active_df):
            return None

        row = self.active_df.iloc[self.current_step]
        self.current_step += 1

        val = row.get('Value', 0)
        priority = min(5, max(1, int((val / self.max_value) * 5) + 1)) if self.max_value > 0 else 3

        # Map activity to GNN embedding
        activity = row.get('Activity', '')
        if activity in self.act_to_idx:
            idx = self.act_to_idx[activity]
            embedding = self.embeddings[idx]
        else:
            embedding = np.zeros(32)

        # Get bottleneck score for this activity
        bottleneck = self.activity_bottleneck.get(activity, 0)

        # Encode domain as a number
        domain_map = {
            'Procurement': 0.2, 'Finance': 0.4, 'Logistics': 0.6,
            'Quality': 0.8, 'Sales': 1.0
        }
        domain = row.get('Domain', 'Unknown')
        domain_encoded = domain_map.get(domain, 0.5)

        return {
            'ID': row.get('Case_ID', row.get('Case ID', '')),
            'Activity': activity,
            'Value': val,
            'Priority_Score': priority,
            'Wait_Time_Simulated': random.randint(1, 48),
            'bottleneck_score': bottleneck,
            'embedding': embedding,
            'domain_encoded': domain_encoded,
        }
