"""
Phase 3C — Multi-Agent RL Environment (MARL)
PettingZoo ParallelEnv with two cooperative agents:
    - jira_agent: handles Logistics, Quality, Sales domains
    - sap_agent: handles Procurement, Finance domains

Each agent has its own backlog of tickets and action space,
but they share a cooperative reward based on total company throughput.

Uses independent policies (separate neural networks per agent).
"""

import functools
import numpy as np
import pandas as pd
import torch
import random
import json
from pettingzoo import ParallelEnv
from gymnasium import spaces


class MARLProcessEnv(ParallelEnv):
    """Multi-Agent Process Optimization Environment.

    Two agents cooperate to maximize total company throughput while
    each controlling only their own department's resources.
    """

    metadata = {"name": "process_optimization_v1", "render_modes": []}

    def __init__(self, df, embeddings_path='node_embeddings.pt'):
        super().__init__()

        self.df = df
        self.possible_agents = ["jira_agent", "sap_agent"]

        # Domain assignments — controls which tickets each agent sees
        self.agent_domains = {
            "jira_agent": ['Logistics', 'Quality', 'Sales'],
            "sap_agent": ['Procurement', 'Finance'],
        }

        # Per-agent config
        self.n_tickets = 5
        self.n_features = 8
        self.max_steps = 500

        # Load GNN embeddings
        try:
            emb_data = torch.load(embeddings_path, weights_only=False)
            self.embeddings = emb_data['embeddings'].numpy()
            self.activity_names = emb_data['activity_names']
            self.n_activities = emb_data['n_activities']
            self.act_to_idx = {name: i for i, name in enumerate(self.activity_names)}
        except Exception:
            self.embeddings = np.zeros((100, 32))
            self.activity_names = []
            self.n_activities = 0
            self.act_to_idx = {}

        # Load bottleneck scores
        try:
            with open('bottleneck_report.json', 'r') as f:
                report = json.load(f)
            self.activity_bottleneck = {
                b['activity']: b.get('bottleneck_score', 0)
                for b in report['bottlenecks']
            }
        except Exception:
            self.activity_bottleneck = {}

        self.max_value = df['Value'].max() if df['Value'].max() > 0 else 1.0

        # State tracking
        self.agents = []
        self.backlogs = {}
        self.current_step = 0
        self.active_df = None
        self.total_value_processed = {}
        self.bottlenecks_avoided = {}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(
            low=0, high=1,
            shape=(self.n_tickets, self.n_features),
            dtype=np.float32
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(self.n_tickets)

    def reset(self, seed=None, options=None):
        """Reset environment and return initial observations for both agents."""
        self.agents = list(self.possible_agents)
        self.current_step = 0
        self.active_df = self.df.sample(frac=1).reset_index(drop=True)

        # Separate backlogs per agent
        self.backlogs = {agent: [] for agent in self.agents}
        self.total_value_processed = {agent: 0 for agent in self.agents}
        self.bottlenecks_avoided = {agent: 0 for agent in self.agents}
        self._agent_df_idx = {agent: 0 for agent in self.agents}

        # Split data by domain (or mock if missing)
        self._agent_dfs = {}
        for agent in self.agents:
            domains = self.agent_domains[agent]
            if 'Domain' in self.active_df.columns:
                agent_df = self.active_df[self.active_df['Domain'].isin(domains)]
            else:
                # Mock domains: SAP agent gets first half, Jira agent gets second
                mid = len(self.active_df) // 2
                if agent == 'sap_agent':
                    agent_df = self.active_df.iloc[:mid]
                else:
                    agent_df = self.active_df.iloc[mid:]
            
            if len(agent_df) == 0:
                agent_df = self.active_df.sample(frac=0.5)
            self._agent_dfs[agent] = agent_df.reset_index(drop=True)

        # Fill initial backlogs
        for agent in self.agents:
            for _ in range(self.n_tickets):
                ticket = self._get_next_ticket(agent)
                if ticket:
                    self.backlogs[agent].append(ticket)

        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        """Execute one step for all agents simultaneously.

        Args:
            actions: dict mapping agent name → action (int)

        Returns:
            observations, rewards, terminations, truncations, infos
        """
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        # Track global throughput this step for cooperative reward
        step_values = {}

        for agent in self.agents:
            backlog = self.backlogs[agent]
            action = actions.get(agent, 0)

            if len(backlog) == 0:
                rewards[agent] = 0
                terminations[agent] = True
                truncations[agent] = False
                infos[agent] = {}
                continue

            valid_action = min(action, len(backlog) - 1)
            selected = backlog.pop(valid_action)

            sel_value = selected['Value']
            sel_bottleneck = selected['bottleneck_score']

            # === Individual reward (rank-based) ===
            all_values = [sel_value] + [t['Value'] for t in backlog]
            sorted_vals = sorted(all_values, reverse=True)
            rank = sorted_vals.index(sel_value)
            n = len(sorted_vals)

            if n > 1:
                indiv_reward = 3.0 - 5.0 * (rank / (n - 1))
            else:
                indiv_reward = 1.0

            # Value bonus
            indiv_reward += (sel_value / self.max_value) * 0.5

            # Bottleneck bonus
            if sel_bottleneck > 0.4:
                indiv_reward += 1.0 * sel_bottleneck

            # SLA penalty (cross-agent: penalize if ANY ticket in either backlog ages)
            max_wait = max(
                (t['Wait_Time_Simulated'] for t in backlog),
                default=0
            )
            if max_wait > 36:
                indiv_reward -= 0.3

            step_values[agent] = sel_value
            self.total_value_processed[agent] += sel_value
            if sel_bottleneck > 0.4:
                self.bottlenecks_avoided[agent] += 1

            # Refill backlog
            new_ticket = self._get_next_ticket(agent)
            if new_ticket:
                backlog.append(new_ticket)

            rewards[agent] = indiv_reward
            terminations[agent] = len(backlog) == 0
            truncations[agent] = self.current_step >= self.max_steps

        # === Cooperative reward component ===
        total_step_value = sum(step_values.values())
        coop_bonus = 0.5 * (total_step_value / self.max_value) if self.max_value > 0 else 0
        for agent in self.agents:
            if agent in rewards:
                rewards[agent] += coop_bonus

        self.current_step += 1

        # Check cross-agent SLA penalty
        for agent in self.agents:
            other_agents = [a for a in self.agents if a != agent]
            for other in other_agents:
                other_max_wait = max(
                    (t['Wait_Time_Simulated'] for t in self.backlogs.get(other, [])),
                    default=0
                )
                if other_max_wait > 48:  # escalated SLA breach
                    rewards[agent] -= 0.2

        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {
            agent: {
                'value_processed': self.total_value_processed.get(agent, 0),
                'bottlenecks_cleared': self.bottlenecks_avoided.get(agent, 0),
            }
            for agent in self.agents
        }

        # Remove terminated agents
        self.agents = [a for a in self.agents
                       if not terminations.get(a, False) and not truncations.get(a, False)]

        return observations, rewards, terminations, truncations, infos

    def _get_observation(self, agent):
        """Build GNN-enriched observation for a specific agent."""
        obs = np.zeros((self.n_tickets, self.n_features), dtype=np.float32)
        backlog = self.backlogs.get(agent, [])

        sorted_backlog = sorted(backlog, key=lambda x: x['Value'], reverse=True)

        for i, ticket in enumerate(sorted_backlog):
            if i >= self.n_tickets:
                break

            emb = ticket.get('embedding', np.zeros(32))
            emb_mean = float(np.mean(emb)) if len(emb) > 0 else 0
            emb_std = float(np.std(emb)) if len(emb) > 0 else 0
            emb_max = float(np.max(emb)) if len(emb) > 0 else 0

            obs[i] = [
                ticket['Value'] / self.max_value,
                ticket['Priority_Score'] / 5.0,
                min(ticket['Wait_Time_Simulated'] / 48.0, 1.0),
                ticket['bottleneck_score'],
                min(max(emb_mean, 0), 1),
                min(max(emb_std, 0), 1),
                min(max(emb_max, 0), 1),
                ticket.get('domain_encoded', 0),
            ]

        self.backlogs[agent] = sorted_backlog
        return obs

    def _get_next_ticket(self, agent):
        """Get next domain-filtered ticket for this agent."""
        agent_df = self._agent_dfs.get(agent)
        if agent_df is None or self._agent_df_idx[agent] >= len(agent_df):
            return None

        row = agent_df.iloc[self._agent_df_idx[agent]]
        self._agent_df_idx[agent] += 1

        val = row.get('Value', 0)
        priority = min(5, max(1, int((val / self.max_value) * 5) + 1)) if self.max_value > 0 else 3

        activity = row.get('Activity', '')
        if activity in self.act_to_idx:
            idx = self.act_to_idx[activity]
            embedding = self.embeddings[idx]
        else:
            embedding = np.zeros(32)

        bottleneck = self.activity_bottleneck.get(activity, 0)

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
