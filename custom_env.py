import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random

class JiraOptimizationEnv(gym.Env):
    """
    ULTRA-OPTIMIZED Environment for maximum AI performance.
    Aggressive reward shaping to teach AI to ALWAYS pick the highest value ticket.
    """
    def __init__(self, df):
        super(JiraOptimizationEnv, self).__init__()
        
        self.df = df
        self.current_step = 0
        self.backlog = []
        
        # Calculate max value for normalization
        self.max_value = df['Value'].max() if df['Value'].max() > 0 else 1.0
        
        # ACTION SPACE: Choose one of the top 5 tickets in the backlog
        self.action_space = spaces.Discrete(5)
        
        # OBSERVATION SPACE: Normalized 5x3 matrix (values between 0-1)
        self.observation_space = spaces.Box(low=0, high=1, shape=(5, 3), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.active_df = self.df.sample(frac=1).reset_index(drop=True)
        self.backlog = [self._get_next_ticket() for _ in range(5)]
        return self._get_observation(), {}

    def step(self, action):
        valid_action = min(action, len(self.backlog) - 1)
        selected_ticket = self.backlog.pop(valid_action)
        
        # === AGGRESSIVE REWARD SHAPING ===
        selected_value = selected_ticket['Value']
        
        # Find position of selected ticket by value (0 = highest value in backlog)
        all_values = [selected_value] + [t['Value'] for t in self.backlog]
        sorted_values = sorted(all_values, reverse=True)
        rank = sorted_values.index(selected_value)  # 0 = best, 4 = worst
        
        # MASSIVE rewards for picking #1, MASSIVE penalties for picking low
        if rank == 0:
            # Picked the BEST ticket! Huge reward
            reward = 10.0 + (selected_value / self.max_value) * 5.0
        elif rank == 1:
            # Second best - small positive
            reward = 2.0
        elif rank == 2:
            # Middle - neutral
            reward = 0.0
        else:
            # Picked a low-value ticket when better ones existed - PUNISH
            reward = -5.0 * rank
        
        # Refill backlog
        if self.current_step < len(self.active_df):
            new_ticket = self._get_next_ticket()
            if new_ticket:
                self.backlog.append(new_ticket)
        
        terminated = len(self.backlog) == 0
        truncated = False
        
        return self._get_observation(), reward, terminated, truncated, {}

    def _get_observation(self):
        """Constructs the SORTED observation - highest value first"""
        obs = np.zeros((5, 3), dtype=np.float32)
        
        # SORT backlog by value (descending) so AI learns position = value
        sorted_backlog = sorted(self.backlog, key=lambda x: x['Value'], reverse=True)
        
        for i, ticket in enumerate(sorted_backlog):
            if i >= 5: break
            obs[i] = [
                ticket['Value'] / self.max_value,
                ticket['Priority_Score'] / 5.0,
                min(ticket['Wait_Time_Simulated'] / 48.0, 1.0)
            ]
        
        # Update backlog order to match observation
        self.backlog = sorted_backlog
        
        return obs

    def _get_next_ticket(self):
        if self.current_step >= len(self.active_df):
            return None
            
        row = self.active_df.iloc[self.current_step]
        self.current_step += 1
        
        val = row.get('Value', 0)
        priority = min(5, max(1, int((val / self.max_value) * 5) + 1)) if self.max_value > 0 else 3
        
        return {
            'ID': row.get('Case ID'),
            'Value': val,
            'Priority_Score': priority,
            'Wait_Time_Simulated': random.randint(1, 48)
        }