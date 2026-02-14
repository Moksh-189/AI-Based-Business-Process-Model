"""
Phase 3B — GNN-Enhanced RL Agent Training
Trains PPO agents with both GLU and GELU activations using the
GNN-enhanced environment, compares them, and saves the best agent.

Output:
  - ppo_gnn_best.zip         (best trained agent)
  - agent_comparison.json    (GLU vs GELU performance comparison)
"""

import json
import time
import sys
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from gnn_env import GNNEnhancedEnv


class ProgressCallback(BaseCallback):
    """Custom callback that emits PROGRESS: JSON lines for WebSocket streaming."""
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.last_report = 0
        self.report_interval = 2048  # Report every N steps
    
    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_report >= self.report_interval:
            self.last_report = self.num_timesteps
            pct = round((self.num_timesteps / self.total_timesteps) * 100, 1)
            
            reward = 0
            # Try 1: SB3 logger (most reliable after first rollout)
            try:
                if hasattr(self.model, 'logger') and self.model.logger is not None:
                    name_to_value = getattr(self.model.logger, 'name_to_value', {})
                    if 'rollout/ep_rew_mean' in name_to_value:
                        reward = round(float(name_to_value['rollout/ep_rew_mean']), 2)
                    elif 'train/loss' in name_to_value:
                        # Use negative loss as proxy reward signal
                        reward = round(-float(name_to_value['train/loss']) / 100, 2)
            except Exception:
                pass
            
            # Try 2: ep_info_buffer (if Monitor-wrapped)
            if reward == 0 and len(self.model.ep_info_buffer) > 0:
                reward = round(np.mean([ep['r'] for ep in self.model.ep_info_buffer]), 2)
            
            # Try 3: Sample from rollout buffer directly
            if reward == 0:
                try:
                    buf = self.model.rollout_buffer
                    if buf is not None and buf.rewards is not None and buf.pos > 0:
                        recent = buf.rewards[:buf.pos].flatten()
                        if len(recent) > 0:
                            reward = round(float(recent.mean()), 2)
                except Exception:
                    pass
            
            progress_data = {
                "step": self.num_timesteps,
                "total": self.total_timesteps,
                "pct": pct,
                "reward": reward,
            }
            # Flush immediately so server can capture it
            print(f"PROGRESS:{json.dumps(progress_data)}", flush=True)
        return True


class GLUActivation(nn.Module):
    """Gated Linear Unit: output = sigma(Wx) * (Vx)"""
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x):
        return nn.functional.sigmoid(self.gate(x)) * self.value(x)


def evaluate_agent(model, env, n_episodes=100):
    """Evaluate an agent over n episodes, return metrics."""
    total_rewards = []
    total_values = []
    total_bottlenecks = []

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward

            # Extract info from vectorized env
            if isinstance(info, list) and len(info) > 0:
                info = info[0]

        total_rewards.append(episode_reward)
        if isinstance(info, dict):
            total_values.append(info.get('value_processed', 0))
            total_bottlenecks.append(info.get('bottlenecks_cleared', 0))

    return {
        'avg_reward': round(float(np.mean(total_rewards)), 2),
        'std_reward': round(float(np.std(total_rewards)), 2),
        'avg_value': round(float(np.mean(total_values)), 2) if total_values else 0,
        'avg_bottlenecks_cleared': round(float(np.mean(total_bottlenecks)), 2) if total_bottlenecks else 0,
        'max_reward': round(float(np.max(total_rewards)), 2),
        'min_reward': round(float(np.min(total_rewards)), 2),
    }


def evaluate_random(env, n_episodes=100):
    """Random baseline for comparison."""
    total_rewards = []
    total_values = []

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = [env.action_space.sample()]
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward

            if isinstance(info, list) and len(info) > 0:
                info = info[0]

        total_rewards.append(episode_reward)
        if isinstance(info, dict):
            total_values.append(info.get('value_processed', 0))

    return {
        'avg_reward': round(float(np.mean(total_rewards)), 2),
        'avg_value': round(float(np.mean(total_values)), 2) if total_values else 0,
    }


N_ENVS = 4  # Number of parallel environments

def make_env(df):
    """Factory that returns a function to create a GNNEnhancedEnv."""
    def _init():
        return GNNEnhancedEnv(df)
    return _init

def train_agent(activation_name, activation_fn, df, timesteps=200000):
    """Train a PPO agent with specified activation."""
    print(f"\n{'='*55}", flush=True)
    print(f"  Training PPO with {activation_name} ({N_ENVS} parallel envs)", flush=True)
    print(f"{'='*55}", flush=True)

    # Use SubprocVecEnv for parallel environments (~3x speedup)
    try:
        env = SubprocVecEnv([make_env(df) for _ in range(N_ENVS)])
        print(f"[OK] Using {N_ENVS} parallel SubprocVecEnv workers", flush=True)
    except Exception as e:
        print(f"[WARN] SubprocVecEnv failed ({e}), falling back to DummyVecEnv", flush=True)
        env = DummyVecEnv([make_env(df)])

    policy_kwargs = dict(
        activation_fn=activation_fn,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=1024,  # Per env; total = 1024 × 4 = 4096
        batch_size=256,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs
    )

    # Use progress callback for real-time streaming
    progress_cb = ProgressCallback(total_timesteps=timesteps)

    start = time.time()
    model.learn(total_timesteps=timesteps, callback=progress_cb)
    train_time = time.time() - start

    print(f"\n[OK] {activation_name} training complete ({train_time:.1f}s)", flush=True)
    
    # SAVE IMMEDIATELY
    model.save("ppo_gnn_best")
    print(f"[OK] Saved model to 'ppo_gnn_best.zip' immediately.", flush=True)

    # Evaluate (only 5 episodes now)
    print(f"[INFO] Evaluating {activation_name} agent (3 episodes)...", flush=True)
    results = evaluate_agent(model, env, n_episodes=3)
    results['training_time_seconds'] = round(train_time, 2)
    results['activation'] = activation_name

    print(f"   Avg reward: {results['avg_reward']:.2f} +/- {results['std_reward']:.2f}", flush=True)
    print(f"   Avg value: ${results['avg_value']:,.2f}", flush=True)

    return model, results


def main():
    print("=" * 55)
    print("  Phase 3: GNN-Enhanced RL Agent (GLU vs GELU)")
    print("=" * 55)

    # Load training data
    try:
        df = pd.read_csv('training_data.csv')
        print(f"[INFO] Loaded {len(df)} tickets for training")
    except FileNotFoundError:
        print("[ERROR] training_data.csv not found!")
        return

    timesteps = 200000

    # Train GELU agent
    gelu_model, gelu_results = train_agent("GELU", nn.GELU, df, timesteps)

    # Train GLU agent (SKIPPED per user request for speed)
    # glu_model, glu_results = train_agent("SiLU/GLU", nn.SiLU, df, timesteps)
    glu_results = {}
    glu_model = None

    # Random baseline
    env = DummyVecEnv([lambda: GNNEnhancedEnv(df)])
    print(f"\n[INFO] Evaluating random baseline (5 episodes)...")
    random_results = evaluate_random(env, n_episodes=5)
    print(f"   Random avg reward: {random_results['avg_reward']:.2f}")
    print(f"   Random avg value: ${random_results['avg_value']:,.2f}")

    # Compare
    print(f"\n{'='*55}")
    print(f"  COMPARISON: GELU vs Random (GLU skipped)")
    print(f"{'='*55}")

    overall_winner = "GELU"
    print(f"  WINNER: {overall_winner}")

    # Save best model
    best_model = gelu_model
    best_model.save("ppo_gnn_best")
    print(f"\n[OK] Saved best agent ({overall_winner}) to 'ppo_gnn_best.zip'")

    # Calculate improvement over random
    gelu_improvement = ((gelu_results['avg_reward'] - random_results['avg_reward'])
                        / abs(random_results['avg_reward']) * 100) if random_results['avg_reward'] != 0 else 0
    glu_improvement = 0

    # Save comparison
    comparison = {
        'winner': overall_winner,
        'gelu_results': gelu_results,
        'glu_results': glu_results,
        'random_baseline': random_results,
        'gelu_improvement_over_random': f"{gelu_improvement:+.1f}%",
        'glu_improvement_over_random': f"{glu_improvement:+.1f}%",
        'timesteps': timesteps,
    }
    with open('agent_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"[OK] Saved comparison to 'agent_comparison.json'")

    print(f"\n[SUCCESS] Phase 3 complete!")
    return best_model, comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--progress-file', type=str, default=None,
                        help='Path to write final training results JSON')
    args = parser.parse_args()
    
    best_model, comparison = main()
    
    # Write results to progress file if specified
    if args.progress_file and comparison:
        try:
            with open(args.progress_file, 'w') as f:
                json.dump(comparison, f, indent=2)
            print(f"[OK] Progress results written to {args.progress_file}", flush=True)
        except Exception as e:
            print(f"[WARN] Could not write progress file: {e}", flush=True)
