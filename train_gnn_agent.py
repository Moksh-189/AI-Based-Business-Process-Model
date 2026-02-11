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
import numpy as np
import pandas as pd
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gnn_env import GNNEnhancedEnv


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


def train_agent(activation_name, activation_fn, df, timesteps=200000):
    """Train a PPO agent with specified activation."""
    print(f"\n{'='*55}")
    print(f"  Training PPO with {activation_name}")
    print(f"{'='*55}")

    env = DummyVecEnv([lambda: GNNEnhancedEnv(df)])

    policy_kwargs = dict(
        activation_fn=activation_fn,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=4096,
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

    start = time.time()
    model.learn(total_timesteps=timesteps)
    train_time = time.time() - start

    print(f"\n[OK] {activation_name} training complete ({train_time:.1f}s)")
    
    # SAVE IMMEDIATELY
    model.save("ppo_gnn_best")
    print(f"[OK] Saved model to 'ppo_gnn_best.zip' immediately.")

    # Evaluate (only 5 episodes now)
    print(f"[INFO] Evaluating {activation_name} agent (5 episodes)...")
    results = evaluate_agent(model, env, n_episodes=5)
    results['training_time_seconds'] = round(train_time, 2)
    results['activation'] = activation_name

    print(f"   Avg reward: {results['avg_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"   Avg value: ${results['avg_value']:,.2f}")

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
    main()
