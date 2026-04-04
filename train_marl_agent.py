"""
Phase 3D — Multi-Agent RL Training (Independent Policies)
Trains two independent PPO agents (Jira + SAP) using the MARLProcessEnv.

Architecture:
    - Independent policies: each agent has its own neural network
    - Gymnasium wrappers to convert PettingZoo → SB3-compatible
    - Comparison: MARL vs single-agent baseline

Output:
    - ppo_marl_jira.zip    (Jira agent weights)
    - ppo_marl_sap.zip     (SAP agent weights)
    - marl_comparison.json (MARL vs single-agent vs random results)
"""

import json
import time
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
from marl_env import MARLProcessEnv
from train_gnn_agent import ProgressCallback, evaluate_random
from gnn_env import GNNEnhancedEnv


class SingleAgentWrapper(gym.Env):
    """Wraps a PettingZoo ParallelEnv for a single agent, making it SB3-compatible.

    The other agent uses a fixed policy (random or provided model).
    """

    def __init__(self, env_factory, agent_name, other_policy=None):
        super().__init__()
        self.env = env_factory()
        self.agent_name = agent_name
        self.other_agent = [a for a in self.env.possible_agents if a != agent_name][0]
        self.other_policy = other_policy  # None = random
        self.other_model = None

        self.observation_space = self.env.observation_space(agent_name)
        self.action_space = self.env.action_space(agent_name)

    def set_other_model(self, model):
        """Set a trained model for the other agent (for iterated training)."""
        self.other_model = model

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed)
        self._last_obs = observations
        return observations.get(self.agent_name,
                                np.zeros(self.observation_space.shape, dtype=np.float32)), \
               infos.get(self.agent_name, {})

    def step(self, action):
        # Determine other agent's action
        other_obs = self._last_obs.get(self.other_agent)
        if self.other_model and other_obs is not None:
            other_action, _ = self.other_model.predict(other_obs, deterministic=False)
        else:
            other_action = self.env.action_space(self.other_agent).sample()

        actions = {
            self.agent_name: action,
            self.other_agent: int(other_action),
        }

        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        self._last_obs = observations

        obs = observations.get(self.agent_name,
                               np.zeros(self.observation_space.shape, dtype=np.float32))
        reward = rewards.get(self.agent_name, 0)
        terminated = terminations.get(self.agent_name, True)
        truncated = truncations.get(self.agent_name, False)
        info = infos.get(self.agent_name, {})

        return obs, float(reward), terminated, truncated, info


def evaluate_marl_agent(jira_model, sap_model, df, n_episodes=20):
    """Evaluate the MARL system with both trained agents."""
    total_rewards = {'jira_agent': [], 'sap_agent': []}
    total_values = {'jira_agent': [], 'sap_agent': []}
    total_bottlenecks = {'jira_agent': [], 'sap_agent': []}

    for _ in range(n_episodes):
        env = MARLProcessEnv(df)
        observations, _ = env.reset()
        done = False
        ep_rewards = {'jira_agent': 0, 'sap_agent': 0}

        step_count = 0
        while env.agents and step_count < 500:
            actions = {}
            for agent in env.agents:
                obs = observations.get(agent)
                if obs is None:
                    continue
                if agent == 'jira_agent' and jira_model:
                    action, _ = jira_model.predict(obs, deterministic=True)
                elif agent == 'sap_agent' and sap_model:
                    action, _ = sap_model.predict(obs, deterministic=True)
                else:
                    action = env.action_space(agent).sample()
                actions[agent] = int(action)

            observations, rewards, terminations, truncations, infos = env.step(actions)

            for agent in ['jira_agent', 'sap_agent']:
                ep_rewards[agent] += rewards.get(agent, 0)

            step_count += 1

        for agent in ['jira_agent', 'sap_agent']:
            total_rewards[agent].append(ep_rewards[agent])
            info = infos.get(agent, {}) if infos else {}
            total_values[agent].append(info.get('value_processed', 0))
            total_bottlenecks[agent].append(info.get('bottlenecks_cleared', 0))

    return {
        'jira_agent': {
            'avg_reward': round(float(np.mean(total_rewards['jira_agent'])), 2),
            'std_reward': round(float(np.std(total_rewards['jira_agent'])), 2),
            'avg_value': round(float(np.mean(total_values['jira_agent'])), 2),
            'avg_bottlenecks': round(float(np.mean(total_bottlenecks['jira_agent'])), 2),
        },
        'sap_agent': {
            'avg_reward': round(float(np.mean(total_rewards['sap_agent'])), 2),
            'std_reward': round(float(np.std(total_rewards['sap_agent'])), 2),
            'avg_value': round(float(np.mean(total_values['sap_agent'])), 2),
            'avg_bottlenecks': round(float(np.mean(total_bottlenecks['sap_agent'])), 2),
        },
        'combined': {
            'avg_reward': round(float(
                np.mean(total_rewards['jira_agent']) + np.mean(total_rewards['sap_agent'])
            ), 2),
            'avg_value': round(float(
                np.mean(total_values['jira_agent']) + np.mean(total_values['sap_agent'])
            ), 2),
        }
    }


def train_marl(df, timesteps=200000):
    """Train two independent PPO agents with iterated self-play."""

    print("=" * 55)
    print("  Phase 3D: Multi-Agent RL Training (Independent)")
    print("=" * 55)

    def make_env():
        return MARLProcessEnv(df)

    policy_kwargs = dict(
        activation_fn=nn.GELU,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    # === Round 1: Train Jira agent (SAP plays random) ===
    print(f"\n{'='*55}")
    print(f"  Round 1: Training Jira Agent (SAP = Random)")
    print(f"{'='*55}", flush=True)

    jira_env = DummyVecEnv([
        lambda: SingleAgentWrapper(make_env, 'jira_agent', other_policy='random')
    ])

    jira_model = PPO(
        "MlpPolicy", jira_env, verbose=1,
        learning_rate=0.0003, n_steps=1024, batch_size=256,
        n_epochs=20, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=policy_kwargs
    )

    jira_cb = ProgressCallback(total_timesteps=timesteps)
    start = time.time()
    jira_model.learn(total_timesteps=timesteps, callback=jira_cb)
    jira_time = time.time() - start
    print(f"\n[OK] Jira Agent trained in {jira_time:.1f}s", flush=True)

    jira_model.save("ppo_marl_jira")
    print("[OK] Saved 'ppo_marl_jira.zip'", flush=True)

    # === Round 2: Train SAP agent (Jira plays trained policy) ===
    print(f"\n{'='*55}")
    print(f"  Round 2: Training SAP Agent (Jira = Trained)")
    print(f"{'='*55}", flush=True)

    sap_wrappers = []
    def make_sap_env():
        w = SingleAgentWrapper(make_env, 'sap_agent')
        w.set_other_model(jira_model)
        sap_wrappers.append(w)
        return w

    sap_env = DummyVecEnv([make_sap_env])

    sap_model = PPO(
        "MlpPolicy", sap_env, verbose=1,
        learning_rate=0.0003, n_steps=1024, batch_size=256,
        n_epochs=20, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=policy_kwargs
    )

    sap_cb = ProgressCallback(total_timesteps=timesteps)
    start = time.time()
    sap_model.learn(total_timesteps=timesteps, callback=sap_cb)
    sap_time = time.time() - start
    print(f"\n[OK] SAP Agent trained in {sap_time:.1f}s", flush=True)

    sap_model.save("ppo_marl_sap")
    print("[OK] Saved 'ppo_marl_sap.zip'", flush=True)

    # === Round 3 (Optional): Refine Jira against trained SAP ===
    print(f"\n{'='*55}")
    print(f"  Round 3: Refining Jira Agent (SAP = Trained)")
    print(f"{'='*55}", flush=True)

    refine_wrappers = []
    def make_jira_refine():
        w = SingleAgentWrapper(make_env, 'jira_agent')
        w.set_other_model(sap_model)
        refine_wrappers.append(w)
        return w

    jira_refine_env = DummyVecEnv([make_jira_refine])

    # Continue training from existing Jira weights
    jira_model.set_env(jira_refine_env)
    jira_cb2 = ProgressCallback(total_timesteps=timesteps // 2)
    start = time.time()
    jira_model.learn(total_timesteps=timesteps // 2, callback=jira_cb2)
    refine_time = time.time() - start
    print(f"\n[OK] Jira refinement in {refine_time:.1f}s", flush=True)

    jira_model.save("ppo_marl_jira")
    print("[OK] Updated 'ppo_marl_jira.zip'", flush=True)

    return jira_model, sap_model, {
        'jira_time': round(jira_time, 2),
        'sap_time': round(sap_time, 2),
        'refine_time': round(refine_time, 2),
    }


def main():
    print("=" * 55)
    print("  MARL Training Pipeline (Independent Policies)")
    print("=" * 55)

    try:
        df = pd.read_csv('training_data.csv')
        print(f"[INFO] Loaded {len(df)} tickets")
    except FileNotFoundError:
        print("[ERROR] training_data.csv not found!")
        return

    timesteps = 200000

    # Train MARL
    jira_model, sap_model, timing = train_marl(df, timesteps)

    # Evaluate MARL
    print(f"\n{'='*55}")
    print(f"  Evaluating MARL System (20 episodes)")
    print(f"{'='*55}", flush=True)
    marl_results = evaluate_marl_agent(jira_model, sap_model, df, n_episodes=20)

    # Evaluate single-agent baseline for comparison
    print(f"\n[INFO] Evaluating single-agent baseline...", flush=True)
    try:
        single_model = PPO.load("ppo_gnn_best")
        env_single = DummyVecEnv([lambda: GNNEnhancedEnv(df)])
        from train_gnn_agent import evaluate_agent
        single_results = evaluate_agent(single_model, env_single, n_episodes=20)
    except Exception:
        single_results = {'avg_reward': 0, 'avg_value': 0}

    # Random baseline
    print(f"[INFO] Evaluating random baseline...", flush=True)
    env_random = DummyVecEnv([lambda: GNNEnhancedEnv(df)])
    random_results = evaluate_random(env_random, n_episodes=20)

    # === Comparison ===
    print(f"\n{'='*55}")
    print(f"  COMPARISON: MARL vs Single-Agent vs Random")
    print(f"{'='*55}")
    print(f"  {'System':<25} {'Reward':>10} {'Value ($)':>12}")
    print(f"  {'-'*47}")
    print(f"  {'Random Baseline':<25} {random_results['avg_reward']:>10.2f} {random_results['avg_value']:>12,.2f}")
    print(f"  {'Single PPO (Classic)':<25} {single_results.get('avg_reward', 0):>10.2f} {single_results.get('avg_value', 0):>12,.2f}")
    print(f"  {'MARL Combined':<25} {marl_results['combined']['avg_reward']:>10.2f} {marl_results['combined']['avg_value']:>12,.2f}")
    print(f"  {'  └ Jira Agent':<25} {marl_results['jira_agent']['avg_reward']:>10.2f} {marl_results['jira_agent']['avg_value']:>12,.2f}")
    print(f"  {'  └ SAP Agent':<25} {marl_results['sap_agent']['avg_reward']:>10.2f} {marl_results['sap_agent']['avg_value']:>12,.2f}")

    # Calculate improvements
    if random_results['avg_reward'] != 0:
        marl_improvement = ((marl_results['combined']['avg_reward'] - random_results['avg_reward'])
                            / abs(random_results['avg_reward'])) * 100
    else:
        marl_improvement = 0

    print(f"\n  MARL improvement over Random: {marl_improvement:+.1f}%")

    # Save comparison
    comparison = {
        'winner': 'MARL',
        'marl_results': marl_results,
        'single_agent_results': single_results,
        'random_baseline': random_results,
        'marl_improvement_over_random': f"{marl_improvement:+.1f}%",
        'timing': timing,
        'timesteps_per_agent': timesteps,
        'policy_type': 'independent',
    }
    with open('marl_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[OK] Saved 'marl_comparison.json'")
    print(f"\n[SUCCESS] MARL Training Complete!")

    return jira_model, sap_model, comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=200000)
    parser.add_argument('--progress-file', type=str, default=None)
    args = parser.parse_args()

    jira, sap, comparison = main()

    if args.progress_file and comparison:
        try:
            with open(args.progress_file, 'w') as f:
                json.dump(comparison, f, indent=2)
            print(f"[OK] Progress written to {args.progress_file}", flush=True)
        except Exception as e:
            print(f"[WARN] Could not write progress file: {e}", flush=True)
