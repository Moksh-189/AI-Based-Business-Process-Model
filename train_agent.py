import pandas as pd
import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from custom_env import JiraOptimizationEnv

def train_brain():
    print("[INFO] ULTRA-TRAINING MODE ACTIVATED")
    print("=" * 50)
    
    try:
        df = pd.read_csv('training_data.csv')
        print(f"   Loaded {len(df)} tickets for training.")
    except FileNotFoundError:
        print("[ERROR] 'training_data.csv' not found.")
        return

    env = DummyVecEnv([lambda: JiraOptimizationEnv(df)])

    # === ULTRA-OPTIMIZED NETWORK ===
    policy_kwargs = dict(
        activation_fn=nn.GELU,
        net_arch=dict(pi=[256, 256], vf=[256, 256])  # Even larger network
    )

    # === OPTIMIZED HYPERPARAMETERS FOR FASTER LEARNING ===
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=0.0003,          # Standard LR
        n_steps=4096,                  # More experience per update
        batch_size=256,                # Larger batches
        n_epochs=20,                   # More epochs per update
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,                # Less exploration (we know what's good)
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs
    )

    print("\n[INFO] STARTING ULTRA-TRAINING...")
    print("   - 200,000 timesteps")
    print("   - Network: 256x256 neurons")
    print("   - Aggressive reward shaping")
    print("   - Sorted observations (position = value rank)")
    print("=" * 50 + "\n")
    
    # EXTENDED TRAINING
    model.learn(total_timesteps=200000)
    
    print("\n" + "=" * 50)
    print("[SUCCESS] ULTRA-TRAINING COMPLETE!")
    model.save("ppo_jira_agent_gelu.zip")
    print("[INFO] AI Model saved.")
    print("=" * 50)

if __name__ == "__main__":
    train_brain()