import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from custom_env import JiraOptimizationEnv
import numpy as np

def run_simulation(model=None, env=None, steps=1000):
    """
    Runs the simulation for a fixed number of steps.
    If 'model' is None, it acts randomly.
    If 'model' is provided, it uses the AI prediction.
    """
    obs, _ = env.reset()
    total_reward = 0
    total_revenue = 0
    
    print(f"   Running for {steps} steps...")
    
    for _ in range(steps):
        if model:
            # AI decides: Predict the best action based on observation
            action, _states = model.predict(obs, deterministic=True)
        else:
            # Random decides: Pick any valid ticket (0-4)
            action = env.action_space.sample()
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # We roughly reverse-engineer revenue from reward for display
        # (Since reward = value/1000 in our env)
        total_revenue += (reward * 1000)
        
        if terminated or truncated:
            obs, _ = env.reset()
            
    return total_revenue

def start_showdown():
    print("🥊 THE SHOWDOWN: Random vs. AI")
    print("---------------------------------")
    
    # 1. Load Data & Env
    df = pd.read_csv('training_data.csv')
    env = JiraOptimizationEnv(df)
    
    # 2. Run Random Agent (Baseline)
    print("\n1️⃣  Running 'Random Intern' Strategy...")
    random_score = run_simulation(model=None, env=env, steps=1000)
    print(f"   👉 Result: ${random_score:,.0f} Revenue Processed")
    
    # 3. Run AI Agent (Your Model)
    print("\n2️⃣  Running 'PPO AI Agent' Strategy...")
    try:
        # Load the GELU model you just trained
        model = PPO.load("ppo_jira_agent_gelu.zip")
        ai_score = run_simulation(model=model, env=env, steps=1000)
        print(f"   👉 Result: ${ai_score:,.0f} Revenue Processed")
        
        # 4. The Verdict
        print("\n---------------------------------")
        if ai_score > random_score:
            improvement = ((ai_score - random_score) / random_score) * 100
            print(f"🏆 WINNER: AI Agent!")
            print(f"📈 Improvement: +{improvement:.1f}% more revenue captured.")
            print("   (The AI successfully learned to prioritize High-Value tickets!)")
        else:
            print("😐 RESULT: Tie or Loss.")
            print("   (The AI might need more training steps or better data differentiation.)")
            
    except FileNotFoundError:
        print("❌ Error: Could not find 'ppo_jira_agent_gelu.zip'. Did you run training?")

if __name__ == "__main__":
    start_showdown()