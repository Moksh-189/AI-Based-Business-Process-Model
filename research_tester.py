import json
import torch
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from graph_builder import build_graph
from gnn_model import ProcessGNN, ProcessGNNWithGLU
from gnn_env import GNNEnhancedEnv
from train_gnn_agent import evaluate_agent, evaluate_random

def evaluate_gnn_accuracy(model_path='gnn_process_model.pt'):
    """
    Evaluates the GNN model's predictive accuracy on the process graph.
    Metrics: MAE, MSE, Pearson Correlation (R).
    Ref: Methodology from Venugopal et al. (Paper 1) & Tax et al. (Paper 3).
    """
    print("\n" + "="*60)
    print("  PART 1: GNN PREDICTIVE ACCURACY EVALUATION")
    print("="*60)
    
    try:
        # Load Data
        data, metadata = build_graph()
        n_act = data.num_activity_nodes
        labels = data.y  # Actual bottleneck scores
        
        # Load Model
        checkpoint = torch.load(model_path)
        in_channels = checkpoint['in_channels']
        edge_dim = checkpoint['edge_dim']
        hidden_channels = checkpoint['hidden_channels']
        out_channels = checkpoint['out_channels']
        model_type = checkpoint['model_type']
        
        print(f"[INFO] Loading {model_type} model from '{model_path}'...")
        
        if model_type == 'GAT+GLU':
            model = ProcessGNNWithGLU(in_channels, hidden_channels, out_channels, edge_dim=edge_dim)
        else:
            model = ProcessGNN(in_channels, hidden_channels, out_channels, edge_dim=edge_dim)
            
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Inference
        with torch.no_grad():
            predictions, _ = model(data.x, data.edge_index, data.edge_attr)
            pred_act = predictions[:n_act]
            
            # Metrics Calculation
            mae = (pred_act - labels).abs().mean().item()
            mse = ((pred_act - labels) ** 2).mean().item()
            
            if labels.std() > 0 and pred_act.std() > 0:
                corr = float(np.corrcoef(pred_act.numpy(), labels.numpy())[0, 1])
            else:
                corr = 0.0
                
        print(f"  [RESULT] Mean Absolute Error (MAE): {mae:.6f}")
        print(f"  [RESULT] Mean Squared Error (MSE):  {mse:.6f}")
        print(f"  [RESULT] Pearson Correlation (R):   {corr:.6f}")
        
        return {
            "mae": round(mae, 6),
            "mse": round(mse, 6),
            "correlation": round(corr, 4),
            "model_type": model_type
        }
        
    except FileNotFoundError:
        print("[ERROR] GNN model file not found!")
        return None
    except Exception as e:
        print(f"[ERROR] GNN evaluation failed: {e}")
        return None

def evaluate_agent_performance(agent_path='ppo_gnn_best.zip', data_path='training_data.csv', episodes=20):
    """
    Evaluates the RL Agent's optimization performance vs Random Baseline.
    Metrics: Reward Improvement %, Value Processed, Bottlenecks Cleared.
    Ref: Methodology from Silvander (Paper 2) & Park et al. (Paper 6).
    """
    print("\n" + "="*60)
    print("  PART 2: AGENT OPTIMIZATION PERFORMANCE EVALUATION")
    print("="*60)
    
    try:
        # Load Environment Data
        df = pd.read_csv(data_path)
        print(f"[INFO] Loaded environment with {len(df)} cases.")
        
        # 1. Evaluate Random Baseline
        print(f"\n[INFO] Running Random Standard Baseline ({episodes} episodes)...")
        env_random = DummyVecEnv([lambda: GNNEnhancedEnv(df)])
        random_metrics = evaluate_random(env_random, n_episodes=episodes)
        print(f"  -> Random Avg Reward: {random_metrics['avg_reward']:.2f}")
        print(f"  -> Random Avg Value:  ${random_metrics['avg_value']:,.2f}")
        
        # 2. Evaluate Trained Agent
        print(f"\n[INFO] Running PPO Agent ({episodes} episodes)...")
        try:
            model = PPO.load(agent_path)
            print(f"[INFO] Loaded PPO Agent from '{agent_path}'")
        except:
            print("[ERROR] PPO agent file not found!")
            return None
            
        env_agent = DummyVecEnv([lambda: GNNEnhancedEnv(df)])
        agent_metrics = evaluate_agent(model, env_agent, n_episodes=episodes)
        print(f"  -> Agent Avg Reward:  {agent_metrics['avg_reward']:.2f}")
        print(f"  -> Agent Avg Value:   ${agent_metrics['avg_value']:,.2f}")
        
        # 3. Comparison
        if random_metrics['avg_reward'] != 0:
            reward_improvement = ((agent_metrics['avg_reward'] - random_metrics['avg_reward']) / abs(random_metrics['avg_reward'])) * 100
        else:
            reward_improvement = 0
            
        if random_metrics['avg_value'] != 0:
            value_improvement = ((agent_metrics['avg_value'] - random_metrics['avg_value']) / random_metrics['avg_value']) * 100
        else:
            value_improvement = 0
            
        print("\n" + "-"*30)
        print("  COMPARISON RESULTS")
        print("-" * 30)
        print(f"  Reward Improvement: {reward_improvement:+.1f}%")
        print(f"  Value Improvement:  {value_improvement:+.1f}%")
        
        return {
            "agent_metrics": agent_metrics,
            "random_metrics": random_metrics,
            "reward_improvement_pct": round(reward_improvement, 1),
            "value_improvement_pct": round(value_improvement, 1)
        }
        
    except Exception as e:
        print(f"[ERROR] Agent evaluation failed: {e}")
        with open('agent_error.log', 'w') as f:
            f.write(str(e))
        return None

def main():
    print("STARTING RESEARCH QUALITY EVALUATION...")
    
    gnn_results = evaluate_gnn_accuracy()
    agent_results = evaluate_agent_performance()
    
    full_report = {
        "gnn_accuracy": gnn_results,
        "agent_performance": agent_results
    }
    
    # Save to JSON
    with open('research_results.json', 'w') as f:
        json.dump(full_report, f, indent=2)
        
    print("\n" + "="*60)
    print(f"  [SUCCESS] Evaluation complete. Results saved to 'research_results.json'")
    print("="*60)

if __name__ == "__main__":
    main()
