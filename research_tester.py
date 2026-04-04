"""
Research Quality Evaluation — Multi-Architecture
Tests all three architectures (Classic GAT, HeteroGNN, MARL) using
the methodology from the research papers:

    Part 1: GNN Predictive Accuracy (MAE, Correlation)
            Ref: Venugopal et al. (2021), Tax et al. (2017)

    Part 2: RL Agent Optimization Performance (Reward, Value)
            Ref: Silvander (2019), Park et al. (2020)

    Part 3: MARL Cooperative Performance (Per-agent metrics)
            Ref: Extended methodology for multi-agent systems

    Part 4: Architecture Comparison (A/B results table)

Output: research_results.json
"""

import json
import torch
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# === Imports for each architecture ===
from graph_builder import build_graph, build_hetero_graph
from gnn_model import ProcessGNN, ProcessGNNWithGLU
from gnn_env import GNNEnhancedEnv
from train_gnn_agent import evaluate_agent, evaluate_random


def evaluate_gnn_accuracy(model_path='gnn_process_model.pt'):
    """
    Evaluates the GNN model's predictive accuracy on the process graph.
    Supports both classic (homogeneous) and hetero architectures.

    Metrics: MAE, MSE, Pearson Correlation (R).
    Ref: Methodology from Venugopal et al. (Paper 1) & Tax et al. (Paper 3).
    """
    print("\n" + "=" * 60)
    print("  PART 1: GNN PREDICTIVE ACCURACY EVALUATION")
    print("=" * 60)

    try:
        checkpoint = torch.load(model_path, weights_only=False)
        architecture = checkpoint.get('architecture', 'classic')
        model_type = checkpoint.get('model_type', 'unknown')

        print(f"[INFO] Architecture: {architecture}, Model: {model_type}")

        if architecture in ('hetero', 'tgn'):
            return _evaluate_hetero_gnn(checkpoint)
        else:
            return _evaluate_classic_gnn(checkpoint)

    except FileNotFoundError:
        print("[ERROR] GNN model file not found!")
        return None
    except Exception as e:
        print(f"[ERROR] GNN evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _evaluate_classic_gnn(checkpoint):
    """Evaluate homogeneous GAT model."""
    data, metadata = build_graph()
    n_act = data.num_activity_nodes
    labels = data.y

    in_channels = checkpoint['in_channels']
    edge_dim = checkpoint['edge_dim']
    model_type = checkpoint['model_type']

    if model_type == 'GAT+GLU':
        model = ProcessGNNWithGLU(in_channels, checkpoint['hidden_channels'],
                                  checkpoint['out_channels'], edge_dim=edge_dim)
    else:
        model = ProcessGNN(in_channels, checkpoint['hidden_channels'],
                           checkpoint['out_channels'], edge_dim=edge_dim)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        predictions, _ = model(data.x, data.edge_index, data.edge_attr)
        pred_act = predictions[:n_act]

        mae = (pred_act - labels).abs().mean().item()
        mse = ((pred_act - labels) ** 2).mean().item()

        if labels.std() > 0 and pred_act.std() > 0:
            corr = float(np.corrcoef(pred_act.numpy(), labels.numpy())[0, 1])
        else:
            corr = 0.0

    print(f"  [RESULT] Architecture: Classic ({model_type})")
    print(f"  [RESULT] Mean Absolute Error (MAE): {mae:.6f}")
    print(f"  [RESULT] Mean Squared Error (MSE):  {mse:.6f}")
    print(f"  [RESULT] Pearson Correlation (R):   {corr:.6f}")

    return {
        "architecture": "classic",
        "model_type": model_type,
        "mae": round(mae, 6),
        "mse": round(mse, 6),
        "correlation": round(corr, 4),
    }


def _evaluate_hetero_gnn(checkpoint):
    """Evaluate heterogeneous GNN model."""
    from hetero_gnn_model import HeteroProcessGNN, HeteroProcessGNNWithGLU

    data, metadata = build_hetero_graph()
    labels = data['activity'].y
    model_type = checkpoint.get('model_type', 'HeteroGAT')

    # Build model
    kwargs = {
        'activity_in': checkpoint['activity_in'],
        'resource_in': checkpoint['resource_in'],
        'hidden': checkpoint['hidden'],
        'out': checkpoint['out'],
        'heads': checkpoint['heads'],
        'dfg_edge_dim': checkpoint['dfg_edge_dim'],
        'ra_edge_dim': checkpoint['ra_edge_dim'],
    }

    if 'GLU' in model_type:
        model = HeteroProcessGNNWithGLU(**kwargs)
    else:
        model = HeteroProcessGNN(**kwargs)

    # Handle TGN wrapping
    architecture = checkpoint.get('architecture', 'hetero')
    if architecture == 'tgn':
        from tgn_model import ProcessTGN
        model = ProcessTGN(
            n_activities=checkpoint['n_activities'],
            n_resources=checkpoint['n_resources'],
            **kwargs,
            memory_dim=checkpoint.get('memory_dim', 32),
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Build input dicts
    x_dict = {'activity': data['activity'].x, 'resource': data['resource'].x}
    edge_index_dict = {}
    edge_attr_dict = {}
    for edge_type in data.edge_types:
        store = data[edge_type]
        edge_index_dict[edge_type] = store.edge_index
        if hasattr(store, 'edge_attr') and store.edge_attr is not None:
            edge_attr_dict[edge_type] = store.edge_attr

    with torch.no_grad():
        predictions, _ = model(x_dict, edge_index_dict, edge_attr_dict)

        mae = (predictions - labels).abs().mean().item()
        mse = ((predictions - labels) ** 2).mean().item()

        if labels.std() > 0 and predictions.std() > 0:
            corr = float(np.corrcoef(predictions.numpy(), labels.numpy())[0, 1])
        else:
            corr = 0.0

    print(f"  [RESULT] Architecture: {architecture} ({model_type})")
    print(f"  [RESULT] Mean Absolute Error (MAE): {mae:.6f}")
    print(f"  [RESULT] Mean Squared Error (MSE):  {mse:.6f}")
    print(f"  [RESULT] Pearson Correlation (R):   {corr:.6f}")

    return {
        "architecture": architecture,
        "model_type": model_type,
        "mae": round(mae, 6),
        "mse": round(mse, 6),
        "correlation": round(corr, 4),
    }


def evaluate_agent_performance(agent_path='ppo_gnn_best.zip',
                                data_path='training_data.csv', episodes=20):
    """
    Evaluates the single-agent RL Agent vs Random Baseline.
    Ref: Silvander (Paper 2) & Park et al. (Paper 6).
    """
    print("\n" + "=" * 60)
    print("  PART 2: SINGLE-AGENT OPTIMIZATION PERFORMANCE")
    print("=" * 60)

    try:
        df = pd.read_csv(data_path)
        print(f"[INFO] Loaded environment with {len(df)} cases.")

        # Random baseline
        print(f"\n[INFO] Running Random Baseline ({episodes} episodes)...")
        env_random = DummyVecEnv([lambda: GNNEnhancedEnv(df)])
        random_metrics = evaluate_random(env_random, n_episodes=episodes)
        print(f"  -> Random Avg Reward: {random_metrics['avg_reward']:.2f}")
        print(f"  -> Random Avg Value:  ${random_metrics['avg_value']:,.2f}")

        # Trained agent
        print(f"\n[INFO] Running PPO Agent ({episodes} episodes)...")
        try:
            model = PPO.load(agent_path)
        except Exception:
            print(f"[WARN] Could not load '{agent_path}', skipping single-agent eval.")
            return {'random_metrics': random_metrics, 'agent_metrics': None}

        env_agent = DummyVecEnv([lambda: GNNEnhancedEnv(df)])
        agent_metrics = evaluate_agent(model, env_agent, n_episodes=episodes)
        print(f"  -> Agent Avg Reward:  {agent_metrics['avg_reward']:.2f}")
        print(f"  -> Agent Avg Value:   ${agent_metrics['avg_value']:,.2f}")

        # Comparison
        if random_metrics['avg_reward'] != 0:
            reward_improvement = ((agent_metrics['avg_reward'] - random_metrics['avg_reward'])
                                  / abs(random_metrics['avg_reward'])) * 100
        else:
            reward_improvement = 0

        print(f"\n  Reward Improvement: {reward_improvement:+.1f}%")

        return {
            "agent_metrics": agent_metrics,
            "random_metrics": random_metrics,
            "reward_improvement_pct": round(reward_improvement, 1),
        }

    except Exception as e:
        print(f"[ERROR] Agent evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_marl_performance(data_path='training_data.csv', episodes=20):
    """
    Evaluates the MARL system with both trained agents.
    Extended methodology for multi-agent cooperative systems.
    """
    print("\n" + "=" * 60)
    print("  PART 3: MULTI-AGENT (MARL) PERFORMANCE EVALUATION")
    print("=" * 60)

    try:
        jira_model = PPO.load("ppo_marl_jira")
        sap_model = PPO.load("ppo_marl_sap")
        print("[INFO] Loaded both MARL agents (Jira + SAP)")
    except Exception:
        print("[WARN] MARL agents not found, skipping MARL evaluation.")
        return None

    try:
        df = pd.read_csv(data_path)
        from train_marl_agent import evaluate_marl_agent
        results = evaluate_marl_agent(jira_model, sap_model, df, n_episodes=episodes)

        print(f"\n  Jira Agent:")
        print(f"    Avg Reward: {results['jira_agent']['avg_reward']:.2f}")
        print(f"    Avg Value:  ${results['jira_agent']['avg_value']:,.2f}")
        print(f"    Bottlenecks Cleared: {results['jira_agent']['avg_bottlenecks']:.1f}")

        print(f"\n  SAP Agent:")
        print(f"    Avg Reward: {results['sap_agent']['avg_reward']:.2f}")
        print(f"    Avg Value:  ${results['sap_agent']['avg_value']:,.2f}")
        print(f"    Bottlenecks Cleared: {results['sap_agent']['avg_bottlenecks']:.1f}")

        print(f"\n  Combined:")
        print(f"    Total Reward: {results['combined']['avg_reward']:.2f}")
        print(f"    Total Value:  ${results['combined']['avg_value']:,.2f}")

        return results

    except Exception as e:
        print(f"[ERROR] MARL evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_comparison_table(gnn_results, agent_results, marl_results):
    """Generate the final A/B comparison table."""
    print("\n" + "=" * 60)
    print("  PART 4: ARCHITECTURE COMPARISON (A/B RESULTS)")
    print("=" * 60)

    print(f"\n  {'='*58}")
    print(f"  {'Component':<22} {'Metric':<18} {'Result':<18}")
    print(f"  {'='*58}")

    if gnn_results:
        arch = gnn_results.get('architecture', 'unknown')
        model = gnn_results.get('model_type', 'unknown')
        print(f"  {'GNN (' + arch + ')':<22} {'MAE':<18} {gnn_results['mae']:<18.6f}")
        print(f"  {'':<22} {'Correlation':<18} {gnn_results['correlation']:<18.4f}")
        print(f"  {'':<22} {'Model Type':<18} {model:<18}")

    if agent_results and agent_results.get('agent_metrics'):
        am = agent_results['agent_metrics']
        rm = agent_results['random_metrics']
        imp = agent_results['reward_improvement_pct']
        print(f"  {'-'*58}")
        print(f"  {'Single PPO':<22} {'Avg Reward':<18} {am['avg_reward']:<18.2f}")
        print(f"  {'':<22} {'Random Baseline':<18} {rm['avg_reward']:<18.2f}")
        print(f"  {'':<22} {'Improvement':<18} {'+' + str(imp) + '%':<18}")

    if marl_results:
        print(f"  {'-'*58}")
        jr = marl_results['jira_agent']
        sr = marl_results['sap_agent']
        cr = marl_results['combined']
        print(f"  {'MARL Jira Agent':<22} {'Avg Reward':<18} {jr['avg_reward']:<18.2f}")
        print(f"  {'MARL SAP Agent':<22} {'Avg Reward':<18} {sr['avg_reward']:<18.2f}")
        print(f"  {'MARL Combined':<22} {'Total Reward':<18} {cr['avg_reward']:<18.2f}")
        val_str = f"${cr['avg_value']:,.2f}"
        print(f"  {'':<22} {'Total Value':<18} {val_str:<18}")

    print(f"  {'='*58}")

    # Determine best system
    systems = {}
    if agent_results and agent_results.get('agent_metrics'):
        systems['Single PPO'] = agent_results['agent_metrics']['avg_reward']
    if marl_results:
        systems['MARL (Combined)'] = marl_results['combined']['avg_reward']

    if systems:
        best = max(systems, key=systems.get)
        print(f"\n  BEST OPTIMIZATION SYSTEM: {best} (reward={systems[best]:.2f})")


def main():
    print("=" * 60)
    print("  RESEARCH QUALITY EVALUATION — MULTI-ARCHITECTURE")
    print("=" * 60)
    print("  Ref: Venugopal (2021), Tax (2017), Silvander (2019),")
    print("       Park (2020), Lischka (2025)")

    gnn_results = evaluate_gnn_accuracy()
    agent_results = evaluate_agent_performance()
    marl_results = evaluate_marl_performance()
    generate_comparison_table(gnn_results, agent_results, marl_results)

    full_report = {
        "gnn_accuracy": gnn_results,
        "single_agent_performance": agent_results,
        "marl_performance": marl_results,
    }

    with open('research_results.json', 'w') as f:
        json.dump(full_report, f, indent=2)

    print("\n" + "=" * 60)
    print("  [SUCCESS] Evaluation complete. Results → 'research_results.json'")
    print("=" * 60)


if __name__ == "__main__":
    main()
