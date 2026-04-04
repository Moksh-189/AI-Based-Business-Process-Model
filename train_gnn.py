"""
Phase 2C — GNN Training (Multi-Mode)
Trains GNN models and compares variants. Supports three modes:

    --mode=classic : Original homogeneous GAT vs GAT+GLU (legacy)
    --mode=hetero  : Heterogeneous GAT vs HeteroGAT+GLU (default)
    --mode=tgn     : Temporal Graph Network (HeteroGNN + temporal memory)

Output:
    - gnn_process_model.pt     (best model weights)
    - node_embeddings.pt       (node embeddings for RL agent / chatbot)
    - gnn_comparison.json      (model comparison results)
"""

import json
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
from graph_builder import build_graph, build_hetero_graph, build_temporal_events
from gnn_model import ProcessGNN, ProcessGNNWithGLU
from hetero_gnn_model import HeteroProcessGNN, HeteroProcessGNNWithGLU
from tgn_model import ProcessTGN


# =====================================================================
#  Classic (Homogeneous) Training — Legacy
# =====================================================================

def train_classic_model(model, data, epochs=300, lr=0.005, model_name="Model"):
    """Train a homogeneous GNN model on the process graph."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    n_act = data.num_activity_nodes
    labels = data.y

    best_loss = float('inf')
    best_state = None

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        predictions, embeddings = model(data.x, data.edge_index, data.edge_attr)
        pred_act = predictions[:n_act]
        loss = criterion(pred_act, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0 or epoch == 1:
            with torch.no_grad():
                model.eval()
                pred, emb = model(data.x, data.edge_index, data.edge_attr)
                pred_a = pred[:n_act]
                mae = (pred_a - labels).abs().mean().item()
                if labels.std() > 0 and pred_a.std() > 0:
                    corr = np.corrcoef(pred_a.numpy(), labels.numpy())[0, 1]
                else:
                    corr = 0.0
                print(f"   [{model_name}] Epoch {epoch:3d} | "
                      f"Loss: {loss.item():.6f} | MAE: {mae:.4f} | Corr: {corr:.4f}")
                model.train()

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred, embeddings = model(data.x, data.edge_index, data.edge_attr)
        pred_act = pred[:n_act]
        final_mae = (pred_act - labels).abs().mean().item()
        final_mse = ((pred_act - labels) ** 2).mean().item()
        if labels.std() > 0 and pred_act.std() > 0:
            final_corr = float(np.corrcoef(pred_act.numpy(), labels.numpy())[0, 1])
        else:
            final_corr = 0.0

    return model, embeddings, {
        "model_name": model_name,
        "best_loss": best_loss,
        "final_mae": round(final_mae, 6),
        "final_mse": round(final_mse, 6),
        "final_correlation": round(final_corr, 4),
        "epochs": epochs,
    }


# =====================================================================
#  Heterogeneous Training
# =====================================================================

def _get_hetero_data_dicts(data):
    """Extract x_dict, edge_index_dict, edge_attr_dict from HeteroData."""
    x_dict = {
        'activity': data['activity'].x,
        'resource': data['resource'].x,
    }
    edge_index_dict = {}
    edge_attr_dict = {}

    for edge_type in data.edge_types:
        store = data[edge_type]
        edge_index_dict[edge_type] = store.edge_index
        if hasattr(store, 'edge_attr') and store.edge_attr is not None:
            edge_attr_dict[edge_type] = store.edge_attr

    return x_dict, edge_index_dict, edge_attr_dict


def train_hetero_model(model, data, epochs=300, lr=0.005, model_name="HeteroGAT"):
    """Train a heterogeneous GNN model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    labels = data['activity'].y
    x_dict, edge_index_dict, edge_attr_dict = _get_hetero_data_dicts(data)

    best_loss = float('inf')
    best_state = None

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        predictions, h = model(x_dict, edge_index_dict, edge_attr_dict)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0 or epoch == 1:
            with torch.no_grad():
                model.eval()
                pred, _ = model(x_dict, edge_index_dict, edge_attr_dict)
                mae = (pred - labels).abs().mean().item()
                if labels.std() > 0 and pred.std() > 0:
                    corr = float(np.corrcoef(pred.numpy(), labels.numpy())[0, 1])
                else:
                    corr = 0.0
                print(f"   [{model_name}] Epoch {epoch:3d} | "
                      f"Loss: {loss.item():.6f} | MAE: {mae:.4f} | Corr: {corr:.4f}")
                model.train()

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred, h = model(x_dict, edge_index_dict, edge_attr_dict)
        final_mae = (pred - labels).abs().mean().item()
        final_mse = ((pred - labels) ** 2).mean().item()
        if labels.std() > 0 and pred.std() > 0:
            final_corr = float(np.corrcoef(pred.numpy(), labels.numpy())[0, 1])
        else:
            final_corr = 0.0

        # Get flat embeddings for RL agent compatibility
        flat_emb, n_act = model.get_flat_embeddings(x_dict, edge_index_dict, edge_attr_dict)

    return model, flat_emb, n_act, {
        "model_name": model_name,
        "best_loss": best_loss,
        "final_mae": round(final_mae, 6),
        "final_mse": round(final_mse, 6),
        "final_correlation": round(final_corr, 4),
        "epochs": epochs,
    }


# =====================================================================
#  TGN Training
# =====================================================================

def train_tgn_model(tgn, data, events, epochs=100, lr=0.005, batch_size=256):
    """Train TGN with sequential event processing."""
    print(f"\n{'='*55}")
    print(f"  Training Temporal Graph Network (TGN)")
    print(f"{'='*55}")

    optimizer = torch.optim.Adam(tgn.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    labels = data['activity'].y
    x_dict, edge_index_dict, edge_attr_dict = _get_hetero_data_dicts(data)

    best_loss = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        tgn.train()
        tgn.reset_memory()
        optimizer.zero_grad()

        # Process events in batches (temporal order)
        n_batches = max(1, len(events) // batch_size)
        epoch_loss = 0

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(events))
            event_batch = events[start:end]

            # Update temporal memory with this batch of events
            tgn.process_events(event_batch)

            # Forward pass with updated memory
            predictions, _ = tgn(x_dict, edge_index_dict, edge_attr_dict)
            loss = criterion(predictions, labels)

            loss.backward()
            epoch_loss += loss.item()

        optimizer.step()
        avg_loss = epoch_loss / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in tgn.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            with torch.no_grad():
                tgn.eval()
                pred, _ = tgn(x_dict, edge_index_dict, edge_attr_dict)
                mae = (pred - labels).abs().mean().item()
                if labels.std() > 0 and pred.std() > 0:
                    corr = float(np.corrcoef(pred.numpy(), labels.numpy())[0, 1])
                else:
                    corr = 0.0
                print(f"   [TGN] Epoch {epoch:3d} | "
                      f"Loss: {avg_loss:.6f} | MAE: {mae:.4f} | Corr: {corr:.4f}")

    tgn.load_state_dict(best_state)
    tgn.eval()
    with torch.no_grad():
        pred, h = tgn(x_dict, edge_index_dict, edge_attr_dict)
        final_mae = (pred - labels).abs().mean().item()
        final_mse = ((pred - labels) ** 2).mean().item()
        if labels.std() > 0 and pred.std() > 0:
            final_corr = float(np.corrcoef(pred.numpy(), labels.numpy())[0, 1])
        else:
            final_corr = 0.0

        static_emb, temporal_mem = tgn.get_temporal_embeddings(
            x_dict, edge_index_dict, edge_attr_dict
        )

    return tgn, static_emb, temporal_mem, {
        "model_name": "TGN",
        "best_loss": best_loss,
        "final_mae": round(final_mae, 6),
        "final_mse": round(final_mse, 6),
        "final_correlation": round(final_corr, 4),
        "epochs": epochs,
    }


# =====================================================================
#  Comparison Helpers
# =====================================================================

def compare_results(results_list):
    """Compare multiple model results and pick the winner."""
    print(f"\n{'='*55}")
    print(f"  MODEL COMPARISON")
    print(f"{'='*55}")
    print(f"  {'Model':<20} {'MAE':>10} {'MSE':>10} {'Corr':>10}")
    print(f"  {'-'*50}")

    for r in results_list:
        print(f"  {r['model_name']:<20} {r['final_mae']:>10.6f} "
              f"{r['final_mse']:>10.6f} {r['final_correlation']:>10.4f}")

    # Score: lower MAE + lower MSE + higher correlation
    best = min(results_list, key=lambda r: r['final_mae'])
    print(f"\n  WINNER (by MAE): {best['model_name']}")
    return best


# =====================================================================
#  Main Entry Points
# =====================================================================

def run_classic(epochs=300):
    """Original homogeneous GAT training (backward compatible)."""
    print("=" * 55)
    print("  Phase 2: GNN Training (Classic — GAT vs GAT+GLU)")
    print("=" * 55 + "\n")

    data, metadata = build_graph()
    in_channels = metadata['n_features']
    edge_dim = metadata['n_edge_features']

    # Train standard GAT
    gat_model = ProcessGNN(in_channels=in_channels, hidden_channels=64,
                           out_channels=32, heads=4, dropout=0.2, edge_dim=edge_dim)
    start = time.time()
    gat_model, gat_emb, gat_results = train_classic_model(
        gat_model, data, epochs=epochs, model_name="GAT"
    )
    gat_results['training_time_seconds'] = round(time.time() - start, 2)

    # Train GAT+GLU
    glu_model = ProcessGNNWithGLU(in_channels=in_channels, hidden_channels=64,
                                  out_channels=32, heads=4, dropout=0.2, edge_dim=edge_dim)
    start = time.time()
    glu_model, glu_emb, glu_results = train_classic_model(
        glu_model, data, epochs=epochs, model_name="GAT+GLU"
    )
    glu_results['training_time_seconds'] = round(time.time() - start, 2)

    # Compare
    best = compare_results([gat_results, glu_results])
    winner = best['model_name']

    if winner == "GAT+GLU":
        best_model, best_emb = glu_model, glu_emb
    else:
        best_model, best_emb = gat_model, gat_emb

    # Save (same format as original)
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'model_type': winner,
        'in_channels': in_channels, 'hidden_channels': 64,
        'out_channels': 32, 'heads': 4, 'edge_dim': edge_dim,
        'results': best,
    }, 'gnn_process_model.pt')

    torch.save({
        'embeddings': best_emb,
        'activity_names': metadata['activity_names'],
        'resource_names': metadata['resource_names'],
        'n_activities': metadata['n_activities'],
        'n_resources': metadata['n_resources'],
        'model_type': winner,
    }, 'node_embeddings.pt')

    comparison = {
        'mode': 'classic',
        'winner': winner,
        'gat_results': gat_results,
        'glu_results': glu_results,
    }
    with open('gnn_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\n[OK] Saved best model ({winner}) to 'gnn_process_model.pt'")
    print(f"[OK] Saved embeddings to 'node_embeddings.pt'")
    print(f"[SUCCESS] Classic Phase 2 complete!")
    return best_model, best_emb, metadata


def run_hetero(epochs=300):
    """Heterogeneous GNN training."""
    print("=" * 55)
    print("  Phase 2: GNN Training (Hetero — HeteroGAT vs HeteroGAT+GLU)")
    print("=" * 55 + "\n")

    data, metadata = build_hetero_graph()
    act_in = metadata['activity_features']
    res_in = metadata['resource_features']
    dfg_edge_dim = metadata['dfg_edge_features']
    ra_edge_dim = metadata['ra_edge_features']

    # Train HeteroGAT
    model1 = HeteroProcessGNN(
        activity_in=act_in, resource_in=res_in, hidden=64, out=32,
        heads=4, dropout=0.2, dfg_edge_dim=dfg_edge_dim, ra_edge_dim=ra_edge_dim
    )
    start = time.time()
    model1, emb1, n_act1, results1 = train_hetero_model(
        model1, data, epochs=epochs, model_name="HeteroGAT"
    )
    results1['training_time_seconds'] = round(time.time() - start, 2)

    # Train HeteroGAT+GLU
    model2 = HeteroProcessGNNWithGLU(
        activity_in=act_in, resource_in=res_in, hidden=64, out=32,
        heads=4, dropout=0.2, dfg_edge_dim=dfg_edge_dim, ra_edge_dim=ra_edge_dim
    )
    start = time.time()
    model2, emb2, n_act2, results2 = train_hetero_model(
        model2, data, epochs=epochs, model_name="HeteroGAT+GLU"
    )
    results2['training_time_seconds'] = round(time.time() - start, 2)

    # Compare
    best = compare_results([results1, results2])
    winner = best['model_name']

    if winner == "HeteroGAT+GLU":
        best_model, best_emb, best_n_act = model2, emb2, n_act2
    else:
        best_model, best_emb, best_n_act = model1, emb1, n_act1

    # Save model
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'model_type': winner,
        'activity_in': act_in, 'resource_in': res_in,
        'hidden': 64, 'out': 32, 'heads': 4,
        'dfg_edge_dim': dfg_edge_dim, 'ra_edge_dim': ra_edge_dim,
        'results': best,
        'architecture': 'hetero',
    }, 'gnn_process_model.pt')

    # Save embeddings in backward-compatible format
    torch.save({
        'embeddings': best_emb,
        'activity_names': metadata['activity_names'],
        'resource_names': metadata['resource_names'],
        'n_activities': metadata['n_activities'],
        'n_resources': metadata['n_resources'],
        'model_type': winner,
        'architecture': 'hetero',
    }, 'node_embeddings.pt')

    comparison = {
        'mode': 'hetero',
        'winner': winner,
        'hetero_gat_results': results1,
        'hetero_glu_results': results2,
    }
    with open('gnn_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\n[OK] Saved best model ({winner}) to 'gnn_process_model.pt'")
    print(f"[OK] Saved embeddings ({best_emb.shape}) to 'node_embeddings.pt'")
    print(f"[SUCCESS] Hetero Phase 2 complete!")
    return best_model, best_emb, metadata


def run_tgn(epochs=100):
    """TGN training (HeteroGNN + temporal memory)."""
    print("=" * 55)
    print("  Phase 2: GNN Training (TGN — Temporal Graph Network)")
    print("=" * 55 + "\n")

    # Build hetero graph + temporal events
    data, metadata = build_hetero_graph()
    events, evt_meta = build_temporal_events()

    act_in = metadata['activity_features']
    res_in = metadata['resource_features']
    dfg_edge_dim = metadata['dfg_edge_features']
    ra_edge_dim = metadata['ra_edge_features']
    n_act = metadata['n_activities']
    n_res = metadata['n_resources']

    # Also train baseline HeteroGAT for comparison
    print(f"\n{'='*55}")
    print(f"  Training HeteroGAT Baseline (for comparison)")
    print(f"{'='*55}")

    baseline = HeteroProcessGNN(
        activity_in=act_in, resource_in=res_in, hidden=64, out=32,
        heads=4, dfg_edge_dim=dfg_edge_dim, ra_edge_dim=ra_edge_dim
    )
    start = time.time()
    baseline, baseline_emb, _, baseline_results = train_hetero_model(
        baseline, data, epochs=epochs, model_name="HeteroGAT"
    )
    baseline_results['training_time_seconds'] = round(time.time() - start, 2)

    # Train TGN
    tgn = ProcessTGN(
        n_activities=n_act, n_resources=n_res,
        activity_in=act_in, resource_in=res_in,
        hidden=64, out=32, memory_dim=32, heads=4,
        dfg_edge_dim=dfg_edge_dim, ra_edge_dim=ra_edge_dim
    )
    start = time.time()
    tgn, static_emb, temporal_mem, tgn_results = train_tgn_model(
        tgn, data, events, epochs=epochs
    )
    tgn_results['training_time_seconds'] = round(time.time() - start, 2)

    # Compare
    best = compare_results([baseline_results, tgn_results])
    winner = best['model_name']

    # Save TGN model
    torch.save({
        'model_state_dict': tgn.state_dict(),
        'model_type': 'TGN',
        'activity_in': act_in, 'resource_in': res_in,
        'hidden': 64, 'out': 32, 'memory_dim': 32, 'heads': 4,
        'dfg_edge_dim': dfg_edge_dim, 'ra_edge_dim': ra_edge_dim,
        'n_activities': n_act, 'n_resources': n_res,
        'results': tgn_results,
        'architecture': 'tgn',
    }, 'gnn_process_model.pt')

    # Save embeddings (use TGN static embeddings for RL compatibility)
    torch.save({
        'embeddings': static_emb,
        'temporal_memory': temporal_mem,
        'activity_names': metadata['activity_names'],
        'resource_names': metadata['resource_names'],
        'n_activities': n_act,
        'n_resources': n_res,
        'model_type': 'TGN',
        'architecture': 'tgn',
    }, 'node_embeddings.pt')

    comparison = {
        'mode': 'tgn',
        'winner': winner,
        'hetero_gat_results': baseline_results,
        'tgn_results': tgn_results,
    }
    with open('gnn_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\n[OK] Saved TGN model to 'gnn_process_model.pt'")
    print(f"[OK] Saved embeddings + temporal memory to 'node_embeddings.pt'")
    print(f"[SUCCESS] TGN Phase 2 complete!")
    return tgn, static_emb, metadata


def main():
    parser = argparse.ArgumentParser(description="GNN Training (Multi-Mode)")
    parser.add_argument('--mode', choices=['classic', 'hetero', 'tgn'],
                        default='hetero',
                        help='Training mode: classic (legacy GAT), hetero (default), tgn')
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args()

    if args.mode == 'classic':
        return run_classic(epochs=args.epochs)
    elif args.mode == 'hetero':
        return run_hetero(epochs=args.epochs)
    elif args.mode == 'tgn':
        return run_tgn(epochs=min(args.epochs, 100))  # TGN converges faster


if __name__ == '__main__':
    main()
