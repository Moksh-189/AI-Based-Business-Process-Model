"""
Phase 2C — GNN Training
Trains both ProcessGNN (GELU) and ProcessGNNWithGLU (GLU) models,
compares their performance, and saves the best model + embeddings.

Output:
  - gnn_process_model.pt   (best model weights)
  - node_embeddings.pt     (node embeddings for RL agent / chatbot)
  - gnn_comparison.json    (GLU vs GELU performance comparison)
"""

import json
import time
import torch
import torch.nn as nn
import numpy as np
from graph_builder import build_graph
from gnn_model import ProcessGNN, ProcessGNNWithGLU


def train_model(model, data, epochs=300, lr=0.005, model_name="Model"):
    """Train a GNN model on the process graph."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    n_act = data.num_activity_nodes
    labels = data.y  # bottleneck scores for activity nodes

    best_loss = float('inf')
    best_state = None
    train_losses = []

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        predictions, embeddings = model(data.x, data.edge_index, data.edge_attr)

        # Only compute loss on activity nodes (first n_act nodes)
        pred_act = predictions[:n_act]
        loss = criterion(pred_act, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        train_losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0 or epoch == 1:
            # Evaluate prediction quality
            with torch.no_grad():
                model.eval()
                pred, emb = model(data.x, data.edge_index, data.edge_attr)
                pred_a = pred[:n_act]

                # Mean Absolute Error
                mae = (pred_a - labels).abs().mean().item()

                # Correlation between predicted and actual
                if labels.std() > 0 and pred_a.std() > 0:
                    corr = np.corrcoef(
                        pred_a.numpy(), labels.numpy()
                    )[0, 1]
                else:
                    corr = 0.0

                print(f"   [{model_name}] Epoch {epoch:3d} | "
                      f"Loss: {loss.item():.6f} | "
                      f"MAE: {mae:.4f} | "
                      f"Corr: {corr:.4f}")
                model.train()

    # Load best weights
    model.load_state_dict(best_state)

    # Final evaluation
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

    results = {
        "model_name": model_name,
        "best_loss": best_loss,
        "final_mae": round(final_mae, 6),
        "final_mse": round(final_mse, 6),
        "final_correlation": round(final_corr, 4),
        "epochs": epochs,
    }

    return model, embeddings, results


def main():
    print("=" * 55)
    print("  Phase 2: GNN Training (GAT vs GAT+GLU)")
    print("=" * 55 + "\n")

    # Step 1: Build the graph
    data, metadata = build_graph()

    in_channels = metadata['n_features']
    edge_dim = metadata['n_edge_features']

    print(f"\n{'='*55}")
    print(f"  Training GAT (GELU activation)")
    print(f"{'='*55}")

    # Step 2: Train standard GAT
    gat_model = ProcessGNN(
        in_channels=in_channels, hidden_channels=64, out_channels=32,
        heads=4, dropout=0.2, edge_dim=edge_dim
    )
    start = time.time()
    gat_model, gat_embeddings, gat_results = train_model(
        gat_model, data, epochs=300, model_name="GAT"
    )
    gat_time = time.time() - start
    gat_results['training_time_seconds'] = round(gat_time, 2)

    print(f"\n{'='*55}")
    print(f"  Training GAT + GLU (Gated Linear Unit)")
    print(f"{'='*55}")

    # Step 3: Train GAT + GLU
    glu_model = ProcessGNNWithGLU(
        in_channels=in_channels, hidden_channels=64, out_channels=32,
        heads=4, dropout=0.2, edge_dim=edge_dim
    )
    start = time.time()
    glu_model, glu_embeddings, glu_results = train_model(
        glu_model, data, epochs=300, model_name="GAT+GLU"
    )
    glu_time = time.time() - start
    glu_results['training_time_seconds'] = round(glu_time, 2)

    # Step 4: Compare and pick winner
    print(f"\n{'='*55}")
    print(f"  COMPARISON: GAT vs GAT+GLU")
    print(f"{'='*55}")
    print(f"  {'Metric':<25} {'GAT':>10} {'GAT+GLU':>10} {'Winner':>10}")
    print(f"  {'-'*55}")

    comparison = {}
    gat_score = 0
    glu_score = 0

    for metric, lower_better in [
        ('final_mae', True), ('final_mse', True), ('final_correlation', False)
    ]:
        g = gat_results[metric]
        l = glu_results[metric]
        if lower_better:
            winner = "GAT" if g < l else "GAT+GLU"
            if g < l: gat_score += 1
            else: glu_score += 1
        else:
            winner = "GAT" if g > l else "GAT+GLU"
            if g > l: gat_score += 1
            else: glu_score += 1
        print(f"  {metric:<25} {g:>10.6f} {l:>10.6f} {winner:>10}")
        comparison[metric] = {"gat": g, "glu": l, "winner": winner}

    print(f"  {'-'*55}")
    overall_winner = "GAT+GLU" if glu_score > gat_score else "GAT"
    print(f"  {'OVERALL WINNER':<25} {'':>10} {'':>10} {overall_winner:>10}")
    print(f"  (GAT: {gat_score} wins, GLU: {glu_score} wins)")

    # Save the winner
    if overall_winner == "GAT+GLU":
        best_model = glu_model
        best_embeddings = glu_embeddings
        best_results = glu_results
    else:
        best_model = gat_model
        best_embeddings = gat_embeddings
        best_results = gat_results

    # Save model
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'model_type': overall_winner,
        'in_channels': in_channels,
        'hidden_channels': 64,
        'out_channels': 32,
        'heads': 4,
        'edge_dim': edge_dim,
        'results': best_results,
    }, 'gnn_process_model.pt')
    print(f"\n[OK] Saved best model ({overall_winner}) to 'gnn_process_model.pt'")

    # Save embeddings
    torch.save({
        'embeddings': best_embeddings,
        'activity_names': metadata['activity_names'],
        'resource_names': metadata['resource_names'],
        'n_activities': metadata['n_activities'],
        'n_resources': metadata['n_resources'],
        'model_type': overall_winner,
    }, 'node_embeddings.pt')
    print(f"[OK] Saved embeddings ({best_embeddings.shape}) to 'node_embeddings.pt'")

    # Save comparison
    full_comparison = {
        "winner": overall_winner,
        "gat_results": gat_results,
        "glu_results": glu_results,
        "comparison": comparison,
    }
    with open('gnn_comparison.json', 'w') as f:
        json.dump(full_comparison, f, indent=2)
    print(f"[OK] Saved comparison to 'gnn_comparison.json'")

    print(f"\n[SUCCESS] Phase 2 complete!")
    return best_model, best_embeddings, metadata


if __name__ == '__main__':
    main()
