"""
Phase 2B — Graph Neural Network Model
Graph Attention Network (GAT) for process prediction.

Tasks:
  1. Bottleneck prediction (regression): predict bottleneck_score per activity node
  2. Node embedding: produce embeddings usable by RL agent and chatbot
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class ProcessGNN(nn.Module):
    """
    2-layer Graph Attention Network for process graph analysis.
    
    Input:  Node features (8-dim) + edge features
    Output: Bottleneck score prediction + node embeddings
    """
    
    def __init__(self, in_channels=8, hidden_channels=64, out_channels=32,
                 heads=4, dropout=0.2, edge_dim=2):
        super().__init__()
        
        self.dropout = dropout
        
        # Layer 1: Multi-head GAT
        self.conv1 = GATConv(
            in_channels, hidden_channels, heads=heads,
            dropout=dropout, edge_dim=edge_dim, add_self_loops=True
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        
        # Layer 2: Single-head GAT (aggregate multi-head)
        self.conv2 = GATConv(
            hidden_channels * heads, out_channels, heads=1, concat=False,
            dropout=dropout, edge_dim=edge_dim, add_self_loops=True
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Prediction head: bottleneck score (regression)
        self.predictor = nn.Sequential(
            nn.Linear(out_channels, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Score in [0, 1]
        )
        
        # Embedding dimension for export
        self.embedding_dim = out_channels
    
    def encode(self, x, edge_index, edge_attr=None):
        """Forward pass through GAT layers to produce node embeddings."""
        # Layer 1
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.elu(x)
        
        return x  # Node embeddings (num_nodes, out_channels)
    
    def forward(self, x, edge_index, edge_attr=None):
        """Full forward: embeddings + prediction."""
        embeddings = self.encode(x, edge_index, edge_attr)
        predictions = self.predictor(embeddings).squeeze(-1)
        return predictions, embeddings
    
    def get_attention_weights(self, x, edge_index, edge_attr=None):
        """Extract attention weights from the first GAT layer."""
        # Run conv1 with return_attention_weights=True
        out, (edge_idx, attention) = self.conv1(
            x, edge_index, edge_attr=edge_attr,
            return_attention_weights=True
        )
        return attention


class ProcessGNNWithGLU(nn.Module):
    """
    GAT + GLU (Gated Linear Unit) variant for comparison with standard GAT.
    
    GLU applies gating: output = σ(W₁x) ⊙ (W₂x)
    This helps suppress noisy features and learn which patterns matter.
    """
    
    def __init__(self, in_channels=8, hidden_channels=64, out_channels=32,
                 heads=4, dropout=0.2, edge_dim=2):
        super().__init__()
        
        self.dropout = dropout
        
        # Layer 1: Multi-head GAT
        self.conv1 = GATConv(
            in_channels, hidden_channels, heads=heads,
            dropout=dropout, edge_dim=edge_dim, add_self_loops=True
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        
        # GLU layer after conv1
        self.glu_gate = nn.Linear(hidden_channels * heads, hidden_channels * heads)
        self.glu_value = nn.Linear(hidden_channels * heads, hidden_channels * heads)
        
        # Layer 2: Single-head GAT
        self.conv2 = GATConv(
            hidden_channels * heads, out_channels, heads=1, concat=False,
            dropout=dropout, edge_dim=edge_dim, add_self_loops=True
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # GLU layer after conv2
        self.glu_gate2 = nn.Linear(out_channels, out_channels)
        self.glu_value2 = nn.Linear(out_channels, out_channels)
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(out_channels, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.embedding_dim = out_channels
    
    def glu(self, x, gate_layer, value_layer):
        """Gated Linear Unit: σ(Wg·x) ⊙ (Wv·x)"""
        gate = torch.sigmoid(gate_layer(x))
        value = value_layer(x)
        return gate * value
    
    def encode(self, x, edge_index, edge_attr=None):
        """Forward through GAT + GLU layers."""
        # Layer 1 + GLU
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = self.glu(x, self.glu_gate, self.glu_value)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2 + GLU
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = self.glu(x, self.glu_gate2, self.glu_value2)
        
        return x
    
    def forward(self, x, edge_index, edge_attr=None):
        """Full forward: embeddings + prediction."""
        embeddings = self.encode(x, edge_index, edge_attr)
        predictions = self.predictor(embeddings).squeeze(-1)
        return predictions, embeddings
