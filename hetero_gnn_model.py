"""
Phase 2B-v2 — Heterogeneous Graph Neural Network
HeteroGNN using separate GATConv per edge type for process graphs.

Architecture:
    - Linear projections per node type → shared hidden dim
    - HeteroConv Layer 1: Multi-head GAT per edge type
    - HeteroConv Layer 2: Single-head GAT (aggregation)
    - Bottleneck prediction head on activity embeddings

Supports:
    - HeteroProcessGNN (ELU activation)
    - HeteroProcessGNNWithGLU (Gated Linear Unit activation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv


class HeteroProcessGNN(nn.Module):
    """
    Heterogeneous Graph Attention Network for process analysis.
    
    Uses separate GATConv layers for:
        ('activity', 'follows', 'activity') — DFG transitions
        ('resource', 'performs', 'activity') — resource assignments
    """
    
    def __init__(self, activity_in=8, resource_in=6, hidden=64, out=32,
                 heads=4, dropout=0.2, dfg_edge_dim=2, ra_edge_dim=1):
        super().__init__()
        self.dropout = dropout
        self.hidden = hidden
        self.out = out
        
        # === Node type projections → shared hidden dim ===
        self.act_proj = nn.Linear(activity_in, hidden)
        self.res_proj = nn.Linear(resource_in, hidden)
        
        # === HeteroConv Layer 1: Multi-head GAT ===
        self.conv1 = HeteroConv({
            ('activity', 'follows', 'activity'): GATConv(
                hidden, hidden, heads=heads, dropout=dropout,
                edge_dim=dfg_edge_dim, add_self_loops=True
            ),
            ('resource', 'performs', 'activity'): GATConv(
                (hidden, hidden), hidden, heads=heads, dropout=dropout,
                edge_dim=ra_edge_dim, add_self_loops=False
            ),
        }, aggr='mean')
        
        self.bn1_act = nn.BatchNorm1d(hidden * heads)
        self.bn1_res = nn.BatchNorm1d(hidden)  # resources only receive via reverse edges
        
        # === HeteroConv Layer 2: Single-head aggregation ===
        self.conv2 = HeteroConv({
            ('activity', 'follows', 'activity'): GATConv(
                hidden * heads, out, heads=1, concat=False,
                dropout=dropout, edge_dim=dfg_edge_dim, add_self_loops=True
            ),
            ('resource', 'performs', 'activity'): GATConv(
                (hidden, hidden * heads), out, heads=1, concat=False,
                dropout=dropout, edge_dim=ra_edge_dim, add_self_loops=False
            ),
        }, aggr='mean')
        
        self.bn2_act = nn.BatchNorm1d(out)
        
        # === Bottleneck prediction head (on activity nodes) ===
        self.predictor = nn.Sequential(
            nn.Linear(out, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.embedding_dim = out
    
    def _project_nodes(self, x_dict):
        """Project each node type into shared hidden dimension."""
        return {
            'activity': F.elu(self.act_proj(x_dict['activity'])),
            'resource': F.elu(self.res_proj(x_dict['resource'])),
        }
    
    def encode(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """Forward pass through HeteroConv layers → node embeddings."""
        # Project to shared dim
        h = self._project_nodes(x_dict)
        
        # Build kwargs for conv with edge_attr
        kwargs = {}
        if edge_attr_dict:
            for key, attr in edge_attr_dict.items():
                kwargs[key] = {'edge_attr': attr}
        
        # Layer 1
        h = self.conv1(h, edge_index_dict, **kwargs) if not kwargs else self._conv_with_attr(self.conv1, h, edge_index_dict, edge_attr_dict)
        
        # BatchNorm + activation per node type
        if 'activity' in h:
            h['activity'] = F.elu(self.bn1_act(h['activity']))
            h['activity'] = F.dropout(h['activity'], p=self.dropout, training=self.training)
        if 'resource' in h:
            # Resource nodes may not get messages from conv1.
            # Keep projected features if missing
            pass
        
        # Layer 2
        h = self._conv_with_attr(self.conv2, h, edge_index_dict, edge_attr_dict)
        
        if 'activity' in h:
            h['activity'] = F.elu(self.bn2_act(h['activity']))
        
        return h
    
    def _conv_with_attr(self, conv_layer, x_dict, edge_index_dict, edge_attr_dict):
        """Run HeteroConv forwarding edge_attr per edge type."""
        # HeteroConv expects edge_attr passed for each edge type
        # We need to pass them as keyword arguments
        kwargs_per_edge = {}
        if edge_attr_dict:
            for edge_type, attr in edge_attr_dict.items():
                if attr is not None and attr.numel() > 0:
                    kwargs_per_edge[edge_type] = {'edge_attr': attr}
        
        return conv_layer(x_dict, edge_index_dict, kwargs_per_edge)
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """Full forward: embeddings + bottleneck predictions."""
        h = self.encode(x_dict, edge_index_dict, edge_attr_dict)
        
        # Predictions only on activity nodes
        act_emb = h.get('activity')
        if act_emb is None:
            raise ValueError("No activity embeddings produced by HeteroConv")
        
        predictions = self.predictor(act_emb).squeeze(-1)
        return predictions, h
    
    def get_flat_embeddings(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """Export embeddings in flat format compatible with RL agent.
        
        Returns:
            embeddings: (n_activities + n_resources, embedding_dim) tensor
            n_activities: int
        """
        h = self.encode(x_dict, edge_index_dict, edge_attr_dict)
        
        act_emb = h.get('activity', torch.zeros(0, self.embedding_dim))
        res_emb = h.get('resource', torch.zeros(0, self.embedding_dim))
        
        # Pad resource embeddings to match activity embedding dim if needed
        if res_emb.shape[-1] != self.embedding_dim and res_emb.numel() > 0:
            pad_proj = nn.Linear(res_emb.shape[-1], self.embedding_dim).to(res_emb.device)
            res_emb = pad_proj(res_emb)
        
        flat = torch.cat([act_emb, res_emb], dim=0)
        return flat, act_emb.shape[0]


class HeteroProcessGNNWithGLU(nn.Module):
    """
    HeteroGNN + Gated Linear Unit (GLU) variant.
    GLU: output = σ(W₁x) ⊙ (W₂x) — helps suppress noisy features.
    """
    
    def __init__(self, activity_in=8, resource_in=6, hidden=64, out=32,
                 heads=4, dropout=0.2, dfg_edge_dim=2, ra_edge_dim=1):
        super().__init__()
        self.dropout = dropout
        self.hidden = hidden
        self.out = out
        
        # Node projections
        self.act_proj = nn.Linear(activity_in, hidden)
        self.res_proj = nn.Linear(resource_in, hidden)
        
        # HeteroConv Layer 1
        self.conv1 = HeteroConv({
            ('activity', 'follows', 'activity'): GATConv(
                hidden, hidden, heads=heads, dropout=dropout,
                edge_dim=dfg_edge_dim, add_self_loops=True
            ),
            ('resource', 'performs', 'activity'): GATConv(
                (hidden, hidden), hidden, heads=heads, dropout=dropout,
                edge_dim=ra_edge_dim, add_self_loops=False
            ),
        }, aggr='mean')
        
        self.bn1_act = nn.BatchNorm1d(hidden * heads)
        
        # GLU layers after conv1 (on activity embeddings)
        self.glu_gate1 = nn.Linear(hidden * heads, hidden * heads)
        self.glu_value1 = nn.Linear(hidden * heads, hidden * heads)
        
        # HeteroConv Layer 2
        self.conv2 = HeteroConv({
            ('activity', 'follows', 'activity'): GATConv(
                hidden * heads, out, heads=1, concat=False,
                dropout=dropout, edge_dim=dfg_edge_dim, add_self_loops=True
            ),
            ('resource', 'performs', 'activity'): GATConv(
                (hidden, hidden * heads), out, heads=1, concat=False,
                dropout=dropout, edge_dim=ra_edge_dim, add_self_loops=False
            ),
        }, aggr='mean')
        
        self.bn2_act = nn.BatchNorm1d(out)
        
        # GLU layers after conv2
        self.glu_gate2 = nn.Linear(out, out)
        self.glu_value2 = nn.Linear(out, out)
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(out, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.embedding_dim = out
    
    def glu(self, x, gate_layer, value_layer):
        """Gated Linear Unit: σ(Wg·x) ⊙ (Wv·x)"""
        return torch.sigmoid(gate_layer(x)) * value_layer(x)
    
    def _project_nodes(self, x_dict):
        return {
            'activity': F.elu(self.act_proj(x_dict['activity'])),
            'resource': F.elu(self.res_proj(x_dict['resource'])),
        }
    
    def _conv_with_attr(self, conv_layer, x_dict, edge_index_dict, edge_attr_dict):
        kwargs_per_edge = {}
        if edge_attr_dict:
            for edge_type, attr in edge_attr_dict.items():
                if attr is not None and attr.numel() > 0:
                    kwargs_per_edge[edge_type] = {'edge_attr': attr}
        return conv_layer(x_dict, edge_index_dict, kwargs_per_edge)
    
    def encode(self, x_dict, edge_index_dict, edge_attr_dict=None):
        h = self._project_nodes(x_dict)
        
        # Layer 1 + GLU
        h = self._conv_with_attr(self.conv1, h, edge_index_dict, edge_attr_dict)
        if 'activity' in h:
            h['activity'] = self.bn1_act(h['activity'])
            h['activity'] = self.glu(h['activity'], self.glu_gate1, self.glu_value1)
            h['activity'] = F.dropout(h['activity'], p=self.dropout, training=self.training)
        
        # Layer 2 + GLU
        h = self._conv_with_attr(self.conv2, h, edge_index_dict, edge_attr_dict)
        if 'activity' in h:
            h['activity'] = self.bn2_act(h['activity'])
            h['activity'] = self.glu(h['activity'], self.glu_gate2, self.glu_value2)
        
        return h
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        h = self.encode(x_dict, edge_index_dict, edge_attr_dict)
        act_emb = h.get('activity')
        if act_emb is None:
            raise ValueError("No activity embeddings produced")
        predictions = self.predictor(act_emb).squeeze(-1)
        return predictions, h
    
    def get_flat_embeddings(self, x_dict, edge_index_dict, edge_attr_dict=None):
        h = self.encode(x_dict, edge_index_dict, edge_attr_dict)
        act_emb = h.get('activity', torch.zeros(0, self.embedding_dim))
        res_emb = h.get('resource', torch.zeros(0, self.embedding_dim))
        if res_emb.shape[-1] != self.embedding_dim and res_emb.numel() > 0:
            pad_proj = nn.Linear(res_emb.shape[-1], self.embedding_dim).to(res_emb.device)
            res_emb = pad_proj(res_emb)
        flat = torch.cat([act_emb, res_emb], dim=0)
        return flat, act_emb.shape[0]
