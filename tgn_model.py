"""
Phase 2D — Temporal Graph Network (TGN)
Adds time-awareness to the HeteroGNN so node embeddings evolve as events stream in.

Architecture:
    - TimeEncode: sinusoidal encoding of time gaps Δt → d-dim vector
    - TemporalMemory: per-node GRU memory updated with each event
    - ProcessTGN: wraps HeteroProcessGNN + TemporalMemory

Reference: Rossi et al. (2020) "Temporal Graph Networks for Deep Learning on Dynamic Graphs"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from hetero_gnn_model import HeteroProcessGNN


class TimeEncode(nn.Module):
    """Learnable sinusoidal time encoding.

    Maps a scalar time gap Δt into a d-dimensional vector using:
        z(Δt) = cos(ω * Δt + φ)
    where ω and φ are learned parameters.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        # Initialize with harmonics for good starting frequencies
        with torch.no_grad():
            self.w.weight.copy_(
                torch.from_numpy(
                    1.0 / 10 ** (torch.arange(0, dim).float() * 2.0 / dim).numpy()
                ).unsqueeze(1)
            )
            self.w.bias.zero_()

    def forward(self, t):
        """
        Args:
            t: (batch,) or (batch, 1) tensor of time deltas
        Returns:
            (batch, dim) time encoding
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)
        return torch.cos(self.w(t.float()))


class TemporalMemory(nn.Module):
    """Per-node GRU memory, updated with each incoming event.

    Each node maintains a memory vector that gets updated when the node
    participates in an event. This gives the GNN a "sense of history."

    Args:
        n_nodes: total number of nodes (activities + resources)
        memory_dim: dimension of per-node memory vector
        message_dim: dimension of incoming message (event features + time encoding)
    """

    def __init__(self, n_nodes, memory_dim=32, message_dim=32):
        super().__init__()
        self.n_nodes = n_nodes
        self.memory_dim = memory_dim

        # Memory storage (not a parameter — updated in-place)
        self.register_buffer('memory', torch.zeros(n_nodes, memory_dim))
        self.register_buffer('last_update', torch.zeros(n_nodes))

        # GRU cell for memory updates
        self.gru = nn.GRUCell(message_dim, memory_dim)

        # Time encoder
        self.time_enc = TimeEncode(memory_dim)

        # Message projector (event features → message_dim)
        self.msg_proj = nn.Linear(1 + memory_dim, message_dim)  # event_feat + time_enc

    def reset(self):
        """Reset all memories to zero (start of new episode)."""
        self.memory.zero_()
        self.last_update.zero_()

    def get_memory(self, node_ids=None):
        """Get memory vectors for specified nodes (or all)."""
        if node_ids is None:
            return self.memory
        return self.memory[node_ids]

    def update(self, node_ids, event_features, timestamps):
        """Update memory for nodes involved in events.

        Args:
            node_ids: (batch,) node indices
            event_features: (batch, feat_dim) event feature vectors
            timestamps: (batch,) event timestamps
        """
        if len(node_ids) == 0:
            return

        # Compute time deltas since last update
        delta_t = timestamps - self.last_update[node_ids]
        delta_t = delta_t.clamp(min=0)  # safety

        # Encode time
        time_features = self.time_enc(delta_t)  # (batch, memory_dim)

        # Build message: [event_features, time_encoding]
        message_input = torch.cat([event_features, time_features], dim=-1)
        message = self.msg_proj(message_input)  # (batch, message_dim)

        # GRU update
        old_memory = self.memory[node_ids].detach()  # detach to avoid backprop through history
        new_memory = self.gru(message, old_memory)

        # Write back
        self.memory[node_ids] = new_memory
        self.last_update[node_ids] = timestamps


class ProcessTGN(nn.Module):
    """Full Temporal Graph Network for process analysis.

    Combines:
        - HeteroProcessGNN for structure-aware embeddings
        - TemporalMemory for time-aware node states

    The temporal memory is concatenated with static node features before
    message passing, giving the GNN both structural AND temporal context.

    Args:
        n_activities: number of activity nodes
        n_resources: number of resource nodes
        activity_in: activity feature dimension
        resource_in: resource feature dimension
        hidden: hidden dimension
        out: output embedding dimension
        memory_dim: temporal memory dimension
    """

    def __init__(self, n_activities, n_resources, activity_in=8, resource_in=6,
                 hidden=64, out=32, memory_dim=32, heads=4, dropout=0.2,
                 dfg_edge_dim=2, ra_edge_dim=1):
        super().__init__()

        self.n_activities = n_activities
        self.n_resources = n_resources
        self.memory_dim = memory_dim
        self.out = out

        # Temporal memory for all nodes
        n_total = n_activities + n_resources
        self.temporal_memory = TemporalMemory(
            n_nodes=n_total,
            memory_dim=memory_dim,
            message_dim=memory_dim
        )

        # HeteroGNN backbone (input dim augmented with memory)
        self.gnn = HeteroProcessGNN(
            activity_in=activity_in + memory_dim,  # static features + memory
            resource_in=resource_in + memory_dim,
            hidden=hidden,
            out=out,
            heads=heads,
            dropout=dropout,
            dfg_edge_dim=dfg_edge_dim,
            ra_edge_dim=ra_edge_dim
        )

        # Embedding dimension
        self.embedding_dim = out

    def reset_memory(self):
        """Reset temporal memory (call at start of each episode/sequence)."""
        self.temporal_memory.reset()

    def _augment_features(self, x_dict):
        """Concatenate temporal memory with static node features."""
        act_mem = self.temporal_memory.get_memory(
            torch.arange(self.n_activities)
        )
        res_mem = self.temporal_memory.get_memory(
            torch.arange(self.n_activities, self.n_activities + self.n_resources)
        )

        return {
            'activity': torch.cat([x_dict['activity'], act_mem], dim=-1),
            'resource': torch.cat([x_dict['resource'], res_mem], dim=-1),
        }

    def process_events(self, events_batch):
        """Process a batch of temporal events, updating node memories.

        Args:
            events_batch: list of dicts with keys:
                ts, src_idx, dst_idx, features
        """
        if not events_batch:
            return

        src_ids = []
        dst_ids = []
        features = []
        timestamps = []

        for evt in events_batch:
            # src is resource (offset by n_activities)
            src_id = self.n_activities + evt['src_idx']
            dst_id = evt['dst_idx']  # activity node

            src_ids.append(src_id)
            dst_ids.append(dst_id)
            features.append(evt['features'])
            timestamps.append(evt['ts'])

        # Convert to tensors
        device = self.temporal_memory.memory.device
        all_ids = torch.tensor(src_ids + dst_ids, dtype=torch.long, device=device)
        feat_tensor = torch.tensor(features + features, dtype=torch.float32, device=device)
        ts_tensor = torch.tensor(timestamps + timestamps, dtype=torch.float32, device=device)

        # Update memory for both source and destination nodes
        self.temporal_memory.update(all_ids, feat_tensor, ts_tensor)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """Forward pass with temporal context.

        1. Augment static features with temporal memory
        2. Run HeteroGNN
        3. Return predictions + embeddings
        """
        augmented = self._augment_features(x_dict)
        predictions, h = self.gnn(augmented, edge_index_dict, edge_attr_dict)
        return predictions, h

    def get_temporal_embeddings(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """Get embeddings with temporal context for RL agent.

        Returns:
            static_embeddings: (n_nodes, out) from GNN
            temporal_memory: (n_nodes, memory_dim) current memory state
        """
        with torch.no_grad():
            _, h = self.forward(x_dict, edge_index_dict, edge_attr_dict)

            act_emb = h.get('activity', torch.zeros(self.n_activities, self.out))
            res_emb = h.get('resource', torch.zeros(self.n_resources, self.out))

            # Pad resource embeddings if needed
            if res_emb.shape[-1] != self.out and res_emb.numel() > 0:
                res_emb = F.linear(res_emb,
                                   torch.eye(self.out)[:res_emb.shape[-1]].T)

            static = torch.cat([act_emb, res_emb], dim=0)
            memory = self.temporal_memory.get_memory()

        return static, memory
