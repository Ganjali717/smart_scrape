"""Relational GNN scorer: per node, a distribution over schema fields + "none"."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

from .graph import NUM_RELATIONS


class RGCNScorer(nn.Module):
    def __init__(self, in_dim: int, num_fields: int, hidden: int = 128, layers: int = 2,
                 dropout: float = 0.2, num_relations: int = NUM_RELATIONS, num_bases: int = 4):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.convs = nn.ModuleList([
            RGCNConv(hidden, hidden, num_relations=num_relations, num_bases=num_bases, aggr="mean")
            for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.dropout = dropout
        # skip connection from the input projection lets the model fall back on node-local evidence
        self.out = nn.Linear(hidden * 2, num_fields + 1)  # last class = none
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, x, edge_index, edge_type):
        h0 = self.inp(x)
        h = h0
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + F.relu(conv(h, edge_index, edge_type)))
            h = F.dropout(h, self.dropout, self.training)
        return self.out(torch.cat([h, h0], dim=-1))

    @torch.no_grad()
    def field_scores(self, x, edge_index, edge_type) -> torch.Tensor:
        """s(n, f) = log p(f|n) - log p(none|n), temperature-calibrated. Shape [N, F]."""
        self.eval()
        logits = self.forward(x, edge_index, edge_type) / self.temperature
        logp = F.log_softmax(logits, dim=-1)
        return logp[:, :-1] - logp[:, -1:]
