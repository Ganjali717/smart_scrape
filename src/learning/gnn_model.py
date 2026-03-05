import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SmartScrapeGNN(nn.Module):
    """
    Graph Neural Network for Web Information Extraction.
    Architecture: 2-layer GCN + Softmax Classifier.

    ВАЖНО: архитектура должна точно совпадать с train_gnn.py:
      conv1: input_dim → 64
      conv2: 64 → 32
      fc:    32 → num_classes
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 3):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)       # 147 → 64
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2) # 64  → 32
        self.fc    = nn.Linear(hidden_dim // 2, num_classes)  # 32 → 3

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        out = self.fc(x)

        return F.log_softmax(out, dim=1)