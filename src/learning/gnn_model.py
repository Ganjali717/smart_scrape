import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SmartScrapeGNN(nn.Module):
    """
    Graph Neural Network for Web Information Extraction.
    Architecture: 2-layer GCN + Softmax Classifier.
    Input: Feature Vectors (Visual + Text + Tag)
    Output: Class Probabilities (Price, Title, Image, Other)
    """

    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SmartScrapeGNN, self).__init__()

        # Первый слой графовой свертки
        # Позволяет узлу "узнать" о соседях
        self.conv1 = GCNConv(input_dim, hidden_dim)

        # Второй слой для углубления абстракции
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Финальный классификатор (линейный слой)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 1. Первая свертка + ReLU (функция активации)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)  # Защита от переобучения

        # 2. Вторая свертка
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # 3. Классификация
        out = self.classifier(x)

        # Возвращаем Log-Softmax (логарифм вероятностей)
        return F.log_softmax(out, dim=1)
