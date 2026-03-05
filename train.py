"""
SmartScrape — GNN Training Script.

Trains SmartScrapeGNN on annotated web page nodes.
Ground-truth labels are provided as a JSON file (see --data argument).

Usage:
    python train.py --data data/books_labeled.json --epochs 50 --output model.pt

Data format (JSON array of page objects):
[
  {
    "url": "https://books.toscrape.com/...",
    "nodes": [
      {"id": "1", "text": "Tipping the Velvet", "tag": "h1",
       "bbox": [200, 50, 600, 40], "label": "title"},
      {"id": "2", "text": "£53.74", "tag": "p",
       "bbox": [200, 120, 100, 30], "label": "price"},
      {"id": "3", "text": "Add to basket", "tag": "button",
       "bbox": [200, 200, 150, 40], "label": "other"}
    ]
  }
]

Labels: "title", "price", "other"
"""

import argparse
import json
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple

from src.learning.features import FeatureEncoder
from src.learning.graph_builder import FitLayoutParser
from src.learning.gnn_model import SmartScrapeGNN

LABEL_MAP = {"price": 0, "title": 1, "other": 2}
CLASSES = ["price", "title", "other"]


def build_graph_from_nodes(
    nodes: List[Dict], encoder: FeatureEncoder, page_w: float = 1200.0, page_h: float = 1080.0
) -> Tuple[Data, List[int]]:
    """
    Convert a list of annotated page nodes into a PyTorch Geometric Data object.
    Returns (graph_data, labels).
    """
    node_features = []
    labels = []

    for node in nodes:
        text = node.get("text", "")
        tag = node.get("tag", "div")
        bbox = node.get("bbox", [0, 0, 0, 0])
        label_str = node.get("label", "other")

        x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        v_feat = encoder.encode_visual(x, y, w, h, page_w, page_h)
        t_feat = encoder.encode_text(text)
        s_feat = encoder.encode_tag(tag)
        full_vec = torch.cat([v_feat, t_feat, s_feat])
        node_features.append(full_vec)
        labels.append(LABEL_MAP.get(label_str, 2))

    x_tensor = torch.stack(node_features)

    # Build KNN edges (k=3) between nodes — mirrors FitLayoutParser._build_knn_edges
    edges = []
    n = len(nodes)
    for i in range(n):
        dists = []
        xi, yi = float(nodes[i]["bbox"][0]), float(nodes[i]["bbox"][1])
        for j in range(n):
            if i == j:
                continue
            xj, yj = float(nodes[j]["bbox"][0]), float(nodes[j]["bbox"][1])
            d = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
            dists.append((d, j))
        dists.sort()
        for _, nb in dists[:3]:
            edges.append([i, nb])
            edges.append([nb, i])
    edges = list(set(tuple(e) for e in edges))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else \
        torch.zeros((2, 0), dtype=torch.long)

    data = Data(x=x_tensor, edge_index=edge_index)
    return data, labels


def train(args):
    print(f"[Train] Loading data from: {args.data}")

    with open(args.data, "r", encoding="utf-8") as f:
        pages = json.load(f)

    print(f"[Train] Loaded {len(pages)} pages.")

    encoder = FeatureEncoder()
    all_data = []
    all_labels = []

    for page in pages:
        nodes = page.get("nodes", [])
        if not nodes:
            continue
        graph_data, labels = build_graph_from_nodes(nodes, encoder)
        all_data.append(graph_data)
        all_labels.append(labels)

    # Train / validation split (80/20 page-level split)
    indices = list(range(len(all_data)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    input_dim = encoder.get_output_dim()
    model = SmartScrapeGNN(input_dim=input_dim, hidden_dim=64, num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    print(f"[Train] Model input_dim={input_dim}, training on {len(train_idx)} pages, "
          f"validating on {len(val_idx)} pages.")

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for idx in train_idx:
            graph = all_data[idx]
            y = torch.tensor(all_labels[idx], dtype=torch.long)
            optimizer.zero_grad()
            out = model(graph)
            loss = F.nll_loss(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for idx in val_idx:
                graph = all_data[idx]
                y = torch.tensor(all_labels[idx], dtype=torch.long)
                out = model(graph)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(train_idx)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Loss: {avg_loss:.4f} | Val Acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output)

    print(f"\n[Train] Training complete.")
    print(f"[Train] Best validation accuracy: {best_val_acc:.3f}")
    print(f"[Train] Model saved to: {args.output}")
    print(f"\n[Train] To use the trained model, ensure SMARTSCRAPE_MODEL_PATH={args.output} "
          f"in your .env file, or set the environment variable.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SmartScrape GNN")
    parser.add_argument(
        "--data",
        type=str,
        default="data/books_labeled.json",
        help="Path to labeled training data (JSON)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model.pt",
        help="Output path for trained model weights (default: model.pt)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"[Train] ERROR: Data file not found: {args.data}")
        print(
            "[Train] Please create labeled training data. See data format in the docstring above.\n"
            "[Train] Quick start: annotate 50+ pages from books.toscrape.com with\n"
            "[Train]   'title', 'price', 'other' labels per node and save as JSON."
        )
        exit(1)

    train(args)
