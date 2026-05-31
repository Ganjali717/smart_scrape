"""
SmartScrape paper-protocol evaluator.

This script is intentionally strict. It does NOT fabricate the paper's 100-page
claim. For the exact paper protocol you must provide at least 81 annotated pages
for the stated split: 40 train, 11 validation, 30 held-out test pages. The paper
mentions 100 annotated pages; the remaining pages may be unused or reserved.

Usage:
    python evaluate_paper_protocol.py --data data/labeled.json --model model.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from src.learning import encoding
from src.learning.features import FeatureEncoder
from src.learning.gnn_model import SmartScrapeGNN
from src.reasoning.solver_fixed import ConstraintSolver

CLASSES = ["price", "title", "other"]
SEED = 42


def split_indices(n_pages: int):
    if n_pages < 81:
        raise ValueError(
            f"Paper protocol requires at least 81 annotated pages for 40/11/30 split; found {n_pages}. "
            "Do not report the paper's 30-page held-out ablation until more pages are annotated."
        )
    rng = np.random.default_rng(SEED)
    idx = np.arange(n_pages)
    rng.shuffle(idx)
    return idx[:40].tolist(), idx[40:51].tolist(), idx[51:81].tolist()


def field_targets(nodes: List[Dict]):
    title = next((n for n in nodes if n.get("label") == "title"), None)
    price = next((n for n in nodes if n.get("label") == "price"), None)
    return title, price


def greedy(raw_nodes, scores):
    result = {}
    for field, j in {"price": 0, "title": 1}.items():
        i = int(np.argmax(scores[:, j]))
        result[field] = {"text": raw_nodes[i].get("text", ""), "bbox": raw_nodes[i].get("bbox", [0, 0, 0, 0])}
    return result


def is_match(expected: str, got: str) -> bool:
    expected = str(expected).replace("£", "").strip().lower()
    got = str(got).replace("£", "").strip().lower()
    return bool(expected) and expected[:8] in got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/labeled_2.json")
    ap.add_argument("--model", default="model.pt")
    args = ap.parse_args()

    pages = json.loads(Path(args.data).read_text(encoding="utf-8"))
    train_idx, val_idx, test_idx = split_indices(len(pages))

    encoder = FeatureEncoder()
    model = SmartScrapeGNN(input_dim=encoder.get_output_dim(), hidden_dim=64, num_classes=3)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()
    solver = ConstraintSolver(CLASSES)

    metrics = {
        "greedy": {"title": 0, "price": 0, "both": 0, "violations": 0},
        "ilp": {"title": 0, "price": 0, "both": 0, "violations": 0},
    }

    for idx in test_idx:
        page = pages[idx]
        nodes = page["nodes"]
        title_node, price_node = field_targets(nodes)
        if not title_node or not price_node:
            continue

        graph, _ = encoding.page_to_graph(nodes)
        with torch.no_grad():
            scores = torch.exp(model(graph)).numpy()

        for mode in ("greedy", "ilp"):
            if mode == "greedy":
                res = greedy(nodes, scores)
            else:
                page_h = max(float(n.get("bbox", [0, 0, 0, 0])[1]) + float(n.get("bbox", [0, 0, 0, 0])[3]) for n in nodes)
                res = solver.solve(nodes, scores, page_h)

            t_ok = is_match(title_node["text"], res.get("title", {}).get("text", ""))
            p_ok = is_match(price_node["text"], res.get("price", {}).get("text", ""))
            metrics[mode]["title"] += int(t_ok)
            metrics[mode]["price"] += int(p_ok)
            metrics[mode]["both"] += int(t_ok and p_ok)
            for field in ("title", "price"):
                bbox = res.get(field, {}).get("bbox", [0, 0, 0, 0])
                if len(bbox) >= 2 and float(bbox[1]) > 0.8 * 1080:
                    metrics[mode]["violations"] += 1

    n = len(test_idx)
    print(f"Paper protocol split: train=40, val=11, heldout={n}")
    for mode in ("greedy", "ilp"):
        m = metrics[mode]
        print(
            f"{mode.upper():6} | Title Acc. {m['title']/n:.2%} | Price Acc. {m['price']/n:.2%} | "
            f"Both {m['both']/n:.2%} | Violations {m['violations']}"
        )


if __name__ == "__main__":
    main()
