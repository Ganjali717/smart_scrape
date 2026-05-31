"""
Shared node-encoding and graph-construction primitives.

SINGLE SOURCE OF TRUTH used by BOTH:
  * training  (train_gnn.py)
  * inference (graph_builder.py / pipeline)

Keeping feature order, visual normalization, and edge construction in one
place guarantees that the graph seen at inference is identical in
construction to the graph the GNN was trained on. Previously the trainer
used 1-D y-distance KNN (bidirectional, no containment) while the runtime
used 2-D Euclidean KNN (unidirectional) + containment, which is a
train/inference skew. This module removes that skew.

Feature layout (146 dims):
  text(128 char-ngram hash)  +  visual(4 normalized bbox)  +  tag(14 one-hot)
"""

import math
import torch

# ── Dimensions ──────────────────────────────────────────────────────────
TEXT_DIM = 128          # char n-gram hash size (matches bert-tiny width)
VISUAL_DIM = 4          # x, y, w, h (normalized)

TAG_MAP = {
    "div": 0, "span": 1, "a": 2, "img": 3, "p": 4,
    "h1": 5, "h2": 6, "li": 7, "ul": 8, "table": 9,
    "form": 10, "input": 11, "button": 12,
}
TAG_DIM = len(TAG_MAP) + 1          # +1 for unknown/other tag  -> 14
INPUT_DIM = TEXT_DIM + VISUAL_DIM + TAG_DIM   # 146

# Fixed visual normalizers. Using CONSTANTS (not per-page estimates) makes the
# normalization identical in training and inference. Values reflect the
# dataset's coordinate space (~1200 px wide, up to ~2400 px tall).
PAGE_W = 1200.0
PAGE_H = 2400.0

# Label space. Dataset uses 'none' for background; map it to class 2.
LABEL_MAP = {"price": 0, "title": 1, "other": 2, "none": 2}
CLASSES = ["price", "title", "other"]


# ── Feature encoders ────────────────────────────────────────────────────
def encode_text(text) -> torch.Tensor:
    """Deterministic char 2-/3-gram hash into a fixed 128-dim L2-normalized vec."""
    vec = [0.0] * TEXT_DIM
    text = str(text).lower().strip()
    if not text:
        return torch.zeros(TEXT_DIM, dtype=torch.float32)
    for n in (2, 3):
        for i in range(len(text) - n + 1):
            h = 0
            for ch in text[i:i + n]:
                h = (h * 31 + ord(ch)) & 0xFFFFFF
            vec[h % TEXT_DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return torch.tensor([v / norm for v in vec], dtype=torch.float32)


def encode_visual(bbox) -> torch.Tensor:
    """Normalize [x, y, w, h] by FIXED page constants (train == inference)."""
    if not bbox or len(bbox) < 4:
        return torch.zeros(VISUAL_DIM, dtype=torch.float32)
    x, y, w, h = (float(b) for b in bbox[:4])
    return torch.tensor([x / PAGE_W, y / PAGE_H, w / PAGE_W, h / PAGE_H],
                        dtype=torch.float32)


def encode_tag(tag) -> torch.Tensor:
    """One-hot HTML tag (lowercased); unknown -> last bucket."""
    idx = TAG_MAP.get(str(tag).lower(), TAG_DIM - 1)
    v = torch.zeros(TAG_DIM, dtype=torch.float32)
    v[idx] = 1.0
    return v


def node_features(node: dict) -> torch.Tensor:
    """Full 146-dim feature vector for one node dict {text, tag, bbox}."""
    return torch.cat([
        encode_text(node.get("text", "")),
        encode_visual(node.get("bbox", [])),
        encode_tag(node.get("tag", "div")),
    ])


# ── Graph construction (the canonical edge builder) ─────────────────────
def _bbox4(node):
    b = node.get("bbox") or [0, 0, 0, 0]
    try:
        return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
    except (TypeError, ValueError, IndexError):
        return [0.0, 0.0, 0.0, 0.0]


def build_edges(raw_nodes, k: int = 3) -> torch.Tensor:
    """
    Hybrid graph used identically in training and inference:
      (1) KNN by 2-D Euclidean distance of bbox top-left, k neighbours,
          BIDIRECTIONAL,
      (2) containment edges (node i geometrically contains node j),
          bidirectional.
    Returns edge_index [2, E]; empty graphs get a single self-loop on node 0.

    Vectorized with numpy (deterministic, same edge set as the naive version).
    """
    import numpy as np

    n = len(raw_nodes)
    if n < 2:
        return torch.tensor([[0], [0]], dtype=torch.long)

    b = np.array([_bbox4(nd) for nd in raw_nodes], dtype=np.float64)  # [n,4]
    x, y, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    src, dst = [], []

    # (1) KNN on top-left corner, 2-D Euclidean, bidirectional
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    np.fill_diagonal(d2, np.inf)
    kk = min(k, n - 1)
    nn = np.argpartition(d2, kk - 1, axis=1)[:, :kk]  # k nearest per row
    rows = np.repeat(np.arange(n), kk)
    cols = nn.reshape(-1)
    src.extend(rows.tolist()); dst.extend(cols.tolist())          # i -> j
    src.extend(cols.tolist()); dst.extend(rows.tolist())          # j -> i

    # (2) containment (i contains j), bidirectional
    x2, y2 = x + w, y + h
    contains = (
        (x[:, None] <= x[None, :]) & (y[:, None] <= y[None, :]) &
        (x2[:, None] >= x2[None, :]) & (y2[:, None] >= y2[None, :])
    )
    np.fill_diagonal(contains, False)
    ci, cj = np.where(contains)
    src.extend(ci.tolist()); dst.extend(cj.tolist())
    src.extend(cj.tolist()); dst.extend(ci.tolist())

    edges = set(zip(src, dst))
    if not edges:
        return torch.tensor([[0], [0]], dtype=torch.long)
    return torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()


def page_to_graph(nodes):
    """Build (Data, labels) for a labeled page. Used by the trainer."""
    from torch_geometric.data import Data
    feats = torch.stack([node_features(nd) for nd in nodes])
    labels = torch.tensor([LABEL_MAP.get(nd.get("label"), 2) for nd in nodes],
                          dtype=torch.long)
    edge_index = build_edges(nodes, k=3)
    return Data(x=feats, edge_index=edge_index), labels
