"""
Typed page graph for the relational GNN.

Relations:
  0  parent -> child   (DOM, or geometric containment for rendered pages without DOM)
  1  child  -> parent
  2  sequence          (previous / next node in reading order, both directions)
  3  spatial kNN       (box centres, both directions; only when the page is rendered)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from .data import Page

NUM_RELATIONS = 4
REL_NAMES = ["dom_down", "dom_up", "sequence", "spatial_knn"]


def build_edges(page: Page, k: int = 3, use_relations=(0, 1, 2, 3)) -> Tuple[torch.Tensor, torch.Tensor]:
    src: List[int] = []
    dst: List[int] = []
    typ: List[int] = []

    def add(a, b, t):
        if t in use_relations:
            src.append(a); dst.append(b); typ.append(t)

    for nd in page.nodes:
        if nd.parent >= 0:
            add(nd.parent, nd.id, 0)
            add(nd.id, nd.parent, 1)
    for i in range(page.n - 1):
        add(i, i + 1, 2)
        add(i + 1, i, 2)

    geo = [nd.id for nd in page.nodes if nd.has_geometry()]
    if len(geo) > 1 and 3 in use_relations:
        c = np.array([[page.nodes[i].bbox[0] + page.nodes[i].bbox[2] / 2,
                       page.nodes[i].bbox[1] + page.nodes[i].bbox[3] / 2] for i in geo])
        d2 = ((c[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        kk = min(k, len(geo) - 1)
        nn = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
        for a in range(len(geo)):
            for b in nn[a]:
                add(geo[a], geo[int(b)], 3)
                add(geo[int(b)], geo[a], 3)

    if not src:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,), dtype=torch.long)
    e = torch.tensor([src, dst], dtype=torch.long)
    t = torch.tensor(typ, dtype=torch.long)
    # deduplicate (same edge, same type)
    key = e[0] * (page.n + 1) * NUM_RELATIONS + e[1] * NUM_RELATIONS + t
    _, idx = np.unique(key.numpy(), return_index=True)
    idx = torch.as_tensor(np.sort(idx))
    return e[:, idx], t[idx]
