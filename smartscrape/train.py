"""Training, calibration and scoring for the RGCN scorer."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .data import Page, text_matches
from .features import FeatureEncoder
from .graph import build_edges
from .model import RGCNScorer
from .schema import Schema


@dataclass
class Encoded:
    page: Page
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    y: torch.Tensor           # [N] class index, num_fields = none


def encode_pages(pages: Sequence[Page], schema: Schema, encoder: FeatureEncoder,
                 relations=(0, 1, 2, 3)) -> List[Encoded]:
    fidx = {f: j for j, f in enumerate(schema.field_names)}
    none = len(fidx)
    out = []
    for p in pages:
        x = torch.from_numpy(encoder.encode(p))
        ei, et = build_edges(p, use_relations=relations)
        y = torch.full((p.n,), none, dtype=torch.long)
        for i, f in p.labels.items():
            if f in fidx:
                y[i] = fidx[f]
        out.append(Encoded(p, x, ei, et, y))
    return out


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _field_hit_rate(model, data: Sequence[Encoded], schema: Schema, device) -> float:
    """Fast validation metric: share of (page, field) where the arg-max node text matches gold."""
    hits = total = 0
    model.eval()
    with torch.no_grad():
        for d in data:
            if d.page.n == 0:
                continue
            s = model.field_scores(d.x.to(device), d.edge_index.to(device), d.edge_type.to(device)).cpu().numpy()
            for j, f in enumerate(schema.field_names):
                gold = d.page.gold.get(f)
                if not gold:
                    continue
                total += 1
                hits += int(text_matches(d.page.nodes[int(s[:, j].argmax())].text, gold))
    return hits / max(1, total)


def train_scorer(train: Sequence[Encoded], val: Sequence[Encoded], schema: Schema, in_dim: int,
                 epochs: int = 30, lr: float = 2e-3, hidden: int = 128, layers: int = 2,
                 pos_weight: float = 20.0, patience: int = 6, seed: int = 0,
                 device: Optional[str] = None, log=print) -> RGCNScorer:
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    nf = len(schema.field_names)
    model = RGCNScorer(in_dim, nf, hidden=hidden, layers=layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    weight = torch.tensor([pos_weight] * nf + [1.0], device=device)

    best, best_state, bad = -1.0, None, 0
    order = list(range(len(train)))
    for ep in range(1, epochs + 1):
        model.train()
        random.shuffle(order)
        tot = 0.0
        for k in order:
            d = train[k]
            if d.page.n == 0:
                continue
            opt.zero_grad()
            out = model(d.x.to(device), d.edge_index.to(device), d.edge_type.to(device))
            loss = F.cross_entropy(out, d.y.to(device), weight=weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
        score = _field_hit_rate(model, val, schema, device) if val else 0.0
        log(f"  epoch {ep:3d}  loss {tot / max(1, len(order)):.4f}  val field-hit {score:.3f}")
        if score > best + 1e-4:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    if val:
        fit_temperature(model, val, device)
    return model.cpu()


def fit_temperature(model: RGCNScorer, val: Sequence[Encoded], device) -> float:
    """Temperature scaling (Guo et al. 2017) on validation nodes, grid search on NLL."""
    model.eval()
    logits, ys = [], []
    with torch.no_grad():
        for d in val:
            if d.page.n:
                logits.append(model(d.x.to(device), d.edge_index.to(device), d.edge_type.to(device)).cpu())
                ys.append(d.y)
    if not logits:
        return 1.0
    L, Y = torch.cat(logits), torch.cat(ys)
    best_t, best_nll = 1.0, float("inf")
    for t in np.exp(np.linspace(np.log(0.25), np.log(8.0), 41)):
        nll = float(F.cross_entropy(L / t, Y))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    model.temperature.fill_(best_t)
    return best_t


def score_pages(model: RGCNScorer, data: Sequence[Encoded]) -> List[np.ndarray]:
    model.eval()
    out = []
    for d in data:
        if d.page.n == 0:
            out.append(np.zeros((0, model.out.out_features - 1), dtype=np.float32))
            continue
        out.append(model.field_scores(d.x, d.edge_index, d.edge_type).numpy())
    return out


# ---------------------------------------------------------------------------
# Model bundles: weights + everything needed to rebuild the scorer
# ---------------------------------------------------------------------------
def save_bundle(model: RGCNScorer, schema: Schema, model_cfg: dict, path: str) -> None:
    torch.save({
        "state_dict": model.state_dict(),
        "fields": schema.field_names,
        "in_dim": model.inp[0].in_features,
        "hidden": model.inp[0].out_features,
        "layers": len(model.convs),
        "sentence_model": (model_cfg or {}).get("sentence_model"),
        "exclude_features": list((model_cfg or {}).get("exclude_features", [])),
    }, path)


def load_bundle(path: str):
    b = torch.load(path, map_location="cpu", weights_only=False)
    model = RGCNScorer(b["in_dim"], len(b["fields"]), hidden=b["hidden"], layers=b["layers"])
    model.load_state_dict(b["state_dict"])
    model.eval()
    return model, b
