"""Metrics: SWDE-style page-level F1, record exact match, violation rate, reliability AUROC."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np

from .data import Page, text_matches
from .schema import Schema
from .solver import Record, violations


def field_outcomes(page: Page, rec: Record, schema: Schema) -> Dict[str, str]:
    """Per field: 'tp', 'fp', 'fn', 'fp+fn' (wrong value), or 'tn' (no gold, nothing predicted)."""
    out = {}
    for f in schema.field_names:
        gold = page.gold.get(f) or []
        pred = rec.fields[f].text or ""
        if gold:
            if pred and text_matches(pred, gold):
                out[f] = "tp"
            elif pred:
                out[f] = "fp+fn"
            else:
                out[f] = "fn"
        else:
            out[f] = "fp" if pred else "tn"
    return out


def record_correct(outcomes: Dict[str, str]) -> bool:
    return all(o in ("tp", "tn") for o in outcomes.values())


def aggregate(pages: Sequence[Page], records: Sequence[Record], schema: Schema) -> Dict[str, float]:
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    exact = viol = 0
    nonempty = grounded = 0
    for p, r in zip(pages, records):
        for c in r.fields.values():
            if c.text:
                nonempty += 1
                grounded += int(c.node is not None)
        oc = field_outcomes(p, r, schema)
        for f, o in oc.items():
            if o == "tp":
                counts[f]["tp"] += 1
            if "fp" in o:
                counts[f]["fp"] += 1
            if "fn" in o:
                counts[f]["fn"] += 1
        exact += int(record_correct(oc))
        viol += int(bool(violations(p, r, schema)))
    res: Dict[str, float] = {}
    f1s = []
    for f in schema.field_names:
        c = counts[f]
        prec = c["tp"] / max(1, c["tp"] + c["fp"])
        rec = c["tp"] / max(1, c["tp"] + c["fn"])
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        res[f"f1/{f}"] = f1
        f1s.append(f1)
    n = max(1, len(pages))
    res["f1/macro"] = float(np.mean(f1s)) if f1s else 0.0
    res["exact_match"] = exact / n
    res["violation_rate"] = viol / n
    res["grounded_rate"] = grounded / max(1, nonempty)   # predicted values that occur verbatim on the page
    res["pages"] = float(len(pages))
    return res


def reliability_auc(risks: Sequence[float], wrong: Sequence[bool]) -> Dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(wrong, dtype=int)
    if y.min() == y.max():
        return {"auroc": float("nan"), "auprc": float("nan"), "error_rate": float(y.mean())}
    return {"auroc": float(roc_auc_score(y, risks)), "auprc": float(average_precision_score(y, risks)),
            "error_rate": float(y.mean())}
