"""
Label-free reliability signals (paper Sect. 4.8).

  stability margin   sigma_f(P) = s(best) - s(second best) among the field's candidates;
                     page score = min over fields. Background nodes do not enter it.
  disagreement       share of fields where the constrained choice differs from the
                     unconstrained arg-max, and the normalised objective gap.

Higher risk = more likely that the record is wrong. `risk()` combines both
into one number in [0, 1] usable for ranking pages for re-annotation.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .solver import Record


def stability(record: Record) -> Dict[str, float]:
    per_field = {}
    for f, c in record.fields.items():
        if c.node is None:
            best = c.competitors[0][1] if c.competitors else 0.0
            # empty optional field: margin = how clearly the best candidate lost to null
            per_field[f] = float(-best) if best < 0 else 0.0
            continue
        second = c.competitors[0][1] if c.competitors else c.score - 10.0
        per_field[f] = float(c.score - second)
    return per_field


def page_stability(record: Record) -> float:
    s = stability(record)
    return float(min(s.values())) if s else 0.0


def disagreement(constrained: Record, unconstrained: Record) -> Dict[str, float]:
    fields = list(constrained.fields)
    differ = sum(1 for f in fields if constrained.fields[f].node != unconstrained.fields[f].node)
    obj_c = sum(c.score for c in constrained.fields.values() if c.node is not None)
    obj_u = sum(c.score for c in unconstrained.fields.values() if c.node is not None)
    gap = max(0.0, obj_u - obj_c) / (abs(obj_u) + 1e-6)
    return {"field_disagreement": differ / max(1, len(fields)), "objective_gap": float(gap)}


def risk(record: Record, unconstrained: Record, margin_scale: float = 2.0) -> float:
    """Single score in [0,1]; higher = less reliable."""
    sig = page_stability(record)
    d = disagreement(record, unconstrained)
    r_margin = float(np.exp(-max(sig, 0.0) / margin_scale))
    return float(np.clip(0.5 * r_margin + 0.25 * d["field_disagreement"] + 0.25 * min(1.0, d["objective_gap"]), 0, 1))
