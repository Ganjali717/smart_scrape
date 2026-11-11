"""
Data-driven constraint calibration.

Hand-written cross-field constraints are only useful if they hold on real
sites. On the training sites we measure, for every non-local constraint, how
often the *gold* record satisfies it, and:

  * proximity      -> max_distance is set to the q-quantile of gold distances
  * ordering       -> kept only if gold records satisfy it on >= min_support of pages
  * exclude_regions, position_band -> kept only if gold nodes satisfy them on >= min_support
  * format, mutual_exclusion -> always kept (definitional)

The report is written next to the results so the paper can state exactly
which constraints were active and why.
"""

from __future__ import annotations

import copy
import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .data import Page
from .schema import ConstraintSpec, Schema
from .solver import node_allowed, structural_distance


def _best_pair_distance(page: Page, a: str, b: str) -> float:
    ia = [i for i, f in page.labels.items() if f == a]
    ib = [i for i, f in page.labels.items() if f == b]
    if not ia or not ib:
        return float("nan")
    return min(structural_distance(page, i, j) for i, j in itertools.product(ia, ib) if i != j)


def _ordering_holds(page: Page, before: str, after: str) -> float:
    ib = [i for i, f in page.labels.items() if f == before]
    ia = [i for i, f in page.labels.items() if f == after]
    if not ia or not ib:
        return float("nan")
    # satisfiable if some gold 'after' node lies at or below some gold 'before' node
    return float(max(page.position(i) for i in ia) >= min(page.position(j) for j in ib))


def _local_holds(page: Page, schema: Schema, c: ConstraintSpec) -> List[float]:
    out = []
    for f in schema.field_names:
        nodes = [i for i, g in page.labels.items() if g == f]
        if not nodes:
            continue
        # satisfied if at least one gold occurrence of the field passes the constraint
        out.append(float(any(node_allowed(page, i, f, schema, [c])[0] for i in nodes)))
    return out


def calibrate(schema: Schema, pages: Sequence[Page], min_support: float = 0.95,
              quantile: float = 0.95, slack: float = 1.0) -> Tuple[Schema, Dict[str, dict]]:
    labelled = [p for p in pages if p.labels]
    new = copy.deepcopy(schema)
    kept: List[ConstraintSpec] = []
    report: Dict[str, dict] = {}
    for c in new.constraints:
        p = c.params
        if c.kind in ("format", "mutual_exclusion"):
            kept.append(c)
            report[c.name] = {"kept": True, "reason": "definitional"}
            continue
        if c.kind == "proximity":
            d = np.array([_best_pair_distance(pg, p["a"], p["b"]) for pg in labelled], dtype=float)
            d = d[~np.isnan(d)]
            if len(d) == 0:
                report[c.name] = {"kept": False, "reason": "no gold pairs"}
                continue
            old = p.get("max_distance")
            p["max_distance"] = float(np.quantile(d, quantile) + slack)
            c.name = f"proximity(a={p['a']},b={p['b']},max_distance={p['max_distance']:.1f})"
            kept.append(c)
            report[c.name] = {"kept": True, "support": float(np.mean(d <= p["max_distance"])),
                              "max_distance_before": old, "max_distance_after": p["max_distance"],
                              "gold_distance_median": float(np.median(d))}
            continue
        if c.kind == "ordering":
            h = np.array([_ordering_holds(pg, p["before"], p["after"]) for pg in labelled], dtype=float)
            h = h[~np.isnan(h)]
        else:  # exclude_regions, position_band
            h = np.array([x for pg in labelled for x in _local_holds(pg, new, c)], dtype=float)
        support = float(h.mean()) if len(h) else 0.0
        ok = support >= min_support
        report[c.name] = {"kept": ok, "support": support}
        if ok:
            kept.append(c)
    new.constraints = kept
    return new, report
