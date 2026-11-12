"""
Constraint-guided record selection.

Given per-node, per-field scores s(n, f) (log-odds of field f against "none"),
select a record

    max  sum_f sum_{n in C_f} s(n,f) * x[n,f]     s.t. constraints of the schema

Three decoders share one interface so that experiments compare like with like:

  * decode_argmax   - unconstrained: best node per field (the "GNN-only" baseline)
  * decode_filtered - node-local constraints only, then best node per field
                      (separates the effect of filtering from joint inference)
  * ConstraintSolver.solve - full ILP with cross-field constraints, null option,
                      ranked relaxation and a provenance trace
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data import Page
from .formats import matches_type
from .schema import ConstraintSpec, Schema


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class FieldChoice:
    node: Optional[int]          # chosen node id, None = field left empty
    text: str
    score: float
    competitors: List[Tuple[int, float]] = field(default_factory=list)  # (node, score), best first


@dataclass
class Record:
    fields: Dict[str, FieldChoice]
    method: str
    status: str = "OPTIMAL"
    objective: float = 0.0
    active: List[str] = field(default_factory=list)
    relaxed: List[str] = field(default_factory=list)

    def values(self) -> Dict[str, str]:
        return {f: c.text for f, c in self.fields.items()}

    def nodes(self) -> Dict[str, Optional[int]]:
        return {f: c.node for f, c in self.fields.items()}

    def provenance(self) -> dict:
        return {
            "method": self.method, "status": self.status, "objective": self.objective,
            "active_constraints": self.active, "relaxed_constraints": self.relaxed,
            "fields": {f: {"node": c.node, "text": c.text, "score": round(c.score, 4),
                           "competitors": [(n, round(s, 4)) for n, s in c.competitors[:3]]}
                       for f, c in self.fields.items()},
        }


# ---------------------------------------------------------------------------
# Structural helpers
# ---------------------------------------------------------------------------
def structural_distance(page: Page, i: int, j: int) -> float:
    """
    Distance between two nodes:
      * DOM tree distance when element ancestry is known (raw HTML),
      * box-centre distance / 100 px when the page is rendered,
      * difference in document order otherwise.
    """
    a, b = page.nodes[i], page.nodes[j]
    if a.ancestors and b.ancestors:
        k = 0
        for x, y in zip(a.ancestors, b.ancestors):
            if x != y:
                break
            k += 1
        return float(len(a.ancestors) + len(b.ancestors) - 2 * k)
    if a.has_geometry() and b.has_geometry():
        ax, ay = a.bbox[0] + a.bbox[2] / 2, a.bbox[1] + a.bbox[3] / 2
        bx, by = b.bbox[0] + b.bbox[2] / 2, b.bbox[1] + b.bbox[3] / 2
        return float(np.hypot(ax - bx, ay - by) / 100.0)
    return float(abs(a.order - b.order))


def node_allowed(page: Page, i: int, fname: str, schema: Schema,
                 constraints: Sequence[ConstraintSpec]) -> Tuple[bool, Optional[str]]:
    """Check node-local constraints. Returns (allowed, name of the blocking constraint)."""
    nd = page.nodes[i]
    for c in constraints:
        if not c.node_local:
            continue
        p = c.params
        if c.kind == "format" and p.get("field") == fname:
            if not matches_type(nd.text, schema.field(fname).type):
                return False, c.name
        elif c.kind == "exclude_regions" and fname in schema.fields_for(p.get("fields")):
            if set(nd.regions) & set(p.get("regions", [])):
                return False, c.name
        elif c.kind == "position_band" and p.get("field") == fname:
            pos = page.position(i)
            if pos > float(p.get("max", 1.0)) or pos < float(p.get("min", 0.0)):
                return False, c.name
    return True, None


def _choice(page: Page, scores: np.ndarray, j: int, node: Optional[int],
            pool: Sequence[int]) -> FieldChoice:
    ranked = sorted(((int(n), float(scores[n, j])) for n in pool), key=lambda t: -t[1])
    if node is None:
        return FieldChoice(None, "", 0.0, ranked[:5])
    return FieldChoice(int(node), page.nodes[node].text, float(scores[node, j]),
                       [r for r in ranked if r[0] != node][:5])


# ---------------------------------------------------------------------------
# Baseline decoders
# ---------------------------------------------------------------------------
def decode_argmax(page: Page, scores: np.ndarray, schema: Schema) -> Record:
    """Best node per field, independently, no constraints (optional fields need score > 0)."""
    out = {}
    pool = list(range(page.n))
    for j, f in enumerate(schema.fields):
        if page.n == 0:
            out[f.name] = FieldChoice(None, "", 0.0)
            continue
        best = int(np.argmax(scores[:, j]))
        if f.cardinality == "optional" and scores[best, j] <= 0:
            out[f.name] = _choice(page, scores, j, None, pool)
        else:
            out[f.name] = _choice(page, scores, j, best, pool)
    return Record(out, method="argmax", active=[])


def decode_filtered(page: Page, scores: np.ndarray, schema: Schema) -> Record:
    """Apply only node-local constraints, then the best node per field."""
    local = [c for c in schema.constraints if c.node_local]
    out = {}
    for j, f in enumerate(schema.fields):
        pool = [i for i in range(page.n) if node_allowed(page, i, f.name, schema, local)[0]]
        if not pool:
            out[f.name] = FieldChoice(None, "", 0.0)
            continue
        best = max(pool, key=lambda i: scores[i, j])
        if f.cardinality == "optional" and scores[best, j] <= 0:
            out[f.name] = _choice(page, scores, j, None, pool)
        else:
            out[f.name] = _choice(page, scores, j, best, pool)
    return Record(out, method="filtered", active=[c.name for c in local])


# ---------------------------------------------------------------------------
# ILP solver
# ---------------------------------------------------------------------------
SOFTENABLE = {"proximity", "ordering"}


class ConstraintSolver:
    """
    soft_lambda: None -> all constraints hard. A number -> cross-field constraints (proximity,
    ordering) become soft: each violated row costs `soft_lambda` in the objective (score units,
    i.e. log-odds). lambda is tuned on validation sites, so constraints that do not transfer to
    new sites are automatically weakened instead of forcing wrong choices.
    """

    def __init__(self, schema: Schema, top_k: int = 20, backend: str = "SCIP", time_limit_ms: int = 2000,
                 soft_lambda: Optional[float] = None):
        self.schema = schema
        self.top_k = top_k
        self.backend = backend
        self.time_limit_ms = time_limit_ms
        self.soft_lambda = soft_lambda

    # candidate sets after node-local constraints
    def candidates(self, page: Page, scores: np.ndarray, constraints: Sequence[ConstraintSpec]
                   ) -> Dict[str, List[int]]:
        cands = {}
        for j, f in enumerate(self.schema.fields):
            allowed = [i for i in range(page.n) if node_allowed(page, i, f.name, self.schema, constraints)[0]]
            allowed.sort(key=lambda i: -scores[i, j])
            cands[f.name] = allowed[: self.top_k]
        return cands

    def solve(self, page: Page, scores: np.ndarray) -> Record:
        cons = list(self.schema.constraints)
        relaxed: List[str] = []
        ranks = sorted({c.rank for c in cons})
        # try the full set, then drop the lowest-rank group each time
        for drop_upto in [None] + ranks:
            active = cons if drop_upto is None else [c for c in cons if c.rank > drop_upto]
            relaxed = [c.name for c in cons if c not in active]
            rec = self._solve_once(page, scores, active)
            if rec is not None:
                rec.relaxed = relaxed
                return rec
        rec = decode_argmax(page, scores, self.schema)
        rec.method, rec.status, rec.relaxed = "ilp", "FALLBACK_ARGMAX", [c.name for c in cons]
        return rec

    def _solve_once(self, page: Page, scores: np.ndarray, active: Sequence[ConstraintSpec]) -> Optional[Record]:
        from ortools.linear_solver import pywraplp

        schema = self.schema
        cands = self.candidates(page, scores, active)
        fidx = {f.name: j for j, f in enumerate(schema.fields)}

        solver = pywraplp.Solver.CreateSolver(self.backend)
        if solver is None:
            raise RuntimeError(f"OR-Tools backend {self.backend} unavailable")
        solver.SetTimeLimit(self.time_limit_ms)

        x: Dict[Tuple[int, str], object] = {}
        for f, pool in cands.items():
            for i in pool:
                x[i, f] = solver.BoolVar(f"x_{i}_{f}")

        # cardinality (never relaxed): "one" -> exactly one if any candidate, "optional" -> at most one
        for fs in schema.fields:
            vars_f = [x[i, fs.name] for i in cands[fs.name]]
            if not vars_f:
                if fs.cardinality == "one":
                    return None  # infeasible under the current constraints -> relax
                continue
            if fs.cardinality == "one":
                solver.Add(sum(vars_f) == 1)
            else:
                solver.Add(sum(vars_f) <= 1)

        lam = self.soft_lambda
        slacks = []

        def add_row(lhs, rhs, soft: bool):
            if soft:
                z = solver.BoolVar(f"z_{len(slacks)}")
                slacks.append(z)
                solver.Add(lhs <= rhs + z)
            else:
                solver.Add(lhs <= rhs)

        for c in active:
            p = c.params
            soft = lam is not None and c.kind in SOFTENABLE
            if c.kind == "mutual_exclusion":
                by_node: Dict[int, list] = {}
                for (i, f), v in x.items():
                    by_node.setdefault(i, []).append(v)
                for vs in by_node.values():
                    if len(vs) > 1:
                        solver.Add(sum(vs) <= 1)
            elif c.kind == "proximity":
                a, b, dmax = p["a"], p["b"], float(p.get("max_distance", 10))
                for i in cands[a]:
                    near = [x[j, b] for j in cands[b] if j != i and structural_distance(page, i, j) <= dmax]
                    add_row(x[i, a], sum(near) if near else 0, soft)
            elif c.kind == "ordering":
                before, after = p["before"], p["after"]
                for i in cands[after]:
                    for j in cands[before]:
                        if i != j and page.position(i) < page.position(j):
                            add_row(x[i, after] + x[j, before], 1, soft)
            # node-local kinds are already applied through the candidate sets

        obj = solver.Objective()
        for (i, f), v in x.items():
            obj.SetCoefficient(v, float(scores[i, fidx[f]]))
        for z in slacks:
            obj.SetCoefficient(z, -float(lam))
        obj.SetMaximization()
        status = solver.Solve()
        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            return None

        out = {}
        for fs in schema.fields:
            j = fidx[fs.name]
            chosen = [i for i in cands[fs.name] if x[i, fs.name].solution_value() > 0.5]
            out[fs.name] = _choice(page, scores, j, chosen[0] if chosen else None, cands[fs.name])
        rec = Record(out, method="ilp" if lam is None else "ilp-soft",
                     status="OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FEASIBLE",
                     objective=float(obj.Value()), active=[c.name for c in active])
        if slacks:
            rec.active = rec.active + [f"soft_violations={int(sum(z.solution_value() > 0.5 for z in slacks))}",
                                       f"lambda={lam}"]
        return rec


def values_to_record(page: Page, values: Dict[str, str], schema: Schema, method: str) -> Record:
    """Wrap a baseline's {field: value} output as a Record; each value is grounded to the first
    node whose normalised text equals it (None when the value does not occur on the page)."""
    from .data import normalize_text

    index: Dict[str, int] = {}
    for nd in page.nodes:
        index.setdefault(normalize_text(nd.text), nd.id)
    out = {}
    for f in schema.field_names:
        v = (values.get(f) or "").strip()
        node = index.get(normalize_text(v)) if v else None
        out[f] = FieldChoice(node, v, 0.0)
    return Record(out, method=method, status="BASELINE")


# ---------------------------------------------------------------------------
# Constraint checking on any record (used to count violations for every method)
# ---------------------------------------------------------------------------
def violations(page: Page, record: Record, schema: Schema) -> List[str]:
    v: List[str] = []
    chosen = record.nodes()
    for fs in schema.fields:
        if fs.cardinality == "one" and not record.fields[fs.name].text:
            v.append(f"cardinality({fs.name})")
    for c in schema.constraints:
        p = c.params
        if c.node_local:
            for f, i in chosen.items():
                if i is None:
                    continue
                ok, _ = node_allowed(page, i, f, schema, [c])
                if not ok:
                    v.append(c.name)
        elif c.kind == "mutual_exclusion":
            used = [i for i in chosen.values() if i is not None]
            if len(used) != len(set(used)):
                v.append(c.name)
        elif c.kind == "proximity":
            i, j = chosen.get(p["a"]), chosen.get(p["b"])
            if i is not None and (j is None or structural_distance(page, i, j) > float(p.get("max_distance", 10))):
                v.append(c.name)
        elif c.kind == "ordering":
            i, j = chosen.get(p["after"]), chosen.get(p["before"])
            if i is not None and j is not None and page.position(i) < page.position(j):
                v.append(c.name)
    return v
