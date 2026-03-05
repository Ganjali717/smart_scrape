from ortools.linear_solver import pywraplp
import numpy as np
from typing import List, Dict, Any

from config import FOOTER_THRESHOLD


class ConstraintSolver:
    """
    Constraint-guided inference (Eq. 1 in the dissertation).

    ILP formulation:
        max  Σ_{i,f}  s(i, f) * x_{i,f}
        s.t. Γ(x) = True

    Constraints:
      Γ1 — Uniqueness:     at most one Price and one Title per page
      Γ2 — Footer trap:    y > 0.80 * page_height  →  not Price/Title
      Γ3 — Product zone:   y > PRODUCT_ZONE_MAX    →  not Price/Title
                           (excludes "You may also like" section)
      Γ4 — Integrity:      each node has exactly one label
    """

    # On books.toscrape.com the product info (title + price) is always
    # in the top ~500px. Everything below is "recommended books" section
    # with duplicate prices. This constant enforces that constraint.
    PRODUCT_ZONE_MAX_PX = 500.0

    def __init__(self, classes: List[str]):
        self.classes    = classes
        self.cls_to_idx = {name: i for i, name in enumerate(classes)}

    def solve(
        self,
        raw_nodes:   List[Dict],
        probs:       np.ndarray,
        page_height: float = 1080.0,
    ) -> Dict[str, Any]:

        num_nodes   = len(raw_nodes)
        num_classes = len(self.classes)

        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            print("[Solver] SCIP unavailable — greedy fallback.")
            return self._greedy_fallback(raw_nodes, probs)

        # ── 1. Variables: x[i, j] ∈ {0, 1} ──────────────────────────
        x = {(i, j): solver.IntVar(0, 1, f"x_{i}_{j}")
             for i in range(num_nodes)
             for j in range(num_classes)}

        price_idx = self.cls_to_idx.get("price")
        title_idx = self.cls_to_idx.get("title")

        # ── 2. Constraints Γ ─────────────────────────────────────────

        # Γ4 — Integrity: each node gets exactly one label
        for i in range(num_nodes):
            solver.Add(solver.Sum([x[i, j] for j in range(num_classes)]) == 1)

        # Γ1 — Uniqueness: at most one Price and one Title per page
        for cls_name in ("title", "price"):
            if cls_name in self.cls_to_idx:
                ci = self.cls_to_idx[cls_name]
                solver.Add(
                    solver.Sum([x[i, ci] for i in range(num_nodes)]) <= 1
                )

        for i, node in enumerate(raw_nodes):
            bbox   = node.get("bbox") or [0, 0, 0, 0]
            node_y = float(bbox[1]) if len(bbox) > 1 else 0.0

            # Γ2 — Footer trap
            if node_y > page_height * FOOTER_THRESHOLD:
                if price_idx is not None:
                    solver.Add(x[i, price_idx] == 0)
                if title_idx is not None:
                    solver.Add(x[i, title_idx] == 0)

            # Γ3 — Product zone
            # Nodes below PRODUCT_ZONE_MAX_PX are in the
            # "recommended / related books" section and must not be
            # selected as Price or Title.
            if node_y > self.PRODUCT_ZONE_MAX_PX:
                if price_idx is not None:
                    solver.Add(x[i, price_idx] == 0)
                if title_idx is not None:
                    solver.Add(x[i, title_idx] == 0)

        # ── 3. Objective: maximise total confidence ───────────────────
        obj = solver.Objective()
        for i in range(num_nodes):
            for j in range(num_classes):
                obj.SetCoefficient(x[i, j], float(probs[i][j]))
        obj.SetMaximization()

        # ── 4. Solve ──────────────────────────────────────────────────
        status = solver.Solve()
        result: Dict[str, Any] = {}

        if status == pywraplp.Solver.OPTIMAL:
            for i in range(num_nodes):
                for j in range(num_classes):
                    if x[i, j].solution_value() > 0.5:
                        cls = self.classes[j]
                        if cls != "other":
                            result[cls] = {
                                "text":       raw_nodes[i].get("text", ""),
                                "confidence": float(probs[i][j]),
                                "bbox":       (raw_nodes[i].get("bbox") or [0,0,0,0]),
                            }
        else:
            print("[Solver] No optimal solution — greedy fallback.")
            return self._greedy_fallback(raw_nodes, probs)

        return result

    # ── Greedy fallback ───────────────────────────────────────────────
    def _greedy_fallback(self, raw_nodes: List[Dict], probs: np.ndarray) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for i, node in enumerate(raw_nodes):
            best_j = int(np.argmax(probs[i]))
            cls    = self.classes[best_j]
            if cls == "other":
                continue
            conf = float(probs[i][best_j])
            if cls not in result or conf > result[cls]["confidence"]:
                result[cls] = {
                    "text":       node.get("text", ""),
                    "confidence": conf,
                    "bbox":       (node.get("bbox") or [0,0,0,0]),
                }
        return result