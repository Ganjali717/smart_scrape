from ortools.linear_solver import pywraplp
import numpy as np
import re
from typing import List, Dict, Any

from config import FOOTER_THRESHOLD, PRODUCT_ZONE_MAX_PX


class ConstraintSolver:
    """
    Constraint-guided inference (paper Eq. 1).

    ILP formulation:
        max  Σ_{i,f}  s(i, f) * x[i,f]
        s.t. Γ(x) = True

    Active constraint set Γ:
      Γ1 — Uniqueness:   Σ_i x[i,Price] ≤ 1; Σ_i x[i,Title] ≤ 1
      Γ2 — Footer trap:  y(n) > 0.8 × PageHeight => x[n,Title] = x[n,Price] = 0
      Γ3 — Product zone: y(n) > 500 px => x[n,Title] = x[n,Price] = 0
      Γ4 — Format:       x[n,Price] = 1 => HasCurrency(n) ∧ IsNumeric(n)
      Integrity:          each node has exactly one class label
    """

    PRICE_PATTERN = re.compile(r"([£$€₼]|AZN)\s*\d+([.,]\d{1,2})?|\d+([.,]\d{1,2})?\s*([£$€₼]|AZN)", re.IGNORECASE)
    NUMBER_PATTERN = re.compile(r"\d+([.,]\d{1,2})?")

    def __init__(self, classes: List[str]):
        self.classes = classes
        self.cls_to_idx = {name: i for i, name in enumerate(classes)}
        self.product_zone_max_px = PRODUCT_ZONE_MAX_PX

    def solve(
        self,
        raw_nodes: List[Dict],
        probs: np.ndarray,
        page_height: float = 1080.0,
    ) -> Dict[str, Any]:
        num_nodes = len(raw_nodes)
        num_classes = len(self.classes)

        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            print("[Solver] SCIP unavailable — greedy fallback.")
            return self._greedy_fallback(raw_nodes, probs, reason="SCIP unavailable")

        x = {
            (i, j): solver.IntVar(0, 1, f"x_{i}_{j}")
            for i in range(num_nodes)
            for j in range(num_classes)
        }

        price_idx = self.cls_to_idx.get("price")
        title_idx = self.cls_to_idx.get("title")

        constraint_trace = []

        # Integrity: every node is assigned exactly one label.
        for i in range(num_nodes):
            solver.Add(solver.Sum([x[i, j] for j in range(num_classes)]) == 1)
        constraint_trace.append({"id": "Integrity", "status": "enforced", "detail": "Σ_j x[i,j] = 1 for every node"})

        # Γ1 — Uniqueness: at most one Title and at most one Price.
        for cls_name in ("title", "price"):
            if cls_name in self.cls_to_idx:
                ci = self.cls_to_idx[cls_name]
                solver.Add(solver.Sum([x[i, ci] for i in range(num_nodes)]) <= 1)
                constraint_trace.append({"id": "Γ1", "status": "enforced", "field": cls_name, "detail": f"Σ_i x[i,{cls_name}] ≤ 1"})

        for i, node in enumerate(raw_nodes):
            bbox = node.get("bbox") or [0, 0, 0, 0]
            node_y = float(bbox[1]) if len(bbox) > 1 else 0.0
            text = str(node.get("text", ""))

            # Γ2 — Footer trap.
            if node_y > page_height * FOOTER_THRESHOLD:
                if price_idx is not None:
                    solver.Add(x[i, price_idx] == 0)
                if title_idx is not None:
                    solver.Add(x[i, title_idx] == 0)
                constraint_trace.append({
                    "id": "Γ2",
                    "status": "enforced",
                    "node_id": node.get("id", i),
                    "detail": f"y={node_y:.1f} > {FOOTER_THRESHOLD}×{page_height:.1f}; title/price forbidden",
                })

            # Γ3 — Product zone.
            if node_y > self.product_zone_max_px:
                if price_idx is not None:
                    solver.Add(x[i, price_idx] == 0)
                if title_idx is not None:
                    solver.Add(x[i, title_idx] == 0)
                constraint_trace.append({
                    "id": "Γ3",
                    "status": "enforced",
                    "node_id": node.get("id", i),
                    "detail": f"y={node_y:.1f} > {self.product_zone_max_px:.1f}px; title/price forbidden",
                })

            # Γ4 — Price format as a HARD ILP constraint.
            # If the text cannot be a price, the price variable is forced to 0.
            if price_idx is not None and not self._has_price_format(text):
                solver.Add(x[i, price_idx] == 0)
                constraint_trace.append({
                    "id": "Γ4",
                    "status": "enforced",
                    "node_id": node.get("id", i),
                    "detail": "price forbidden: missing currency symbol or numeric token",
                })

        obj = solver.Objective()
        for i in range(num_nodes):
            for j in range(num_classes):
                obj.SetCoefficient(x[i, j], float(probs[i][j]))
        obj.SetMaximization()

        status = solver.Solve()
        result: Dict[str, Any] = {
            "_solver": {
                "backend": "OR-Tools SCIP",
                "status": self._status_name(status),
                "objective_value": float(obj.Value()) if status == pywraplp.Solver.OPTIMAL else None,
                "constraints": constraint_trace,
                "violated": [],
                "fallback_used": False,
            }
        }

        if status == pywraplp.Solver.OPTIMAL:
            for i in range(num_nodes):
                for j in range(num_classes):
                    if x[i, j].solution_value() > 0.5:
                        cls = self.classes[j]
                        if cls != "other":
                            result[cls] = {
                                "text": raw_nodes[i].get("text", ""),
                                "confidence": float(probs[i][j]),
                                "bbox": raw_nodes[i].get("bbox") or [0, 0, 0, 0],
                                "node_id": raw_nodes[i].get("id", i),
                                "solver_variable": f"x[{i},{cls}]",
                                "constraints": constraint_trace,
                            }
            return result

        print("[Solver] No optimal solution — greedy fallback.")
        return self._greedy_fallback(raw_nodes, probs, reason=f"ILP status={self._status_name(status)}")

    def _has_price_format(self, text: str) -> bool:
        return bool(self.PRICE_PATTERN.search(text)) and bool(self.NUMBER_PATTERN.search(text))

    @staticmethod
    def _status_name(status: int) -> str:
        names = {
            pywraplp.Solver.OPTIMAL: "OPTIMAL",
            pywraplp.Solver.FEASIBLE: "FEASIBLE",
            pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
            pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
            pywraplp.Solver.ABNORMAL: "ABNORMAL",
            pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
        }
        return names.get(status, f"UNKNOWN({status})")

    def _greedy_fallback(self, raw_nodes: List[Dict], probs: np.ndarray, reason: str = "fallback") -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "_solver": {
                "backend": "greedy fallback",
                "status": "FALLBACK",
                "reason": reason,
                "constraints": [],
                "violated": ["no_constraint_enforcement"],
                "fallback_used": True,
            }
        }
        for i, node in enumerate(raw_nodes):
            best_j = int(np.argmax(probs[i]))
            cls = self.classes[best_j]
            if cls == "other":
                continue
            conf = float(probs[i][best_j])
            if cls not in result or conf > result[cls]["confidence"]:
                result[cls] = {
                    "text": node.get("text", ""),
                    "confidence": conf,
                    "bbox": node.get("bbox") or [0, 0, 0, 0],
                    "node_id": node.get("id", i),
                    "solver_variable": f"greedy[{i},{cls}]",
                    "constraints": [],
                }
        return result
