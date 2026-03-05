from ortools.linear_solver import pywraplp
import numpy as np
from typing import List, Dict, Any

from config import FOOTER_THRESHOLD


class ConstraintSolver:
    """
    Implements the constraint-guided inference (Eq. 1 in the paper).
    Ensures that the extracted data satisfies Schema Constraints (Gamma).

    ILP formulation (Appendix A.2 of the dissertation):
        max  Σ_{i,f}  s(i, f) * x_{i,f}
        s.t. Γ(x) = True
    """

    def __init__(self, classes: List[str]):
        self.classes = classes  # e.g., ['price', 'title', 'other']
        self.cls_to_idx = {name: i for i, name in enumerate(classes)}

    def solve(
        self, raw_nodes: List[Dict], probs: np.ndarray, page_height: float = 1080.0
    ) -> Dict[str, Any]:
        """
        Solves the Integer Linear Programming (ILP) problem.
        Maximize: Sum(Confidence)
        Subject to:
          Γ1 — Uniqueness:  at most one Price and one Title per page
          Γ2 — Geometry:    Title/Price cannot appear in the footer zone
          Γ3 — Integrity:   each node is assigned exactly one class
        """
        num_nodes = len(raw_nodes)
        num_classes = len(self.classes)

        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            print("[Solver] SCIP solver not available, falling back to greedy.")
            return self._greedy_fallback(raw_nodes, probs)

        # --- 1. Binary decision variables: x[i, j] ∈ {0, 1} ---
        x = {}
        for i in range(num_nodes):
            for j in range(num_classes):
                x[i, j] = solver.IntVar(0, 1, f"x_{i}_{j}")

        # --- 2. Constraints (Gamma) ---

        # Γ3 — Integrity: each node gets exactly one label (including 'other')
        for i in range(num_nodes):
            solver.Add(solver.Sum([x[i, j] for j in range(num_classes)]) == 1)

        # Γ1 — Uniqueness: at most one Price and one Title per page (Section 6.7.1)
        for target_name in ["title", "price"]:
            if target_name in self.cls_to_idx:
                t_idx = self.cls_to_idx[target_name]
                solver.Add(solver.Sum([x[i, t_idx] for i in range(num_nodes)]) <= 1)

        # Γ2 — Geometry / Footer Trap (Section 6.3):
        # Nodes in the bottom (1 - FOOTER_THRESHOLD) of the page cannot be
        # Title or Price. Threshold = 0.80 → bottom 20% is footer zone.
        limit_y = page_height * FOOTER_THRESHOLD

        price_idx = self.cls_to_idx.get("price")
        title_idx = self.cls_to_idx.get("title")

        for i in range(num_nodes):
            # bbox формата [x, y, w, h]
            # Берем y координату
            node_y = raw_nodes[i]["bbox"][1]

            if node_y > limit_y:
                # Если узел слишком низко, принудительно запрещаем ему быть Price или Title
                if price_idx is not None:
                    solver.Add(x[i, price_idx] == 0)
                if title_idx is not None:
                    solver.Add(x[i, title_idx] == 0)

        # --- 3. Objective Function ---
        # Максимизируем общую уверенность системы: Max sum(prob * x)
        objective = solver.Objective()
        for i in range(num_nodes):
            for j in range(num_classes):
                objective.SetCoefficient(x[i, j], float(probs[i][j]))

        objective.SetMaximization()

        # --- 4. Solve ---
        status = solver.Solve()

        result = {}
        if status == pywraplp.Solver.OPTIMAL:
            # Собираем результаты
            for i in range(num_nodes):
                for j in range(num_classes):
                    if x[i, j].solution_value() > 0.5:
                        class_name = self.classes[j]
                        if class_name != "other":
                            result[class_name] = {
                                "text": raw_nodes[i]["text"],
                                "confidence": float(probs[i][j]),
                                "bbox": raw_nodes[i]["bbox"],
                            }
        else:
            print(
                "[Solver] No optimal solution found. Constraints might be too strict."
            )

        return result

    def _greedy_fallback(self, raw_nodes, probs):
        """Запасной вариант, если солвер не сработал"""
        result = {}
        for i, node in enumerate(raw_nodes):
            best_j = int(np.argmax(probs[i]))
            cls = self.classes[best_j]
            if cls == "other":
                continue

            conf = float(probs[i][best_j])
            if cls not in result or conf > result[cls]["confidence"]:
                result[cls] = {
                    "text": node["text"],
                    "confidence": conf,
                    "bbox": node["bbox"],
                }
        return result
