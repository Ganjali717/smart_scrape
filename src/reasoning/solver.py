from ortools.linear_solver import pywraplp
import numpy as np
from typing import List, Dict, Any


class ConstraintSolver:
    """
    Implements the constraint-guided inference (Eq. 1 in the paper).
    Ensures that the extracted data satisfies Schema Constraints (Gamma).
    """

    def __init__(self, classes: List[str]):
        self.classes = classes  # e.g., ['price', 'title', 'other']
        # Индексы классов для удобства
        self.cls_to_idx = {name: i for i, name in enumerate(classes)}

    def solve(self, raw_nodes: List[Dict], probs: np.ndarray) -> Dict[str, Any]:
        """
        Solves the Integer Linear Programming (ILP) problem.
        Maximize: Sum(Confidence)
        Subject to: Integrity Constraints
        """
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            print("[Solver] SCIP solver not available.")
            return {}

        num_nodes = len(raw_nodes)
        num_classes = len(self.classes)

        # --- 1. Variables ---
        # x[i, j] = 1 if node i is assigned to class j
        x = {}
        for i in range(num_nodes):
            for j in range(num_classes):
                x[i, j] = solver.IntVar(0, 1, f"x_{i}_{j}")

        # --- 2. Constraints ---

        # A. Каждый узел имеет ровно 1 класс (или 'other')
        for i in range(num_nodes):
            solver.Add(solver.Sum([x[i, j] for j in range(num_classes)]) == 1)

        # B. Schema Constraint: Только ОДИН Title на странице
        if "title" in self.cls_to_idx:
            t_idx = self.cls_to_idx["title"]
            solver.Add(solver.Sum([x[i, t_idx] for i in range(num_nodes)]) == 1)

        # C. Schema Constraint: Только ОДНА Price (для карточки товара)
        if "price" in self.cls_to_idx:
            p_idx = self.cls_to_idx["price"]
            solver.Add(solver.Sum([x[i, p_idx] for i in range(num_nodes)]) == 1)

        # --- 3. Objective Function ---
        # Максимизируем суммарную вероятность выбранных классов
        objective = solver.Objective()
        for i in range(num_nodes):
            for j in range(num_classes):
                # probs[i][j] - это вероятность от нейросети
                objective.SetCoefficient(x[i, j], float(probs[i][j]))

        objective.SetMaximization()

        # --- 4. Solve ---
        status = solver.Solve()

        result = {}
        if status == pywraplp.Solver.OPTIMAL:
            print("[Solver] Optimal solution found satisfying all constraints.")
            for i in range(num_nodes):
                for j in range(num_classes):
                    if x[i, j].solution_value() > 0.5:
                        class_name = self.classes[j]
                        if class_name != "other":
                            # Сохраняем результат
                            node_info = raw_nodes[i]
                            result[class_name] = {
                                "text": node_info["text"],
                                "confidence": float(probs[i][j]),
                                "bbox": node_info["bbox"],
                            }
        else:
            print("[Solver] No solution found compatible with constraints.")

        return result
