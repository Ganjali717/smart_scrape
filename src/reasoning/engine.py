from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# В реальном пакете можно заменить на:
# from .structure import ...
from src.reasoning.structure import (
    Schema,
    Field,
    NodeCandidate,
    ExtractionResult,
    Selector,
    DefaultSelector,
)


@dataclass
class InferenceResult:
    """
    Результат вывода для одной страницы.

    field_results: поле -> ExtractionResult
    stability: σ(P) – стабильность страницы.
    """

    field_results: Dict[str, ExtractionResult]
    stability: float


class InferenceEngine:
    """
    Constrained Inference Engine (Section V.C).

    Решает задачу:

        max_S  Σ_f φ_f(S)
        s.t.   Γ(S) = True

    где:
      - S – присваивание полей кандидатов,
      - φ_f(S) – скор модели для поля f,
      - Γ(S) – конъюнкция ограничений схемы.
    """

    def __init__(self) -> None:
        # В будущем сюда можно пробрасывать backend ILP/SMT и т.п.
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, page_graph: Any, schema: Schema) -> InferenceResult:
        """
        Запустить вывод для одной страницы.

        page_graph:
          Произвольный объект, который умеют читать Selector'ы (например,
          через page_graph.candidates_by_field или GNN-выходы).

        schema:
          Декларативная схема (Schema) с полями и ограничениями Γ.
        """
        # 1. Кандидаты для каждого поля через Selector.
        field_to_candidates: Dict[str, List[NodeCandidate]] = {}
        for field_name, field in schema.fields.items():
            selector = field.selector or DefaultSelector(field_name)
            candidates = selector.propose(page_graph)
            field_to_candidates[field_name] = candidates

        # 2. Жадное присваивание с проверкой Γ.
        assignment: Dict[str, Optional[NodeCandidate]] = {}
        page_context = self._build_page_context(page_graph, field_to_candidates)

        for field_name, field in schema.fields.items():
            candidates = field_to_candidates.get(field_name, [])
            chosen: Optional[NodeCandidate] = None

            # Жадно перебираем кандидатов по убыванию score.
            for cand in candidates:
                trial_assignment = dict(assignment)
                trial_assignment[field_name] = cand

                if schema.check_all_constraints(trial_assignment, page_context):
                    chosen = cand
                    assignment[field_name] = cand
                    break

            # Если ничто не удовлетворяет Γ, но поле обязательное – берём top-1.
            if chosen is None and candidates:
                if field.required:
                    chosen = candidates[0]
                    assignment[field_name] = chosen
                else:
                    assignment[field_name] = None

        # 3. Формируем ExtractionResult с proof / justification.
        field_results: Dict[str, ExtractionResult] = {}
        for field_name, field in schema.fields.items():
            cand = assignment.get(field_name)
            if cand is None:
                field_results[field_name] = ExtractionResult(
                    field_name=field_name,
                    value=None,
                    confidence=0.0,
                    proof={
                        "node_ids": [],
                        "constraints": [],
                    },
                )
                continue

            satisfied, violated = self._evaluate_constraints_per_field(
                schema, field_name, assignment, page_context
            )

            field_results[field_name] = ExtractionResult(
                field_name=field_name,
                value=getattr(cand, "text", None),
                confidence=float(getattr(cand, "score", 0.0)),
                proof={
                    "node_ids": [cand.node_id],
                    "constraints": satisfied,
                    "violated": violated,
                },
            )

        # 4. σ(P): минимальный margin по полям.
        stability = self.compute_stability_score(field_to_candidates)

        return InferenceResult(field_results=field_results, stability=stability)

    # ------------------------------------------------------------------
    # Stability / drift (Section V.D)
    # ------------------------------------------------------------------

    def compute_stability_score(
        self,
        field_to_candidates: Dict[str, List[NodeCandidate]],
    ) -> float:
        """
        σ(P) = min_f σ_f, где σ_f = score_best(f) - score_second(f).

        Если кандидатов меньше двух, второй скор считаем 0.
        """
        margins: List[float] = []

        for field_name, candidates in field_to_candidates.items():
            if not candidates:
                margins.append(0.0)
                continue

            best = float(getattr(candidates[0], "score", 0.0))
            second = (
                float(getattr(candidates[1], "score", 0.0))
                if len(candidates) > 1
                else 0.0
            )
            margins.append(max(0.0, best - second))

        if not margins:
            return 0.0

        return min(margins)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_page_context(
        self,
        page_graph: Any,
        field_to_candidates: Dict[str, List[NodeCandidate]],
    ) -> Dict[str, Any]:
        """
        Строим лёгкий контекст страницы (сейчас только page_height),
        нужный для VisualConstraint.
        """
        page_height = getattr(page_graph, "page_height", None)

        if page_height is None:
            max_bottom = 0.0
            for candidates in field_to_candidates.values():
                for cand in candidates:
                    bbox = getattr(cand, "bbox", None)
                    if bbox:
                        _, y, _, h = bbox
                        max_bottom = max(max_bottom, float(y + h))
            if max_bottom > 0.0:
                page_height = max_bottom

        return {"page_height": page_height}

    def _evaluate_constraints_per_field(
        self,
        schema: Schema,
        field_name: str,
        assignment: Dict[str, Optional[NodeCandidate]],
        page_context: Dict[str, Any],
    ) -> Tuple[List[str], List[str]]:
        """
        Для заданного поля считаем, какие ограничения Γ удовлетворены / нарушены
        при финальном присваивании – для заполнения proof.
        """
        satisfied: List[str] = []
        violated: List[str] = []

        # Field-level
        field: Field = schema.fields[field_name]
        for c in field.constraints:
            ok = c.check(assignment, page_context)
            (satisfied if ok else violated).append(c.name)

        # Schema-level, которые применимы к этому полю.
        for c in schema.constraints:
            if not c.applies_to(field_name):
                continue
            ok = c.check(assignment, page_context)
            (satisfied if ok else violated).append(c.name)

        return satisfied, violated
