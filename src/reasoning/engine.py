from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# In a real package, this could be replaced with:
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
    Inference result for a single page.

    field_results: mapping from field name to ExtractionResult.
    stability: σ(P) – page stability score.
    """

    field_results: Dict[str, ExtractionResult]
    stability: float


class InferenceEngine:
    """
    Constrained Inference Engine (Section V.C).

    Solves the optimization problem:

        max_S  Σ_f φ_f(S)
        s.t.   Γ(S) = True

    where:
      - S is the assignment of candidates to fields,
      - φ_f(S) is the model score for field f,
      - Γ(S) is the conjunction of schema constraints.
    """

    def __init__(self) -> None:
        # Future enhancement: integrate ILP/SMT backends here, etc.
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, page_graph: Any, schema: Schema) -> InferenceResult:
        """
        Run inference for a single page.

        page_graph:
          An arbitrary object that Selectors can read (e.g., via
          page_graph.candidates_by_field or GNN outputs).

        schema:
          A declarative Schema containing fields and constraints Γ.
        """
        # 1. Propose candidates for each field using its Selector.
        field_to_candidates: Dict[str, List[NodeCandidate]] = {}
        for field_name, field in schema.fields.items():
            selector = field.selector or DefaultSelector(field_name)
            candidates = selector.propose(page_graph)
            field_to_candidates[field_name] = candidates

        # 2. Greedy assignment with Γ constraint checking.
        assignment: Dict[str, Optional[NodeCandidate]] = {}
        page_context = self._build_page_context(page_graph, field_to_candidates)

        for field_name, field in schema.fields.items():
            candidates = field_to_candidates.get(field_name, [])
            chosen: Optional[NodeCandidate] = None

            # Greedily iterate over candidates in descending order of score.
            for cand in candidates:
                trial_assignment = dict(assignment)
                trial_assignment[field_name] = cand

                if schema.check_all_constraints(trial_assignment, page_context):
                    chosen = cand
                    assignment[field_name] = cand
                    break

            # If no candidate satisfies Γ, but the field is required – pick the top-1.
            if chosen is None and candidates:
                if field.required:
                    chosen = candidates[0]
                    assignment[field_name] = chosen
                else:
                    assignment[field_name] = None

        # 3. Formulate the ExtractionResult with proof / justification.
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

        # 4. Compute σ(P): the stability score across fields.
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
        σ(P) = E_{n∈P} [ P(ŷ₁|n) − P(ŷ₂|n) ]

        Per the dissertation (Section 6.4 / Appendix A.3), stability is the
        *mean* margin across all candidate nodes on the page, not the minimum.
        Using minimum would make σ overly sensitive to a single low-confidence
        node and would not correlate well with overall extraction correctness.

        Implementation: for each field, collect per-candidate margins; then
        average over all candidates across all fields.
        """
        all_margins: List[float] = []

        for candidates in field_to_candidates.values():
            for i, cand in enumerate(candidates):
                best = float(getattr(cand, "score", 0.0))
                # Find second-best score among other candidates for the same field
                second = 0.0
                for j, other in enumerate(candidates):
                    if j != i:
                        s = float(getattr(other, "score", 0.0))
                        if s > second:
                            second = s
                margin = max(0.0, best - second)
                all_margins.append(margin)

        if not all_margins:
            return 0.0

        return float(sum(all_margins) / len(all_margins))  # mean, not min

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_page_context(
        self,
        page_graph: Any,
        field_to_candidates: Dict[str, List[NodeCandidate]],
    ) -> Dict[str, Any]:
        """
        Builds a lightweight page context (currently only page_height),
        which is required for VisualConstraints.
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
        Evaluates which constraints Γ are satisfied or violated for a given field
        under the final assignment, in order to populate the proof.
        """
        satisfied: List[str] = []
        violated: List[str] = []

        # Field-level constraints
        field: Field = schema.fields[field_name]
        for c in field.constraints:
            ok = c.check(assignment, page_context)
            (satisfied if ok else violated).append(c.name)

        # Schema-level constraints that apply to this specific field
        for c in schema.constraints:
            if not c.applies_to(field_name):
                continue
            ok = c.check(assignment, page_context)
            (satisfied if ok else violated).append(c.name)

        return satisfied, violated