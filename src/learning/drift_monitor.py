# src/learning/drift_monitor.py

"""
Drift monitoring and active learning utilities for SmartScrape.

This module operationalizes the stability notion σ_f(P) discussed around
Theorem 1 and Eq. (1) in the paper:

- Stability σ is estimated via margin sampling:
  σ = E_i[ p̂_top1(i) - p̂_top2(i) ],
  where p̂_topk(i) — k-я по величине вероятность по узлу i.

- Template drift is heuristically detected when σ drops below a threshold.
- A lightweight JSON history of σ is maintained to approximate the
  long-term regime of the underlying template distribution Δ.
- ActiveLearningManager simulates the “learn–validate–repair” loop by
  emitting human-labeling queries when drift is detected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _default_history_path() -> Path:
    """
    Default location for drift history.

    For a prototype, we keep the file next to this module, which mimics an
    on-disk approximation of the monitoring store used in Section VIII.
    """
    return Path(__file__).resolve().parent / "drift_history.json"


def _default_query_log_path() -> Path:
    return Path(__file__).resolve().parent / "active_learning_queries.json"


@dataclass
class DriftMonitor:
    """
    Monitors stability σ(P) for a page and performs simple drift detection.

    Stability is computed via margin sampling, which is a standard
    uncertainty metric and aligns with the stability score σ_f(P) mentioned
    before Theorem 1: pages with low σ are likely to be affected by
    template drift Δ, even if Eq. (1) is still optimised locally.

    Attributes
    ----------
    stability_threshold : float
        If E[margin] < stability_threshold, the page is flagged as drifted.
    history_path : Path
        Where to store historical stability scores (JSON).
    max_history : int
        Maximum number of past scores to retain.
    """

    stability_threshold: float = 0.6
    history_path: Path = field(default_factory=_default_history_path)
    max_history: int = 1000

    # internal state (list[float]) maintained across calls
    _history: List[float] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.history_path = Path(self.history_path)
        self._load_history()

    # ---------------- core computations ----------------

    @staticmethod
    def compute_node_margins(probs: np.ndarray) -> np.ndarray:
        """
        Compute per-node margin scores: p_top1 - p_top2.

        Parameters
        ----------
        probs : np.ndarray
            Array of shape [num_nodes, num_classes] with class probabilities
            or unnormalised scores (monotone transforms are acceptable).

        Returns
        -------
        np.ndarray
            Array of shape [num_nodes] with margin scores in [0, 1]
            (assuming probs are valid probabilities).
        """
        if probs.size == 0:
            return np.asarray([], dtype=float)

        if probs.ndim != 2 or probs.shape[1] < 2:
            # Degenerate case: cannot compute a meaningful margin.
            # We conservatively treat all nodes as maximally uncertain.
            return np.zeros(probs.shape[0], dtype=float)

        # Use partitioning for efficiency: we only need the two largest values.
        # For each row i, we compute the largest and second-largest scores.
        # Note: we work on a copy to avoid mutating the input.
        scores = np.array(probs, dtype=float)
        # argpartition to get indices of top2; then sort those two.
        # For clarity (and since num_classes is small), we simply sort:
        sorted_scores = np.sort(scores, axis=1)
        top1 = sorted_scores[:, -1]
        top2 = sorted_scores[:, -2]
        margins = top1 - top2
        return margins

    def compute_page_stability(self, probs: np.ndarray) -> float:
        """
        Aggregate node-level margins into a page-level stability score.

        This is the empirical counterpart of σ(P) in Theorem 1: if σ(P) is
        high, the learned selector class is likely to be Δ-stable on P.
        """
        margins = self.compute_node_margins(probs)
        if margins.size == 0:
            return 0.0
        return float(np.mean(margins))

    # ---------------- history management ----------------

    def _load_history(self) -> None:
        if not self.history_path.exists():
            self._history = []
            return

        try:
            raw = json.loads(self.history_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and "scores" in raw:
                scores = raw["scores"]
            elif isinstance(raw, list):
                scores = raw
            else:
                scores = []

            self._history = [float(s) for s in scores if isinstance(s, (int, float))]
        except Exception:
            # Corrupted or incompatible file — reset history.
            self._history = []

    def _save_history(self) -> None:
        try:
            scores = self._history[-self.max_history :]
            payload = {"scores": scores}
            self.history_path.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        except Exception:
            # For a prototype we silently ignore persistence failures.
            pass

    # ---------------- public API ----------------

    def evaluate(
        self,
        probs: np.ndarray,
        solver_result: Optional[Dict[str, Any]] = None,
        page_url: Optional[str] = None,
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Compute stability σ, detect drift, and update history.

        Parameters
        ----------
        probs : np.ndarray
            Node-level probabilities/scores from the GNN + priors.
        solver_result : dict, optional
            Final structured prediction for the page (title/price/etc.).
            Included only for context in diagnostic metadata.
        page_url : str, optional
            URL of the page, used for reporting and traceability.

        Returns
        -------
        stability : float
            Average margin-based stability score for the page.
        drift_detected : bool
            True if stability < stability_threshold.
        context : dict
            Diagnostic metadata (current σ, historical mean, counts, URL).
        """
        stability = self.compute_page_stability(probs)

        historical_mean: Optional[float]
        if self._history:
            historical_mean = float(np.mean(self._history))
        else:
            historical_mean = None

        drift_detected = stability < self.stability_threshold

        # Update history (we treat every page as another sample from Δ).
        self._history.append(stability)
        self._save_history()

        context: Dict[str, Any] = {
            "page_url": page_url,
            "stability": stability,
            "stability_threshold": self.stability_threshold,
            "history_size": len(self._history),
            "history_mean": historical_mean,
            "drift_detected": drift_detected,
        }
        # Optionally attach a shallow snapshot of solver_result for debugging.
        if solver_result is not None:
            context["solver_keys"] = list(solver_result.keys())

        return stability, drift_detected, context


@dataclass
class ActiveLearningManager:
    """
    Simulated active learning controller.

    In the full learn–validate–repair loop, pages with low σ would be sent
    to human annotators; their labels would then update Eq. (1) via another
    round of training. Here we only emit a query stub and log a mock
    “retraining trigger”.

    Attributes
    ----------
    query_log_path : Path
        Where to store the list of query requests.
    """

    query_log_path: Path = field(default_factory=_default_query_log_path)

    _queries: List[Dict[str, Any]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.query_log_path = Path(self.query_log_path)
        self._load_queries()

    def _load_queries(self) -> None:
        if not self.query_log_path.exists():
            self._queries = []
            return
        try:
            raw = json.loads(self.query_log_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                self._queries = raw
            else:
                self._queries = []
        except Exception:
            self._queries = []

    def _save_queries(self) -> None:
        try:
            self.query_log_path.write_text(
                json.dumps(self._queries, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    # ---------------- public API ----------------

    def create_query_request(
        self,
        page_url: str,
        stability_score: float,
        drift_context: Dict[str, Any],
        solver_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build a human-review request for a drifted page.

        This structure is intentionally verbose; in a production setting
        it would be sent to an annotation system / labeling UI.
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        request: Dict[str, Any] = {
            "page_url": page_url,
            "stability_score": stability_score,
            "reason": "DRIFT_DETECTED: σ below threshold (cf. Theorem 1).",
            "drift_context": drift_context,
            "timestamp_utc": timestamp,
        }

        # Attach a compact snapshot of the solver output, if available.
        if solver_result is not None:
            request["prediction_summary"] = {
                k: {
                    "text": v.get("text"),
                    "confidence": v.get("confidence"),
                }
                for k, v in solver_result.items()
                if isinstance(v, dict) and "text" in v
            }

        return request

    def handle_drift(
        self,
        page_url: str,
        stability_score: float,
        drift_context: Dict[str, Any],
        solver_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Handle a drift event by:

        1. Enqueuing a human-annotation query.
        2. Emitting a mock “retraining trigger” message.

        This is the glue that closes the learn–validate–repair loop in
        Section V without actually retraining models.
        """
        query = self.create_query_request(
            page_url=page_url,
            stability_score=stability_score,
            drift_context=drift_context,
            solver_result=solver_result,
        )
        self._queries.append(query)
        self._save_queries()

        # Mock retraining trigger — in a full system this would enqueue
        # a job for the training pipeline.
        print(
            f"[ActiveLearning] Retraining trigger (mock) for page={page_url!r}, "
            f"sigma={stability_score:.3f}"
        )

        return query
