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
    Mimics an on-disk approximation of the monitoring store (Section VIII).
    """
    return Path(__file__).resolve().parent / "drift_history.json"


def _default_query_log_path() -> Path:
    return Path(__file__).resolve().parent / "active_learning_queries.json"


@dataclass
class DriftMonitor:
    """
    Monitors stability σ(P) for a page and performs simple drift detection.
    """

    stability_threshold: float = 0.6
    history_path: Path = field(default_factory=_default_history_path)
    max_history: int = 1000

    # internal state maintained across calls
    _history: List[float] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.history_path = Path(self.history_path)
        self._load_history()

    # ---------------- core computations ----------------

    @staticmethod
    def compute_node_margins(probs: np.ndarray) -> np.ndarray:
        """
        Compute per-node margin scores: p_top1 - p_top2.
        """
        if probs.size == 0:
            return np.asarray([], dtype=float)

        if probs.ndim != 2 or probs.shape[1] < 2:
            return np.zeros(probs.shape[0], dtype=float)

        scores = np.array(probs, dtype=float)
        sorted_scores = np.sort(scores, axis=1)
        top1 = sorted_scores[:, -1]
        top2 = sorted_scores[:, -2]
        margins = top1 - top2
        return margins

    def compute_page_stability(self, probs: np.ndarray) -> float:
        """
        Aggregate node-level margins into a page-level stability score.
        Empirical counterpart of σ(P) in Theorem 1.
        """
        margins = self.compute_node_margins(probs)
        if margins.size == 0:
            return 0.0
        return float(np.mean(margins))

    # ---------------- public API ----------------

    def evaluate_simple(
        self, stability_score: float, page_url: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Simplified entry point used by the InferenceEngine pipeline.

        Accepts a pre-calculated stability_score (σ) from the solver engine
        instead of raw probabilities. This allows the InferenceEngine to
        handle the mathematical details of σ calculation directly.
        """
        # 1. Detect Drift based on Threshold (Theorem 1)
        drift_detected = stability_score < self.stability_threshold

        # 2. Update History
        self._history.append(stability_score)
        self._save_history()

        # 3. Build Context for UI/Logs
        historical_mean = float(np.mean(self._history)) if self._history else 0.0

        context = {
            "page_url": page_url,
            "stability": stability_score,
            "stability_threshold": self.stability_threshold,
            "history_size": len(self._history),
            "history_mean": historical_mean,
            "drift_detected": drift_detected,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        return drift_detected, context

    def evaluate(
        self,
        probs: np.ndarray,
        solver_result: Optional[Dict[str, Any]] = None,
        page_url: Optional[str] = None,
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Compute stability σ from raw probs, detect drift, and update history.
        Legacy method for direct GNN usage.
        """
        stability = self.compute_page_stability(probs)
        drift_detected, context = self.evaluate_simple(stability, page_url or "unknown")

        if solver_result is not None:
            context["solver_keys"] = list(solver_result.keys())

        return stability, drift_detected, context

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
            self._history = []

    def _save_history(self) -> None:
        try:
            scores = self._history[-self.max_history :]
            payload = {"scores": scores}
            self.history_path.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        except Exception:
            pass


@dataclass
class ActiveLearningManager:
    """
    Simulated active learning controller.
    Emits human-labeling queries when drift is detected.
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

    def create_query_request(
        self,
        page_url: str,
        stability_score: float,
        drift_context: Dict[str, Any],
        solver_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        timestamp = datetime.utcnow().isoformat() + "Z"
        request: Dict[str, Any] = {
            "page_url": page_url,
            "stability_score": stability_score,
            "reason": "DRIFT_DETECTED: σ below threshold (cf. Theorem 1).",
            "drift_context": drift_context,
            "timestamp_utc": timestamp,
        }
        if solver_result is not None:
            request["prediction_summary"] = {
                k: v for k, v in solver_result.items() if k != "_meta"
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
        Handle a drift event by enqueuing a human-annotation query.
        """
        query = self.create_query_request(
            page_url=page_url,
            stability_score=stability_score,
            drift_context=drift_context,
            solver_result=solver_result,
        )
        self._queries.append(query)
        self._save_queries()

        print(
            f"[ActiveLearning] Retraining trigger (mock) for page={page_url!r}, "
            f"sigma={stability_score:.3f}"
        )
        return query
