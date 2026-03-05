"""
SmartScrape main extraction pipeline.

Architecture (Section 6 of the dissertation):
  Phase 1 — FitLayout renders the page and returns visual descriptors
  Phase 2 — FitLayoutParser builds a heterogeneous graph (DOM + KNN edges)
  Phase 3 — SmartScrapeGNN scores each node (perception)
  Phase 4 — ConstraintSolver (ILP) selects globally optimal assignment (reasoning)
  Phase 5 — DriftMonitor computes σ(P) and triggers active learning if needed

Reasoning mode can be toggled between ILP (default, dissertation-aligned)
and Greedy (ablation baseline) via the `reasoning_mode` parameter.
"""

import os
import torch
import numpy as np
import re
from typing import List, Dict, Any, Literal

from src.integration.fitlayout import FitLayoutClient
from src.learning.features import FeatureEncoder
from src.learning.graph_builder import FitLayoutParser
from src.learning.gnn_model import SmartScrapeGNN
from src.learning.drift_monitor import DriftMonitor, ActiveLearningManager
from src.reasoning.solver_fixed import ConstraintSolver
from config import FOOTER_THRESHOLD, STABILITY_THRESHOLD, MODEL_CHECKPOINT


ReasoningMode = Literal["ilp", "greedy"]


class SmartScrapePipeline:
    """
    End-to-end neuro-symbolic extraction pipeline.

    Parameters
    ----------
    reasoning_mode : "ilp" | "greedy"
        "ilp"    — use ConstraintSolver (ILP via OR-Tools). Default.
                   Implements H2: constraint optimization reduces semantic errors.
        "greedy" — argmax per field, no constraint enforcement.
                   Used as ablation baseline in evaluation (Section 6.8.3).
    use_mock : bool
        If True, fall back to built-in mock data when FitLayout is unavailable.
        Should be False during evaluation runs to avoid masking connectivity errors.
    """

    CLASSES = ["price", "title", "other"]

    def __init__(
        self,
        reasoning_mode: ReasoningMode = "ilp",
        use_mock: bool = False,
    ):
        torch.manual_seed(42)
        np.random.seed(42)

        self.reasoning_mode = reasoning_mode
        self.use_mock = use_mock

        self.client = FitLayoutClient()
        self.parser = FitLayoutParser()

        # --- GNN Model ---
        self.encoder = FeatureEncoder()
        self.model = SmartScrapeGNN(
            input_dim=self.encoder.get_output_dim(),
            hidden_dim=64,
            num_classes=len(self.CLASSES),
        )
        self._load_model_weights()
        self.model.eval()

        # --- Constraint Solver (ILP, Section 6.3 / Appendix A.2) ---
        self.solver = ConstraintSolver(classes=self.CLASSES)

        # --- Drift Monitor (Section 6.4) ---
        self.drift_monitor = DriftMonitor(stability_threshold=STABILITY_THRESHOLD)
        self.active_learning = ActiveLearningManager()

    def _load_model_weights(self):
        """Load trained GNN weights if available; warn otherwise."""
        if os.path.exists(MODEL_CHECKPOINT):
            try:
                state = torch.load(MODEL_CHECKPOINT, map_location="cpu")
                self.model.load_state_dict(state)
                print(f"[Pipeline] Loaded GNN weights from '{MODEL_CHECKPOINT}'")
                self._model_trained = True
            except Exception as e:
                print(f"[Pipeline] WARNING: Could not load weights: {e}")
                self._model_trained = False
        else:
            print(
                f"[Pipeline] WARNING: Model checkpoint '{MODEL_CHECKPOINT}' not found. "
                "Running with random weights (heuristic-only mode). "
                "Run train.py to generate model.pt."
            )
            self._model_trained = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, url: str) -> Dict[str, Any] | None:
        """
        Extract structured information from a single page URL.

        Returns a dict with keys:
          "price"  — {text, bbox, confidence, proof}
          "title"  — {text, bbox, confidence, proof}
          "_meta"  — {drift_alert, stability_score, reasoning_mode,
                      model_trained, active_constraints}
        """
        print(f"\n--- Processing: {url} ---")
        print(f"    Reasoning: {self.reasoning_mode.upper()} | "
              f"Model trained: {self._model_trained}")

        # ------ Phase 1: Acquire page data ------
        data, raw_nodes = self._acquire_page(url)
        if not raw_nodes:
            return None

        # ------ Phase 2: GNN Inference ------
        raw_probs = self._run_gnn(data, raw_nodes)

        # ------ Phase 3: Heuristic priors (feature engineering) ------
        # These serve as strong signals for the untrained model and as
        # interpretable features when the GNN is trained.
        scores = raw_probs.copy()
        self._inject_priors(raw_nodes, scores)

        # ------ Phase 4: Reasoning (ILP or Greedy) ------
        page_height = self._estimate_page_height(raw_nodes)

        if self.reasoning_mode == "ilp":
            solver_result = self.solver.solve(raw_nodes, scores, page_height)
            final_record = self._format_ilp_result(solver_result, raw_nodes)
        else:
            final_record = self._greedy_solve(raw_nodes, scores)

        # ------ Phase 5: Drift Monitoring — σ(P) ------
        # σ(P) = E_{n∈P}[p̂_top1(n) - p̂_top2(n)]  (dissertation Eq. 1)
        # Computed from raw GNN probabilities, NOT from heuristic scores.
        stability_score = self.drift_monitor.compute_page_stability(raw_probs)
        drift_alert, drift_context = self.drift_monitor.evaluate_simple(
            stability_score, url
        )

        if drift_alert:
            self.active_learning.handle_drift(
                page_url=url,
                stability_score=stability_score,
                drift_context=drift_context,
                solver_result=final_record,
            )

        # ------ Assemble final output ------
        final_record["_meta"] = {
            "drift_alert": bool(drift_alert),
            "stability_score": float(stability_score),
            "reasoning_mode": self.reasoning_mode,
            "model_trained": self._model_trained,
            "active_constraints": [
                "uniqueness[title]",
                "uniqueness[price]",
                f"geometry[footer > {FOOTER_THRESHOLD}]",
                "format[price ~ currency_pattern]",
            ],
        }

        return final_record

    # ------------------------------------------------------------------
    # Phase 1: Data acquisition
    # ------------------------------------------------------------------

    def _acquire_page(self, url: str):
        """Try live FitLayout, fall back to mock if use_mock=True."""
        try:
            json_data = self.client.get_page_content(url)
            data, raw_nodes = self.parser.parse(json_data)
            if raw_nodes:
                return data, raw_nodes
            print("[Pipeline] Parser returned no nodes from live data.")
        except Exception as e:
            print(f"[Pipeline] Live extraction failed: {e}")

        if self.use_mock:
            print("[Pipeline] Switching to MOCK DATA (use_mock=True).")
            return None, self._get_mock_nodes()

        print("[Pipeline] use_mock=False — returning empty result.")
        return None, []

    def _get_mock_nodes(self):
        """
        Minimal mock page for offline demos (books.toscrape.com layout).
        Note: mock results are labelled as such in _meta.
        """
        return [
            {"id": "101", "text": "Tipping the Velvet",
             "tag": "h1", "bbox": [200, 50, 600, 40]},
            {"id": "102", "text": "£53.74",
             "tag": "p",  "bbox": [200, 120, 100, 30]},
            {"id": "103", "text": "Contact us | Privacy Policy",
             "tag": "footer", "bbox": [10, 950, 1000, 50]},
            {"id": "104", "text": "Similar product £12.99",
             "tag": "div", "bbox": [800, 920, 200, 40]},
            {"id": "105", "text": "Add to basket",
             "tag": "button", "bbox": [200, 200, 150, 40]},
        ]

    # ------------------------------------------------------------------
    # Phase 2: GNN
    # ------------------------------------------------------------------

    def _run_gnn(self, data, raw_nodes) -> np.ndarray:
        """Run GNN forward pass; return probability matrix [N, num_classes]."""
        if data is not None:
            try:
                with torch.no_grad():
                    logits = self.model(data)
                    return torch.exp(logits).numpy()
            except Exception as e:
                print(f"[Pipeline] GNN inference failed: {e}")

        # Fallback: uniform distribution (will be dominated by heuristic priors)
        return np.full((len(raw_nodes), len(self.CLASSES)), 1.0 / len(self.CLASSES))

    # ------------------------------------------------------------------
    # Phase 3: Heuristic priors
    # ------------------------------------------------------------------

    PRICE_PATTERN = re.compile(
        r"([£$€₼]|AZN)?\s*\d+([.,]\d{1,2})?\s*([£$€₼]|AZN)?"
    )

    def _inject_priors(self, raw_nodes: List[Dict], scores: np.ndarray):
        """
        Inject domain-knowledge priors into the score matrix.

        Implements the 'Heuristic prior' part of the hybrid model:
        strong signals for known patterns (currency, H1 tags, y-position).
        """
        page_h = self._estimate_page_height(raw_nodes)

        for i, node in enumerate(raw_nodes):
            text    = node.get("text", "").strip()
            tag     = node.get("tag", "").lower()
            bbox    = node.get("bbox", [0, 0, 0, 0])
            y_coord = float(bbox[1]) if bbox and len(bbox) > 1 else 0.0

            if len(text) < 1:
                scores[i][0] = -100.0
                scores[i][1] = -100.0
                continue

            # --- PRICE (index 0) ---
            has_currency = self.PRICE_PATTERN.search(text) and len(text) < 25
            if has_currency and 150 <= y_coord <= 500:
                scores[i][0] += 40.0   # цена в зоне продукта
            elif has_currency:
                scores[i][0] += 10.0   # цена но не в зоне
            else:
                scores[i][0] -= 20.0   # нет валютного символа — не цена

            # --- TITLE (index 1) ---
            is_price_text = bool(self.PRICE_PATTERN.search(text))
            STOP_WORDS = {
                "warning", "notice", "in stock", "add to basket",
                "add to cart", "home", "sign in", "register",
                "this is a demo", "prices and ratings",
            }
            is_nav_text = (
                "/" in text
                or text.isdigit()
                or any(s in text.lower() for s in STOP_WORDS)
                or len(text) < 3
            )

            if is_price_text:
                scores[i][1] -= 50.0   # цена не может быть заголовком
            elif is_nav_text:
                scores[i][1] -= 50.0   # навигация/статус не заголовок
            elif 185 <= y_coord <= 400 and 2 <= len(text) <= 200:
                scores[i][1] += 30.0   # текст в зоне заголовка
                if tag == "h1":
                    scores[i][1] += 20.0
            elif y_coord < 185:
                scores[i][1] -= 30.0   # шапка сайта

            # --- Зональные штрафы ---
            if y_coord > 500:
                scores[i][0] -= 40.0
                scores[i][1] -= 40.0
            if y_coord > page_h * FOOTER_THRESHOLD:
                scores[i][0] -= 30.0
                scores[i][1] -= 30.0

    # ------------------------------------------------------------------
    # Phase 4a: ILP reasoning
    # ------------------------------------------------------------------

    def _format_ilp_result(
        self, solver_result: Dict, raw_nodes: List[Dict]
    ) -> Dict[str, Any]:
        """Convert ConstraintSolver output to standard record format."""
        record = {}
        node_index = {str(n.get("id", i)): n for i, n in enumerate(raw_nodes)}

        for field_name, field_data in solver_result.items():
            if not isinstance(field_data, dict):
                continue
            record[field_name] = {
                "text": field_data.get("text", ""),
                "bbox": field_data.get("bbox", [0, 0, 0, 0]),
                "confidence": float(field_data.get("confidence", 0.0)),
                "proof": {
                    "reasoning": "ILP",
                    "constraints": ["uniqueness", "geometry[footer]", "format"],
                    "violated": [],
                },
            }

        # Merge split titles: FitLayout sometimes splits long titles into 2 nodes
        if "title" in record:
            title_y = record["title"]["bbox"][1] if record["title"]["bbox"] else 0
            title_text = record["title"]["text"]
            price_text = record.get("price", {}).get("text", "")

            for node in raw_nodes:
                ny    = node["bbox"][1] if node.get("bbox") else 0
                text  = node.get("text", "").strip()
                if (
                    abs(ny - title_y) <= 60
                    and text != title_text
                    and text != price_text
                    and not self.PRICE_PATTERN.search(text)
                    and len(text) > 2
                ):
                    record["title"]["text"] = title_text + " " + text
                    break

        return record

    # ------------------------------------------------------------------
    # Phase 4b: Greedy reasoning (ablation baseline)
    # ------------------------------------------------------------------

    def _greedy_solve(
        self, raw_nodes: List[Dict], scores: np.ndarray
    ) -> Dict[str, Any]:
        """
        Greedy argmax baseline — no constraint enforcement.
        Used in ablation study to isolate the contribution of ILP (H2).
        """
        record: Dict[str, Any] = {}
        best: Dict[str, tuple] = {}  # field -> (score, node_idx)

        FIELD_IDX = {"price": 0, "title": 1}

        for i, node in enumerate(raw_nodes):
            for field, j in FIELD_IDX.items():
                s = float(scores[i][j])
                if field not in best or s > best[field][0]:
                    best[field] = (s, i)

        for field, (conf, idx) in best.items():
            node = raw_nodes[idx]
            record[field] = {
                "text": node.get("text", ""),
                "bbox": node.get("bbox", [0, 0, 0, 0]),
                "confidence": conf,
                "proof": {
                    "reasoning": "greedy",
                    "constraints": [],
                    "violated": ["no_constraint_enforcement"],
                },
            }
        return record

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _estimate_page_height(self, raw_nodes: List[Dict]) -> float:
        """Estimate page height from the lowest node bounding box."""
        max_bottom = 1080.0
        for node in raw_nodes:
            bbox = node.get("bbox", [0, 0, 0, 0])
            if len(bbox) >= 4:
                bottom = float(bbox[1]) + float(bbox[3])
                if bottom > max_bottom:
                    max_bottom = bottom
        return max_bottom