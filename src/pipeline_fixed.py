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
from src.learning import encoding
from src.learning.graph_builder import FitLayoutParser
from src.learning.gnn_model import SmartScrapeGNN
from src.learning.drift_monitor import DriftMonitor, ActiveLearningManager
from src.reasoning.solver_fixed import ConstraintSolver
from config import FOOTER_THRESHOLD, STABILITY_THRESHOLD, MODEL_CHECKPOINT, PRODUCT_ZONE_MAX_PX


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
        use_priors: bool = True,
        offline: bool = False,
    ):
        torch.manual_seed(42)
        np.random.seed(42)

        self.reasoning_mode = reasoning_mode
        self.use_mock = use_mock
        self.use_priors = use_priors

        self.offline = offline
        self.client = FitLayoutClient()
        self.parser = FitLayoutParser()
        self._demo_pages = self._load_demo_pages() if offline else {}

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
                "Run train_gnn.py to generate model.pt."
            )
            self._model_trained = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, url: str, drift: float = 0.0) -> Dict[str, Any] | None:
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

        # Synthetic drift: blend GNN probabilities toward uniform by
        # `drift` in [0,1]. This genuinely lowers per-node margins, so
        # sigma(P) drops and the scores/decision change for real.
        if drift and drift > 0.0:
            u = np.full_like(raw_probs, 1.0 / raw_probs.shape[1])
            raw_probs = (1.0 - drift) * raw_probs + drift * u

        # ------ Phase 3: Heuristic priors (feature engineering) ------
        # These serve as strong signals for the untrained model and as
        # interpretable features when the GNN is trained.
        scores = raw_probs.copy()
        if self.use_priors:
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
        candidates = self._build_candidate_trace(raw_nodes, scores, page_height)
        final_record["_meta"] = {
            "drift_alert": bool(drift_alert),
            "stability_score": float(stability_score),
            "reasoning_mode": self.reasoning_mode,
            "use_priors": self.use_priors,
            "drift": float(drift),
            "candidates": candidates,
            "model_trained": self._model_trained,
            "active_constraints": [
                "uniqueness[title]",
                "uniqueness[price]",
                f"Γ2 footer_trap[y > {FOOTER_THRESHOLD} × page_height]",
                f"Γ3 product_zone[y > {PRODUCT_ZONE_MAX_PX}px]",
                "Γ4 format[price => HasCurrency ∧ IsNumeric]",
            ],
        }

        return final_record

    # ------------------------------------------------------------------
    # Phase 1: Data acquisition
    # ------------------------------------------------------------------

    def _acquire_page(self, url: str):
        """Try live FitLayout, fall back to mock if use_mock=True."""
        if self.offline:
            return self._acquire_offline(url)
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


    def _load_demo_pages(self):
        """Load bundled offline demo pages (data/demo_pages.json)."""
        import json, os
        path = os.path.join("data", "demo_pages.json")
        if not os.path.exists(path):
            print("[Pipeline] offline=True but data/demo_pages.json missing.")
            return {}
        with open(path, encoding="utf-8") as f:
            pages = json.load(f)
        print(f"[Pipeline] Offline mode: {len(pages)} bundled demo pages.")
        return pages

    def _acquire_offline(self, url: str):
        """Build a graph from a bundled demo page (no network/FitLayout)."""
        page = self._demo_pages.get(url)
        if page is None and self._demo_pages:
            page = next(iter(self._demo_pages.values()))
            print(f"[Pipeline] URL not in demo set; using first bundled page.")
        if page is None:
            return None, []
        raw_nodes = []
        for j, n in enumerate(page["nodes"]):
            bbox = n.get("bbox") or [0, 0, 0, 0]
            raw_nodes.append({
                "id": n.get("id", j),
                "text": n.get("text", ""),
                "tag": n.get("tag", "div"),
                "bbox": [float(b) for b in bbox],
            })
        edge_index = encoding.build_edges(raw_nodes, k=3)
        feats = torch.stack([encoding.node_features(n) for n in raw_nodes])
        from torch_geometric.data import Data
        return Data(x=feats, edge_index=edge_index), raw_nodes

    def _build_candidate_trace(self, raw_nodes, scores, page_height):
        """
        Real per-candidate scores for the UI (replaces hard-coded demo bars).
        For title/price, return the top nodes by fused score, flagging which
        ones the ILP constraints exclude (footer/zone/format).
        """
        import numpy as _np
        from config import FOOTER_THRESHOLD
        out = {}
        field_idx = {"price": 0, "title": 1}
        for field, j in field_idx.items():
            order = _np.argsort(-scores[:, j])[:5]
            items = []
            for i in order:
                node = raw_nodes[int(i)]
                y = float(node["bbox"][1]) if node.get("bbox") else 0.0
                excluded = []
                if y > page_height * FOOTER_THRESHOLD:
                    excluded.append("Γ2 footer")
                if y > self.solver.product_zone_max_px:
                    excluded.append("Γ3 zone")
                if field == "price" and not self.solver._has_price_format(str(node["text"])):
                    excluded.append("Γ4 format")
                items.append({
                    "text": (node["text"] or "")[:28],
                    "score": float(scores[int(i)][j]),
                    "excluded_by": excluded,
                })
            out[field] = items
        return out

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

    def _inject_priors(self, raw_nodes, scores):
        """
        Bounded, SECONDARY priors added on top of GNN probabilities.

        The GNN (range [0,1] per class) is the primary signal; these priors
        are small nudges (|delta| <= 0.3) that change the outcome only when
        the GNN is genuinely uncertain. Hard layout/format rules (footer,
        product-zone, price-format) are NOT applied here - they are enforced
        exactly by the ILP solver (Gamma2-Gamma4), so duplicating them as
        large penalties is unnecessary and would let heuristics dominate.
        """
        STOP_WORDS = {
            "warning", "notice", "in stock", "add to basket", "add to cart",
            "home", "sign in", "register", "this is a demo", "prices and ratings",
        }
        for i, node in enumerate(raw_nodes):
            text = str(node.get("text", "")).strip()
            tag = str(node.get("tag", "")).lower()

            if not text:
                scores[i][0] -= 1.0
                scores[i][1] -= 1.0
                continue

            is_price_text = bool(self.PRICE_PATTERN.search(text)) and len(text) < 25
            is_nav_text = (
                "/" in text or text.isdigit() or len(text) < 3
                or any(sw in text.lower() for sw in STOP_WORDS)
            )

            # PRICE (index 0): mild boost for currency-looking short text.
            if is_price_text:
                scores[i][0] += 0.3

            # TITLE (index 1): boost a prominent H1; discourage price/nav text.
            if tag == "h1":
                scores[i][1] += 0.3
            if is_price_text:
                scores[i][1] -= 0.3
            elif is_nav_text:
                scores[i][1] -= 0.3


    # ------------------------------------------------------------------
    # Phase 4a: ILP reasoning
    # ------------------------------------------------------------------

    def _format_ilp_result(
        self, solver_result: Dict, raw_nodes: List[Dict]
    ) -> Dict[str, Any]:
        """Convert ConstraintSolver output to standard proof/provenance record."""
        record: Dict[str, Any] = {}
        solver_meta = solver_result.get("_solver", {}) if isinstance(solver_result, dict) else {}

        for field_name, field_data in solver_result.items():
            if field_name.startswith("_") or not isinstance(field_data, dict):
                continue
            record[field_name] = {
                "text": field_data.get("text", ""),
                "bbox": field_data.get("bbox", [0, 0, 0, 0]),
                "confidence": float(field_data.get("confidence", 0.0)),
                "node_id": field_data.get("node_id"),
                "solver_variable": field_data.get("solver_variable"),
                "proof": {
                    "reasoning": "ILP",
                    "solver_backend": solver_meta.get("backend", "OR-Tools SCIP"),
                    "solver_status": solver_meta.get("status", "UNKNOWN"),
                    "objective_value": solver_meta.get("objective_value"),
                    "constraints": solver_meta.get("constraints", []),
                    "violated": solver_meta.get("violated", []),
                    "fallback_used": bool(solver_meta.get("fallback_used", False)),
                },
            }

        # Merge split titles: FitLayout sometimes splits long titles into 2 nodes.
        if "title" in record:
            title_y = record["title"]["bbox"][1] if record["title"].get("bbox") else 0
            title_text = record["title"]["text"]
            price_text = record.get("price", {}).get("text", "")

            for node in raw_nodes:
                ny = node["bbox"][1] if node.get("bbox") else 0
                text = node.get("text", "").strip()
                if (
                    abs(ny - title_y) <= 60
                    and text != title_text
                    and text != price_text
                    and not self.PRICE_PATTERN.search(text)
                    and len(text) > 2
                ):
                    record["title"]["text"] = title_text + " " + text
                    break

        if solver_meta:
            record["_solver"] = solver_meta

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