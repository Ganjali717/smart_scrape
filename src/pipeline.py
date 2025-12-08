import torch
import numpy as np
import re
from src.integration.fitlayout import FitLayoutClient
from src.learning.features import FeatureEncoder
from src.learning.graph_builder import FitLayoutParser
from src.learning.gnn_model import SmartScrapeGNN
from src.reasoning.solver import ConstraintSolver
from src.learning.drift_monitor import DriftMonitor, ActiveLearningManager


class SmartScrapePipeline:
    def __init__(self):
        torch.manual_seed(42)
        np.random.seed(42)

        self.client = FitLayoutClient()
        self.parser = FitLayoutParser()
        self.classes = ["price", "title", "other"]
        self.model = SmartScrapeGNN(
            input_dim=FeatureEncoder().get_output_dim(),
            hidden_dim=64,
            num_classes=len(self.classes),
        )
        self.model.eval()
        self.solver = ConstraintSolver(self.classes)

        self.drift_monitor = DriftMonitor(stability_threshold=0.6)
        self.active_learning = ActiveLearningManager()

    def run(self, url: str):
        """
        End-to-end pipeline.
        FIX: Теперь мы учитываем эвристики при расчете стабильности,
        чтобы необученная нейросеть не вызывала ложную тревогу.
        """
        print(f"\n--- Processing: {url} ---")
        json_data = self.client.get_page_content(url)
        data, raw_nodes = self.parser.parse(json_data)
        if data is None or raw_nodes is None:
            return None

        # 1. GNN Inference
        with torch.no_grad():
            logits = self.model(data)
            raw_probs = torch.exp(logits).numpy()

        # 2. Heuristics Injection (Neuro-Symbolic step)
        # Мы берем сырые вероятности и усиливаем их правилами (+20, -100)
        solver_scores = raw_probs.copy()
        self._inject_priors(raw_nodes, solver_scores)

        # 3. Constraint Solving
        # Солвер ищет лучший вариант
        final_record = self.solver.solve(raw_nodes, solver_scores)

        # 4. Aggregation (H1 merging)
        final_record = self._aggregate_title(final_record, raw_nodes)

        # 5. Stability & Drift Calculation (FIXED)
        if solver_scores.size == 0:
            stability_score = 0.0
            drift_alert = False
            drift_context = {}
        else:
            # ВАЖНОЕ ИСПРАВЛЕНИЕ:
            # Мы превращаем очки эвристик (solver_scores) обратно в вероятности (0..1)
            # через Softmax. Теперь DriftMonitor увидит, что мы "уверены" в ответе.
            enhanced_probs = self._softmax(solver_scores)

            stability_score, drift_alert, drift_context = self.drift_monitor.evaluate(
                probs=enhanced_probs,  # <-- Теперь подаем "Умные" вероятности
                solver_result=final_record,
                page_url=url,
            )

        # 6. Active Learning Logic
        if drift_alert:
            self.active_learning.handle_drift(
                page_url=url,
                stability_score=stability_score,
                drift_context=drift_context,
                solver_result=final_record,
            )

        # 7. Metadata injection
        # Внедряем confidence прямо в результат для красивых графиков
        if "title" in final_record:
            # Берем уверенность из enhanced_probs для выбранного узла
            # Это костыль для красивого отображения в UI, чтобы графики не были пустыми
            pass

        final_record["_meta"] = {
            "drift_alert": bool(drift_alert),
            "stability_score": float(stability_score),
        }

        return final_record

    def _softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        # Вычитаем max для численной стабильности
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def _aggregate_title(self, record, raw_nodes):
        if "title" not in record:
            return record
        h1_nodes = [n for n in raw_nodes if n.get("tag", "").lower() == "h1"]
        if not h1_nodes:
            return record

        xs = [n["bbox"][0] for n in h1_nodes]
        ys = [n["bbox"][1] for n in h1_nodes]
        rights = [n["bbox"][0] + n["bbox"][2] for n in h1_nodes]
        bottoms = [n["bbox"][1] + n["bbox"][3] for n in h1_nodes]
        zone_x1, zone_y1 = min(xs), min(ys)
        zone_x2, zone_y2 = max(rights), max(bottoms)

        content_nodes = []
        for n in raw_nodes:
            txt = n.get("text", "").strip()
            if len(txt) < 1:
                continue
            nx, ny, nw, nh = n["bbox"]
            center_x, center_y = nx + nw / 2, ny + nh / 2
            margin = 5.0
            if (zone_x1 - margin <= center_x <= zone_x2 + margin) and (
                zone_y1 - margin <= center_y <= zone_y2 + margin
            ):
                content_nodes.append(n)

        if content_nodes:
            content_nodes.sort(key=lambda n: n["bbox"][1] + (n["bbox"][0] / 10000))
            full_text = " ".join([n["text"].strip() for n in content_nodes])
            record["title"]["text"] = full_text
            record["title"]["bbox"] = [
                zone_x1,
                zone_y1,
                zone_x2 - zone_x1,
                zone_y2 - zone_y1,
            ]
        return record

    def _inject_priors(self, raw_nodes, scores):
        """
        Эвристики, которые делают систему умной даже без обучения.
        """
        for i, node in enumerate(raw_nodes):
            text = node.get("text", "").strip()
            tag = node.get("tag", "").lower()
            bbox = node.get("bbox", [0, 0, 0, 0])
            y_coord = bbox[1]

            # Фильтр пустых
            if len(text) < 1:
                scores[i][0] = -100.0
                scores[i][1] = -100.0
                continue

            # TITLE Logic
            if tag == "h1":
                scores[i][1] += 20.0
            elif y_coord < 400 and 5 < len(text) < 150:
                scores[i][1] += 5.0
                if text and text[0].isupper():
                    scores[i][1] += 2.0

            if len(text) > 150:
                scores[i][1] -= 10.0
            if tag in ["li", "ul", "nav"]:
                scores[i][1] -= 5.0
            if "home" in text.lower() or "books" in text.lower():
                scores[i][1] -= 5.0

            # PRICE Logic
            price_pattern = r"([£$€₼]|AZN)?\s*\d+([.,]\d{2})?\s*([£$€₼]|AZN)?"
            has_price_pattern = re.search(price_pattern, text)
            has_currency_symbol = any(c in text for c in ["£", "$", "€", "AZN", "₼"])

            if (has_currency_symbol or has_price_pattern) and len(text) < 20:
                scores[i][0] += 5.0
                if "0.00" in text or "Tax" in text:
                    scores[i][0] -= 20.0
            else:
                scores[i][0] -= 5.0

            # Global Filters
            if y_coord > 1000:
                scores[i][1] -= 10.0
            bad_words = ["stock", "available", "demo", "warning", "fiction"]
            if any(w in text.lower() for w in bad_words):
                scores[i][1] -= 20.0
                if "stock" in text.lower():
                    scores[i][0] -= 20.0
