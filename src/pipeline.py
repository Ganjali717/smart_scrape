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
        End-to-end pipeline for a single page.

        In addition to the original extraction result, this method now
        computes a stability score σ(P) (margin-based) and performs a
        simple drift check, as motivated by Theorem 1 and Eq. (1) in the
        SmartScrape paper.
        """

        json_data = self.client.get_page_content(url)
        data, raw_nodes = self.parser.parse(json_data)
        if data is None or raw_nodes is None:
            return None

        with torch.no_grad():
            logits = self.model(data)
            # log_softmax -> exp to obtain probabilities; any monotone scores
            # are acceptable for margin-based σ.
            probs = torch.exp(logits).numpy()

        # Inject hand-crafted priors (heuristics); this modifies probs in-place
        # but still preserves a meaningful ranking for margin-based stability.
        self._inject_priors(raw_nodes, probs)

        # Constrained inference as in Eq. (1).
        final_record = self.solver.solve(raw_nodes, probs)

        # Post-processing: aggregate title near <h1>.
        final_record = self._aggregate_title(final_record, raw_nodes)

        # --- NEW: stability estimation and drift detection (Theorem 1) ---
        if probs.size == 0:
            stability_score = 0.0
            drift_alert = False
            drift_context = {
                "page_url": url,
                "stability": stability_score,
                "stability_threshold": self.drift_monitor.stability_threshold,
                "history_size": 0,
                "history_mean": None,
                "drift_detected": False,
            }
        else:
            stability_score, drift_alert, drift_context = self.drift_monitor.evaluate(
                probs=probs,
                solver_result=final_record,
                page_url=url,
            )

        # Active learning hook: if drift is detected, emit a human-review query
        # and simulate a retraining trigger to conceptually close the
        # learn–validate–repair loop from Section V.
        if drift_alert:
            self.active_learning.handle_drift(
                page_url=url,
                stability_score=stability_score,
                drift_context=drift_context,
                solver_result=final_record,
            )

        # Attach metadata to the record while preserving backward compatibility
        # with existing consumers that expect keys like "title" and "price".
        final_record["drift_alert"] = bool(drift_alert)
        final_record["stability_score"] = float(stability_score)

        return final_record

    def _aggregate_title(self, record, raw_nodes):
        """
        Post-Processing: Visual Text Harvesting.
        Мы определяем зону H1, а затем собираем ВЕСЬ текст, который попадает в эту зону.
        Это решает проблему, когда H1 - пустой контейнер, а текст лежит внутри.
        """
        if "title" not in record:
            return record

        # 1. Ищем узлы H1, чтобы определить "Зону Интереса"
        h1_nodes = [n for n in raw_nodes if n.get("tag", "").lower() == "h1"]

        # Если H1 не найдены, доверяем Solver
        if not h1_nodes:
            return record

        # 2. Вычисляем границы Зоны (Bounding Box всех H1)
        xs = [n["bbox"][0] for n in h1_nodes]
        ys = [n["bbox"][1] for n in h1_nodes]
        rights = [n["bbox"][0] + n["bbox"][2] for n in h1_nodes]
        bottoms = [n["bbox"][1] + n["bbox"][3] for n in h1_nodes]

        zone_x1, zone_y1 = min(xs), min(ys)
        zone_x2, zone_y2 = max(rights), max(bottoms)

        # 3. "Пылесос": Ищем ЛЮБЫЕ узлы с текстом внутри этой зоны
        content_nodes = []
        for n in raw_nodes:
            txt = n.get("text", "").strip()
            if len(txt) < 1:
                continue  # Пропускаем пустышки

            # Проверяем, попадает ли центр узла в Зону H1
            nx, ny, nw, nh = n["bbox"]
            center_x = nx + nw / 2
            center_y = ny + nh / 2

            # Добавляем небольшой допуск (margin), если текст чуть вылезает
            margin = 5.0
            if (zone_x1 - margin <= center_x <= zone_x2 + margin) and (
                zone_y1 - margin <= center_y <= zone_y2 + margin
            ):
                content_nodes.append(n)

        # Если мы нашли контент внутри H1, используем его
        if content_nodes:
            print(
                f"[Aggregation] Found {len(content_nodes)} text nodes inside H1 zone. Merging..."
            )

            # 4. Сортируем (сверху-вниз, слева-направо)
            content_nodes.sort(key=lambda n: n["bbox"][1] + (n["bbox"][0] / 10000))

            # 5. Склеиваем
            full_text = " ".join([n["text"].strip() for n in content_nodes])

            # 6. Обновляем результат
            record["title"]["text"] = full_text

            # Обновляем bbox до общего размера зоны
            record["title"]["bbox"] = [
                zone_x1,
                zone_y1,
                zone_x2 - zone_x1,
                zone_y2 - zone_y1,
            ]

        return record

    def _inject_priors(self, raw_nodes, probs):
        """
        Logic with H1 boost and empty text filtering.
        """
        for i, node in enumerate(raw_nodes):
            text = node.get("text", "").strip()
            tag = node.get("tag", "").lower()
            bbox = node.get("bbox", [0, 0, 0, 0])
            y_coord = bbox[1]

            # Фильтр пустых
            if len(text) < 1:
                probs[i][0] = -100.0
                probs[i][1] = -100.0
                continue

            # TITLE Logic
            if tag == "h1":
                probs[i][1] += 20.0
            elif y_coord < 400 and 5 < len(text) < 150:
                probs[i][1] += 5.0
                if text[0].isupper():
                    probs[i][1] += 2.0

            if len(text) > 150:
                probs[i][1] -= 10.0
            if tag in ["li", "ul", "nav"]:
                probs[i][1] -= 5.0
            if "home" in text.lower() or "books" in text.lower():
                probs[i][1] -= 5.0

            # PRICE Logic
            import re

            price_pattern = r"([£$€₼]|AZN)?\s*\d+([.,]\d{2})?\s*([£$€₼]|AZN)?"
            has_price_pattern = re.search(price_pattern, text)

            # Проверяем наличие символов валют вручную (для надежности)
            has_currency_symbol = (
                "£" in text
                or "$" in text
                or "€" in text
                or "AZN" in text
                or "₼" in text
            )

            # ГЛАВНОЕ ИЗМЕНЕНИЕ ЗДЕСЬ:
            # Если (есть символ ВАЛЮТЫ ИЛИ сработал REGEX) И текст короткий
            if (has_currency_symbol or has_price_pattern) and len(text) < 20:
                probs[i][0] += 15.0

                # Фильтр мусора
                if "0.00" in text or "Tax" in text:
                    probs[i][0] -= 20.0
            else:
                probs[i][0] -= 5.0

            # Global Filters
            if y_coord > 1000:
                probs[i][1] -= 10.0
            bad_words = ["stock", "available", "demo", "warning", "fiction"]
            if any(w in text.lower() for w in bad_words):
                probs[i][1] -= 20.0
                if "stock" in text.lower():
                    probs[i][0] -= 20.0
