import torch
import numpy as np
import re
from src.integration.fitlayout import FitLayoutClient
from src.learning.graph_builder import FitLayoutParser
from src.learning.gnn_model import SmartScrapeGNN
from src.reasoning.solver import ConstraintSolver


class SmartScrapePipeline:
    def __init__(self):
        torch.manual_seed(42)
        np.random.seed(42)

        self.client = FitLayoutClient()
        self.parser = FitLayoutParser()
        self.classes = ["price", "title", "other"]
        self.model = SmartScrapeGNN(
            input_dim=146, hidden_dim=64, num_classes=len(self.classes)
        )
        self.model.eval()
        self.solver = ConstraintSolver(self.classes)

    def run(self, url: str):
        print(f"\n=== Processing: {url} ===")

        json_data = self.client.get_page_content(url)
        data, raw_nodes = self.parser.parse(json_data)
        if not data:
            return None

        with torch.no_grad():
            logits = self.model(data)
            probs = torch.exp(logits).numpy()

        self._inject_priors(raw_nodes, probs)

        print("\n[Pipeline] Running Constraint Solver...")
        final_record = self.solver.solve(raw_nodes, probs)

        # --- НОВЫЙ ШАГ: АГРЕГАЦИЯ ФРАГМЕНТОВ ---
        # Склеиваем разорванный текст (например, H1 на двух строках)
        final_record = self._aggregate_title(final_record, raw_nodes)

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

            price_pattern = r"[£$€]?\d+\.\d{2}"
            has_price_pattern = re.search(price_pattern, text)

            if has_price_pattern and len(text) < 20:
                probs[i][0] += 15.0
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
