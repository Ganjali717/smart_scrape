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

        # 1. Агрегация Заголовка (уже было)
        final_record = self._aggregate_title(final_record, raw_nodes)

        # 2. НОВОЕ: Агрегация Цены (Склеиваем цифру и валюту)
        final_record = self._aggregate_price(final_record, raw_nodes)

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

    def _aggregate_price(self, record, raw_nodes):
        """
        Post-Processing: Price Merge.
        Если Solver выбрал только значок валюты (AZN, $),
        находим число, стоящее рядом (обычно слева).
        """
        if "price" not in record:
            return record

        text = record["price"]["text"].strip()
        bbox = record["price"]["bbox"]  # [x, y, w, h]

        # Проверяем: если в тексте НЕТ цифр (значит поймали только валюту)
        import re

        if not re.search(r"\d", text):
            print(
                f"[Aggregation] Price node '{text}' has no digits. Looking for neighbors..."
            )

            # Центр текущего узла по Y
            target_y = bbox[1] + bbox[3] / 2
            target_x = bbox[0]

            best_neighbor = None
            min_dist = 200.0  # Ищем в радиусе 200 пикселей

            for n in raw_nodes:
                n_text = n.get("text", "").strip()

                # Сосед должен содержать цифры
                if not re.search(r"\d", n_text):
                    continue

                # Игнорируем слишком длинный текст (это описание)
                if len(n_text) > 20:
                    continue

                nx, ny, nw, nh = n["bbox"]
                n_cy = ny + nh / 2

                # 1. Проверка по вертикали (должны быть на одной строке)
                # Допуск 20 пикселей вверх/вниз
                if abs(n_cy - target_y) > 20:
                    continue

                # 2. Проверка по горизонтали (ищем ближайшего слева или справа)
                # Расстояние между краями
                dist = 1000

                # Если сосед слева (Цифра ... AZN)
                if nx < target_x:
                    dist = target_x - (nx + nw)
                # Если сосед справа ($ ... Цифра)
                else:
                    dist = nx - (target_x + bbox[2])

                if 0 <= dist < min_dist:
                    min_dist = dist
                    best_neighbor = n

            if best_neighbor:
                print(f"[Aggregation] Merging price with: {best_neighbor['text']}")

                # Склеиваем текст.
                # Если сосед слева: "100" + " " + "AZN"
                if best_neighbor["bbox"][0] < bbox[0]:
                    full_text = f"{best_neighbor['text']} {text}"
                    # Расширяем bbox влево
                    new_x = best_neighbor["bbox"][0]
                    new_w = (bbox[0] + bbox[2]) - new_x
                else:
                    full_text = f"{text} {best_neighbor['text']}"
                    # Расширяем bbox вправо
                    new_x = bbox[0]
                    new_w = (
                        best_neighbor["bbox"][0] + best_neighbor["bbox"][2]
                    ) - new_x

                record["price"]["text"] = full_text
                record["price"]["bbox"] = [new_x, bbox[1], new_w, bbox[3]]

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

            # ============================
            # 2. ЛОГИКА ЦЕНЫ (PRICE)
            # ============================

            # Улучшенный Regex:
            # 1. Поддержка пробелов в тысячах (1 290)
            # 2. Поддержка чешского формата (,-) и Kč
            # 3. Стандартные валюты
            import re

            # Ищем: (Валюта)(Число с пробелами/точками)(Валюта/,-)
            price_pattern = (
                r"([£$€₼]|AZN)?\s*[\d\s]+([.,-]\d{0,2})?\s*([£$€₼]|AZN|Kč|,-)?"
            )

            # Проверка: есть ли цифры вообще?
            has_digits = re.search(r"\d", text)
            # Проверка паттерна
            has_price_pattern = re.search(price_pattern, text)

            # Список валют (добавили Kč и ,-)
            currency_symbols = ["£", "$", "€", "₼", "AZN", "Kč", ",-"]
            has_currency = any(c in text for c in currency_symbols)

            if has_digits and (has_currency or has_price_pattern) and len(text) < 30:
                probs[i][0] += 15.0

                # Если явная валюта - еще буст
                if has_currency:
                    probs[i][0] += 5.0
            else:
                probs[i][0] -= 5.0

            # --- ОТРИЦАТЕЛЬНЫЕ ФИЛЬТРЫ (CONSTRAINTS) ---

            # A. Фильтр Скидок, Экономии и РАССРОЧКИ (Installments)
            # Мы убиваем цену, если видим слова-маркеры скидок или ежемесячных платежей.
            bad_price_words = [
                "save",
                "discount",
                "off",
                "sleva",
                "ušetříte",
                "difference",  # Скидки
                "monthly",
                "měsíčně",
                "month",
                "mesicne",
                "from",  # Рассрочка и "от"
            ]

            if any(w in text.lower() for w in bad_price_words):
                probs[i][0] -= 30.0  # Это не полная цена товара

            # B. Фильтр телефонов
            phone_pattern = r"(\(\d{3}\))|(\+\d+)|(\d{3}-\d{2}-\d{2})"
            if re.search(phone_pattern, text) or text.count("-") >= 2:
                # Исключение: чешская цена "100,-" имеет одно тире, это ок.
                # Но телефоны имеют 2+ тире или скобки.
                if not text.strip().endswith(",-"):
                    probs[i][0] -= 30.0

            # C. Фильтр мусора
            if "0.00" in text or "Tax" in text or "star" in text.lower():
                probs[i][0] -= 20.0

            # Global Filters
            if y_coord > 1000:
                probs[i][1] -= 10.0
            bad_words = ["stock", "available", "demo", "warning", "fiction"]
            if any(w in text.lower() for w in bad_words):
                probs[i][1] -= 20.0
                if "stock" in text.lower():
                    probs[i][0] -= 20.0
