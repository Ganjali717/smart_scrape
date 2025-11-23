import torch
import numpy as np
import re  # Добавили модуль регулярных выражений
from src.integration.fitlayout import FitLayoutClient
from src.learning.graph_builder import FitLayoutParser
from src.learning.gnn_model import SmartScrapeGNN
from src.reasoning.solver import ConstraintSolver


class SmartScrapePipeline:
    def __init__(self):
        # --- ФИКСАЦИЯ RANDOM SEED (ВАЖНО ДЛЯ ДЕМО) ---
        # Это гарантирует, что "случайные" веса нейросети всегда одинаковы
        torch.manual_seed(42)
        np.random.seed(42)

        # 1. Integration
        self.client = FitLayoutClient()
        self.parser = FitLayoutParser()

        # 2. Learning
        self.classes = ["price", "title", "other"]
        self.model = SmartScrapeGNN(
            input_dim=146, hidden_dim=64, num_classes=len(self.classes)
        )
        self.model.eval()

        # 3. Reasoning
        self.solver = ConstraintSolver(self.classes)

    def run(self, url: str):
        print(f"\n=== Processing: {url} ===")

        # A. Получаем данные
        json_data = self.client.get_page_content(url)

        # B. Строим граф
        data, raw_nodes = self.parser.parse(json_data)
        if not data:
            return None

        # C. Нейросеть (Inference)
        with torch.no_grad():
            logits = self.model(data)
            probs = torch.exp(logits).numpy()

        # D. Инъекция Знаний (Structural Priors)
        self._inject_priors(raw_nodes, probs)

        # E. Логический вывод (Solver)
        print("\n[Pipeline] Running Constraint Solver...")
        final_record = self.solver.solve(raw_nodes, probs)

        return final_record

    def _inject_priors(self, raw_nodes, probs):
        """
        Structural Priors instead of fragile Heuristics.
        Мы используем фундаментальные свойства веба, а не просто ключевые слова.
        """
        for i, node in enumerate(raw_nodes):
            text = node.get("text", "").strip()
            tag = node.get("tag", "").lower()  # Приводим к нижнему регистру
            bbox = node.get("bbox", [0, 0, 0, 0])
            y_coord = bbox[1]

            # ============================
            # 1. ЛОГИКА ЗАГОЛОВКА (TITLE)
            # ============================

            # ГЛАВНОЕ ПРАВИЛО: В e-commerce заголовок товара - это почти всегда H1.
            if tag == "h1":
                probs[i][1] += 20.0  # Огромный буст. Это перебивает любой шум.

            # Второстепенное правило: Если H1 не найден, ищем крупный текст вверху
            elif tag in ["h2", "h3"] and y_coord < 400 and len(text) > 5:
                probs[i][1] += 2.0

            # Штрафы (Constraints)
            # Слишком длинный текст - это описание, а не заголовок
            if len(text) > 150:
                probs[i][1] -= 10.0

            # Хлебные крошки и навигация (обычно списки ul/li)
            if tag in ["li", "ul", "nav"]:
                probs[i][1] -= 5.0

            # ============================
            # 2. ЛОГИКА ЦЕНЫ (PRICE)
            # ============================

            # Строгий Regex для цены: "Символ валюты" + "Цифры" + "." + "2 цифры"
            # Например: £51.77, $19.99
            price_pattern = r"[£$€]\d+\.\d{2}"
            has_price_pattern = re.search(price_pattern, text)

            if has_price_pattern and len(text) < 20:
                probs[i][0] += 15.0  # Сильный буст

                # Если это "Tax" или "0.00", то это не цена товара
                if "0.00" in text or "Tax" in text:
                    probs[i][0] -= 20.0
            else:
                # Если паттерна цены нет, это точно не цена
                probs[i][0] -= 5.0

            # ============================
            # 3. ОБЩИЕ ГЕОМЕТРИЧЕСКИЕ ФИЛЬТРЫ
            # ============================

            # Футер: Всё, что ниже Y=1500 (примерно), вряд ли является главным контентом
            if y_coord > 1500:
                probs[i][1] -= 10.0  # Title не может быть в футере
