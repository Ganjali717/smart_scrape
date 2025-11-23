import re
import torch
import networkx as nx
from torch_geometric.data import Data
from .features import FeatureEncoder


class FitLayoutParser:
    """
    Парсит JSON-LD от FitLayout и строит граф для GNN.
    """

    def __init__(self):
        self.encoder = FeatureEncoder()

    def parse(self, json_data: dict):
        """
        Главный метод: JSON -> PyTorch Geometric Data
        """
        graphs = json_data.get("@graph", [])

        # Индекс всех объектов по @id: нужен, чтобы раскрывать b:bounds -> rect-объект
        id_index = {
            item["@id"]: item
            for item in graphs
            if isinstance(item, dict) and "@id" in item
        }

        # Размеры страницы (b:width/b:height у объекта Page)
        page_w, page_h = self._get_page_size(graphs)

        nodes = []
        raw_nodes = []
        extracted_items = []

        # Проходим по всем элементам (учитываем вариант с вложенными @graph)
        for g in graphs:
            inner_items = (
                g.get("@graph", []) if isinstance(g, dict) and "@graph" in g else [g]
            )
            for item in inner_items:
                if not isinstance(item, dict):
                    continue

                bounds_ref = self._find_key_ending_with(item, "bounds")
                if not bounds_ref:
                    continue

                bbox = self._resolve_bbox(bounds_ref, id_index)
                if not bbox:
                    continue

                x, y, w, h = bbox
                extracted_items.append((item, x, y, w, h))

        print(
            f"[GraphBuilder] Найдено {len(extracted_items)} элементов с координатами."
        )

        # 3. Создаём фичи для каждого элемента
        for item, x, y, w, h in extracted_items:
            # текст
            text = self._find_key_ending_with(item, "text") or ""

            # html-тег (в артефакте это b:htmlTagName)
            tag = (
                self._find_key_ending_with(item, "htmlTagName")
                or self._find_key_ending_with(item, "htmlTag")
                or "div"
            )

            # --- Encoding ---
            v_feat = self.encoder.encode_visual(x, y, w, h, page_w, page_h)
            t_feat = self.encoder.encode_text(text)
            s_feat = self.encoder.encode_tag(tag)

            full_vector = torch.cat([v_feat, t_feat, s_feat])
            nodes.append(full_vector)

            raw_nodes.append(
                {
                    "text": text,
                    "tag": tag,
                    "bbox": [x, y, w, h],
                    "id": item.get("@id", "unknown"),
                }
            )

        if not nodes:
            print("[Error] Не удалось построить узлы. Проверьте формат JSON.")
            return None, None

        x_tensor = torch.stack(nodes)
        edge_index = self._build_knn_edges(raw_nodes, k=3)
        data = Data(x=x_tensor, edge_index=edge_index)

        return data, raw_nodes

    # --------- helpers ---------

    def _find_key_ending_with(self, d: dict, suffix: str):
        """Поиск значения по суффиксу ключа (b:text, b:htmlTagName, b:bounds и т.д.)."""
        for k, v in d.items():
            if k.endswith(suffix):
                return v
        return None

    def _resolve_bbox(self, bounds_ref, id_index):
        """
        Приводит значение b:bounds к (x, y, w, h).
        Возможные варианты:
        - строка "x,y,w,h" или "x y w h"
        - строка-IRI "r:art66#b0-rect-b"
        - словарь {"@id": "..."} на rect-объект
        - сразу rect-объект с b:positionX / b:positionY / b:width / b:height
        """
        # вариант 1: строка
        if isinstance(bounds_ref, str):
            coords = self._parse_coords_string(bounds_ref)
            if coords:
                return coords

            rect = id_index.get(bounds_ref)
            if rect:
                return self._bbox_from_rect(rect)
            return None

        # вариант 2: dict
        if isinstance(bounds_ref, dict):
            if "@id" in bounds_ref:
                rect = id_index.get(bounds_ref["@id"], bounds_ref)
            else:
                rect = bounds_ref
            return self._bbox_from_rect(rect)

        return None

    def _bbox_from_rect(self, rect: dict):
        x = self._get_numeric(rect, "positionX")
        y = self._get_numeric(rect, "positionY")
        w = self._get_numeric(rect, "width")
        h = self._get_numeric(rect, "height")
        if None in (x, y, w, h):
            return None
        return float(x), float(y), float(w), float(h)

    def _parse_coords_string(self, s: str):
        """Парсинг строк 'x,y,w,h' или 'x y w h'."""
        parts = re.split(r"[,\s]+", s.strip())
        if len(parts) != 4:
            return None
        try:
            return tuple(float(p) for p in parts)
        except ValueError:
            return None

    def _get_numeric(self, obj: dict, suffix: str):
        """Берёт число из поля вида b:width / b:height / b:positionX и т.п."""
        for k, v in obj.items():
            if k.endswith(suffix):
                if isinstance(v, dict):
                    v = v.get("@value", v)
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
        return None

    def _get_page_size(self, graphs):
        """Ищет объект Page и берёт b:width/b:height; иначе дефолт 1920x1080."""
        page_w = page_h = None
        for item in graphs:
            if not isinstance(item, dict):
                continue
            if not str(item.get("@type", "")).endswith("Page"):
                continue

            w = self._get_numeric(item, "width")
            h = self._get_numeric(item, "height")
            if w:
                page_w = w
            if h:
                page_h = h
            break

        if page_w is None:
            page_w = 1920.0
        if page_h is None:
            page_h = 1080.0
        return page_w, page_h

    def _build_knn_edges(self, nodes_info, k=3):
        """Строит связи между визуально близкими элементами."""
        edges = []
        num_nodes = len(nodes_info)
        if num_nodes < 2:
            return torch.tensor([[0], [0]], dtype=torch.long)

        for i in range(num_nodes):
            dists = []
            xi, yi, _, _ = nodes_info[i]["bbox"]

            for j in range(num_nodes):
                if i == j:
                    continue
                xj, yj, _, _ = nodes_info[j]["bbox"]
                dist = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
                dists.append((dist, j))

            dists.sort(key=lambda x: x[0])
            for _, neighbor_idx in dists[:k]:
                edges.append([i, neighbor_idx])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index
