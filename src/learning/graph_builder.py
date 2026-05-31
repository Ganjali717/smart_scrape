import re
import torch
import networkx as nx
from torch_geometric.data import Data
from .features import FeatureEncoder
from . import encoding


class FitLayoutParser:
    """
    Parses JSON-LD from FitLayout and builds a graph for GNN.
    """

    def __init__(self):
        self.encoder = FeatureEncoder()

    def parse(self, json_data: dict):
        """
        Main method: JSON -> PyTorch Geometric Data
        """
        graphs = json_data.get("@graph", [])

        # Flat list of all elements (top level + nested @graphs)
        flat_items = []
        for g in graphs:
            if isinstance(g, dict) and "@graph" in g:
                flat_items.append(g)
                inner = g.get("@graph", [])
                if isinstance(inner, list):
                    for inner_item in inner:
                        if isinstance(inner_item, dict):
                            flat_items.append(inner_item)
            else:
                flat_items.append(g)

        # Index of all objects by @id: needed to resolve b:bounds -> rect object
        id_index = {
            item["@id"]: item
            for item in flat_items
            if isinstance(item, dict) and "@id" in item
        }

        # Page dimensions (b:width/b:height of the Page object)
        page_w, page_h = self._get_page_size(flat_items)

        nodes = []
        raw_nodes = []
        extracted_items = []

        # Iterate through all elements (handling the nested @graph variant)
        for g in graphs:
            inner_items = (
                g.get("@graph", []) if isinstance(g, dict) and "@graph" in g else [g]
            )
            for item in inner_items:
                if not isinstance(item, dict):
                    continue

                # ── FIX: get raw bounds value (dict or str), NOT converted to string ──
                bounds_ref = None
                for k, v in item.items():
                    if k.endswith("bounds"):
                        bounds_ref = v  # keep as-is: may be {'@id': '...'} or str
                        break

                if not bounds_ref:
                    continue

                bbox = self._resolve_bbox(bounds_ref, id_index)
                if not bbox:
                    continue

                x, y, w, h = bbox
                extracted_items.append((item, x, y, w, h))

        print(
            f"[GraphBuilder] Found {len(extracted_items)} elements with coordinates."
        )

        # 3. Create features for each element
        for item, x, y, w, h in extracted_items:
            # text
            text = self._find_key_ending_with(item, "text") or ""

            # html tag (in the artifact this is b:htmlTagName)
            tag = (
                self._find_key_ending_with(item, "htmlTagName")
                or self._find_key_ending_with(item, "htmlTag")
                or "div"
            )

            # --- Encoding ---
            # Feature order MUST match the paper and training code:
            # 128-dim text hash + 4-dim visual bbox + 15-dim tag one-hot = 147 dims.
            # Do not change this order without retraining model.pt.
            t_feat = self.encoder.encode_text(text)
            v_feat = self.encoder.encode_visual(x, y, w, h, page_w, page_h)
            s_feat = self.encoder.encode_tag(tag)

            full_vector = torch.cat([t_feat, v_feat, s_feat])
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
            print("[Error] Failed to build nodes. Check the JSON format.")
            return None, None

        x_tensor = torch.stack(nodes)
        edge_index = encoding.build_edges(raw_nodes, k=3)
        data = Data(x=x_tensor, edge_index=edge_index)

        return data, raw_nodes

    # --------- helpers ---------

    def _find_key_ending_with(self, d: dict, suffix: str) -> str | None:
        """Finds a value by key suffix (b:text, b:htmlTagName, etc.)."""
        for k, v in d.items():
            if k.endswith(suffix):
                # FitLayout values often come wrapped in {"@value": "..."} — unwrap it
                if isinstance(v, dict) and "@value" in v:
                    return str(v["@value"])
                if isinstance(v, dict):
                    return None  # don't stringify dicts (e.g. bounds refs)
                return str(v)
        return None

    def _resolve_bbox(self, bounds_ref, id_index):
        """
        Resolves the b:bounds value to (x, y, w, h).
        Possible variants:
        - string "x,y,w,h" or "x y w h"
        - IRI string "r:art66#b0-rect-b"
        - dictionary {"@id": "..."} pointing to a rect object
        - direct rect object with b:positionX / b:positionY / b:width / b:height
        """
        # variant 1: string
        if isinstance(bounds_ref, str):
            coords = self._parse_coords_string(bounds_ref)
            if coords:
                return coords
            rect = id_index.get(bounds_ref)
            if rect:
                return self._bbox_from_rect(rect)
            return None

        # variant 2: dict {"@id": "r:art157#a1-rect-b"}
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
        if x is None or y is None or w is None or h is None:
            return None
        return float(x), float(y), float(w), float(h)

    def _parse_coords_string(self, s: str):
        """Parses strings like 'x,y,w,h' or 'x y w h'."""
        parts = re.split(r"[,\s]+", s.strip())
        if len(parts) != 4:
            return None
        try:
            return tuple(float(p) for p in parts)
        except ValueError:
            return None

    def _get_numeric(self, obj: dict, suffix: str):
        """Extracts a number from fields like b:width / b:height / b:positionX, etc."""
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
        """Finds the Page object and gets b:width/b:height; otherwise defaults to 1920x1080."""
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

