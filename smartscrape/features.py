"""
Node features. Deterministic and identical at training and inference time.

Blocks (dims):
  text char 2-4 gram hash ......... 256
  previous-node text word hash ....  64   (label/value pairs: "Price:" -> "$12.99")
  tag path hash (last 4 tags) .....  32
  tag one-hot .....................  len(TAGS)+1
  hand-crafted ....................  16
  geometry ........................   6   (zeros + flag when the page is not rendered)
Optional: sentence embeddings (sentence-transformers) appended when enabled.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from .data import Page
from .formats import is_date, is_number, is_price

TAGS = ["div", "span", "a", "p", "li", "td", "th", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
        "b", "strong", "em", "i", "font", "label", "button", "option", "dd", "dt", "small", "big",
        "u", "center", "title", "img", "table", "ul", "form", "input", "select", "textarea"]
_TAG_IDX = {t: i for i, t in enumerate(TAGS)}

_char = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=256,
                          alternate_sign=False, norm="l2", lowercase=True)
_word = HashingVectorizer(analyzer="word", n_features=64, alternate_sign=False, norm="l2",
                          lowercase=True, token_pattern=r"(?u)\b\w+\b|[$£€:]")
_path = HashingVectorizer(analyzer="word", n_features=32, alternate_sign=False, norm="l2",
                          lowercase=True, token_pattern=r"[^/]+")

N_HAND = 16
N_GEOM = 6
BASE_DIM = 256 + 64 + 32 + (len(TAGS) + 1) + N_HAND + N_GEOM


class FeatureEncoder:
    def __init__(self, sentence_model: Optional[str] = None, device: str = "cpu"):
        self.sentence_model_name = sentence_model
        self._st = None
        self.device = device

    @property
    def dim(self) -> int:
        return BASE_DIM + (self._st_dim() if self.sentence_model_name else 0)

    def _st_dim(self) -> int:
        self._load_st()
        return self._st.get_sentence_embedding_dimension()

    def _load_st(self):
        if self._st is None and self.sentence_model_name:
            from sentence_transformers import SentenceTransformer  # optional dependency
            self._st = SentenceTransformer(self.sentence_model_name, device=self.device)

    def encode(self, page: Page) -> np.ndarray:
        n = page.n
        if n == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        texts = [nd.text for nd in page.nodes]
        prev = [""] + texts[:-1]
        paths = ["/".join(nd.xpath.split("/")[-4:]) if nd.xpath else nd.tag for nd in page.nodes]

        blocks = [
            _char.transform(texts).toarray(),
            _word.transform(prev).toarray(),
            _path.transform(paths).toarray(),
            self._tags(page),
            self._hand(page),
            self._geom(page),
        ]
        if self.sentence_model_name:
            self._load_st()
            blocks.append(self._st.encode(texts, batch_size=256, normalize_embeddings=True,
                                          show_progress_bar=False))
        return np.concatenate(blocks, axis=1).astype(np.float32)

    @staticmethod
    def _tags(page: Page) -> np.ndarray:
        m = np.zeros((page.n, len(TAGS) + 1), dtype=np.float32)
        for i, nd in enumerate(page.nodes):
            m[i, _TAG_IDX.get(nd.tag, len(TAGS))] = 1.0
        return m

    @staticmethod
    def _hand(page: Page) -> np.ndarray:
        m = np.zeros((page.n, N_HAND), dtype=np.float32)
        max_depth = max([nd.depth for nd in page.nodes] + [1])
        for i, nd in enumerate(page.nodes):
            t = nd.text.strip()
            L = max(1, len(t))
            words = t.split()
            m[i] = [
                math.log1p(len(t)) / 6.0,
                min(len(words), 50) / 50.0,
                sum(c.isdigit() for c in t) / L,
                sum(c.isupper() for c in t) / L,
                float(any(c in t for c in "$£€¥₼")),
                float(is_price(t)),
                float(is_date(t)),
                float(is_number(t)),
                (sum(1 for w in words if w[:1].isupper()) / max(1, len(words))),
                float(t.endswith(":")),
                page.position(i),
                nd.depth / max_depth,
                float("footer" in nd.regions),
                float("nav" in nd.regions),
                float("header" in nd.regions),
                float("aside" in nd.regions),
            ]
        return m

    @staticmethod
    def _geom(page: Page) -> np.ndarray:
        m = np.zeros((page.n, N_GEOM), dtype=np.float32)
        if not page.has_geometry():
            return m
        H = page.page_height()
        W = max((nd.bbox[0] + nd.bbox[2]) for nd in page.nodes if nd.has_geometry()) or 1.0
        sizes = sorted({nd.font_size for nd in page.nodes if nd.font_size})
        rank = {s: k / max(1, len(sizes) - 1) for k, s in enumerate(sizes)}
        for i, nd in enumerate(page.nodes):
            if not nd.has_geometry():
                continue
            x, y, w, h = nd.bbox
            m[i] = [1.0, x / W, y / H, w / W, h / H, rank.get(nd.font_size, 0.0)]
        return m
