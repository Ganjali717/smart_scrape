"""
Feature encoding for SmartScrape nodes.

Uses a lightweight hash-based text encoder instead of BERT to avoid
heavy dependencies (sentencepiece, transformers download).
When the GNN is trained, this encoder produces consistent fixed-size
vectors from text, geometry, and HTML tag features.
"""

import re
import math
import torch
from typing import Optional


class FeatureEncoder:
    """
    Lightweight feature encoder — no transformers/sentencepiece required.

    Text is encoded via character n-gram hashing into a fixed-size vector.
    This is deterministic, fast, and produces the same dimensionality as
    bert-tiny (128) so the GNN architecture is unchanged.
    """

    TEXT_DIM = 128   # matches bert-tiny hidden size
    VISUAL_DIM = 4   # x, y, w, h (normalized)

    TAG_MAP = {
        "div": 0, "span": 1, "a": 2, "img": 3, "p": 4,
        "h1": 5, "h2": 6, "li": 7, "ul": 8, "table": 9,
        "form": 10, "input": 11, "button": 12,
    }

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        # model_name kept for API compatibility but ignored
        self.text_dim = self.TEXT_DIM
        self.visual_dim = self.VISUAL_DIM
        self.tag_dim = len(self.TAG_MAP) + 1  # +1 for 'other'
        print("[FeatureEncoder] Using lightweight hash encoder (no BERT needed).")

    # ------------------------------------------------------------------

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text into a 128-dim vector using character n-gram hashing.
        Deterministic, no external models needed.
        """
        vec = [0.0] * self.TEXT_DIM
        text = str(text).lower().strip()

        if not text:
            return torch.zeros(self.TEXT_DIM, dtype=torch.float32)

        # Character 2-grams and 3-grams → hash into vector buckets
        for n in (2, 3):
            for i in range(len(text) - n + 1):
                gram = text[i: i + n]
                # Simple polynomial hash
                h = 0
                for ch in gram:
                    h = (h * 31 + ord(ch)) & 0xFFFFFF
                idx = h % self.TEXT_DIM
                vec[idx] += 1.0

        # L2-normalize
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vec = [v / norm for v in vec]

        return torch.tensor(vec, dtype=torch.float32)

    def encode_visual(self, x, y, w, h, page_w, page_h) -> torch.Tensor:
        """Normalize bounding box coordinates to [0, 1]."""
        page_w = float(page_w) if page_w and page_w > 0 else 1920.0
        page_h = float(page_h) if page_h and page_h > 0 else 1080.0
        return torch.tensor(
            [x / page_w, y / page_h, w / page_w, h / page_h],
            dtype=torch.float32,
        )

    def encode_tag(self, tag_name: str) -> torch.Tensor:
        """One-hot encode HTML tag."""
        idx = self.TAG_MAP.get(str(tag_name).lower(), len(self.TAG_MAP))
        vec = torch.zeros(self.tag_dim, dtype=torch.float32)
        vec[idx] = 1.0
        return vec

    def get_output_dim(self) -> int:
        return self.text_dim + self.visual_dim + self.tag_dim