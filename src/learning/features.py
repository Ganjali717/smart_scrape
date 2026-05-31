"""
Feature encoding for SmartScrape nodes.

Thin wrapper around `encoding.py` (the single source of truth shared by the
trainer and the runtime). Kept as a class for API compatibility with the
existing pipeline/graph-builder call sites.
"""

from typing import Optional
import torch

from . import encoding


class FeatureEncoder:
    """Lightweight, deterministic encoder — no transformers/sentencepiece."""

    TEXT_DIM = encoding.TEXT_DIM
    VISUAL_DIM = encoding.VISUAL_DIM
    TAG_MAP = encoding.TAG_MAP

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.text_dim = encoding.TEXT_DIM
        self.visual_dim = encoding.VISUAL_DIM
        self.tag_dim = encoding.TAG_DIM
        print("[FeatureEncoder] Using shared hash encoder (encoding.py, dim=%d)."
              % encoding.INPUT_DIM)

    def encode_text(self, text: str) -> torch.Tensor:
        return encoding.encode_text(text)

    def encode_visual(self, x, y, w, h, page_w=None, page_h=None) -> torch.Tensor:
        # page_w/page_h accepted for backward compat but IGNORED:
        # normalization uses fixed constants so train == inference.
        return encoding.encode_visual([x, y, w, h])

    def encode_tag(self, tag_name: str) -> torch.Tensor:
        return encoding.encode_tag(tag_name)

    def get_output_dim(self) -> int:
        return encoding.INPUT_DIM
