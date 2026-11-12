"""Single-page extraction with a trained model: the production entry point."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .data import Page
from .features import FeatureEncoder
from .loaders import page_from_html, pages_from_fitlayout
from .reliability import page_stability, risk
from .schema import Schema
from .solver import ConstraintSolver, decode_argmax, violations
from .train import encode_pages, load_bundle, score_pages


class Extractor:
    def __init__(self, model, bundle: dict, schema: Schema, top_k: int = 20, risk_threshold: float = 0.5):
        if list(bundle["fields"]) != schema.field_names:
            raise ValueError(f"model fields {bundle['fields']} != schema fields {schema.field_names}")
        self.model, self.schema = model, schema
        self.encoder = FeatureEncoder(sentence_model=bundle.get("sentence_model"), exclude=bundle.get("exclude_features", []))
        self.solver = ConstraintSolver(schema, top_k=top_k)
        self.risk_threshold = risk_threshold

    @staticmethod
    def load(model_path: str, schema_path: str, **kw) -> "Extractor":
        model, bundle = load_bundle(model_path)
        return Extractor(model, bundle, Schema.load(schema_path), **kw)

    def extract_page(self, page: Page) -> dict:
        enc = encode_pages([page], self.schema, self.encoder)
        scores = score_pages(self.model, enc)[0]
        rec = self.solver.solve(page, scores)
        base = decode_argmax(page, scores, self.schema)
        r = risk(rec, base)
        return {
            "page": page.page_id, "url": page.url,
            "record": rec.values(),
            "reliability": {"risk": round(r, 4), "stability": round(page_stability(rec), 4),
                            "needs_review": r >= self.risk_threshold},
            "violations": violations(page, rec, self.schema),
            "provenance": rec.provenance(),
        }

    def extract_html_file(self, path: str) -> dict:
        with open(path, "rb") as f:
            html = f.read().decode("utf-8", errors="replace")
        return self.extract_page(page_from_html(html, os.path.basename(path), site="local"))

    def extract_url(self, url: str) -> dict:
        # FitLayout rendering (legacy client + parser from src/)
        from src.integration.fitlayout import FitLayoutClient
        from src.learning.graph_builder import FitLayoutParser

        raw = FitLayoutClient().get_page_content(url)
        _, nodes = FitLayoutParser().parse(raw)
        return self.extract_page(pages_from_fitlayout(nodes, page_id=url, site="live", url=url))
