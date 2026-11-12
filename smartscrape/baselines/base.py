from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

from ..data import Page
from ..schema import Schema


class BaselineUnavailable(RuntimeError):
    """Raised when an optional dependency or service is missing."""


@dataclass
class BaselineStats:
    pages: int = 0
    seconds: float = 0.0
    fit_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    failures: int = 0
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["ms_per_page"] = 1000 * self.seconds / max(1, self.pages)
        d.pop("notes")
        return d


class Baseline:
    name = "baseline"
    needs_site_training = False   # True: must see pages of the test site (wrappers)

    def __init__(self, schema: Schema, **options):
        self.schema = schema
        self.options = options
        self.stats = BaselineStats()

    # -- public API ---------------------------------------------------------
    def fit(self, pages: Sequence[Page]) -> "Baseline":
        t = time.perf_counter()
        self._fit([p for p in pages if p.source])
        self.stats.fit_seconds += time.perf_counter() - t
        return self

    def extract(self, page: Page) -> Dict[str, str]:
        if not page.source:
            raise ValueError(f"{self.name}: page {page.page_id} has no raw HTML (drifted copy?)")
        t = time.perf_counter()
        try:
            out = self._extract(page)
        except BaselineUnavailable:
            raise
        except Exception as e:  # a baseline failing on one page counts as empty output
            self.stats.failures += 1
            self.stats.notes.append(f"{page.page_id}: {type(e).__name__}: {e}")
            out = {}
        self.stats.seconds += time.perf_counter() - t
        self.stats.pages += 1
        return {f: str(out.get(f, "") or "").strip() for f in self.schema.field_names}

    # -- to implement -------------------------------------------------------
    def _fit(self, pages: Sequence[Page]) -> None:
        pass

    def _extract(self, page: Page) -> Dict[str, str]:
        raise NotImplementedError


def read_html(page: Page) -> str:
    with open(page.source, "rb") as f:
        return f.read().decode("utf-8", errors="replace")


BASELINES: Dict[str, Callable[..., Baseline]] = {}


def register(name: str):
    def deco(cls):
        BASELINES[name] = cls
        cls.name = name
        return cls
    return deco


def make_baseline(name: str, schema: Schema, **options) -> Baseline:
    from . import xpath_wrapper, autoscraper_baseline, llm  # noqa: F401  (registration)
    if name not in BASELINES:
        raise KeyError(f"unknown baseline {name!r}; available: {sorted(BASELINES)}")
    return BASELINES[name](schema, **options)
