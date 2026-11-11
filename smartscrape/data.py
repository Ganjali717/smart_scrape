"""
Core data model for SmartScrape v2.

A page is a list of *text nodes*: DOM elements that carry their own visible
text. Each node keeps enough structure for the graph (parent / ancestors,
document order) and optional rendered geometry (bbox from FitLayout).
Geometry is optional on purpose: SWDE ships raw HTML without CSS, while
FitLayout-rendered pages carry real boxes. Every component works with both.
"""

from __future__ import annotations

import html as _html
import json
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence


@dataclass
class Node:
    id: int                      # index within the page (0..N-1)
    text: str
    tag: str
    order: int                   # position in document order (0..N-1)
    depth: int = 0               # DOM depth of the element
    parent: int = -1             # nearest ancestor that is also a text node (-1 = none)
    ancestors: List[int] = field(default_factory=list)  # DOM-element ancestry ids (page-local element ids)
    regions: List[str] = field(default_factory=list)    # semantic regions: "footer", "nav", "header", "aside", "form"
    bbox: Optional[List[float]] = None                   # [x, y, w, h] in px, if rendered
    font_size: Optional[float] = None
    font_weight: Optional[float] = None
    xpath: str = ""

    def has_geometry(self) -> bool:
        return self.bbox is not None and len(self.bbox) >= 4


@dataclass
class Page:
    page_id: str
    site: str
    nodes: List[Node]
    # gold values per field (list of acceptable strings), SWDE style
    gold: Dict[str, List[str]] = field(default_factory=dict)
    # node-level labels derived from gold: node id -> field name
    labels: Dict[int, str] = field(default_factory=dict)
    url: str = ""
    snapshot: str = ""           # e.g. Wayback timestamp for drift benchmarks

    @property
    def n(self) -> int:
        return len(self.nodes)

    def has_geometry(self) -> bool:
        return any(nd.has_geometry() for nd in self.nodes)

    def page_height(self) -> float:
        if not self.has_geometry():
            return float(max(1, self.n))
        return max((nd.bbox[1] + nd.bbox[3]) for nd in self.nodes if nd.has_geometry()) or 1.0

    def position(self, i: int) -> float:
        """Relative vertical position in [0,1]: bbox y if rendered, else document order."""
        nd = self.nodes[i]
        if nd.has_geometry():
            return float(nd.bbox[1]) / self.page_height()
        return nd.order / max(1, self.n - 1)

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["labels"] = {str(k): v for k, v in self.labels.items()}
        return d

    @staticmethod
    def from_dict(d: dict) -> "Page":
        nodes = [Node(**nd) for nd in d["nodes"]]
        return Page(
            page_id=d["page_id"], site=d["site"], nodes=nodes,
            gold=d.get("gold", {}),
            labels={int(k): v for k, v in d.get("labels", {}).items()},
            url=d.get("url", ""), snapshot=d.get("snapshot", ""),
        )


# ---------------------------------------------------------------------------
# Text normalisation and matching (SWDE convention: exact match after
# whitespace / entity / case normalisation).
# ---------------------------------------------------------------------------
_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    s = _html.unescape(_html.unescape(str(s)))
    s = unicodedata.normalize("NFKC", s)
    s = _WS.sub(" ", s).strip().lower()
    return s


def text_matches(pred: str, gold_values: Sequence[str]) -> bool:
    p = normalize_text(pred)
    if not p:
        return False
    return any(p == normalize_text(g) for g in gold_values if normalize_text(g))


def assign_labels(page: Page) -> None:
    """Mark every node whose text equals a gold value of a field."""
    page.labels = {}
    norm_gold = {f: {normalize_text(v) for v in vals if normalize_text(v)} for f, vals in page.gold.items()}
    for nd in page.nodes:
        t = normalize_text(nd.text)
        if not t:
            continue
        for f, vals in norm_gold.items():
            if t in vals and nd.id not in page.labels:
                page.labels[nd.id] = f


# ---------------------------------------------------------------------------
# Persistence: JSON-lines cache of parsed pages
# ---------------------------------------------------------------------------
def save_pages(pages: Sequence[Page], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")


def load_pages(path: str) -> List[Page]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(Page.from_dict(json.loads(line)))
    return out
