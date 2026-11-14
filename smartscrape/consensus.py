"""
Site-level template consensus (transductive, label-free).

Pages of one site are generated from one template, so the correct node for a
field sits in the same structural slot on (almost) every page. A page-level
model ignores this; a classic wrapper relies on it entirely. Consensus decoding
combines the two without any labels from the target site:

    1. decode every page of the site with the base decoder (filtered / ILP);
    2. vote, per field, over the structural keys of the chosen nodes;
    3. add  beta * support(key)  to each node's field score and decode again;
    4. repeat for a fixed number of rounds.

A node's key is its tag path plus its position relative to the nearest
preceding piece of template text (a string that recurs on most pages of the
site, e.g. the label "Price:"), measured in text nodes. A key whose
votes are dominated by one repeated text is ignored: a slot that always holds
the same string is boilerplate, not a field value.

The same statistics give a page-level reliability signal (how many fields left
the site's consensus slot) and, once the votes are stable, they *are* a wrapper:
the winning key per field can be exported as a selector for cheap re-extraction.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data import Page, normalize_text
from .schema import Schema
from .solver import Record

_DIGITS = re.compile(r"\d+")

Key = Tuple[str, str]


def site_anchors(pages: Sequence[Page], min_share: float = 0.5, min_pages: int = 3) -> set:
    """Template text: normalised strings that occur on at least `min_share` of a site's pages."""
    df = Counter()
    for p in pages:
        df.update({normalize_text(nd.text or "") for nd in p.nodes} - {""})
    need = max(min_pages, min_share * len(pages))
    return {t for t, c in df.items() if c >= need and len(t) <= 60}


def node_keys(page: Page, anchors: Optional[set] = None) -> List[Key]:
    """Key = (tag path, nearest preceding template text, distance to it in text nodes, capped).
    Without anchors the preceding node's digit-normalised text is used instead."""
    keys: List[Key] = []
    last, last_i, prev = "", -1, ""
    for i, nd in enumerate(page.nodes):
        if anchors is None:
            keys.append((nd.xpath or nd.tag or "", prev))
        else:
            keys.append((nd.xpath or nd.tag or "", f"{last}|{min(i - last_i, 12)}"))
        t = normalize_text(nd.text or "")
        prev = _DIGITS.sub("#", t)[:40]
        if anchors is not None and t in anchors:
            last, last_i = t[:60], i
    return keys


class SiteConsensus:
    """Vote statistics of one site, built from decoded records of its pages."""

    def __init__(self, schema: Schema, max_text_share: float = 0.8, min_votes: int = 5):
        self.schema = schema
        self.max_text_share = max_text_share
        self.min_votes = min_votes
        self.full: Dict[str, Dict[Key, float]] = {}
        self.path: Dict[str, Dict[str, float]] = {}
        self.top: Dict[str, Optional[Key]] = {}
        self.n_pages = 0

    def fit(self, pages: Sequence[Page], records: Sequence[Record], keys: Sequence[List[Key]]):
        n = max(1, len(pages))
        self.n_pages = len(pages)
        for f in self.schema.field_names:
            votes_full, votes_path = Counter(), Counter()
            texts_full, texts_path = defaultdict(Counter), defaultdict(Counter)
            for p, r, ks in zip(pages, records, keys):
                c = r.fields[f]
                if c.node is None or not c.text:
                    continue
                k = ks[c.node]
                t = normalize_text(c.text)
                votes_full[k] += 1
                votes_path[k[0]] += 1
                texts_full[k][t] += 1
                texts_path[k[0]][t] += 1
            self.full[f] = {k: v / n for k, v in votes_full.items()
                            if not self._boilerplate(v, texts_full[k])}
            self.path[f] = {k: v / n for k, v in votes_path.items()
                            if not self._boilerplate(v, texts_path[k])}
            self.top[f] = max(self.full[f], key=self.full[f].get) if self.full[f] else None
        return self

    def _boilerplate(self, votes: int, texts: Counter) -> bool:
        return votes >= self.min_votes and texts.most_common(1)[0][1] / votes > self.max_text_share

    def bonus(self, page: Page, keys: List[Key]) -> np.ndarray:
        out = np.zeros((page.n, len(self.schema.field_names)), dtype=np.float32)
        for j, f in enumerate(self.schema.field_names):
            full, path = self.full.get(f, {}), self.path.get(f, {})
            if not full and not path:
                continue
            for i, k in enumerate(keys):
                out[i, j] = 0.5 * (full.get(k, 0.0) + path.get(k[0], 0.0))
        return out

    def agreement(self, rec: Record, keys: List[Key]) -> float:
        """Fraction of filled fields whose node sits in the site's winning slot (1.0 if none filled)."""
        hits = total = 0
        for f, c in rec.fields.items():
            if c.node is None or not c.text or self.top.get(f) is None:
                continue
            total += 1
            hits += int(keys[c.node] == self.top[f])
        return hits / total if total else 1.0

    def wrapper(self) -> Dict[str, Optional[Key]]:
        return dict(self.top)


def consensus_decode(pages: Sequence[Page], scores: Sequence[np.ndarray],
                     decode: Callable[[Page, np.ndarray], Record], schema: Schema,
                     beta: float, rounds: int = 2, **kw):
    """Decode all pages with site consensus. Returns (records, {site: SiteConsensus}, keys)."""
    by_site: Dict[str, List[int]] = defaultdict(list)
    for i, p in enumerate(pages):
        by_site[p.site].append(i)
    anchors = {site: site_anchors([pages[i] for i in idx]) for site, idx in by_site.items()}
    keys = [node_keys(p, anchors[p.site]) for p in pages]
    recs = [decode(p, s) for p, s in zip(pages, scores)]
    cons: Dict[str, SiteConsensus] = {}

    def refit():
        for site, idx in by_site.items():
            cons[site] = SiteConsensus(schema, **kw).fit([pages[i] for i in idx], [recs[i] for i in idx],
                                                          [keys[i] for i in idx])

    refit()
    if beta <= 0:
        return recs, cons, keys
    for _ in range(rounds):
        recs = [decode(p, s + beta * cons[p.site].bonus(p, k)) if p.n else decode(p, s)
                for p, s, k in zip(pages, scores, keys)]
        refit()
    for r in recs:
        r.method = f"{r.method}+cons"
    return recs, cons, keys
