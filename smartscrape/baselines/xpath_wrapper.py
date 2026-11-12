"""
Induced XPath wrapper: the classic per-site selector, built automatically so
that no hand-tuning skill enters the comparison.

fit():   for every training page of a site and every field, take the elements
         whose own text equals a gold value and record their absolute XPath;
         keep the most frequent path per (site, field).
extract(): evaluate that path on the new page. With `generalize=True` (default)
         a second path without positional indices is tried when the exact one
         returns nothing, the usual first repair a scraper maintainer makes.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, Sequence

from ..data import Page, normalize_text
from .base import Baseline, read_html, register

_IDX = re.compile(r"\[\d+\]")


def _own_text(el) -> str:
    return re.sub(r"\s+", " ", el.text or "").strip()


@register("xpath")
class XPathWrapper(Baseline):
    needs_site_training = True

    def _fit(self, pages: Sequence[Page]) -> None:
        from lxml import html as lh

        counts: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
        for p in pages:
            root = lh.fromstring(read_html(p))
            tree = root.getroottree()
            gold = {f: {normalize_text(v) for v in vals} for f, vals in p.gold.items()}
            for el in root.iter():
                if not isinstance(el.tag, str):
                    continue
                t = normalize_text(_own_text(el))
                if not t:
                    continue
                for f, vals in gold.items():
                    if t in vals:
                        counts[p.site][f][tree.getpath(el)] += 1
        self.paths = {site: {f: c.most_common(1)[0][0] for f, c in fields.items()}
                      for site, fields in counts.items()}

    def _extract(self, page: Page) -> Dict[str, str]:
        from lxml import html as lh

        paths = getattr(self, "paths", {}).get(page.site, {})
        if not paths:
            return {}
        root = lh.fromstring(read_html(page))
        out = {}
        for f, path in paths.items():
            hits = root.getroottree().xpath(path)
            if not hits and self.options.get("generalize", True):
                hits = root.getroottree().xpath(_IDX.sub("", path))
            for el in hits:
                t = _own_text(el) or re.sub(r"\s+", " ", el.text_content()).strip()
                if t:
                    out[f] = t
                    break
        return out
