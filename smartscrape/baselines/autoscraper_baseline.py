"""
AutoScraper baseline (github.com/alirezamika/autoscraper): learns extraction
rules from example values. One scraper per site, trained with
`build(html=..., wanted_dict={field: [gold value]})` on the training pages
and applied with `get_result_exact(..., group_by_alias=True)`.
"""

from __future__ import annotations

import html as _html
from collections import defaultdict
from typing import Dict, Sequence

from ..data import Page
from .base import Baseline, BaselineUnavailable, read_html, register


@register("autoscraper")
class AutoScraperBaseline(Baseline):
    needs_site_training = True

    def _fit(self, pages: Sequence[Page]) -> None:
        try:
            from autoscraper import AutoScraper
        except ImportError as e:
            raise BaselineUnavailable("pip install autoscraper") from e
        by_site = defaultdict(list)
        for p in pages:
            by_site[p.site].append(p)
        self.scrapers = {}
        max_pages = int(self.options.get("max_train_pages_per_site", 5))
        for site, ps in by_site.items():
            sc = AutoScraper()
            for p in ps[:max_pages]:
                # SWDE ground truth keeps HTML entities (e.g. &#34;); AutoScraper matches visible text
                wanted = {f: [_html.unescape(_html.unescape(vals[0])).strip()]
                          for f, vals in p.gold.items() if vals and f in self.schema.field_names}
                if wanted:
                    sc.build(html=read_html(p), wanted_dict=wanted, update=True)
            self.scrapers[site] = sc

    def _extract(self, page: Page) -> Dict[str, str]:
        sc = getattr(self, "scrapers", {}).get(page.site)
        if sc is None:
            return {}
        res = sc.get_result_exact(html=read_html(page), group_by_alias=True)
        out = {}
        for f, vals in (res or {}).items():
            # guard against rules that matched a whole container: keep short values only
            vals = [v.strip() for v in vals if v and len(v.strip()) <= 300]
            out[f] = vals[0] if vals else ""
        return out
