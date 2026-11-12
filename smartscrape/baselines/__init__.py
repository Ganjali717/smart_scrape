"""
External baselines behind one interface:

    b = make_baseline("xpath", schema, **options)
    b.fit(train_pages)                  # pages of the same site(s) with gold values
    values = b.extract(page)            # {field: str}; "" = not found
    b.stats                             # timing / token counters for the cost table

Every baseline reads the page's raw HTML (`Page.source`), never our parsed
nodes, so no baseline profits from SmartScrape's own preprocessing.
"""

from .base import Baseline, BaselineUnavailable, make_baseline, BASELINES  # noqa: F401
