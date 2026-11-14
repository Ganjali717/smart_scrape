"""
Input-level synthetic template drift.

Every perturbation changes the *page* (DOM structure, tags, order, content
around the record) and keeps the record itself unchanged. Nothing touches
model outputs, unlike the ICWE prototype that blended probabilities.

  wrapper_insertion   extra wrapper elements around a fraction of nodes
  tag_rename          tags swapped for visually equivalent ones (span<->div, b->strong ...)
  sibling_reorder     reorder nodes that share a parent
  distractor_block    inject a "related products" block with look-alike field values

Two levels:
  level="page"  every decision is drawn independently per page (noise-like drift);
  level="site"  every decision is a deterministic function of (site, seed, structural
                slot), so the same edit hits the same template slot on every page of the
                site, as a real redesign does (e.g. "wrap every div/span in two divs").
"""

from __future__ import annotations

import copy
import random
from typing import Callable, Dict, List, Optional, Sequence

from .data import Node, Page, normalize_text
from .formats import is_price

_TAG_SWAP = {"span": "div", "div": "span", "b": "strong", "strong": "b", "td": "div", "h1": "div",
             "h2": "div", "p": "div", "font": "span", "li": "div", "a": "span"}


def _rebuild(page: Page, nodes: List[Node], id_map: Dict[int, int]) -> Page:
    """Re-index nodes 0..N-1 in list order, remap parents and labels."""
    for new_id, nd in enumerate(nodes):
        nd.id = new_id
        nd.order = new_id
    for nd in nodes:
        nd.parent = id_map.get(nd.parent, -1) if nd.parent >= 0 else -1
    labels = {id_map[i]: f for i, f in page.labels.items() if i in id_map}
    return Page(page.page_id, page.site, nodes, dict(page.gold), labels, page.url, page.snapshot, page.source)


def _slot_rng(rng, keyed, kind: str, key: str) -> random.Random:
    return keyed(kind, key) if keyed is not None else rng


def wrapper_insertion(page: Page, severity: float, rng: random.Random, keyed=None) -> Page:
    p = copy.deepcopy(page)
    next_elem = 10_000_000 + rng.randrange(1_000_000)
    for nd in p.nodes:
        rng = _slot_rng(rng, keyed, "wrap", f"{nd.xpath}|{nd.depth}") if keyed else rng
        if rng.random() < severity and nd.ancestors:
            extra = rng.randint(1, 3)
            pos = rng.randrange(len(nd.ancestors) + 1)
            new = list(range(next_elem, next_elem + extra))
            next_elem += extra
            nd.ancestors = nd.ancestors[:pos] + new + nd.ancestors[pos:]
            nd.depth += extra
            nd.xpath = "div/" * extra + nd.xpath
    return p


def tag_rename(page: Page, severity: float, rng: random.Random, keyed=None) -> Page:
    p = copy.deepcopy(page)
    for nd in p.nodes:
        rng = _slot_rng(rng, keyed, "tag", f"{nd.xpath}|{nd.depth}") if keyed else rng
        if rng.random() < severity and nd.tag in _TAG_SWAP:
            new = _TAG_SWAP[nd.tag]
            nd.xpath = nd.xpath.rsplit("/", 1)[0] + "/" + new if "/" in nd.xpath else new
            nd.tag = new
    return p


def sibling_reorder(page: Page, severity: float, rng: random.Random, keyed=None) -> Page:
    p = copy.deepcopy(page)
    groups: Dict[int, List[int]] = {}
    for nd in p.nodes:
        groups.setdefault(nd.parent, []).append(nd.id)
    new_order = list(range(p.n))
    for parent, ids in groups.items():
        if keyed is not None:
            first = p.nodes[ids[0]]
            rng = keyed("reorder", f"{first.xpath}|{first.depth}|{len(ids)}")
        if len(ids) > 1 and rng.random() < severity:
            shuffled = ids[:]
            rng.shuffle(shuffled)
            for slot, nid in zip(sorted(ids), shuffled):
                new_order[slot] = nid
    nodes = [p.nodes[i] for i in new_order]
    id_map = {old: new for new, old in enumerate(new_order)}
    return _rebuild(p, nodes, id_map)


def distractor_block(page: Page, severity: float, rng: random.Random,
                     pool_titles: Optional[Sequence[str]] = None, keyed=None) -> Page:
    """
    Insert a block of look-alike items (title + price pairs). With probability 0.5 the
    block carries no region hint (class names changed in the redesign), so region
    constraints cannot simply filter it out.
    """
    p = copy.deepcopy(page)
    if p.n == 0:
        return p
    n_items = max(1, int(round(1 + severity * 5)))
    gold_vals = {normalize_text(v) for vals in p.gold.values() for v in vals}
    labeled = sorted(p.labels)
    anchor = labeled[-1] if labeled else p.n // 2
    # place the block either just after the record or early on the page
    if keyed is None:   # page level: unchanged from the original protocol (reproducibility)
        insert_at = (min(p.n, anchor + 1 + rng.randint(0, 10)) if rng.random() < 0.7
                     else rng.randint(0, max(0, anchor)))
        flagged = rng.random() < 0.5
    else:               # site level: one layout decision for the whole site
        lay = keyed("distractor", "layout")
        after, offset, flagged = lay.random() < 0.7, lay.randint(0, 10), lay.random() < 0.5
        insert_at = min(p.n, anchor + 1 + offset) if after else min(anchor, offset)
    base_anc = list(p.nodes[min(insert_at, p.n - 1)].ancestors[:-1]) if p.nodes[min(insert_at, p.n - 1)].ancestors else []
    block_elem = 20_000_000 + rng.randrange(1_000_000)
    new_nodes: List[Node] = []
    for k in range(n_items):
        title = None
        if pool_titles:
            for _ in range(10):
                cand = rng.choice(pool_titles)
                if normalize_text(cand) not in gold_vals:
                    title = cand
                    break
        title = title or f"Similar item {rng.randint(100, 999)}"
        price = f"${rng.randint(5, 900)}.{rng.randint(0, 99):02d}"
        for text, tag in ((title, "a"), (price, "span")):
            new_nodes.append(Node(id=-1, text=text, tag=tag, order=0,
                                  depth=len(base_anc) + 2, parent=-1,
                                  ancestors=base_anc + [block_elem, block_elem + 1 + k],
                                  regions=["aside"] if flagged else [], xpath=f"div/div/{tag}"))
    old_ids = list(range(p.n))
    nodes = p.nodes[:insert_at] + new_nodes + p.nodes[insert_at:]
    id_map = {old: (old if old < insert_at else old + len(new_nodes)) for old in old_ids}
    return _rebuild(p, nodes, id_map)


PERTURBATIONS: Dict[str, Callable] = {
    "wrapper_insertion": wrapper_insertion,
    "tag_rename": tag_rename,
    "sibling_reorder": sibling_reorder,
    "distractor_block": distractor_block,
}


def apply_drift(page: Page, kinds: Sequence[str], severity: float, seed: int,
                pool_titles: Optional[Sequence[str]] = None, level: str = "page") -> Page:
    rng = random.Random(f"{page.page_id}:{seed}")
    keyed = None
    if level == "site":
        def keyed(kind: str, key: str, _site=page.site, _seed=seed) -> random.Random:
            return random.Random(f"{_site}:{_seed}:{kind}:{key}")
    elif level != "page":
        raise ValueError(f"unknown drift level {level}")
    out = page
    for k in kinds:
        fn = PERTURBATIONS[k]
        out = (fn(out, severity, rng, pool_titles, keyed=keyed) if k == "distractor_block"
               else fn(out, severity, rng, keyed=keyed))
    if out is page:
        out = copy.copy(page)
    # the raw HTML no longer matches the perturbed page: baselines that read HTML must not see it
    out.source = ""
    return out
