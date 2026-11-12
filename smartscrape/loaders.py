"""
Dataset loaders. Each returns a list of `Page` objects with gold values and
node labels assigned.

  * load_swde_vertical  - SWDE raw HTML + ground truth parquet (abdo-Mansour/SWDE)
  * load_legacy_json    - the original SmartScrape annotation format
                          (data/labeled*.json: nodes with text/tag/bbox/label)
  * pages_from_fitlayout - FitLayout-rendered pages (nodes with real boxes)
"""

from __future__ import annotations

import ast
import glob
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence

from .data import Node, Page, assign_labels

# Tags whose text is never visible content
_SKIP_TAGS = {"script", "style", "noscript", "head", "title", "meta", "link", "iframe", "svg", "template"}
_REGION_TAGS = {"footer": "footer", "nav": "nav", "header": "header", "aside": "aside", "form": "form"}
_REGION_HINTS = {
    "footer": re.compile(r"foot", re.I),
    "nav": re.compile(r"\b(nav|menu|breadcrumb|topbar)", re.I),
    "header": re.compile(r"\b(header|masthead)", re.I),
    "aside": re.compile(r"(sidebar|related|recommend|also[-_ ]?(bought|viewed)|similar)", re.I),
}
_WS = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Generic HTML -> Page
# ---------------------------------------------------------------------------
def page_from_html(html_text: str, page_id: str, site: str, gold: Optional[Dict[str, List[str]]] = None,
                   url: str = "") -> Page:
    """Parse raw HTML into text nodes. No rendering: geometry stays None."""
    from lxml import html as lxml_html

    try:
        root = lxml_html.fromstring(html_text)
    except Exception:
        root = lxml_html.fromstring("<html><body></body></html>")

    nodes: List[Node] = []
    elem_ids: Dict[int, int] = {}   # id(element) -> element index (for ancestry)
    elem_counter = [0]

    def elem_index(e) -> int:
        k = id(e)
        if k not in elem_ids:
            elem_ids[k] = elem_counter[0]
            elem_counter[0] += 1
        return elem_ids[k]

    # stack of (element, nearest text-node ancestor id, regions)
    def walk(e, depth: int, text_parent: int, regions: tuple, ancestry: tuple):
        tag = e.tag if isinstance(e.tag, str) else ""
        tag_l = tag.lower()
        if not tag or tag_l in _SKIP_TAGS:
            return
        attrs = " ".join([e.get("id", "") or "", e.get("class", "") or ""])
        new_regions = set(regions)
        if tag_l in _REGION_TAGS:
            new_regions.add(_REGION_TAGS[tag_l])
        for name, rx in _REGION_HINTS.items():
            if attrs and rx.search(attrs):
                new_regions.add(name)
        regions_t = tuple(sorted(new_regions))
        anc = ancestry + (elem_index(e),)

        own = _WS.sub(" ", e.text or "").strip()
        my_id = text_parent
        if own:
            nd = Node(id=len(nodes), text=own[:500], tag=tag_l, order=len(nodes), depth=depth,
                      parent=text_parent, ancestors=list(anc), regions=list(regions_t),
                      xpath=_short_xpath(e))
            nodes.append(nd)
            my_id = nd.id
        for child in e:
            walk(child, depth + 1, my_id, regions_t, anc)
            # tail text belongs to the parent element visually
            tail = _WS.sub(" ", child.tail or "").strip() if isinstance(child.tag, str) else ""
            if tail:
                nd = Node(id=len(nodes), text=tail[:500], tag=tag_l, order=len(nodes), depth=depth,
                          parent=my_id if my_id >= 0 else text_parent, ancestors=list(anc),
                          regions=list(regions_t), xpath=_short_xpath(e) + "/text()")
                nodes.append(nd)

    body = root.find(".//body")
    walk(body if body is not None else root, 0, -1, (), ())
    page = Page(page_id=page_id, site=site, nodes=nodes, gold=gold or {}, url=url)
    assign_labels(page)
    return page


def _short_xpath(e) -> str:
    parts = []
    cur = e
    for _ in range(6):
        if cur is None or not isinstance(cur.tag, str):
            break
        parts.append(cur.tag.lower())
        cur = cur.getparent()
    return "/".join(reversed(parts))


# ---------------------------------------------------------------------------
# SWDE
# ---------------------------------------------------------------------------
def _parse_gt(gt: str) -> Dict[str, List[str]]:
    try:
        d = ast.literal_eval(gt)
    except Exception:
        d = json.loads(gt)
    return {k: [str(x) for x in (v if isinstance(v, list) else [v])] for k, v in d.items()}


def load_swde_vertical(html_root: str, gt_parquet: str, vertical: str,
                       sites: Optional[Sequence[str]] = None,
                       max_pages_per_site: Optional[int] = None) -> List[Page]:
    """
    html_root:  directory that contains <vertical>/<vertical>-<site>(<n>)/<0000>.htm
    gt_parquet: data/<vertical>-00000-of-00001.parquet from abdo-Mansour/SWDE
    """
    import pyarrow.parquet as pq

    df = pq.read_table(gt_parquet).to_pandas()
    gt_by_id = {row.website_id: _parse_gt(row.gt) for row in df.itertuples()}

    pages: List[Page] = []
    site_dirs = sorted(glob.glob(os.path.join(html_root, vertical, f"{vertical}-*")))
    for sd in site_dirs:
        site = re.match(rf"{vertical}-(.+)\(\d+\)", os.path.basename(sd)).group(1)
        if sites and site not in sites:
            continue
        files = sorted(glob.glob(os.path.join(sd, "*.htm")))
        if max_pages_per_site:
            files = files[:max_pages_per_site]
        for fp in files:
            num = os.path.splitext(os.path.basename(fp))[0]
            pid = f"{site}_{num}"
            gold = gt_by_id.get(pid)
            if gold is None:
                continue
            with open(fp, "rb") as fh:
                raw = fh.read()
            text = raw.decode("utf-8", errors="replace")
            pg = page_from_html(text, pid, site, gold)
            pg.source = os.path.abspath(fp)
            pages.append(pg)
    return pages


# ---------------------------------------------------------------------------
# Flat node lists (legacy annotations and FitLayout output)
# ---------------------------------------------------------------------------
def page_from_flat_nodes(raw_nodes: Sequence[dict], page_id: str, site: str,
                         gold: Optional[Dict[str, List[str]]] = None,
                         use_node_labels: bool = False, url: str = "") -> Page:
    """
    Build a Page from nodes that carry {text, tag, bbox} but no DOM tree
    (FitLayout output, legacy JSON). Structure is recovered from geometry:
    parent = smallest box that contains the node.
    """
    kept = []
    for rn in raw_nodes:
        text = _WS.sub(" ", str(rn.get("text", ""))).strip()
        bbox = rn.get("bbox")
        try:
            bbox = [float(b) for b in bbox[:4]] if bbox else None
        except (TypeError, ValueError):
            bbox = None
        kept.append((rn, text, bbox))

    # order by reading position (y, then x) when boxes exist
    if any(b is not None for _, _, b in kept):
        kept.sort(key=lambda t: ((t[2] or [0, 0])[1], (t[2] or [0, 0])[0]))

    nodes: List[Node] = []
    orig_index = []
    for rn, text, bbox in kept:
        if not text:
            continue
        nodes.append(Node(id=len(nodes), text=text[:500], tag=str(rn.get("tag", "div")).lower(),
                          order=len(nodes), bbox=bbox,
                          font_size=_num(rn.get("font_size")), font_weight=_num(rn.get("font_weight")),
                          regions=list(rn.get("regions", []))))
        orig_index.append(rn)

    # containment-based parent
    for nd in nodes:
        if not nd.has_geometry():
            continue
        x, y, w, h = nd.bbox
        best, best_area = -1, float("inf")
        for other in nodes:
            if other.id == nd.id or not other.has_geometry():
                continue
            ox, oy, ow, oh = other.bbox
            if ox <= x and oy <= y and ox + ow >= x + w and oy + oh >= y + h and ow * oh < best_area \
                    and (ow * oh) > (w * h):
                best, best_area = other.id, ow * oh
        nd.parent = best

    page = Page(page_id=page_id, site=site, nodes=nodes, gold=gold or {}, url=url)
    if use_node_labels:
        page.labels = {}
        for nd, rn in zip(nodes, orig_index):
            lab = rn.get("label")
            if lab and lab not in ("none", "other"):
                page.labels[nd.id] = lab
    else:
        assign_labels(page)
    return page


def _num(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def load_legacy_json(path: str, site: Optional[str] = None) -> List[Page]:
    """Original SmartScrape format: [{url, nodes:[{id,tag,bbox,text,label}], title?, price?}]"""
    from urllib.parse import urlparse

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    pages = []
    for i, p in enumerate(raw):
        gold: Dict[str, List[str]] = {}
        for nd in p["nodes"]:
            lab = nd.get("label")
            if lab and lab not in ("none", "other"):
                gold.setdefault(lab, []).append(str(nd.get("text", "")))
        s = site or urlparse(p.get("url", "")).netloc or "unknown"
        pg = page_from_flat_nodes(p["nodes"], page_id=f"{s}_{i:04d}", site=s, gold=gold,
                                  use_node_labels=True, url=p.get("url", ""))
        pages.append(pg)
    return pages


def pages_from_fitlayout(raw_nodes: Sequence[dict], page_id: str, site: str,
                         gold: Optional[Dict[str, List[str]]] = None, url: str = "") -> Page:
    """Convert FitLayoutParser.parse() output (list of {id,text,tag,bbox}) into a Page."""
    return page_from_flat_nodes(raw_nodes, page_id, site, gold=gold, url=url)
