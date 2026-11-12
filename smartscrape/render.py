"""
Batch rendering with FitLayout and conversion of its output into `Page`s.

FitLayout's CLI (docker image `fitlayout/fitlayout-cli`) renders pages in
headless Chromium (puppeteer backend) and exports a box tree as XML:

    <page>
      <box tagName="body" positionX=".." positionY=".." width=".." height=".."
           visualX=".." ... fontsize=".." fontweight=".." color=".." bgcolor=".." ...>
        <box tagName="#text" ... >Canon PowerShot S90</box>
        ...

`parse_fitlayout_xml` turns that tree into a Page that has BOTH structure
(element ancestry of the box tree) and geometry (boxes, font size/weight),
so every graph relation and geometric constraint becomes available.

`render_batch` drives the docker CLI with its BATCH command over a list of
(page_id, url) pairs. Local HTML files (SWDE) are served by a temporary
HTTP server that the container reaches through host.docker.internal.
Public URLs (e.g. Wayback Machine snapshots for drift experiments) are
rendered directly.
"""

from __future__ import annotations

import http.server
import os
import re
import shutil
import socketserver
import subprocess
import threading
import xml.etree.ElementTree as ET
from functools import partial
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .data import Node, Page, assign_labels

_WS = re.compile(r"\s+")
_REGION_TAGS = {"footer": "footer", "nav": "nav", "header": "header", "aside": "aside", "form": "form"}
_SKIP_TAGS = {"script", "style", "noscript", "head", "title", "meta", "link", "template"}


# ---------------------------------------------------------------------------
# XML -> Page
# ---------------------------------------------------------------------------
def _f(attrs: dict, key: str) -> Optional[float]:
    v = attrs.get(key)
    try:
        return float(v) if v not in (None, "", "null") else None
    except ValueError:
        return None


def _weight(v: Optional[str]) -> Optional[float]:
    if v is None:
        return None
    try:
        w = float(v)
        return w / 100.0 if w > 10 else w   # FitLayout reports 0..1 or CSS 100..900 depending on version
    except ValueError:
        return {"bold": 0.7, "normal": 0.4}.get(str(v).lower())


def parse_fitlayout_xml(xml_text: str, page_id: str, site: str,
                        gold: Optional[Dict[str, List[str]]] = None, url: str = "",
                        use_visual_bounds: bool = True) -> Page:
    # FitLayout escapes '>' before '&', which yields '&amp;gt;' sequences; they decode to '&gt;'
    # and are undone by normalize_text() (double unescape) during matching.
    root = ET.fromstring(xml_text)
    nodes: List[Node] = []
    elem_counter = [0]

    def walk(box, ancestry: tuple, depth: int, text_parent: int, regions: tuple, parent_tag: str):
        tag = (box.get("tagName") or "").lower()
        if tag in _SKIP_TAGS:
            return
        is_text = tag == "#text"
        new_regions = set(regions)
        if tag in _REGION_TAGS:
            new_regions.add(_REGION_TAGS[tag])
        regions_t = tuple(sorted(new_regions))

        if is_text:
            text = _WS.sub(" ", "".join(box.itertext())).strip()
            if text:
                pre = "visual" if use_visual_bounds else "position"
                x = _f(box.attrib, f"{pre}X")
                y = _f(box.attrib, f"{pre}Y")
                w = _f(box.attrib, f"{pre}Width" if use_visual_bounds else "width")
                h = _f(box.attrib, f"{pre}Height" if use_visual_bounds else "height")
                bbox = [x, y, w, h] if None not in (x, y, w, h) else None
                nd = Node(id=len(nodes), text=text[:500], tag=parent_tag or "span", order=len(nodes),
                          depth=depth, parent=text_parent, ancestors=list(ancestry),
                          regions=list(regions_t), bbox=bbox,
                          font_size=_f(box.attrib, "fontsize"),
                          font_weight=_weight(box.get("fontweight")),
                          xpath="/".join(([parent_tag] if parent_tag else [])))
                nodes.append(nd)
            return

        my_elem = elem_counter[0]
        elem_counter[0] += 1
        anc = ancestry + (my_elem,)
        # the first text node inside an element acts as the "text parent" for deeper elements
        first_text_id = None
        for child in box:
            if child.tag != "box":
                continue
            before = len(nodes)
            walk(child, anc, depth + 1,
                 first_text_id if first_text_id is not None else text_parent,
                 regions_t, tag)
            if first_text_id is None and len(nodes) > before and (child.get("tagName") or "").lower() == "#text":
                first_text_id = before

    for top in root:
        if top.tag == "box":
            walk(top, (), 0, -1, (), "")

    _fill_xpaths(nodes)
    page = Page(page_id=page_id, site=site, nodes=nodes, gold=gold or {}, url=url)
    assign_labels(page)
    return page


def _fill_xpaths(nodes: List[Node]) -> None:
    # tag path is not exported by FitLayout; keep the parent tag as a short path
    for nd in nodes:
        nd.xpath = nd.xpath or nd.tag


# ---------------------------------------------------------------------------
# HTML preparation for offline rendering
# ---------------------------------------------------------------------------
_BASE_RE = re.compile(r"<base\b[^>]*>", re.I)
_SCRIPT_RE = re.compile(r"<script\b.*?</script\s*>", re.I | re.S)
_EXT_LINK_RE = re.compile(r"<link\b[^>]*href\s*=\s*[\"']?https?://[^>]*>", re.I)
_EXT_SRC_RE = re.compile(r"(\s(?:src|background)\s*=\s*)([\"']?)https?://[^\"'\s>]*\2", re.I)


def prepare_offline_html(html: str) -> str:
    """
    Make an archived page render fast and deterministically without network:
    drop <base>, scripts and external style sheets, blank external image sources.
    Layout then comes from inline styles, tables and HTML defaults (SWDE pages
    do not ship their CSS anyway).
    """
    html = _BASE_RE.sub("", html)
    html = _SCRIPT_RE.sub("", html)
    html = _EXT_LINK_RE.sub("", html)
    html = _EXT_SRC_RE.sub(r'\1""', html)
    return html


# ---------------------------------------------------------------------------
# Local HTTP server (the container reaches it via host.docker.internal)
# ---------------------------------------------------------------------------
class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def serve_directory(directory: str, port: int) -> socketserver.TCPServer:
    handler = partial(_QuietHandler, directory=directory)
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    httpd = socketserver.ThreadingTCPServer(("0.0.0.0", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


# ---------------------------------------------------------------------------
# Docker CLI batch rendering
# ---------------------------------------------------------------------------
def render_batch(items: Sequence[Tuple[str, str]], out_dir: str, threads: int = 4, timeout_s: int = 120,
                 width: int = 1200, height: int = 800, image: str = "fitlayout/fitlayout-cli",
                 skip_existing: bool = True, log=print) -> Dict[str, str]:
    """
    items: (page_id, url). Writes <out_dir>/xml/<page_id>.xml. Returns {page_id: xml_path} for
    the pages that exist after the run.
    """
    if shutil.which("docker") is None:
        raise RuntimeError("docker is not on PATH; install Docker Desktop or use --mode api")
    out_dir = os.path.abspath(out_dir)
    xml_dir = os.path.join(out_dir, "xml")
    os.makedirs(xml_dir, exist_ok=True)
    todo = [(pid, u) for pid, u in items
            if not (skip_existing and os.path.exists(os.path.join(xml_dir, f"{_safe(pid)}.xml")))]
    log(f"[render] {len(items)} pages, {len(items) - len(todo)} already rendered, {len(todo)} to go")
    if todo:
        with open(os.path.join(out_dir, "batch_input.tsv"), "w", encoding="utf-8", newline="\n") as f:
            for pid, u in todo:
                f.write(f"{u}\t{_safe(pid)}\n")
        with open(os.path.join(out_dir, "batch_commands.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write(f"RENDER -b puppeteer -W {width} -H {height} $1\n")
            f.write("EXPORT -f xml -o /out/xml/$2.xml\n")
        cmd = ["docker", "run", "--rm",
               "--add-host=host.docker.internal:host-gateway",
               "-v", f"{out_dir}:/out",
               image,
               "BATCH", "-i", "/out/batch_input.tsv", "-p", str(threads), "-t", str(timeout_s),
               "/out/batch_commands.txt"]
        log("[render] " + " ".join(cmd))
        subprocess.run(cmd, check=False)
    done = {}
    for pid, _ in items:
        p = os.path.join(xml_dir, f"{_safe(pid)}.xml")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            done[pid] = p
    log(f"[render] {len(done)}/{len(items)} pages available")
    return done


def _safe(pid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", pid)


# ---------------------------------------------------------------------------
# Remote FitLayout server (REST API) for a few public URLs
# ---------------------------------------------------------------------------
def render_via_api(items: Sequence[Tuple[str, str]], log=print) -> Dict[str, List[dict]]:
    """Render public URLs through the configured FitLayout REST server (see config.py / .env)."""
    from src.integration.fitlayout import FitLayoutClient
    from src.learning.graph_builder import FitLayoutParser

    client, parser = FitLayoutClient(), FitLayoutParser()
    out = {}
    for pid, url in items:
        try:
            raw = client.get_page_content(url)
            _, nodes = parser.parse(raw)
            out[pid] = nodes
        except Exception as e:  # keep going on individual failures
            log(f"[render-api] {pid}: {e}")
    return out
