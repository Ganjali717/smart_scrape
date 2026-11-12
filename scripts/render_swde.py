"""
Render SWDE pages with FitLayout (docker) and save them as rendered Pages.

    docker pull fitlayout/fitlayout-cli
    python scripts/render_swde.py camera --max-pages-per-site 200 --threads 6

Steps
  1. copy each page to data/render/<vertical>/site/<id>.html with offline fixes
     (no <base>, scripts, external CSS or images) so rendering needs no network
  2. serve that folder on http://host.docker.internal:<port>/ from this process
  3. run `fitlayout-cli BATCH` (puppeteer backend) -> data/render/<vertical>/xml/<id>.xml
  4. parse the XML box trees (geometry + structure + fonts), attach SWDE ground truth,
     write data/render/<vertical>/pages.jsonl  (dataset type `jsonl` in experiment configs)

Re-running skips pages that are already rendered.
"""

import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smartscrape.data import save_pages  # noqa: E402
from smartscrape.loaders import _parse_gt  # noqa: E402
from smartscrape.render import parse_fitlayout_xml, prepare_offline_html, render_batch, serve_directory  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vertical")
    ap.add_argument("--html-root", default="data/swde/html")
    ap.add_argument("--gt", default=None, help="default: data/swde/<vertical>-00000-of-00001.parquet")
    ap.add_argument("--out", default=None, help="default: data/render/<vertical>")
    ap.add_argument("--sites", nargs="*")
    ap.add_argument("--max-pages-per-site", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="host.docker.internal")
    a = ap.parse_args()

    import pyarrow.parquet as pq

    gt_path = a.gt or f"data/swde/{a.vertical}-00000-of-00001.parquet"
    out = a.out or os.path.join("data", "render", a.vertical)
    site_dir = os.path.join(out, "site")
    os.makedirs(site_dir, exist_ok=True)
    df = pq.read_table(gt_path).to_pandas()
    gold = {row.website_id: _parse_gt(row.gt) for row in df.itertuples()}

    # 1. offline copies
    items, meta = [], {}
    for sd in sorted(glob.glob(os.path.join(a.html_root, a.vertical, f"{a.vertical}-*"))):
        site = re.match(rf"{a.vertical}-(.+)\(\d+\)", os.path.basename(sd)).group(1)
        if a.sites and site not in a.sites:
            continue
        files = sorted(glob.glob(os.path.join(sd, "*.htm")))[: a.max_pages_per_site]
        for fp in files:
            pid = f"{site}_{os.path.splitext(os.path.basename(fp))[0]}"
            if pid not in gold:
                continue
            dst = os.path.join(site_dir, f"{pid}.html")
            if not os.path.exists(dst):
                with open(fp, "rb") as f:
                    html = f.read().decode("utf-8", errors="replace")
                with open(dst, "w", encoding="utf-8") as f:
                    f.write(prepare_offline_html(html))
            items.append((pid, f"http://{a.host}:{a.port}/{pid}.html"))
            meta[pid] = site
    print(f"[prepare] {len(items)} pages in {site_dir}")

    # 2-3. serve and render
    httpd = serve_directory(site_dir, a.port)
    try:
        done = render_batch(items, out, threads=a.threads, timeout_s=a.timeout)
    finally:
        httpd.shutdown()

    # 4. parse
    pages, failed = [], 0
    for pid, path in done.items():
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                pages.append(parse_fitlayout_xml(f.read(), pid, meta[pid], gold[pid]))
        except Exception as e:
            failed += 1
            print(f"[parse] {pid}: {e}")
    out_jsonl = os.path.join(out, "pages.jsonl")
    save_pages(pages, out_jsonl)
    with_geo = sum(1 for p in pages if p.has_geometry())
    print(f"[done] {len(pages)} pages ({with_geo} with geometry, {failed} parse errors) -> {out_jsonl}")


if __name__ == "__main__":
    main()
