"""
Command line:

  python -m smartscrape run      configs/experiments/<exp>.yaml
  python -m smartscrape extract  --model results/<exp>/model_seed0.pt --schema configs/schemas/<s>.yaml \
                                 (--html page.html | --url https://...)  [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="smartscrape")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run an experiment config")
    r.add_argument("config")
    e = sub.add_parser("extract", help="extract one record from a page")
    e.add_argument("--model", required=True)
    e.add_argument("--schema", required=True)
    g = e.add_mutually_exclusive_group(required=True)
    g.add_argument("--html", help="local HTML file (parsed without rendering)")
    g.add_argument("--url", help="URL rendered through FitLayout (needs FITLAYOUT_TOKEN)")
    e.add_argument("--json", help="write the record with provenance to this file")
    a = ap.parse_args(argv)

    if a.cmd == "run":
        from .experiment import run
        run(a.config)
        return 0

    from .extract import Extractor
    ex = Extractor.load(a.model, a.schema)
    result = ex.extract_html_file(a.html) if a.html else ex.extract_url(a.url)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
