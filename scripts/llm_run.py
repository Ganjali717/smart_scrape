"""
Run an LLM extractor on the fixed evaluation sample (100 pages per site, all sites) and store
one JSON line per page: values, tokens, wall-clock seconds. Resumable: pages already in the
output file are skipped, so the run can be stopped and restarted at any time.

    ollama pull qwen2.5:14b
    python scripts/llm_run.py --model qwen2.5:14b                       # all four verticals
    python scripts/llm_run.py --model qwen2.5:14b --verticals job --per-site 50

The page sample is configs/sample_100_per_site.json (the same pages the template-inference
analysis uses), so LLM and our method are scored on identical pages.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

from smartscrape.baselines.llm import LLMBaseline  # noqa: E402
from smartscrape.experiment import load_dataset  # noqa: E402
from smartscrape.schema import Schema  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", default="ollama", choices=["ollama", "openai", "anthropic"])
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--verticals", nargs="+", default=["camera", "book", "auto", "job"])
    ap.add_argument("--per-site", type=int, default=100)
    ap.add_argument("--input", default="text", choices=["text", "html"])
    ap.add_argument("--max-chars", type=int, default=24000)
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--sample", default="configs/sample_100_per_site.json")
    ap.add_argument("--out", default="results/llm")
    a = ap.parse_args()

    sample = json.load(open(a.sample))
    tag = a.model.replace("/", "_").replace(":", "_")
    os.makedirs(os.path.join(a.out, tag), exist_ok=True)
    for v in a.verticals:
        cfg = yaml.safe_load(open(f"configs/experiments/swde_{v}_semfeat.yaml"))
        schema = Schema.load(cfg["schema"])
        want = sample[v]
        pages = {p.page_id: p for p in load_dataset(cfg["dataset"], cfg.get("cache_dir", "cache")) if p.page_id in set(want)}
        per_site = {}
        todo = []
        for pid in want:
            p = pages.get(pid)
            if p is None:
                continue
            if per_site.get(p.site, 0) >= a.per_site:
                continue
            per_site[p.site] = per_site.get(p.site, 0) + 1
            todo.append(p)
        path = os.path.join(a.out, tag, f"{v}.jsonl")
        done = set()
        if os.path.exists(path):
            done = {json.loads(l)["page"] for l in open(path, encoding="utf-8") if l.strip()}
        opts = {"backend": a.backend, "model": a.model, "input": a.input, "max_chars": a.max_chars,
                "num_ctx": a.num_ctx, "temperature": 0.0}
        if a.base_url:
            opts["base_url"] = a.base_url
        llm = LLMBaseline(schema, **opts)
        print(f"[{v}] {len(todo)} pages, {len(done)} already done -> {path}", flush=True)
        t_start = time.time()
        with open(path, "a", encoding="utf-8") as f:
            for n, p in enumerate(todo):
                if p.page_id in done:
                    continue
                pt0, ct0, fail0 = llm.stats.prompt_tokens, llm.stats.completion_tokens, llm.stats.failures
                t0 = time.perf_counter()
                values = llm.extract(p)
                dt = time.perf_counter() - t0
                rec = {"page": p.page_id, "site": p.site, "values": values, "seconds": dt,
                       "prompt_tokens": llm.stats.prompt_tokens - pt0,
                       "completion_tokens": llm.stats.completion_tokens - ct0,
                       "error": llm.stats.notes[-1] if llm.stats.failures > fail0 else None}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if n % 25 == 0:
                    el = time.time() - t_start
                    print(f"  {n + 1}/{len(todo)}  {dt:.1f}s/page  elapsed {el / 60:.1f} min", flush=True)
        print(f"[{v}] done, failures {llm.stats.failures}", flush=True)


if __name__ == "__main__":
    main()
