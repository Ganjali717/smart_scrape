"""
Efficiency benchmark (experiment E7): per-stage latency, throughput, peak memory.

    python -m smartscrape bench --model results/<exp>/model_seed0.pt \
        --schema results/<exp>/schema_calibrated_seed0.yaml \
        --pages cache/pages_swde_<hash>.jsonl --sizes 100 1000 10000 --out results/bench.csv

For each corpus size N (pages are re-used cyclically when N exceeds the
corpus) it measures, per page and per stage:
    features+graph   encode_pages (CPU)
    gnn              scorer forward pass (GPU if available)
    ilp              constrained decoding
plus total wall time, pages/second, peak process RSS and peak CUDA memory.
Warm-up pages are excluded. The results also include page size (number of
nodes) so latency can be plotted against page size.
"""

from __future__ import annotations

import csv
import gc
import os
import statistics
import time
from typing import Dict, List, Sequence

import numpy as np
import torch

from .data import Page, load_pages
from .features import FeatureEncoder
from .schema import Schema
from .solver import ConstraintSolver, decode_argmax
from .train import encode_pages, load_bundle


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 2**20
    except ImportError:
        return float("nan")


class PeakRSS:
    """Samples RSS in a background thread; psutil has no portable peak counter on Windows."""

    def __init__(self, interval: float = 0.05):
        import threading
        self.interval, self.peak, self._stop = interval, _rss_mb(), threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, _rss_mb())
            time.sleep(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, _rss_mb())


def profile_pages(pages: Sequence[Page], model, schema: Schema, encoder: FeatureEncoder,
                  device: str, top_k: int = 20, warmup: int = 5) -> Dict[str, float]:
    solver = ConstraintSolver(schema, top_k=top_k)
    model = model.to(device).eval()
    t_feat, t_gnn, t_ilp, sizes = [], [], [], []
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    with PeakRSS() as mem:
        t_all = time.perf_counter()
        for k, p in enumerate(pages):
            t0 = time.perf_counter()
            d = encode_pages([p], schema, encoder)[0]
            t1 = time.perf_counter()
            with torch.no_grad():
                s = model.field_scores(d.x.to(device), d.edge_index.to(device), d.edge_type.to(device))
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                s = s.cpu().numpy()
            t2 = time.perf_counter()
            solver.solve(p, s) if p.n else decode_argmax(p, s, schema)
            t3 = time.perf_counter()
            if k >= warmup:
                t_feat.append(t1 - t0); t_gnn.append(t2 - t1); t_ilp.append(t3 - t2); sizes.append(p.n)
        wall = time.perf_counter() - t_all
    n = max(1, len(t_feat))
    tot = np.array(t_feat) + np.array(t_gnn) + np.array(t_ilp)
    return {
        "pages": len(pages),
        "ms_features_graph": 1000 * float(np.mean(t_feat)) if t_feat else float("nan"),
        "ms_gnn": 1000 * float(np.mean(t_gnn)) if t_gnn else float("nan"),
        "ms_ilp": 1000 * float(np.mean(t_ilp)) if t_ilp else float("nan"),
        "ms_total_p50": 1000 * float(np.percentile(tot, 50)) if len(tot) else float("nan"),
        "ms_total_p95": 1000 * float(np.percentile(tot, 95)) if len(tot) else float("nan"),
        "pages_per_second": len(pages) / wall,
        "peak_rss_mb": mem.peak,
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / 2**20) if device.startswith("cuda") else 0.0,
        "mean_nodes": float(np.mean(sizes)) if sizes else 0.0,
        "corr_nodes_latency": float(np.corrcoef(sizes, tot)[0, 1]) if len(sizes) > 2 else float("nan"),
        "device": device,
    }


def run_bench(model_path: str, schema_path: str, pages_path: str, sizes: Sequence[int],
              out_csv: str, device: str = None, log=print) -> List[dict]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, bundle = load_bundle(model_path)
    schema = Schema.load(schema_path)
    if list(bundle["fields"]) != schema.field_names:
        raise ValueError("model and schema fields differ")
    encoder = FeatureEncoder(sentence_model=bundle.get("sentence_model"))
    corpus = load_pages(pages_path)
    rows = []
    for n in sizes:
        pages = [corpus[i % len(corpus)] for i in range(n)]
        gc.collect()
        r = {"n": n, **profile_pages(pages, model, schema, encoder, device)}
        rows.append(r)
        log(f"[bench] n={n:>7}  {r['pages_per_second']:.1f} pages/s  p50 {r['ms_total_p50']:.1f} ms  "
            f"(feat {r['ms_features_graph']:.1f} / gnn {r['ms_gnn']:.1f} / ilp {r['ms_ilp']:.1f})  "
            f"RSS {r['peak_rss_mb']:.0f} MB  VRAM {r['peak_vram_mb']:.0f} MB")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows
