"""
End-to-end experiment runner.

    python -m smartscrape run configs/experiments/swde_camera.yaml

For every seed: split sites -> train the scorer -> score test pages (clean and
under each drift setting) -> decode with every method -> metrics. Writes:

    <out>/results.csv        one row per (seed, setting, method)
    <out>/summary.csv        mean and std over seeds
    <out>/reliability.json   AUROC/AUPRC of the risk signals per setting
    <out>/provenance_sample.json  a few full ILP records with their audit trail
    <out>/model_seed<k>.pt   trained scorers
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
import torch
import yaml

from .data import Page, load_pages, save_pages, text_matches
from .calibrate import calibrate
from .drift import apply_drift
from .evaluate import aggregate, field_outcomes, record_correct, reliability_auc
from .features import FeatureEncoder
from .reliability import disagreement, page_stability, risk
from .schema import Schema, save_schema
from .solver import ConstraintSolver, decode_argmax, decode_filtered
from .train import encode_pages, save_bundle, score_pages, train_scorer


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_dataset(cfg: dict, cache_dir: str, log=print) -> List[Page]:
    key = hashlib.md5(json.dumps({**cfg, "_cache_version": 2}, sort_keys=True).encode()).hexdigest()[:10]
    path = os.path.join(cache_dir, f"pages_{cfg['type']}_{key}.jsonl")
    if os.path.exists(path):
        log(f"[data] cached pages: {path}")
        return _sanitize(load_pages(path))
    t0 = time.time()
    if cfg["type"] == "swde":
        from .loaders import load_swde_vertical
        pages = load_swde_vertical(cfg["html_root"], cfg["gt_parquet"], cfg["vertical"],
                                   sites=cfg.get("sites"), max_pages_per_site=cfg.get("max_pages_per_site"))
    elif cfg["type"] == "legacy_json":
        from .loaders import load_legacy_json
        pages = load_legacy_json(cfg["path"], site=cfg.get("site"))
        if cfg.get("max_pages"):
            pages = pages[: cfg["max_pages"]]
    elif cfg["type"] == "jsonl":
        pages = load_pages(cfg["path"])
    else:
        raise ValueError(f"unknown dataset type {cfg['type']}")
    os.makedirs(cache_dir, exist_ok=True)
    save_pages(pages, path)
    log(f"[data] parsed {len(pages)} pages in {time.time() - t0:.1f}s -> {path}")
    return pages


def _sanitize(pages: List[Page]) -> List[Page]:
    """Drop SWDE '<NULL>' gold values from pages cached before the loader filtered them."""
    for p in pages:
        clean = {f: [v for v in vals if v.strip() and v.strip().upper() != "<NULL>"] for f, vals in p.gold.items()}
        p.gold = {f: v for f, v in clean.items() if v}
        p.labels = {i: f for i, f in p.labels.items() if f in p.gold}
    return pages


def label_coverage(pages: Sequence[Page], schema: Schema) -> Dict[str, float]:
    """Share of (page, field with gold) where some node text matches gold: an upper bound on F1."""
    cov = {}
    for f in schema.field_names:
        have = [p for p in pages if p.gold.get(f)]
        cov[f] = sum(1 for p in have if f in set(p.labels.values())) / max(1, len(have))
    return cov


def split_sites(pages: Sequence[Page], split: dict, seed: int):
    sites = sorted({p.site for p in pages})
    rng = random.Random(seed)
    if split["type"] == "cross_site":
        rng.shuffle(sites)
        k, v = split.get("train_sites", 3), split.get("val_sites", 1)
        tr, va, te = set(sites[:k]), set(sites[k:k + v]), set(sites[k + v:])
        return ([p for p in pages if p.site in tr], [p for p in pages if p.site in va],
                [p for p in pages if p.site in te])
    if split["type"] == "in_site":
        # every site contributes its first pages to training/validation and the rest to test;
        # this is the setting where per-site wrappers (XPath, AutoScraper) are applicable
        k, v = split.get("train_per_site", 20), split.get("val_per_site", 5)
        by_site = {}
        for p in pages:
            by_site.setdefault(p.site, []).append(p)
        tr, va, te = [], [], []
        for s in sorted(by_site):
            ps = sorted(by_site[s], key=lambda p: p.page_id)
            rng.shuffle(ps)
            tr += ps[:k]; va += ps[k:k + v]; te += ps[k + v:]
        return tr, va, te
    if split["type"] == "random":
        idx = list(range(len(pages)))
        rng.shuffle(idx)
        a = int(len(idx) * split.get("train", 0.6))
        b = a + int(len(idx) * split.get("val", 0.1))
        return ([pages[i] for i in idx[:a]], [pages[i] for i in idx[a:b]], [pages[i] for i in idx[b:]])
    raise ValueError(split["type"])


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------
def tune_soft_lambda(schema: Schema, pages: Sequence[Page], scores: Sequence[np.ndarray], top_k: int,
                     grid: Sequence[float], log=print) -> float:
    """Pick the penalty for soft cross-field constraints by macro F1 on validation sites."""
    best_lam, best_f1 = None, -1.0
    for lam in grid:
        solver = ConstraintSolver(schema, top_k=top_k, soft_lambda=float(lam))
        f1 = aggregate(pages, [solver.solve(p, s) for p, s in zip(pages, scores)], schema)["f1/macro"]
        log(f"  [soft] lambda={lam:<6} val F1 {f1:.3f}")
        if f1 > best_f1 + 1e-9:
            best_lam, best_f1 = float(lam), f1
    return best_lam


def build_methods(schema: Schema, top_k: int, variants: Sequence[str], raw_schema: Schema = None,
                  soft_lambda: float = None):
    """Return {name: callable(page, scores) -> Record}. `schema` may be calibrated; `raw_schema` is the
    hand-written one (used by 'ilp-uncalibrated')."""
    methods = {}
    raw_schema = raw_schema or schema
    for v in variants:
        if v == "ilp-uncalibrated":
            methods[v] = ConstraintSolver(raw_schema, top_k=top_k).solve
            continue
        if v == "ilp-soft":
            methods[v] = ConstraintSolver(schema, top_k=top_k, soft_lambda=soft_lambda).solve
            continue
        if v == "argmax":
            methods[v] = lambda p, s: decode_argmax(p, s, schema)
        elif v == "filtered":
            methods[v] = lambda p, s: decode_filtered(p, s, schema)
        elif v == "ilp":
            solver = ConstraintSolver(schema, top_k=top_k)
            methods[v] = solver.solve
        elif v.startswith("ilp-"):
            # ablation: ilp-<kind>+<kind>... keeps only the listed constraint kinds
            kinds = set(v[4:].split("+"))
            solver = ConstraintSolver(schema.subset(kinds=kinds), top_k=top_k)
            methods[v] = solver.solve
        else:
            raise ValueError(f"unknown method {v}")
    return methods


def run_baseline(name: str, opts: dict, schema: Schema, train: Sequence[Page], test: Sequence[Page],
                 split_type: str, seed: int, log=print) -> List[dict]:
    from .baselines import BaselineUnavailable, make_baseline
    from .solver import values_to_record

    kind = opts.get("type", name)
    try:
        b = make_baseline(kind, schema, **{k: v for k, v in opts.items() if k != "type"})
    except (KeyError, BaselineUnavailable) as e:
        log(f"  [baseline] {name}: skipped ({e})")
        return []
    if b.needs_site_training and split_type == "cross_site":
        log(f"  [baseline] {name}: skipped (needs pages of the test site; use split in_site or temporal)")
        return []
    pages = [p for p in test if p.source]
    if len(pages) < len(test):
        log(f"  [baseline] {name}: {len(test) - len(pages)} test pages have no raw HTML and are skipped")
    try:
        b.fit(train)
        recs = [values_to_record(p, b.extract(p), schema, f"baseline:{name}") for p in pages]
    except BaselineUnavailable as e:
        log(f"  [baseline] {name}: unavailable ({e})")
        return []
    m = aggregate(pages, recs, schema)
    st = b.stats.as_dict()
    m.update({"ms_per_page": st["ms_per_page"], "fit_seconds": st["fit_seconds"],
              "prompt_tokens": st["prompt_tokens"], "completion_tokens": st["completion_tokens"],
              "failures": st["failures"]})
    cost = getattr(b, "cost_usd", lambda: None)()
    if cost is not None:
        m["usd_per_1k_pages"] = 1000 * cost / max(1, len(pages))
    log(f"  [{'clean':>18}] {'baseline:' + name:<28} F1 {m['f1/macro']:.3f}  EM {m['exact_match']:.3f}  "
        f"viol {m['violation_rate']:.3f}  grounded {m['grounded_rate']:.3f}  {m['ms_per_page']:.0f} ms/page")
    return [{"seed": seed, "setting": "clean", "method": f"baseline:{name}", **m}]


def old_sigma(scores: np.ndarray) -> float:
    """ICWE-version stability: mean over ALL nodes of top1 - top2 class probability."""
    if scores.shape[0] == 0:
        return 0.0
    logits = np.concatenate([scores, np.zeros((scores.shape[0], 1))], axis=1)
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    s = np.sort(p, axis=1)
    return float((s[:, -1] - s[:, -2]).mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(config_path: str, log=print) -> str:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_dir = cfg.get("output", os.path.join("results", cfg["name"]))
    os.makedirs(out_dir, exist_ok=True)
    schema = Schema.load(cfg["schema"])
    pages = load_dataset(cfg["dataset"], cfg.get("cache_dir", "cache"), log)
    log(f"[data] {len(pages)} pages, {len({p.site for p in pages})} sites; "
        f"label coverage {json.dumps({k: round(v, 3) for k, v in label_coverage(pages, schema).items()})}")

    mcfg = cfg.get("model", {})
    encoder = FeatureEncoder(sentence_model=mcfg.get("sentence_model"),
                             exclude=mcfg.get("exclude_features", []))
    methods_cfg = cfg.get("methods", ["argmax", "filtered", "ilp"])
    top_k = cfg.get("top_k", 20)
    drift_settings = cfg.get("drift", [])
    rows: List[dict] = []
    rel_rows: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    provenance = []

    for seed in cfg.get("seeds", [0]):
        tr, va, te = split_sites(pages, cfg["split"], seed)
        tr = [p for p in tr if p.labels]  # train only on pages with at least one labelled node
        log(f"\n=== seed {seed}: train {len(tr)} pages ({sorted({p.site for p in tr})}), "
            f"val {len(va)} ({sorted({p.site for p in va})}), test {len(te)} ===")
        t0 = time.time()
        enc_tr = encode_pages(tr, schema, encoder)
        enc_va = encode_pages(va, schema, encoder)
        log(f"[train] encoded in {time.time() - t0:.1f}s, feature dim {encoder.dim}")
        model = train_scorer(enc_tr, enc_va, schema, encoder.dim,
                             epochs=mcfg.get("epochs", 30), lr=mcfg.get("lr", 2e-3),
                             hidden=mcfg.get("hidden", 128), layers=mcfg.get("layers", 2),
                             pos_weight=mcfg.get("pos_weight", 20.0), patience=mcfg.get("patience", 6),
                             seed=seed, log=log)
        save_bundle(model, schema, mcfg, os.path.join(out_dir, f"model_seed{seed}.pt"))
        log(f"[train] done in {time.time() - t0:.1f}s, temperature {float(model.temperature):.2f}")

        title_pool = {}
        for p in pages:
            for f in schema.field_names:
                if schema.field(f).type == "text" and schema.field(f).cardinality == "one":
                    title_pool.setdefault(p.site, []).extend(p.gold.get(f, [])[:1])

        settings = [("clean", None)] + [(d["name"], d) for d in drift_settings]
        ccfg = cfg.get("calibrate", {"enabled": True})
        if ccfg.get("enabled", True):
            seed_schema, report = calibrate(schema, list(tr) + list(va),
                                            min_support=ccfg.get("min_support", 0.95),
                                            quantile=ccfg.get("quantile", 0.95))
            with open(os.path.join(out_dir, f"calibration_seed{seed}.json"), "w") as f:
                json.dump(report, f, indent=2)
            save_schema(seed_schema, os.path.join(out_dir, f"schema_calibrated_seed{seed}.yaml"))
            log(f"[calibrate] active: {[c.name for c in seed_schema.constraints]}")
        else:
            seed_schema = schema
        soft_lambda = None
        if "ilp-soft" in methods_cfg:
            val_scores = score_pages(model, enc_va)
            soft_lambda = tune_soft_lambda(seed_schema, va, val_scores, top_k,
                                           cfg.get("soft_lambda_grid", [0.25, 0.5, 1, 2, 4, 8]), log)
            log(f"[soft] chosen lambda = {soft_lambda}")
            with open(os.path.join(out_dir, f"soft_lambda_seed{seed}.json"), "w") as f:
                json.dump({"lambda": soft_lambda}, f)
        methods = build_methods(seed_schema, top_k, methods_cfg, raw_schema=schema, soft_lambda=soft_lambda)
        for sname, d in settings:
            if d is None:
                test_pages = te
            else:
                test_pages = [apply_drift(p, d["kinds"], d.get("severity", 0.3), seed,
                                          pool_titles=title_pool.get(p.site)) for p in te]
            if d is None:
                for bname, bopts in (cfg.get("baselines") or {}).items():
                    rows.extend(run_baseline(bname, bopts or {}, seed_schema, list(tr) + list(va), test_pages,
                                             cfg["split"]["type"], seed, log))
            enc_te = encode_pages(test_pages, schema, encoder)
            scores = score_pages(model, enc_te)
            for mname, fn in methods.items():
                t1 = time.time()
                recs = [fn(p, s) for p, s in zip(test_pages, scores)]
                metrics = aggregate(test_pages, recs, seed_schema)
                metrics["violation_rate_handwritten"] = aggregate(test_pages, recs, schema)["violation_rate"]
                metrics["ms_per_page"] = 1000 * (time.time() - t1) / max(1, len(recs))
                rows.append({"seed": seed, "setting": sname, "method": mname, **metrics})
                log(f"  [{sname:>18}] {mname:<28} F1 {metrics['f1/macro']:.3f}  EM {metrics['exact_match']:.3f}  "
                    f"viol {metrics['violation_rate']:.3f}")
                if mname == "ilp":
                    base = [decode_argmax(p, s, schema) for p, s in zip(test_pages, scores)]
                    for p, r, b, s in zip(test_pages, recs, base, scores):
                        wrong = not record_correct(field_outcomes(p, r, schema))
                        dis = disagreement(r, b)
                        rel_rows[sname]["wrong"].append(wrong)
                        rel_rows[sname]["risk"].append(risk(r, b))
                        rel_rows[sname]["neg_margin"].append(-page_stability(r))
                        rel_rows[sname]["disagreement"].append(dis["field_disagreement"] + dis["objective_gap"])
                        rel_rows[sname]["old_sigma_neg"].append(-old_sigma(s))
                    if sname == "clean" and len(provenance) < 5:
                        for p, r in list(zip(test_pages, recs))[:5 - len(provenance)]:
                            provenance.append({"page": p.page_id, "gold": p.gold, **r.provenance()})

    _write_rows(os.path.join(out_dir, "results.csv"), rows)
    _write_summary(os.path.join(out_dir, "summary.csv"), rows)
    rel = {}
    for sname, cols in rel_rows.items():
        rel[sname] = {sig: reliability_auc(cols[sig], cols["wrong"])
                      for sig in ("risk", "neg_margin", "disagreement", "old_sigma_neg")}
    with open(os.path.join(out_dir, "reliability.json"), "w") as f:
        json.dump(rel, f, indent=2)
    with open(os.path.join(out_dir, "provenance_sample.json"), "w") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False)
    log(f"\n[done] results in {out_dir}")
    log(json.dumps(rel, indent=1))
    return out_dir


def _write_rows(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)


def _write_summary(path, rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["setting"], r["method"])].append(r)
    metrics = []
    for r in rows:
        metrics += [k for k in r if k not in ("seed", "setting", "method") and k not in metrics]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "method", "n_seeds"] + [f"{m}_{s}" for m in metrics for s in ("mean", "std")])
        for (s, m), rs in groups.items():
            vals = []
            for k in metrics:
                a = np.array([r[k] for r in rs if k in r], dtype=float)
                vals += ([round(float(a.mean()), 4), round(float(a.std()), 4)] if len(a) else ["", ""])
            w.writerow([s, m, len(rs)] + vals)
