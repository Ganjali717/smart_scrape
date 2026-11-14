"""
Site-level adaptive constraint gating, evaluated on trained models without retraining.

Hypothesis: whether cross-field constraints help depends on the target site. A label-free
statistic tells us which sites they fit: the share of a site's pages on which the hard ILP
overrides the scorer's own (filtered) choice. If the constraints fit the site, they only
fix occasional mistakes (low override rate); if they do not fit, they override the scorer
on most pages (high override rate) and we fall back to the filtered decoder.

The threshold tau is chosen on the validation sites (which have labels), then applied
unchanged to the test sites. An oracle that picks the better decoder per test site with
labels is reported as an upper bound.

    python scripts/adaptive_gate.py --config configs/experiments/swde_camera_semfeat.yaml \
        --models results/swde_camera_semfeat --out results/swde_camera_semfeat/adaptive.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from smartscrape.calibrate import calibrate  # noqa: E402
from smartscrape.data import text_matches  # noqa: E402
from smartscrape.experiment import load_dataset, split_sites  # noqa: E402
from smartscrape.features import FeatureEncoder  # noqa: E402
from smartscrape.schema import Schema  # noqa: E402
from smartscrape.solver import ConstraintSolver, decode_filtered, violations  # noqa: E402
from smartscrape.train import encode_pages, load_bundle, score_pages  # noqa: E402


GAP_TAUS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 1e9]


def outcomes(page, rec, schema):
    """Per-field (tp, fp, fn) counts with the SWDE convention used in evaluate.py."""
    res = {}
    for f in schema.field_names:
        gold = page.gold.get(f) or []
        pred = rec.fields[f].text or ""
        ok = bool(pred) and bool(gold) and text_matches(pred, gold)
        res[f] = (int(ok), int(bool(pred) and not ok), int(bool(gold) and not ok))
    return res


def f1_from(rows, schema):
    f1s = []
    for f in schema.field_names:
        tp = sum(r[f][0] for r in rows); fp = sum(r[f][1] for r in rows); fn = sum(r[f][2] for r in rows)
        p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
        f1s.append(2 * p * r / max(1e-9, p + r))
    return float(np.mean(f1s))


def analyse(pages, model, encoder, schema, solver, log):
    """Per page: outcomes of filtered and ILP, whether ILP overrode, violations of each."""
    per_page = []
    for k, p in enumerate(pages):
        d = encode_pages([p], schema, encoder)
        s = score_pages(model, d)[0]
        rf = decode_filtered(p, s, schema)
        ri = solver.solve(p, s)
        # score the scorer gives up when the ILP replaces its choice (summed over fields, log-odds units)
        gap = 0.0
        for j, f in enumerate(schema.field_names):
            a, b = rf.fields[f].node, ri.fields[f].node
            if a != b:
                sa = float(s[a, j]) if a is not None else 0.0
                sb = float(s[b, j]) if b is not None else 0.0
                gap += max(0.0, sa - sb)
        per_page.append({
            "site": p.site, "gap": gap,
            "override": int(any(rf.fields[f].node != ri.fields[f].node for f in schema.field_names)),
            "filtered": outcomes(p, rf, schema), "ilp": outcomes(p, ri, schema),
            "viol_filtered": int(bool(violations(p, rf, schema))),
            "viol_ilp": int(bool(violations(p, ri, schema))),
        })
        if (k + 1) % 500 == 0:
            log(f"    {k + 1}/{len(pages)} pages")
    return per_page


def gated(per_page, schema, use_ilp_site):
    rows, viol = [], 0
    for r in per_page:
        m = "ilp" if use_ilp_site[r["site"]] else "filtered"
        rows.append(r[m])
        viol += r["viol_ilp"] if m == "ilp" else r["viol_filtered"]
    return f1_from(rows, schema), viol / max(1, len(per_page))


def site_rates(per_page, key="override"):
    by = defaultdict(list)
    for r in per_page:
        by[r["site"]].append(r[key])
    return {s: float(np.mean(v)) for s, v in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", required=True, help="folder with model_seed*.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="*")
    a = ap.parse_args()
    log = print

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    schema = Schema.load(cfg["schema"])
    pages = load_dataset(cfg["dataset"], cfg.get("cache_dir", "cache"), log)
    mcfg = cfg.get("model", {})
    encoder = FeatureEncoder(sentence_model=mcfg.get("sentence_model"), exclude=mcfg.get("exclude_features", []))
    ccfg = cfg.get("calibrate", {})
    report = {}

    for seed in (a.seeds if a.seeds is not None else cfg.get("seeds", [0])):
        tr, va, te = split_sites(pages, cfg["split"], seed)
        tr = [p for p in tr if p.labels]
        log(f"=== seed {seed}: val {sorted({p.site for p in va})}, test {sorted({p.site for p in te})}")
        model, _ = load_bundle(os.path.join(a.models, f"model_seed{seed}.pt"))
        cal, _ = calibrate(schema, list(tr) + list(va), min_support=ccfg.get("min_support", 0.95),
                           quantile=ccfg.get("quantile", 0.95))
        solver = ConstraintSolver(cal, top_k=cfg.get("top_k", 20))
        log("  validation sites ...")
        pv = analyse(va, model, encoder, cal, solver, log)
        log("  test sites ...")
        pt = analyse(te, model, encoder, cal, solver, log)

        rt = site_rates(pt)
        gv, gt = site_rates(pv, "gap"), site_rates(pt, "gap")
        res = {"test_site_override_rate": rt, "test_site_gap": gt, "val_site_gap": gv,
               "filtered": gated(pt, cal, {s: False for s in rt}),
               "ilp": gated(pt, cal, {s: True for s in rt})}
        # label-free gate on the mean score gap per site; threshold chosen on validation sites
        best_tau, best_f1 = None, -1
        for t in GAP_TAUS:
            f1, _ = gated(pv, cal, {s: g < t for s, g in gv.items()})
            if f1 > best_f1 + 1e-9:
                best_tau, best_f1 = t, f1
        res["tau_gap"] = best_tau
        res["adaptive"] = gated(pt, cal, {s: g < best_tau for s, g in gt.items()})
        # sensitivity: every fixed threshold on test (reported, not used for selection)
        res["adaptive_by_tau"] = {str(t): gated(pt, cal, {s: g < t for s, g in gt.items()}) for t in GAP_TAUS}
        # per-site F1 of both decoders + oracle upper bound
        per_site = {}
        for s in rt:
            sub = [r for r in pt if r["site"] == s]
            per_site[s] = {"override_rate": rt[s], "gap": gt[s],
                           "f1_filtered": f1_from([r["filtered"] for r in sub], cal),
                           "f1_ilp": f1_from([r["ilp"] for r in sub], cal), "pages": len(sub)}
        res["per_site"] = per_site
        res["oracle"] = gated(pt, cal, {s: v["f1_ilp"] >= v["f1_filtered"] for s, v in per_site.items()})
        report[f"seed{seed}"] = res
        log(f"  tau_gap={best_tau}  filtered F1 {res['filtered'][0]:.3f} viol {res['filtered'][1]:.3f} | "
            f"ilp F1 {res['ilp'][0]:.3f} viol {res['ilp'][1]:.3f} | adaptive F1 {res['adaptive'][0]:.3f} "
            f"viol {res['adaptive'][1]:.3f} | oracle F1 {res['oracle'][0]:.3f}")
        for s, v in sorted(per_site.items(), key=lambda kv: kv[1]["override_rate"]):
            log(f"    {s:<12} override {v['override_rate']:.2f}  gap {gt[s]:.2f}  F1 filtered {v['f1_filtered']:.3f}  "
                f"ilp {v['f1_ilp']:.3f}  ({v['pages']} pages)")
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(report, open(a.out, "w"), indent=2)

    for m in ("filtered", "ilp", "adaptive", "oracle"):
        f1 = [report[k][m][0] for k in report]
        v = [report[k][m][1] for k in report]
        log(f"[mean over seeds] {m:<9} F1 {np.mean(f1):.3f} ± {np.std(f1):.3f}   violations {np.mean(v):.3f}")


if __name__ == "__main__":
    main()
