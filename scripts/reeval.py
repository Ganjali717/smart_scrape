"""
Re-evaluate finished experiments from their saved models (no retraining).

    python scripts/reeval.py configs/experiments/swde_book_semfeat.yaml [more configs ...]

For every seed it reloads model_seed<k>.pt, the calibrated schema and the soft-constraint
lambda written by `smartscrape run`, rebuilds the identical split and drift, and adds:

1. Template inference (smartscrape.template.TemplateEM) on top of filtered / ILP / soft ILP:
   label-free, no tuning (u fixed a priori at 0.05; --u-grid adds a sensitivity sweep).
   A site-level drift setting (same edit on every page of a site, like a real redesign)
   is added next to the page-level drift settings of the config.
2. Reliability against graded targets instead of "record has any error" (86-98 % of
   pages on SWDE, which saturates AUROC):
     - spearman  : rank correlation of the signal with the fraction of wrong fields
     - auroc_half: AUROC for "at least half of the fields are wrong"
     - auroc_any : the old target, for continuity
   and at site level: rank correlation of the mean signal of a site with its macro F1
   (the monitoring question: which site broke?).

Writes into the experiment's results directory:
    reeval_results.csv, reeval_summary.csv, reeval_reliability.json,
    reeval_pages.csv (one row per page/method/setting), reeval_sites.csv (one row per test site)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smartscrape.template import TemplateEM  # noqa: E402
from smartscrape.drift import apply_drift  # noqa: E402
from smartscrape.evaluate import aggregate, field_outcomes  # noqa: E402
from smartscrape.experiment import _write_rows, _write_summary, load_dataset, old_sigma, split_sites  # noqa: E402
from smartscrape.features import FeatureEncoder  # noqa: E402
from smartscrape.reliability import disagreement, page_stability, risk  # noqa: E402
from smartscrape.schema import Schema  # noqa: E402
from smartscrape.solver import ConstraintSolver, decode_argmax, decode_filtered  # noqa: E402
from smartscrape.train import encode_pages, load_bundle, score_pages  # noqa: E402

SIGNALS = ("risk", "neg_margin", "disagreement", "old_sigma_neg", "template_dis", "risk+template")
REL_METHODS = ("ilp", "ilp-soft+em")


def frac_wrong(page, rec, schema) -> float:
    oc = field_outcomes(page, rec, schema)
    return sum(o not in ("tp", "tn") for o in oc.values()) / max(1, len(oc))


def decoders(schema, top_k, lam):
    return {
        "filtered": lambda p, s: decode_filtered(p, s, schema),
        "ilp": ConstraintSolver(schema, top_k=top_k).solve,
        "ilp-soft": ConstraintSolver(schema, top_k=top_k, soft_lambda=lam).solve,
    }


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, dtype=int)
    return float("nan") if y.min() == y.max() else float(roc_auc_score(y, s))


def spearman(a, b):
    from scipy.stats import spearmanr
    if len(set(a)) < 2 or len(set(b)) < 2:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def reeval(config_path, u_grid, log=print):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_dir = cfg.get("output", os.path.join("results", cfg["name"]))
    raw_schema = Schema.load(cfg["schema"])
    pages = load_dataset(cfg["dataset"], cfg.get("cache_dir", "cache"), log)
    mcfg = cfg.get("model", {})
    encoder = FeatureEncoder(sentence_model=mcfg.get("sentence_model"), exclude=mcfg.get("exclude_features", []))
    top_k = cfg.get("top_k", 20)
    settings = [("clean", None)] + [(d["name"], d) for d in cfg.get("drift", [])]
    for d in cfg.get("drift", []):
        if d["name"] == "all_combined":
            settings.append(("all_combined_site", {**d, "level": "site"}))

    rows, page_rows, site_rows = [], [], []
    for seed in cfg.get("seeds", [0]):
        mpath = os.path.join(out_dir, f"model_seed{seed}.pt")
        if not os.path.exists(mpath):
            log(f"[skip] {mpath} missing")
            continue
        t0 = time.time()
        model, _ = load_bundle(mpath)
        schema = Schema.load(os.path.join(out_dir, f"schema_calibrated_seed{seed}.yaml"))
        lam_path = os.path.join(out_dir, f"soft_lambda_seed{seed}.json")
        lam = json.load(open(lam_path))["lambda"] if os.path.exists(lam_path) else 1.0
        _, va, te = split_sites(pages, cfg["split"], seed)
        log(f"\n=== {cfg['name']} seed {seed}: val {sorted({p.site for p in va})}, "
            f"test {sorted({p.site for p in te})}, lambda {lam} ===")

        title_pool = {}
        for p in pages:
            for f in raw_schema.field_names:
                if raw_schema.field(f).type == "text" and raw_schema.field(f).cardinality == "one":
                    title_pool.setdefault(p.site, []).extend(p.gold.get(f, [])[:1])

        decs = decoders(schema, top_k, lam)
        for sname, d in settings:
            test = te if d is None else [apply_drift(p, d["kinds"], d.get("severity", 0.3), seed,
                                                     pool_titles=title_pool.get(p.site),
                                                     level=d.get("level", "page")) for p in te]
            scores = score_pages(model, encode_pages(test, raw_schema, encoder))
            base_argmax = [decode_argmax(p, s, schema) for p, s in zip(test, scores)]
            rescored = {}
            for u in u_grid:
                t1 = time.time()
                rescored[u] = TemplateEM(schema, u=u).rescore(test, scores) + (time.time() - t1,)
            for name, dec in decs.items():
                variants = [(name, None)] + [(name + "+em" + ("" if u == u_grid[0] else f"@u{u}"), u)
                                             for u in u_grid]
                for variant, u in variants:
                    t1 = time.time()
                    sc = scores if u is None else rescored[u][0]
                    recs = [dec(p, s) for p, s in zip(test, sc)]
                    m = aggregate(test, recs, schema)
                    m["violation_rate_handwritten"] = aggregate(test, recs, raw_schema)["violation_rate"]
                    extra = 0.0 if u is None else rescored[u][3]
                    m["ms_per_page"] = 1000 * (time.time() - t1 + extra) / max(1, len(recs))
                    rows.append({"seed": seed, "setting": sname, "method": variant, **m})
                    log(f"  [{sname:>18}] {variant:<18} F1 {m['f1/macro']:.3f}  viol {m['violation_rate']:.3f}")
                    if variant not in REL_METHODS:
                        continue
                    tpls, keys = rescored[u_grid[0]][1], rescored[u_grid[0]][2]
                    by_site = defaultdict(list)
                    for p, r, b, s, k in zip(test, recs, base_argmax, scores, keys):
                        dis = disagreement(r, b)
                        rk = risk(r, b)
                        tpl = tpls[p.site]
                        td = _template_dis(r, k, tpl)
                        fw = frac_wrong(p, r, schema)
                        by_site[p.site].append(fw)
                        page_rows.append({
                            "seed": seed, "setting": sname, "method": variant, "site": p.site, "page": p.page_id,
                            "frac_wrong": fw, "risk": rk, "neg_margin": -page_stability(r),
                            "disagreement": dis["field_disagreement"] + dis["objective_gap"],
                            "old_sigma_neg": -old_sigma(s), "template_dis": td,
                            "risk+template": 0.5 * rk + 0.5 * td,
                        })
                    for site, fws in by_site.items():
                        site_rows.append({"seed": seed, "setting": sname, "method": variant, "site": site,
                                          "mean_frac_wrong": float(np.mean(fws)),
                                          "template_pi": tpls[site].reliability(),
                                          "pages": len(fws)})
        log(f"  seed {seed} done in {time.time() - t0:.0f}s")

    _write_rows(os.path.join(out_dir, "reeval_results.csv"), rows)
    _write_summary(os.path.join(out_dir, "reeval_summary.csv"), rows)
    _write_rows(os.path.join(out_dir, "reeval_pages.csv"), page_rows)
    _write_rows(os.path.join(out_dir, "reeval_sites.csv"), site_rows)
    rel = reliability_report(page_rows, rows)
    rel["_site_template_pi"] = site_template_report(site_rows)
    with open(os.path.join(out_dir, "reeval_reliability.json"), "w") as f:
        json.dump(rel, f, indent=2)
    log(json.dumps(rel, indent=1))
    return out_dir


def _template_dis(rec, keys, tpl) -> float:
    """Fraction of filled fields whose node is outside the site's inferred template slot."""
    hit = tot = 0
    for f, c in rec.fields.items():
        t = tpl.fields.get(f)
        if c.node is None or not c.text or t is None or t.slot is None:
            continue
        tot += 1
        hit += int(keys[c.node] == t.slot)
    return 1.0 - hit / tot if tot else 0.0


def site_template_report(site_rows):
    """Does a low template adherence (pi) flag the sites where extraction is bad?"""
    out = {}
    groups = defaultdict(list)
    for r in site_rows:
        groups[(r["setting"], r["method"])].append(r)
    for (sname, method), rs in groups.items():
        out.setdefault(sname, {})[method] = {
            "n_sites": len(rs),
            "spearman_negpi_vs_error": spearman([-r["template_pi"] for r in rs], [r["mean_frac_wrong"] for r in rs]),
            "auroc_bad_site": auc([r["mean_frac_wrong"] >= 0.5 for r in rs], [-r["template_pi"] for r in rs]),
        }
    return out


def reliability_report(page_rows, rows):
    groups = defaultdict(list)
    for r in page_rows:
        groups[(r["setting"], r["method"])].append(r)
    rel = {}
    for (sname, method), rs in groups.items():
        fw = [r["frac_wrong"] for r in rs]
        out = {"pages": len(rs), "mean_frac_wrong": float(np.mean(fw)),
               "rate_any": float(np.mean([x > 0 for x in fw])), "rate_half": float(np.mean([x >= 0.5 for x in fw]))}
        # site level: mean signal per (seed, site) vs that site's mean fraction of wrong fields
        by_site = defaultdict(list)
        for r in rs:
            by_site[(r["seed"], r["site"])].append(r)
        site_err = [np.mean([r["frac_wrong"] for r in v]) for v in by_site.values()]
        for sig in SIGNALS:
            s = [r[sig] for r in rs]
            site_sig = [np.mean([r[sig] for r in v]) for v in by_site.values()]
            out[sig] = {"spearman": spearman(s, fw), "auroc_half": auc([x >= 0.5 for x in fw], s),
                        "auroc_any": auc([x > 0 for x in fw], s),
                        "site_spearman": spearman(site_sig, site_err), "n_sites": len(site_err)}
        rel.setdefault(sname, {})[method] = out
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--u-grid", type=float, nargs="+", default=[0.05, 0.01],
                    help="first value is the main setting; the rest are a sensitivity sweep")
    a = ap.parse_args()
    for c in a.configs:
        t0 = time.time()
        try:
            reeval(c, a.u_grid)
        except Exception as e:  # keep going with the next config
            import traceback
            traceback.print_exc()
            print(f"##### {c} FAILED: {e}")
            continue
        print(f"##### {c} re-evaluated in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
