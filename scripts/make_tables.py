"""
Collect results of several experiments into the paper's main tables.

    python scripts/make_tables.py results/swde_camera_semfeat results/swde_book_semfeat \
        results/swde_auto_semfeat results/swde_job_semfeat --out results/tables

Writes:
  main_f1.csv / main_f1.tex           macro F1 (mean ± std over seeds) per vertical and method, clean
  violations.csv                      share of records violating the schema in force
  drift.csv                           macro F1 per drift setting
  reliability.csv / reliability.tex   AUROC of each risk signal per vertical and setting
"""

import argparse
import csv
import json
import os
from collections import OrderedDict

METHOD_NAMES = OrderedDict([
    ("baseline:xpath", "XPath wrapper"),
    ("argmax", "GNN only"),
    ("filtered", "GNN + node filters"),
    ("ilp-uncalibrated", "ILP, hand-set constraints"),
    ("ilp", "ILP, calibrated (ours)"),
    ("ilp-soft", "ILP, soft constraints (ours)"),
])
SIGNALS = OrderedDict([("risk", "Risk (ours)"), ("disagreement", "Disagreement"),
                       ("neg_margin", "Stability margin"), ("old_sigma_neg", "$\\sigma(P)$ (ICWE)")])


def _tex(s: str) -> str:
    return s if "$" in s else s.replace("_", "\\_")


def load(exp_dir):
    rows = list(csv.DictReader(open(os.path.join(exp_dir, "summary.csv"))))
    rel = json.load(open(os.path.join(exp_dir, "reliability.json")))
    return rows, rel


def cell(row, key, scale=100):
    m, s = row.get(f"{key}_mean"), row.get(f"{key}_std")
    if m in (None, ""):
        return ""
    return f"{scale * float(m):.1f} ± {scale * float(s):.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--out", default="results/tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = OrderedDict((os.path.basename(e.rstrip("/\\")), load(e)) for e in a.experiments)
    names = list(data)

    def table(metric, setting="clean"):
        out = []
        for m, label in METHOD_NAMES.items():
            r = [label]
            for n in names:
                row = next((x for x in data[n][0] if x["method"] == m and x["setting"] == setting), None)
                r.append(cell(row, metric) if row else "")
            if any(r[1:]):
                out.append(r)
        return out

    def write_csv(path, header, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    def write_tex(path, header, rows, caption, label):
        cols = "l" + "c" * (len(header) - 1)
        lines = ["\\begin{table}[t]", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\centering",
                 "\\scriptsize", f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}", "\\toprule",
                 " & ".join(f"\\textbf{{{_tex(h)}}}" for h in header) + " \\\\", "\\midrule"]
        for r in rows:
            lines.append(" & ".join(_tex(x).replace("±", "$\\pm$") for x in r) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")

    header = ["Method"] + names
    f1 = table("f1/macro")
    write_csv(os.path.join(a.out, "main_f1.csv"), header, f1)
    write_tex(os.path.join(a.out, "main_f1.tex"), header, f1,
              "Cross-site macro F1 (\\%, mean $\\pm$ std over seeds) on unseen sites.", "tab:main-f1")
    write_csv(os.path.join(a.out, "violations.csv"), header, table("violation_rate"))

    settings = sorted({x["setting"] for n in names for x in data[n][0]})
    drift_rows = []
    for n in names:
        for s in settings:
            for m, label in METHOD_NAMES.items():
                row = next((x for x in data[n][0] if x["method"] == m and x["setting"] == s), None)
                if row:
                    drift_rows.append([n, s, label, cell(row, "f1/macro"), cell(row, "violation_rate")])
    write_csv(os.path.join(a.out, "drift.csv"), ["experiment", "setting", "method", "macro F1", "violations"], drift_rows)

    rel_rows = []
    for n in names:
        for s, sig in data[n][1].items():
            rel_rows.append([n, s] + [f"{sig[k]['auroc']:.3f}" if k in sig else "" for k in SIGNALS]
                            + [f"{sig['risk']['error_rate']:.3f}"])
    rh = ["experiment", "setting"] + list(SIGNALS.values()) + ["error rate"]
    write_csv(os.path.join(a.out, "reliability.csv"), rh, rel_rows)
    write_tex(os.path.join(a.out, "reliability.tex"), rh, rel_rows,
              "AUROC of label-free signals for detecting wrong records.", "tab:reliability")
    for r in f1:
        print("  ".join(f"{x:<28}" if i == 0 else f"{x:>14}" for i, x in enumerate(r)))
    print(f"tables written to {a.out}")


if __name__ == "__main__":
    main()
