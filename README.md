# SmartScrape v2

**Constraint-guided neural web information extraction.**

A relational graph neural network scores the text nodes of a web page; an
integer linear program then picks the best *record* that satisfies a
declarative schema: one title, one price, the price near the title, nothing
from the footer, and so on. Every record comes with an audit trail
(provenance) and a label-free risk score that flags pages for review.

```
page (raw HTML or FitLayout render)
  └─ text nodes + DOM / reading-order / spatial relations      smartscrape/loaders.py, graph.py
       └─ R-GCN scorer: s(n, f) = log p(f|n) − log p(none|n)    smartscrape/model.py, train.py
            └─ constraint library compiled to an ILP             smartscrape/schema.py, solver.py
                 ├─ record + provenance                          smartscrape/extract.py
                 └─ reliability: stability margin, disagreement  smartscrape/reliability.py
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # pick the CUDA build for your GPU
pip install -r requirements.txt
pytest -q tests/                                         # 14 tests, < 1 s
```

## Run an experiment

```bash
python scripts/download_swde.py camera                  # ~10 MB; or: all
python -m smartscrape run configs/experiments/swde_camera_quick.yaml   # ~3 min on CPU
python -m smartscrape run configs/experiments/swde_camera.yaml         # full: 3 seeds, all pages, GPU
```

Outputs in `results/<name>/`:

| File | Content |
|---|---|
| `results.csv` | one row per seed × setting (clean / each drift) × method: per-field F1, macro F1, exact match, violation rate, ms/page |
| `summary.csv` | mean and std over seeds |
| `reliability.json` | AUROC / AUPRC of each risk signal for predicting a wrong record |
| `calibration_seed*.json` | which constraints survived calibration on the training sites, with their support |
| `provenance_sample.json` | full ILP records with competitors, active and relaxed constraints |
| `model_seed*.pt` | trained scorer bundles, usable by `extract` |

### Methods compared

| Name | What it does |
|---|---|
| `argmax` | best node per field, no constraints (GNN-only baseline) |
| `filtered` | node-local constraints only (format, regions), then best node per field |
| `ilp` | full constraint set, parameters calibrated on the training sites |
| `ilp-uncalibrated` | same constraints with the hand-written parameters |
| `ilp-<kind>+<kind>` | ablation keeping only the listed constraint kinds |

### Drift settings

Applied to the **input page**, never to model outputs: `wrapper_insertion`,
`tag_rename`, `sibling_reorder`, `distractor_block` (look-alike "related
products" block, half of the time without any region hint), and combinations.

## Extract from a page

```bash
python -m smartscrape extract --model results/swde_camera/model_seed0.pt \
    --schema results/swde_camera/schema_calibrated_seed0.yaml --html some_page.html
python -m smartscrape extract ... --url https://example.com/product   # rendered by FitLayout (FITLAYOUT_TOKEN in .env)
```

Returns the record, its violations (normally none), a risk score with a
`needs_review` flag, and the provenance trace.

## Schemas

`configs/schemas/*.yaml` declare fields (`text`, `price`, `number`, `date`,
`url`; cardinality `one` or `optional`) and constraints: `mutual_exclusion`,
`format`, `exclude_regions`, `position_band`, `proximity`, `ordering`. Each
constraint has a `rank`; when a page makes the program infeasible, constraints
are relaxed from the lowest rank up and the relaxation is recorded.

## First measurements (quick config, CPU, 1 seed — not paper numbers)

SWDE camera, 60 pages per site, trained on 3 sites, tested on 6 unseen sites:

| Method | macro F1 clean | violation rate clean | macro F1, all drifts combined |
|---|---|---|---|
| argmax | 0.589 | 0.642 | 0.376 |
| filtered | 0.600 | 0.503 | 0.378 |
| ilp (calibrated) | 0.599 | 0.250 | 0.357 |
| ilp-uncalibrated | 0.544 | 0.014 | 0.306 |

What this already shows: hand-set cross-field constraints remove violations
but cost F1 because they do not hold on every site (e.g. Amazon puts the
price above the title); calibration on training sites recovers the F1. The
scorer is the bottleneck in this small run (price F1 ≈ 0.31). The full
configuration on GPU is the next step.

## Legacy code

`src/`, `app.py`, `annotate.py`, `train_gnn.py` are the ICWE prototype and are
kept unchanged for reference. New work goes into `smartscrape/`.
