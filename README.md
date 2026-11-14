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
pytest -q tests/                                         # 21 tests
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

## Render with FitLayout (visual features)

SWDE ships HTML only, so the main SWDE run uses DOM structure and reading
order. To add real geometry, fonts and spatial edges, render the pages with
FitLayout's docker CLI (puppeteer backend):

```bash
docker pull fitlayout/fitlayout-cli                                   # Docker Desktop must be running
python scripts/render_swde.py camera --sites newegg --max-pages-per-site 3      # smoke test: 3 pages
python scripts/render_swde.py camera --max-pages-per-site 200 --threads 6       # ~2,000 pages
python -m smartscrape run configs/experiments/swde_camera_rendered200.yaml      # rendered
python -m smartscrape run configs/experiments/swde_camera_html200.yaml          # same pages, HTML only
```

The script copies each page with offline fixes (no `<base>`, scripts or
external CSS/images, which no longer exist for 2011 pages), serves them on
`http://host.docker.internal:8765/` from the Python process, runs
`fitlayout-cli BATCH`, and parses the exported XML box trees into
`data/render/camera/pages.jsonl`. Re-running skips rendered pages. On first
run Windows may ask to allow Python through the firewall: allow it on
private networks so the container can reach the server.

The two `*200` configs share pages, splits and schema, so the difference
between them is exactly what rendering adds.

## Several verticals overnight, then the paper tables

```bash
python scripts/download_swde.py book auto job
python scripts/run_suite.py configs/experiments/swde_book_semfeat.yaml \
    configs/experiments/swde_auto_semfeat.yaml configs/experiments/swde_job_semfeat.yaml
python scripts/make_tables.py results/swde_camera_semfeat results/swde_book_semfeat \
    results/swde_auto_semfeat results/swde_job_semfeat --out results/tables
```

`run_suite.py` keeps going if one config fails and logs to `results/<name>.log`.
`make_tables.py` writes CSV and LaTeX (booktabs) tables: main F1, violations,
drift and reliability. `scripts/adaptive_gate.py` re-scores test sites with
trained models to analyse, per site, when cross-field constraints help.

## Baselines (external tools, same data, same metrics)

`smartscrape/baselines/` wraps external extractors behind one interface
(`fit(train_pages)`, `extract(page) -> {field: value}`). Baselines read the
page's raw HTML, never SmartScrape's parsed nodes, and are scored by the same
`evaluate.py`, including the violation rate and `grounded_rate` (share of
predicted values that occur verbatim on the page, i.e. no hallucination).

| Name | Tool | Needs pages of the test site |
|---|---|---|
| `xpath` | induced per-site XPath wrapper (lxml), with the usual index-free fallback | yes |
| `autoscraper` | AutoScraper, rules learned from example values (experimental: fails on SWDE's archived HTML) | yes |
| `llm` | our transparent LLM extractor: `ollama` (local), `openai` (any compatible endpoint), `anthropic` | no |
| `scrapegraphai` | ScrapeGraphAI `SmartScraperGraph` (thin adapter, not yet run end-to-end) | no |
| `crawl4ai` | Crawl4AI `LLMExtractionStrategy` (thin adapter, not yet run end-to-end) | no |

Per-site wrappers only make sense when they may see the test site, so they run
in the `in_site` split (`configs/experiments/swde_camera_insite.yaml`) and in
temporal drift experiments; in `cross_site` they are skipped with a log line.
Synthetic drift changes the parsed page, not the HTML file, so baselines are
not run on synthetic drift settings. LLM cost is computed only from the prices
you put in the config, with the date you looked them up.

## Efficiency benchmark (E7)

```bash
python -m smartscrape bench --model results/swde_camera_insite/model_seed0.pt \
    --schema results/swde_camera_insite/schema_calibrated_seed0.yaml \
    --pages cache/pages_swde_<hash>.jsonl --sizes 100 1000 10000 --out results/bench.csv
```

Per page and per stage (features+graph, GNN, ILP): mean and p50/p95 latency,
pages/second, peak RSS, peak VRAM, and the correlation of latency with page size.

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
