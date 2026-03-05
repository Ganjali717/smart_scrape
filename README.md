# SmartScrape

A neuro-symbolic web information extraction framework.
Built for the dissertation **"Information Extraction from WWW"**
at Brno University of Technology, Faculty of Information Technology.

## Architecture

```
Phase 1: FitLayout (Puppeteer) → visual page rendering
Phase 2: FitLayoutParser       → heterogeneous graph (DOM + KNN edges)
Phase 3: SmartScrapeGNN        → node scoring (neural perception)
Phase 4: ConstraintSolver      → ILP-based constrained assignment (symbolic reasoning)
Phase 5: DriftMonitor          → σ(P) stability score + active learning
```

## Setup

```bash
pip install -r requirements.txt

# Configure credentials
cp .env.example .env
# Edit .env and set your FITLAYOUT_TOKEN
```

## Train the GNN

```bash
# Prepare labeled data (see train.py for format)
python train.py --data data/books_labeled.json --epochs 50 --output model.pt
```

## Run the demo

```bash
streamlit run app.py
```

## Ablation study

Use the **Reasoning Mode** toggle in the sidebar to compare:
- `ILP` — full neuro-symbolic pipeline with constraint enforcement (Γ)
- `Greedy` — neural-only baseline without constraints

This directly tests dissertation hypothesis **H2**: constraint optimization
reduces semantic errors (duplicate fields, footer traps).

## Key parameters (config.py)

| Parameter | Value | Reference |
|---|---|---|
| `FOOTER_THRESHOLD` | 0.80 | Dissertation Section 6.3 |
| `STABILITY_THRESHOLD` | 0.60 | Dissertation Section 6.4 |
| `k` (KNN edges) | 3 | Dissertation Section 6.1 |

## Dissertation hypotheses

- **H1** — Hybrid graph (DOM + KNN) maintains higher F1 under drift than DOM-only
- **H2** — ILP constraints reduce semantic errors vs unconstrained greedy
- **H3** — σ(P) correlates with extraction correctness and detects drift
