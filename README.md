# 🕸️ SmartScrape

**A Neuro-Symbolic Framework for Web Information Extraction**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

SmartScrape combines a **Graph Neural Network (GNN)** for node scoring with an **Integer Linear Programming (ILP)** constraint solver to extract structured information from web pages — reliably, auditably, and without breaking when page templates change.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **GNN Inference** | 2-layer GCN encodes DOM nodes using text, visual geometry, and tag features |
| 🔒 **ILP Constraint Solver** | Enforces uniqueness, footer exclusion, product zone, and format constraints |
| 📊 **Proof-Carrying Output** | Every record includes which constraints were applied and whether any were violated |
| 📡 **Drift Detection** | Stability metric σ(P) monitors extraction confidence and triggers active learning |
| 🔬 **Ablation Study** | Toggle between ILP and Greedy modes to see the constraint benefit live |
| 🏷️ **Annotation Tool** | Built-in Streamlit labeling UI to build your own training dataset |

---

## 🏗️ Architecture

```
URL
 │
 ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  FitLayout API  │────▶│  Feature Graph   │────▶│  GNN (GCN)  │
│ (Puppeteer/VIPS)│     │  Builder (KNN)   │     │  147-dim    │
└─────────────────┘     └──────────────────┘     └──────┬──────┘
                                                         │ scores
                                                         ▼
                                                ┌─────────────────┐
                                                │   ILP Solver    │
                                                │  (OR-Tools)     │
                                                │  Γ1 Uniqueness  │
                                                │  Γ2 Footer trap │
                                                │  Γ3 Product zone│
                                                │  Γ4 Format      │
                                                └────────┬────────┘
                                                         │
                                                         ▼
                                               ┌──────────────────┐
                                               │  Proof-Carrying  │
                                               │  Record + σ(P)   │
                                               └──────────────────┘
```

---

## 📈 Results

Evaluated on **51 annotated pages** from books.toscrape.com:

| Method | Title Acc. | Price Acc. | Both Correct | Violations |
|---|---|---|---|---|
| Greedy (no Γ) | 83% | 80% | 77% | 4 |
| **SmartScrape ILP (Γ)** | **97%** | **97%** | **93%** | **0** |

The ILP solver eliminates all constraint violations and improves accuracy by +14 pp (title) and +17 pp (price) over the greedy baseline.

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/yourusername/smartscrape.git
cd smartscrape
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux / Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your FitLayout token:
# FITLAYOUT_TOKEN=Bearer eyJ...
```

### 3. Run the demo

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 🏋️ Training Your Own Model

### Step 1 — Annotate pages

```bash
streamlit run annotate.py
```

Label each page node as `title`, `price`, or `other`. Saves to `data/labeled.json`.
Aim for at least 20-50 pages for good results.

### Step 2 — Train the GNN

```bash
python train_gnn.py
```

Output: `model.pt` — copy this to the project root.

### Step 3 — Run with trained model

```bash
streamlit run app.py
```

The header will show **Model trained: True** when the model is loaded correctly.

---

## 📁 Project Structure

```
smartscrape/
├── app.py                      # Main Streamlit demo application
├── annotate.py                 # Data annotation tool
├── train_gnn.py                # GNN training script
├── config.py                   # Configuration (reads from .env)
├── model.pt                    # Trained GNN weights (not in repo)
├── data/
│   └── labeled.json            # Annotated training data
└── src/
    ├── pipeline_fixed.py       # Main extraction pipeline
    ├── integration/
    │   └── fitlayout.py        # FitLayout API client
    ├── learning/
    │   ├── gnn_model.py        # GCN architecture
    │   ├── features.py         # Node feature encoder
    │   ├── graph_builder.py    # Graph construction (KNN edges)
    │   └── drift_monitor.py    # σ(P) stability metric
    └── reasoning/
        ├── solver_fixed.py     # ILP constraint solver (OR-Tools)
        └── engine.py           # Greedy inference (ablation baseline)
```

---

## 🔧 Configuration

| Variable | Default | Description |
|---|---|---|
| `FITLAYOUT_TOKEN` | — | FitLayout API Bearer token (required) |
| `FITLAYOUT_API_URL` | `https://layout.fit.vutbr.cz/api` | FitLayout endpoint |
| `SMARTSCRAPE_MODEL_PATH` | `model.pt` | Path to trained GNN weights |
| `FOOTER_THRESHOLD` | `0.80` | Footer zone cutoff (fraction of page height) |
| `STABILITY_THRESHOLD` | `0.60` | σ(P) threshold for drift detection |

---

## 🎮 Demo Scenarios

Once the app is running, you can explore:

1. **Live Extraction** — paste any `books.toscrape.com` URL and watch the pipeline extract title and price with bounding-box overlay
2. **Constraint Proof** — inspect the proof object showing active constraints Γ1–Γ4
3. **Drift Simulation** — use the chaos engineering slider to inject noise and watch σ(P) drop below the threshold
4. **Ablation Study** — toggle ILP ↔ Greedy in the sidebar and see accuracy change in real time
5. **Batch Evaluation** — run the benchmark over all 51 labeled pages and view F1 scores and σ(P) distribution

---

## 🧮 Formal Constraints

The ILP solver enforces:

```prolog
Γ1 UNIQUENESS:    ∑ x[i, Price] ≤ 1,   ∑ x[i, Title] ≤ 1
Γ2 FOOTER TRAP:   y(n) > 0.8 × PageHeight  ⇒  x[n, Title] = x[n, Price] = 0
Γ3 PRODUCT ZONE:  y(n) > 500px           ⇒  x[n, Title] = x[n, Price] = 0
Γ4 FORMAT:        x[n, Price] = 1        ⇒  HasCurrency(n) ∧ IsNumeric(n)
```

---

## 📄 Citation

If you use SmartScrape in your research, please cite:

```bibtex
@inproceedings{imanov2026smartscrape,
  title     = {SmartScrape: A Neuro-Symbolic Web Information Extraction Demo},
  author    = {Imanov, Ganjali},
  booktitle = {Proceedings of the 26th International Conference on Web Engineering (ICWE)},
  year      = {2026},
  address   = {Lyon, France},
  publisher = {Springer}
}
```

---

## 🙏 Acknowledgements

This work was developed at the Faculty of Information Technology, Brno University of Technology.
Thanks to doc. Ing. Radek Burget, Ph.D. for supervision and to Prof. RNDr. Alexandr Meduna, CSc. for guidance.

The system integrates with [FitLayout](https://github.com/FitLayout/FitLayout) — a web page segmentation and analysis framework developed at BUT FIT.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
