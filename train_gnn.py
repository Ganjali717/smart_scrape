"""
SmartScrape GNN training.

Uses the SHARED encoding module (src/learning/encoding.py) so the graph the
model trains on is constructed identically to the graph seen at inference.

Run:  python train_gnn.py   ->   model.pt
"""

import json
import random
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from src.learning import encoding
from src.learning.gnn_model import SmartScrapeGNN

DATA_FILE = "data/labeled_2.json"
OUTPUT = "model.pt"
EPOCHS = 50
LR = 0.001
SEED = 42


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)

    print(f"Loading data: {DATA_FILE}")
    with open(DATA_FILE, encoding="utf-8") as f:
        pages = json.load(f)

    graphs, labels_list, kept = [], [], []
    for idx, p in enumerate(pages):
        if not p.get("nodes"):
            continue
        g, l = encoding.page_to_graph(p["nodes"])
        graphs.append(g)
        labels_list.append(l)
        kept.append(idx)

    total_nodes = sum(l.shape[0] for l in labels_list)
    print(f"Pages: {len(graphs)} | Nodes: {total_nodes} | Feat dim: {encoding.INPUT_DIM}")

    pos = list(range(len(graphs)))
    tr, val = train_test_split(pos, test_size=0.2, random_state=SEED)
    print(f"Train: {len(tr)} pages | Val: {len(val)} pages\n")

    # Dataset is ~99% 'other'; weight title/price up so the model is not
    # rewarded for predicting background everywhere.
    class_weight = torch.tensor([5.0, 5.0, 1.0])  # [price, title, other]

    model = SmartScrapeGNN(input_dim=encoding.INPUT_DIM, hidden_dim=64, num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    best_score = -1.0

    def evaluate(split):
        model.eval()
        correct = total = 0
        tp = {0: 0, 1: 0, 2: 0}; fn = {0: 0, 1: 0, 2: 0}
        with torch.no_grad():
            for i in split:
                preds = model(graphs[i]).argmax(dim=1)
                y = labels_list[i]
                correct += (preds == y).sum().item(); total += y.shape[0]
                for c in range(3):
                    tp[c] += ((preds == c) & (y == c)).sum().item()
                    fn[c] += ((preds != c) & (y == c)).sum().item()
        acc = correct / total
        r_price = tp[0] / (tp[0] + fn[0]) if tp[0] + fn[0] else 0.0
        r_title = tp[1] / (tp[1] + fn[1]) if tp[1] + fn[1] else 0.0
        return acc, r_title, r_price

    for epoch in range(1, EPOCHS + 1):
        model.train(); total_loss = 0.0
        for i in tr:
            optimizer.zero_grad()
            out = model(graphs[i])
            loss = F.nll_loss(out, labels_list[i], weight=class_weight)
            loss.backward(); optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            acc, r_title, r_price = evaluate(val)
            score = (r_title + r_price) / 2
            print(f"Epoch {epoch:3d} | Loss {total_loss/len(tr):.3f} | "
                  f"NodeAcc {acc:.3f} | Recall title {r_title:.2f} price {r_price:.2f}")
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), OUTPUT)

    print(f"\nBest (title+price recall)/2: {best_score:.3f}")
    print(f"Model saved: {OUTPUT}")

    model.load_state_dict(torch.load(OUTPUT)); model.eval()
    page_ok = title_ok = price_ok = 0
    for i in val:
        nodes = pages[kept[i]]["nodes"]
        with torch.no_grad():
            preds = model(graphs[i]).argmax(dim=1).tolist()
        pred_title = next((nodes[j].get("text", "") for j, p in enumerate(preds) if p == 1), "")
        pred_price = next((nodes[j].get("text", "") for j, p in enumerate(preds) if p == 0), "")
        true_title = next((n.get("text", "") for n in nodes if n.get("label") == "title"), "")
        true_price = next((n.get("text", "") for n in nodes if n.get("label") == "price"), "")
        t_ok = bool(true_title) and true_title.strip()[:15].lower() in pred_title.lower()
        p_ok = bool(true_price) and true_price.strip()[:6] in pred_price
        title_ok += t_ok; price_ok += p_ok; page_ok += (t_ok and p_ok)
    n = len(val)
    print(f"\n-- GNN-only (no priors, no ILP) on {n} val pages --")
    print(f"Title:        {title_ok}/{n} = {title_ok/n*100:.0f}%")
    print(f"Price:        {price_ok}/{n} = {price_ok/n*100:.0f}%")
    print(f"Both correct: {page_ok}/{n} = {page_ok/n*100:.0f}%")


if __name__ == "__main__":
    main()
