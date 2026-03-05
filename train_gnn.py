"""
SmartScrape GNN Training Script
Запуск: python train_gnn.py
Результат: model.pt (скопируй в корень проекта SmartScrape)
"""

import json, math, random, torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split

# ── Config ──────────────────────────────────────────────────
DATA_FILE  = "data/labeled.json"   # путь к датасету
OUTPUT     = "model.pt"             # куда сохранить
EPOCHS     = 100
LR         = 0.001
SEED       = 42

# ── Feature encoding (должно совпадать с features.py) ───────
TEXT_DIM = 128
TAG_MAP  = {"div":0,"span":1,"a":2,"img":3,"p":4,
            "h1":5,"h2":6,"li":7,"ul":8,"table":9,
            "form":10,"input":11,"button":12}
TAG_DIM  = len(TAG_MAP) + 1
INPUT_DIM = TEXT_DIM + 4 + TAG_DIM   # 147

def encode_text(text):
    vec  = [0.0] * TEXT_DIM
    text = str(text).lower().strip()
    if not text:
        return torch.zeros(TEXT_DIM)
    for n in (2, 3):
        for i in range(len(text) - n + 1):
            h = 0
            for ch in text[i:i+n]:
                h = (h * 31 + ord(ch)) & 0xFFFFFF
            vec[h % TEXT_DIM] += 1.0
    norm = math.sqrt(sum(v*v for v in vec)) or 1.0
    return torch.tensor([v/norm for v in vec], dtype=torch.float32)

def encode_visual(bbox, pw=1200.0, ph=1080.0):
    if not bbox or len(bbox) < 4:
        return torch.zeros(4)
    x,y,w,h = [float(b) for b in bbox[:4]]
    return torch.tensor([x/pw, y/ph, w/pw, h/ph], dtype=torch.float32)

def encode_tag(tag):
    idx = TAG_MAP.get(str(tag).lower(), TAG_DIM-1)
    v   = torch.zeros(TAG_DIM)
    v[idx] = 1.0
    return v

LABEL_MAP = {"price": 0, "title": 1, "other": 2}

def page_to_graph(nodes):
    feats, labels = [], []
    for node in nodes:
        f = torch.cat([
            encode_text(node['text']),
            encode_visual(node.get('bbox', [])),
            encode_tag(node.get('tag', 'div')),
        ])
        feats.append(f)
        labels.append(LABEL_MAP.get(node['label'], 2))

    x  = torch.stack(feats)
    n  = len(nodes)
    ys = [float(nd.get('bbox',[0,0])[1]) if nd.get('bbox') else 0 for nd in nodes]

    # KNN edges k=3 по близости y-координат
    edges = []
    for i in range(n):
        dists = sorted([(abs(ys[i]-ys[j]), j) for j in range(n) if j != i])[:3]
        for _, j in dists:
            edges += [[i, j], [j, i]]
    edges = list({tuple(e) for e in edges})
    ei = (torch.tensor(edges, dtype=torch.long).t().contiguous()
          if edges else torch.zeros(2, 0, dtype=torch.long))

    return Data(x=x, edge_index=ei), torch.tensor(labels, dtype=torch.long)


# ── Model ────────────────────────────────────────────────────
class SmartScrapeGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(INPUT_DIM, 64)
        self.conv2 = GCNConv(64, 32)
        self.fc    = torch.nn.Linear(32, 3)

    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = F.relu(self.conv1(x, ei))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, ei))
        return F.log_softmax(self.fc(x), dim=1)


# ── Main ─────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    random.seed(SEED)

    print(f"Loading data: {DATA_FILE}")
    with open(DATA_FILE, encoding='utf-8') as f:
        pages = json.load(f)

    graphs, labels_list = [], []
    for p in pages:
        if not p['nodes']:
            continue
        g, l = page_to_graph(p['nodes'])
        graphs.append(g)
        labels_list.append(l)

    total_nodes = sum(l.shape[0] for l in labels_list)
    print(f"Pages: {len(graphs)} | Nodes: {total_nodes}")

    idx = list(range(len(graphs)))
    tr, val = train_test_split(idx, test_size=0.2, random_state=SEED)
    print(f"Train: {len(tr)} pages | Val: {len(val)} pages\n")

    model     = SmartScrapeGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    best_acc  = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for i in tr:
            optimizer.zero_grad()
            out  = model(graphs[i])
            loss = F.nll_loss(out, labels_list[i])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            model.eval()
            correct = total = 0
            tp = {0:0, 1:0, 2:0}
            fn = {0:0, 1:0, 2:0}
            with torch.no_grad():
                for i in val:
                    preds = model(graphs[i]).argmax(dim=1)
                    y     = labels_list[i]
                    correct += (preds == y).sum().item()
                    total   += y.shape[0]
                    for cls in range(3):
                        tp[cls] += ((preds == cls) & (y == cls)).sum().item()
                        fn[cls] += ((preds != cls) & (y == cls)).sum().item()

            acc     = correct / total
            r_title = tp[1] / (tp[1] + fn[1]) if (tp[1] + fn[1]) > 0 else 0
            r_price = tp[0] / (tp[0] + fn[0]) if (tp[0] + fn[0]) > 0 else 0

            print(f"Epoch {epoch:3d} | Loss: {total_loss/len(tr):.3f} | "
                  f"Val Acc: {acc:.3f} | "
                  f"Recall title: {r_title:.2f} | Recall price: {r_price:.2f}")

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), OUTPUT)

    print(f"\n{'='*50}")
    print(f"Best validation accuracy: {best_acc:.3f}")
    print(f"Model saved: {OUTPUT}")

    # ── Per-page evaluation ──────────────────────────────────
    print(f"\n── Per-page results (val set) ──")
    model.load_state_dict(torch.load(OUTPUT))
    model.eval()
    page_correct = 0

    for i in val:
        url   = pages[i]['url'].split('/')[4][:28]
        nodes = pages[i]['nodes']
        with torch.no_grad():
            preds = model(graphs[i]).argmax(dim=1).tolist()

        pred_title = next((nodes[j]['text'][:35] for j,p in enumerate(preds) if p==1), 'NOT FOUND')
        pred_price = next((nodes[j]['text'][:15] for j,p in enumerate(preds) if p==0), 'NOT FOUND')
        true_title = next((n['text'][:35] for n in nodes if n['label']=='title'), '?')
        true_price = next((n['text'][:15] for n in nodes if n['label']=='price'), '?')

        t_ok = true_title[:15].lower() in pred_title.lower()
        p_ok = true_price[:8] in pred_price
        ok   = t_ok and p_ok
        page_correct += ok

        status = "✅" if ok else "❌"
        print(f"  {status} {url}")
        if not t_ok:
            print(f"       TITLE pred: '{pred_title}'")
            print(f"       TITLE true: '{true_title}'")
        if not p_ok:
            print(f"       PRICE pred: '{pred_price}'")
            print(f"       PRICE true: '{true_price}'")

    n_val = len(val)
    print(f"\n{'='*50}")
    print(f"Page-level accuracy: {page_correct}/{n_val} = {page_correct/n_val*100:.0f}%")
    print(f"\n✅ Done! Copy '{OUTPUT}' to your SmartScrape project root.")
    print(f"   Then run: streamlit run app.py")


if __name__ == "__main__":
    main()
