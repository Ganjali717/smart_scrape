"""
SmartScrape — Data Annotation Tool
Run: streamlit run annotate.py
"""

import streamlit as st
import json
from pathlib import Path

from src.integration.fitlayout import FitLayoutClient
from src.learning.graph_builder import FitLayoutParser

st.set_page_config(page_title="SmartScrape Annotator", page_icon="🏷️", layout="wide")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
LABELS_FILE = DATA_DIR / "labeled_2.json"

LABEL_OPTIONS = ["other", "title", "price"]
LABEL_COLORS  = {"title": "#1565C0", "price": "#2E7D32", "other": "#555"}


def load_pages():
    if LABELS_FILE.exists():
        return json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    return []

def save_pages(pages):
    LABELS_FILE.write_text(
        json.dumps(pages, indent=2, ensure_ascii=False), encoding="utf-8"
    )


@st.cache_data(show_spinner="Loading nodes from FitLayout...")
def fetch_nodes(url: str):
    try:
        client = FitLayoutClient()
        parser = FitLayoutParser()
        json_data = client.get_page_content(url)
        _, nodes = parser.parse(json_data)
        if nodes:
            return nodes
    except Exception as e:
        st.warning(f"FitLayout error: {e}")
    return None


def init_annotations(nodes, existing_labels: dict):
    for i, node in enumerate(nodes):
        nid = str(node.get("id", i))
        key = f"ann_{nid}"
        if key not in st.session_state:
            st.session_state[key] = existing_labels.get(nid, "other")

def get_annotations(nodes) -> dict:
    result = {}
    for i, node in enumerate(nodes):
        nid = str(node.get("id", i))
        result[nid] = st.session_state.get(f"ann_{nid}", "other")
    return result

def set_label(node_id: str, label: str):
    st.session_state[f"ann_{node_id}"] = label


QUICK_URLS = [
    "https://books.toscrape.com/catalogue/tipping-the-velvet_999/index.html",
    "https://books.toscrape.com/catalogue/sapiens-a-brief-history-of-humankind_996/index.html",
    "https://books.toscrape.com/catalogue/the-house-by-the-lake_846/index.html",
    "https://books.toscrape.com/catalogue/sharp-objects_997/index.html",
    "https://books.toscrape.com/catalogue/soumission_998/index.html",
    "https://books.toscrape.com/catalogue/the-dirty-little-secrets-of-getting-your-dream-job_994/index.html",
    "https://books.toscrape.com/catalogue/shakespeares-sonnets_989/index.html",
    "https://books.toscrape.com/catalogue/the-boys-in-the-boat-nine-americans-and-their-epic-quest-for-gold-at-the-1936-berlin-olympics_992/index.html",
    "https://books.toscrape.com/catalogue/its-only-the-himalayas_981/index.html",
    "https://books.toscrape.com/catalogue/libertarianism-for-beginners_982/index.html",
    "https://books.toscrape.com/catalogue/mesaerion-the-best-science-fiction-stories-1800-1849_983/index.html",
    "https://books.toscrape.com/catalogue/olio_984/index.html",
    "https://books.toscrape.com/catalogue/our-band-could-be-your-life-scenes-from-the-american-indie-underground-1981-1991_985/index.html",
    "https://books.toscrape.com/catalogue/rip-it-up-and-start-again_986/index.html",
    "https://books.toscrape.com/catalogue/scott-pilgrims-precious-little-life_987/index.html",
]


st.title("🏷️ SmartScrape — Annotation Tool")

pages = load_pages()
labeled_urls = {p["url"] for p in pages}

with st.sidebar:
    st.header("📊 Progress")
    st.metric("Pages labeled", len(pages))
    st.metric("Total nodes", sum(len(p["nodes"]) for p in pages))
    st.progress(min(len(pages) / 50, 1.0), text=f"{len(pages)}/50 pages")
    st.divider()

    st.subheader("📚 Quick URLs")
    for u in QUICK_URLS:
        slug = u.split("/")[4][:28]
        done = u in labeled_urls
        label = f"✅ {slug}" if done else f"📄 {slug}"
        if st.button(label, key=f"btn_{u}", use_container_width=True):
            for k in list(st.session_state.keys()):
                if isinstance(k, str) and k.startswith("ann_"):
                    del st.session_state[k]
            st.session_state["current_url"] = u
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear ALL data", type="secondary"):
        save_pages([])
        st.rerun()


if "current_url" not in st.session_state:
    st.session_state["current_url"] = QUICK_URLS[0]

url = st.text_input("Page URL:", value=st.session_state["current_url"])

if st.button("📥 Load this URL", type="primary"):
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith("ann_"):
            del st.session_state[k]
    st.session_state["current_url"] = url
    st.rerun()

current_url = st.session_state["current_url"]

raw_nodes = fetch_nodes(current_url)
if not raw_nodes:
    st.error("No nodes loaded. Check FitLayout or network.")
    st.stop()

existing_page = next((p for p in pages if p["url"] == current_url), None)
existing_labels = (
    {n["id"]: n["label"] for n in existing_page["nodes"]}
    if existing_page else {}
)
if existing_page:
    st.info("ℹ️ Already labeled. Re-label and Save to overwrite.")

init_annotations(raw_nodes, existing_labels)

st.markdown(f"**{len(raw_nodes)} nodes** loaded.")
st.markdown("---")

ann = get_annotations(raw_nodes)
n_title = sum(1 for v in ann.values() if v == "title")
n_price = sum(1 for v in ann.values() if v == "price")

c1, c2, c3 = st.columns(3)
c1.metric("title", n_title, delta="✅" if n_title == 1 else "need 1")
c2.metric("price", n_price, delta="✅" if n_price == 1 else "need 1")
c3.metric("other", sum(1 for v in ann.values() if v == "other"))

st.markdown("---")

col_f1, col_f2 = st.columns([3, 1])
with col_f1:
    filter_text = st.text_input("🔍 Filter:", placeholder="Search node text...")
with col_f2:
    hide_empty = st.checkbox("Hide empty", value=True)

st.markdown("---")
st.subheader("🏷️ Label nodes")
st.caption("T / P / O buttons for quick tagging. The dropdown also works.")

for i, node in enumerate(raw_nodes):
    nid   = str(node.get("id", i))
    text  = node.get("text", "").strip()
    tag   = node.get("tag", "div")
    bbox  = node.get("bbox", [0, 0, 0, 0])
    y_pos = int(bbox[1]) if len(bbox) > 1 else 0

    if hide_empty and len(text) < 2:
        continue
    if filter_text and filter_text.lower() not in text.lower():
        continue

    current = st.session_state.get(f"ann_{nid}", "other")
    color   = LABEL_COLORS[current]

    col_text, col_select, col_btns = st.columns([4, 1, 1])

    with col_text:
        st.markdown(
            f"<div style='border-left:4px solid {color};padding:4px 10px;"
            f"background:#1a1a2e;border-radius:4px;margin:2px 0'>"
            f"<small style='color:#888'>#{i} &lt;{tag}&gt; y={y_pos}px</small><br>"
            f"<span style='color:#eee;font-size:13px'>{text[:130]}"
            f"{'…' if len(text)>130 else ''}</span></div>",
            unsafe_allow_html=True,
        )

    with col_select:
        st.selectbox(
            "label",
            options=LABEL_OPTIONS,
            index=LABEL_OPTIONS.index(current),
            key=f"ann_{nid}",
            label_visibility="collapsed",
        )

    with col_btns:
        b1, b2, b3 = st.columns(3)

        b1.button(
            "T",
            key=f"T_{nid}",
            type="primary" if current == "title" else "secondary",
            on_click=set_label,
            args=(nid, "title"),
        )

        b2.button(
            "P",
            key=f"P_{nid}",
            type="primary" if current == "price" else "secondary",
            on_click=set_label,
            args=(nid, "price"),
        )

        b3.button(
            "O",
            key=f"O_{nid}",
            type="primary" if current == "other" else "secondary",
            on_click=set_label,
            args=(nid, "other"),
        )

st.markdown("---")

ann = get_annotations(raw_nodes)
n_title = sum(1 for v in ann.values() if v == "title")
n_price = sum(1 for v in ann.values() if v == "price")
ready   = (n_title == 1 and n_price == 1)

if not ready:
    st.warning(f"⚠️ Need exactly 1 title and 1 price. Currently: {n_title} title, {n_price} price.")

col_save, col_info = st.columns([1, 3])

with col_save:
    if st.button("💾 Save Page", type="primary", disabled=not ready):
        labeled_nodes = []
        for idx, node in enumerate(raw_nodes):
            nid  = str(node.get("id", idx))
            text = node.get("text", "").strip()
            if len(text) < 1:
                continue
            labeled_nodes.append({
                "id":    nid,
                "text":  text,
                "tag":   node.get("tag", "div"),
                "bbox":  node.get("bbox", [0, 0, 0, 0]),
                "label": ann.get(nid, "other"),
            })
        pages = [p for p in pages if p["url"] != current_url]
        pages.append({"url": current_url, "nodes": labeled_nodes})
        save_pages(pages)
        st.success(f"✅ Saved! Total pages: {len(pages)}")
        st.balloons()

with col_info:
    if ready:
        t_node = next(
            (n for n in raw_nodes if ann.get(str(n.get("id", ""))) == "title"), None
        )
        p_node = next(
            (n for n in raw_nodes if ann.get(str(n.get("id", ""))) == "price"), None
        )
        if t_node and p_node:
            st.info(
                f"**title** → {t_node.get('text','')[:50]}  |  "
                f"**price** → {p_node.get('text','')[:20]}"
            )

with st.expander("📤 Export & Train"):
    st.code(f"python train_gnn.py --data {LABELS_FILE} --epochs 50 --output model.pt")
    if pages:
        st.download_button(
            "⬇️ Download labeled.json",
            data=json.dumps(pages, indent=2, ensure_ascii=False),
            file_name="labeled.json",
            mime="application/json",
        )