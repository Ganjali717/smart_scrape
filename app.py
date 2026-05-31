"""
SmartScrape — Streamlit demo.

Offline mode (default ON) runs entirely from bundled pages
(data/demo_pages.json) so the demo never depends on the FitLayout backend,
network, or a live token. Turn it off to query a live FitLayout repository.
"""

import time
import json
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from typing import Literal, cast

from src.pipeline_fixed import SmartScrapePipeline
from config import API_BASE_URL, FOOTER_THRESHOLD, STABILITY_THRESHOLD, PRODUCT_ZONE_MAX_PX

st.set_page_config(page_title="SmartScrape AI", page_icon="🕸️", layout="wide")


@st.cache_resource
def load_pipeline(reasoning_mode, use_mock, offline, use_priors):
    return SmartScrapePipeline(
        reasoning_mode=reasoning_mode, use_mock=use_mock,
        offline=offline, use_priors=use_priors,
    )


@st.cache_data
def load_demo_pages():
    try:
        with open("data/demo_pages.json", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ─────────────────────────── SIDEBAR ───────────────────────────
with st.sidebar:
    st.header("🎮 Control Panel")
    mode = st.radio("Mode:", ["🔴 Live Extraction", "📊 Batch Evaluation"])
    st.divider()

    offline = st.checkbox("Offline demo (bundled pages)", value=True,
                          help="Runs without the FitLayout backend / token.")

    st.subheader("🧠 Reasoning")
    reasoning_mode = st.radio(
        "Solver:", ["ilp", "greedy"],
        format_func=lambda x: "ILP (constraints Γ)" if x == "ilp" else "Greedy (ablation)",
    )
    use_priors = st.checkbox("Use heuristic priors (bounded)", value=True,
                             help="Off = GNN-only ablation (no priors).")
    st.divider()

    if mode == "🔴 Live Extraction":
        st.subheader("⚡ Synthetic drift")
        simulate_drift = st.checkbox("Simulate template drift (Δ)")
        drift_severity = st.slider("Drift severity", 0.0, 0.9, 0.4,
                                   disabled=not simulate_drift)
    else:
        simulate_drift = False
        drift_severity = 0.0

    st.divider()
    if offline:
        st.info("Offline mode: bundled demo pages.")
    else:
        st.success(f"Backend:\n`{API_BASE_URL}`")

selected_mode = cast(Literal["ilp", "greedy"], reasoning_mode)
pipeline = load_pipeline(selected_mode, False, offline, use_priors)
demo_pages = load_demo_pages()


def real_landscape(meta):
    """Plot REAL candidate scores from this page (not hard-coded bars).
    Red = excluded by a Γ constraint; green = admissible."""
    cands = meta.get("candidates", {})
    labels, scores, colors, tags = [], [], [], []
    for field in ("title", "price"):
        for c in cands.get(field, [])[:3]:
            labels.append(f"{field[:1].upper()}: {c['text'] or '∅'}")
            scores.append(c["score"])
            excl = c["excluded_by"]
            colors.append("#d62728" if excl else "#2ca02c")
            tags.append(",".join(excl) if excl else "admissible")
    if not labels:
        st.caption("No candidates to display.")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(range(len(labels)), scores, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Fused score (GNN prob + bounded prior)")
    ax.axvline(0, color="black", linewidth=0.8)
    for b, t in zip(bars, tags):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f"  {t}",
                va="center", fontsize=7, color="#555")
    ax.set_title("Optimization Landscape (real candidate scores)")
    st.pyplot(fig)


# ─────────────────────── MODE 1: LIVE ───────────────────────
if mode == "🔴 Live Extraction":
    st.title("🕸️ SmartScrape: Live Demo")
    st.markdown(
        f"**Reasoning:** `{reasoning_mode.upper()}` | "
        f"**Priors:** `{use_priors}` | **Model trained:** `{pipeline._model_trained}` | "
        f"**Footer τ:** `{FOOTER_THRESHOLD}`"
    )

    if offline and demo_pages:
        url = st.selectbox("Demo page:", list(demo_pages.keys()))
    else:
        url = st.text_input(
            "Target URL:",
            value="https://books.toscrape.com/catalogue/tipping-the-velvet_999/index.html",
        )

    if st.button("🚀 Analyze Page", type="primary"):
        drift = drift_severity if simulate_drift else 0.0
        with st.status("Running SmartScrape pipeline...", expanded=False) as status:
            t0 = time.time()
            result = pipeline.run(url, drift=drift)
            t1 = time.time()
            status.update(label="Extraction complete", state="complete")

        if not result:
            st.error("Extraction failed (no nodes). In live mode check the backend/token.")
            st.stop()

        meta = result["_meta"]
        if meta.get("candidates", {}).get("price") is not None and offline:
            page = demo_pages.get(url, {})
            if page.get("synthetic"):
                st.warning("⚠️ Synthetic stress-test page (footer distractors injected).")

        col1, col2 = st.columns([1, 1.4])
        with col1:
            st.subheader("📄 Extracted record")
            st.json({k: result[k] for k in ("title", "price") if k in result} | {"_meta": meta})
            st.metric("Latency", f"{t1 - t0:.2f} s")
        with col2:
            st.subheader("👁️ Visual verification")
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.set_ylim(1200, 0); ax.set_xlim(0, 1280)
            ax.grid(True, linestyle="--", alpha=0.3)
            footer_y = 1080 * FOOTER_THRESHOLD
            ax.axhspan(footer_y, 1080, alpha=0.08, color="red")
            ax.axhline(footer_y, color="red", linestyle="--", alpha=0.4)
            ax.text(10, footer_y - 8, f"Footer zone (Γ2, τ={FOOTER_THRESHOLD})",
                    color="red", fontsize=8, alpha=0.7)
            colmap = {"price": "green", "title": "royalblue"}
            for label in ("title", "price"):
                if label not in result:
                    continue
                x, y, w, h = result[label]["bbox"]
                ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=2,
                             edgecolor=colmap[label], facecolor="none"))
                ax.text(x, y - 8, f"{label.upper()} ({result[label]['confidence']:.2f})",
                        color=colmap[label], fontsize=9, weight="bold")
            st.pyplot(fig)

        st.markdown("---")
        st.header("🧮 Constraints & solver trace")
        with st.expander("1. Optimization landscape (real candidate scores)", expanded=True):
            real_landscape(meta)
            st.code(
                "Γ1 UNIQUENESS:   Σ x[i,Price] ≤ 1 ,  Σ x[i,Title] ≤ 1\n"
                f"Γ2 FOOTER:       y(n) > {FOOTER_THRESHOLD}·PageHeight ⇒ x[n,Title]=x[n,Price]=0\n"
                f"Γ3 PRODUCT ZONE: y(n) > {PRODUCT_ZONE_MAX_PX:.0f}px ⇒ x[n,Title]=x[n,Price]=0  (relaxable)\n"
                "Γ4 FORMAT:       x[n,Price]=1 ⇒ HasCurrency(n) ∧ IsNumeric(n)\n"
                "INTEGRITY:       Σ_j x[i,j] = 1   ∀i",
                language="text",
            )
        with st.expander("2. Drift σ(P)", expanded=True):
            sigma = meta["stability_score"]
            alert = meta["drift_alert"]
            c1, c2 = st.columns(2)
            with c1:
                st.latex(r"\sigma(P)=\mathbb{E}_{n\in P}\big[p_{\hat y_1}(n)-p_{\hat y_2}(n)\big]")
                st.metric("σ(P)", f"{sigma:.3f}",
                          delta="stable" if not alert else "drift",
                          delta_color="normal" if not alert else "inverse")
                st.caption(f"Threshold τ = {STABILITY_THRESHOLD} · preliminary signal")
            with c2:
                if alert:
                    st.error(f"🚨 σ < {STABILITY_THRESHOLD} → page queued for re-annotation")
                else:
                    st.success("✅ stable")
        with st.expander("3. ILP vs Greedy", expanded=False):
            st.caption(
                "On this controlled single-template set the GNN alone is already "
                "accurate; ILP's role here is to GUARANTEE schema validity (0 Γ "
                "violations) and emit the proof object, which matters under drift "
                "and at scale — not to raise accuracy on clean pages."
            )

# ─────────────────────── MODE 2: BATCH ───────────────────────
elif mode == "📊 Batch Evaluation":
    st.title("📊 Batch evaluation")
    if offline:
        urls = [(u, demo_pages[u].get("title", ""), str(demo_pages[u].get("price", "")))
                for u in demo_pages if not demo_pages[u].get("synthetic")]
        st.markdown(f"Offline: {len(urls)} bundled pages · Reasoning `{reasoning_mode.upper()}`")
    else:
        st.warning("Live batch needs the FitLayout backend; switch to offline for a safe demo.")
        urls = []

    if urls and st.button("▶️ Run benchmark", type="primary"):
        rows = []
        ct = cp = cb = viol = 0
        bar = st.progress(0)
        for i, (u, exp_t, exp_p) in enumerate(urls):
            r = pipeline.run(u)
            pt = r.get("title", {}).get("text", "") if r else ""
            pp = r.get("price", {}).get("text", "") if r else ""
            sigma = r.get("_meta", {}).get("stability_score", 0.0) if r else 0.0
            pbb = r.get("price", {}).get("bbox", [0, 0, 0, 0]) if r else [0, 0, 0, 0]
            if pbb[1] > 1080 * FOOTER_THRESHOLD:
                viol += 1
            t_ok = bool(exp_t) and exp_t.strip()[:8].lower() in pt.lower()
            p_ok = bool(exp_p) and exp_p.replace("£", "")[:5] in pp.replace("£", "")
            ct += t_ok; cp += p_ok; cb += t_ok and p_ok
            rows.append({"Page": u.split("/")[4][:22] if "/" in u else u,
                         "Got Title": pt[:28] or "—", "T": "✅" if t_ok else "❌",
                         "Got Price": pp or "—", "P": "✅" if p_ok else "❌",
                         "σ(P)": round(sigma, 3)})
            bar.progress((i + 1) / len(urls))
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        n = len(urls)
        a, b, c, d = st.columns(4)
        a.metric("Title", f"{ct/n*100:.0f}%")
        b.metric("Price", f"{cp/n*100:.0f}%")
        c.metric("Both", f"{cb/n*100:.0f}%")
        d.metric("Γ violations", str(viol))
        st.caption("Controlled single-template (books.toscrape) set — not a robustness/drift benchmark.")
