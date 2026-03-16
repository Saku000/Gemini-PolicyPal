"""
PolicyPal — Interactive ChatGPT-Style Q&A with Optimized Dashboard
Focus: Real-time user feedback and clear data visualization
"""
import os
import sys
import json
import shutil
import pathlib
import subprocess
import numpy as np
from scipy.spatial.distance import cosine
from typing import Dict, Any
import pdfplumber

import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv

# ── 1. SETUP & CSS ────────────────────────────────────────────────────────────
load_dotenv()
st.set_page_config(
    page_title="PolicyPal — Your Insurance, Simplified",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

try:
    current_dir = pathlib.Path(__file__).parent
    css_path = current_dir / "styles.css"
    _css = css_path.read_text(encoding="utf-8")

    # 强制修正对话框边框对齐和间距
    ui_fixes = """
    [data-testid="stChatInput"] { 
        margin-bottom: 2.5rem !important;
        border-radius: 20px !important;
    }
    .stMainBlockContainer { padding-bottom: 5rem !important; }
    """
    _fonts = '<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">'
    st.markdown(_fonts, unsafe_allow_html=True)
    st.markdown(f"<style>{_css}\n{ui_fixes}</style>", unsafe_allow_html=True)
except Exception as e:
    st.warning(f"styles.css loading error: {e}")

def _get_api_key():
    k = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    if k: return k
    st.error("⚠️ Add GEMINI_API_KEY to your .env file.")
    st.stop()

API_KEY = _get_api_key()

# ── 2. BACKEND IMPORTS ────────────────────────────────────────────────────────
import core
from policy_paths import PROJECT_ROOT, POLICY_A_DIR, POLICY_B_DIR, COMPARE_DIR
from prod_index import build_policy_index
from auto_analysis import analyze_policy_document
from prod_compare import build_policy_summary, compare_policies_prod
from compare_policies import build_radar_chart, compare_policies_llm, DIMENSIONS

# ── 3. STATE & UTILS ──────────────────────────────────────────────────────────
def open_folder(path: pathlib.Path):
    abs_path = str(path.resolve())
    path.mkdir(parents=True, exist_ok=True)
    try:
        if sys.platform == "win32": os.startfile(abs_path)
        elif sys.platform == "darwin": subprocess.Popen(["open", abs_path])
        else: subprocess.Popen(["xdg-open", abs_path])
    except Exception as e:
        st.error(f"Cannot open folder: {e}")

for k, v in {
    "page": "dashboard",
    "analysis": None, "policy_text": None,
    "chat_history": [], "comparison": None,
    "cmp_name_a": "Policy A", "cmp_name_b": "Policy B",
    "compare_last_answer": "", "a_store": "", "b_store": ""
}.items():
    if k not in st.session_state: st.session_state[k] = v

QA_PDF_DIR = PROJECT_ROOT / "data" / "qa_policies"
QA_CHUNKS_PATH = PROJECT_ROOT / "storage" / "qa_parsed_chunks.json"
QA_VECTOR_STORE_PATH = PROJECT_ROOT / "storage" / "qa_vector_store.json"
for p in [QA_PDF_DIR, QA_CHUNKS_PATH.parent]: p.mkdir(parents=True, exist_ok=True)

def extract_text_from_folder(folder_path: str, max_chars: int = 40000) -> str:
    text_parts = []
    folder = pathlib.Path(folder_path)
    if not folder.exists() or not folder.is_dir(): return ""
    for pdf_file in folder.glob("*.pdf"):
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text: text_parts.append(page_text)
        except Exception: pass
    return "\n".join(text_parts)[:max_chars]

def build_qa_index_from_folder(folder_path: str):
    payload = core.step3_ingest_to_json(input_dir=str(folder_path), output_path=str(QA_CHUNKS_PATH))
    chunks = payload.get("chunks", [])
    if chunks:
        ids = [c["chunk_id"] for c in chunks]; docs = [c["text"] for c in chunks]
        metadatas = [{"doc_name": c.get("doc_name", ""), "page_start": core.extract_page_range(c.get("text", ""))[0] or -1} for c in chunks]
        embeddings = core.embed_texts_openai(docs, api_key=API_KEY)
        store = {"ids": ids, "documents": docs, "metadatas": metadatas, "embeddings": embeddings}
        with open(QA_VECTOR_STORE_PATH, "w", encoding="utf-8") as f: json.dump(store, f)
    else:
        raise ValueError("No text extracted.")

def query_rag(question: str):
    if not QA_VECTOR_STORE_PATH.exists(): return "Index not found.", []
    with open(QA_VECTOR_STORE_PATH, "r", encoding="utf-8") as f: store = json.load(f)
    q_emb = core.embed_texts_openai([question], api_key=API_KEY)[0]
    dists = [float(cosine(q_emb, emb)) for emb in store["embeddings"]]
    idx = np.argsort(dists)[:3]
    retrieval = {"ids": [[store["ids"][i] for i in idx]], "documents": [[store["documents"][i] for i in idx]], "metadatas": [[store["metadatas"][i] for i in idx]], "distances": [[dists[i] for i in idx]]}
    context_text, sources, _ = core._build_context_from_retrieval(retrieval)
    from google.genai import types
    client = core._openai_client(API_KEY)
    resp = client.models.generate_content(model=core.CHAT_MODEL, contents=f"Question: {question}\n\nContext: {context_text}", config=types.GenerateContentConfig(system_instruction="Use context only. Keep it conversational.", temperature=0.2))
    return getattr(resp, "text", "").strip(), sources

# ── 4. VISUAL COMPONENTS ──────────────────────────────────────────────────────
def pal_svg(size=44, state="default"):
    s = size; h = int(s * 1.2)
    grad = f'<linearGradient id="pg{s}{state}" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#4F46E5"/><stop offset="100%" stop-color="#7C3AED"/></linearGradient>'
    shield = f'<path d="M{s*.5} {s*.04} C{s*.5} {s*.04} {s*.94} {s*.15} {s*.97} {s*.22} L{s*.97} {s*.5} C{s*.97} {s*.78} {s*.5} {s*1.15} {s*.5} {s*1.15} C{s*.5} {s*1.15} {s*.03} {s*.78} {s*.03} {s*.5} L{s*.03} {s*.22} C{s*.06} {s*.15} {s*.5} {s*.04} {s*.5} {s*.04}Z" fill="url(#pg{s}{state})"/>'
    return f'<svg width="{s}" height="{h}" viewBox="0 0 {s} {h}" xmlns="http://www.w3.org/2000/svg"><defs>{grad}</defs>{shield}</svg>'

def donut_chart(areas):
    labels = list(areas.keys()); values = list(areas.values())
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.55, textinfo="label+percent", marker=dict(colors=["#6366F1", "#8B5CF6", "#06B6D4", "#10B981"])))
    fig.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(color="#A89FCC")), paper_bgcolor="rgba(0,0,0,0)", height=450, margin=dict(t=20, b=100, l=20, r=20))
    return fig

def sparkle(): return '<svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M6 1L7 5L11 6L7 7L6 11L5 7L1 6L5 5Z" fill="#A78BFA"/></svg>'

# ── 5. NAVIGATION ─────────────────────────────────────────────────────────────
def render_nav():
    logo_col, c1, c2, c3 = st.columns([1.2, 1, 1, 1])
    with logo_col: st.markdown(f'<div class="pp-logo" style="height:44px;display:flex;align-items:center;gap:10px;padding-left:8px">{pal_svg(28)} PolicyPal</div>', unsafe_allow_html=True)
    with c1:
        if st.button("📊  Dashboard", key="n1", use_container_width=True, type="primary" if st.session_state.page == "dashboard" else "secondary"):
            st.session_state.page = "dashboard"; st.rerun()
    with c2:
        if st.button("⚖️  Compare", key="n2", use_container_width=True, type="primary" if st.session_state.page == "compare" else "secondary"):
            st.session_state.page = "compare"; st.rerun()
    with c3:
        if st.button("💬  Ask Pal", key="n3", use_container_width=True, type="primary" if st.session_state.page == "ask" else "secondary"):
            st.session_state.page = "ask"; st.rerun()

# ── 6. PAGES ──────────────────────────────────────────────────────────────────

def page_dashboard():
    an = st.session_state.analysis
    st.markdown('<div class="pp-page"><div class="orb-tr"></div><div class="orb-bl"></div>', unsafe_allow_html=True)

    if an is None:
        st.markdown(f'''<div class="hero-wrap"><div class="hero-h">Your insurance,<br><span class="hero-grad">simplified.</span></div></div>''', unsafe_allow_html=True)
        uc, _ = st.columns([2, 1])
        with uc:
            st.markdown(f'''<div class="upload-zone-wrapper"><div class="upload-zone-inner"><div>{pal_svg(64)}</div><div class="upload-card-text"><h3>Analyze Policy Folder</h3><p>Reads from <code>data/qa_policies</code></p></div></div></div>''', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("📂 Open QA Folder", use_container_width=True): open_folder(QA_PDF_DIR)
            with c2:
                if st.button("✨ Analyze & Index", type="primary", use_container_width=True):
                    with st.spinner("Extracting & Indexing..."):
                        text = extract_text_from_folder(str(QA_PDF_DIR))
                        if text:
                            st.session_state.policy_text = text
                            st.session_state.analysis = analyze_policy_document(text, API_KEY)
                            build_qa_index_from_folder(str(QA_PDF_DIR))
                            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Banner with Action Controls
    st.markdown(f'''<div class="policy-banner">
      <div class="banner-gradient-bar"></div>
      {pal_svg(52)}
      <div style="flex-grow:1">
        <div class="policy-name">{an.get("insurer", "Your Policy")}</div>
        <div class="policy-meta">Indexed from folder &nbsp;·&nbsp; <span class="active">Active</span></div>
      </div>
      <div style="display:flex; gap:10px;">
        <button onclick="window.location.reload()" style="background:rgba(255,255,255,0.08); border:1px solid rgba(167,139,250,0.3); color:white; border-radius:12px; padding:10px 20px; cursor:pointer; font-weight:600;">🔄 Re-run Analysis</button>
      </div>
    </div>''', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="stat-card sc-1"><div class="stat-label">Deductible</div><div class="stat-value">{an.get("deductible","—")}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="stat-card sc-2"><div class="stat-label">Premium</div><div class="stat-value">{an.get("monthly_premium") or an.get("annual_premium") or "—"}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="stat-card sc-3"><div class="stat-label">Out-of-Pocket</div><div class="stat-value">{an.get("out_of_pocket_max","—")}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="stat-card sc-4"><div class="stat-label">Score</div><div class="stat-value">{an.get("risk_score",5)}/10</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="gap-md"></div>', unsafe_allow_html=True)

    # Summary Row
    st.markdown(f'''<div class="summary-card">
      <div class="sum-header"><div><h3>Plain English Summary</h3></div></div>
      <div class="sum-text">{an.get("plain_summary","")}</div>
      <div class="ideal-for">💡 <strong>Ideal for:</strong> {an.get("who_its_good_for","")}</div>
    </div>''', unsafe_allow_html=True)

    st.markdown('<div class="gap-lg"></div>', unsafe_allow_html=True)

    # Full Width Chart to prevent text cutting
    st.markdown('<div class="cc"><div class="cc-title">Coverage Composition</div>', unsafe_allow_html=True)
    areas = an.get("coverage_areas", {})
    if areas: st.plotly_chart(donut_chart(areas), use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="gap-lg"></div>', unsafe_allow_html=True)
    if st.button("🗑️ Reset and Analyze New Policy Content", use_container_width=True):
        st.session_state.analysis = None
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def page_compare():
    st.markdown('<div class="pp-page"><div class="orb-tr"></div><div class="orb-bl"></div>', unsafe_allow_html=True)
    st.markdown('<div class="cmp-headline">Policy <span>Comparison</span></div>', unsafe_allow_html=True)
    c_left, c_right = st.columns([1, 2], gap="large")
    with c_left:
        st.markdown('<div class="cc">', unsafe_allow_html=True)
        st.subheader("📁 Policy Folders")
        ca, cb = st.columns(2)
        with ca:
            if st.button("📂 Open A", use_container_width=True): open_folder(POLICY_A_DIR)
            na = st.text_input("Label A", value=st.session_state.cmp_name_a)
        with cb:
            if st.button("📂 Open B", use_container_width=True): open_folder(POLICY_B_DIR)
            nb = st.text_input("Label B", value=st.session_state.cmp_name_b)
        if st.button("✨ Run Comparison", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                idx_a = build_policy_index(str(POLICY_A_DIR), na, API_KEY, str(COMPARE_DIR))
                idx_b = build_policy_index(str(POLICY_B_DIR), nb, API_KEY, str(COMPARE_DIR))
                st.session_state.a_store, st.session_state.b_store = idx_a.store_path, idx_b.store_path
                st.session_state.comparison = compare_policies_llm(build_policy_summary(na, idx_a.store_path, API_KEY), build_policy_summary(nb, idx_b.store_path, API_KEY), API_KEY)
                st.session_state.cmp_name_a, st.session_state.cmp_name_b = na, nb
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with c_right:
        cmp = st.session_state.comparison
        if cmp:
            st.plotly_chart(build_radar_chart(cmp, st.session_state.cmp_name_a, st.session_state.cmp_name_b), use_container_width=True)
            q = st.text_area("Detailed Query", placeholder="Ask a cross-policy question...")
            if st.button("💬 Retrieve & Compare", type="primary"):
                with st.spinner("Searching..."):
                    st.session_state.compare_last_answer = compare_policies_prod(st.session_state.cmp_name_a, st.session_state.a_store, st.session_state.cmp_name_b, st.session_state.b_store, q, API_KEY)
            if st.session_state.compare_last_answer: st.markdown(st.session_state.compare_last_answer, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def page_ask():
    st.markdown('<div class="pp-page"><div class="orb-tr"></div><div class="orb-bl"></div>', unsafe_allow_html=True)

    # Header area
    head_col, btn_col = st.columns([3, 1])
    with head_col: st.markdown('<div class="hero-h" style="font-size:2.5rem; margin-bottom:1rem;">Ask <span class="hero-grad">Pal</span></div>', unsafe_allow_html=True)
    with btn_col:
        if st.button("📂 Open QA Folder", use_container_width=True): open_folder(QA_PDF_DIR)

    if not st.session_state.policy_text:
        st.info("Please index your documents in the Dashboard first.")
        return

    # ── 1. INPUT HANDLER ──
    # 用户输入后立即存入 history 并刷新
    user_q = st.chat_input("Ask about your policy documents...")
    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        st.rerun()

    # ── 2. RENDER HISTORY ──
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            _, c2 = st.columns([1, 4])
            with c2: st.markdown(f'<div style="display:flex;gap:10px;justify-content:flex-end;margin-bottom:14px"><div class="bubble-user">{msg["content"]}</div></div>', unsafe_allow_html=True)
        else:
            c1, _ = st.columns([4, 1])
            with c1:
                content = msg["content"]
                sources_html = ""
                if "sources" in msg and msg["sources"]:
                    pages = [str(s.get("page_start", "?")) for s in msg["sources"]]
                    sources_html = f'<div class="cite-pill">{sparkle()} Sources: Pages {", ".join(pages)}</div>'
                st.markdown(f'<div style="display:flex;gap:12px;margin-bottom:14px">{pal_svg(44)}<div class="bubble-pal">{content}{sources_html}</div></div>', unsafe_allow_html=True)

    # ── 3. ASYNC LOADING LOGIC ──
    # 如果最后一条是用户的，显示加载气泡并请求后台
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        c1, _ = st.columns([4, 1])
        with c1:
            placeholder = st.empty()
            placeholder.markdown(f'''
                <div style="display:flex;gap:12px;align-items:flex-start;margin-bottom:14px">
                  {pal_svg(44)}
                  <div class="bubble-pal">
                    <div class="typing-bubble">
                      <div class="t-dot"></div><div class="t-dot"></div><div class="t-dot"></div>
                    </div>
                  </div>
                </div>
            ''', unsafe_allow_html=True)

            # 后台逻辑
            last_q = st.session_state.chat_history[-1]["content"]
            ans, sources = query_rag(last_q)

            # 更新 history 并再次刷新
            st.session_state.chat_history.append({"role": "assistant", "content": ans, "sources": sources})
            st.rerun()

    st.markdown('<div style="height:100px"></div>', unsafe_allow_html=True) # 防止被底部输入框遮挡
    st.markdown("</div>", unsafe_allow_html=True)

# ── 7. EXECUTE ────────────────────────────────────────────────────────────────
render_nav()
if st.session_state.page == "dashboard": page_dashboard()
elif st.session_state.page == "compare": page_compare()
elif st.session_state.page == "ask": page_ask()