"""
RAG Knowledge Assistant — Streamlit Demo
4 tabs: Chat, Hallucination Demo, Vector DB Comparison, System Internals
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# Ensure current directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main { background-color: #080c10; }
.rag-answer {
    background: #0d2137;
    border-left: 3px solid #00d4ff;
    padding: 16px;
    border-radius: 8px;
}
.vanilla-answer {
    background: #1a0d0d;
    border-left: 3px solid #ff4444;
    padding: 16px;
    border-radius: 8px;
}
.source-chip {
    display: inline-block;
    background: #1a2940;
    border: 1px solid #00d4ff44;
    border-radius: 4px;
    padding: 2px 8px;
    margin: 2px;
    font-size: 12px;
    color: #00d4ff;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD RAG SYSTEM
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="🧠 Building knowledge base...")
def load_rag_system():
    from rag_system import RAGSystem
    system = RAGSystem(
        chunk_strategy="sentence_aware",
        retrieval_strategy="dense",
        top_k=5
    )
    system.index()
    return system


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 RAG Assistant")
    st.markdown("---")

    system = load_rag_system()
    stats = system.stats_

    st.markdown("### Knowledge Base")
    st.metric("Documents", stats["n_documents"])
    st.metric("Chunks", stats["n_chunks"])
    st.metric("Embedding Dim", stats["embedding_dim"])

    st.markdown("### Settings")

    store_choice = st.radio(
        "Vector Store",
        ["faiss", "chroma"],
        format_func=lambda x: "FAISS (In-Memory)" if x == "faiss" else "ChromaDB"
    )

    retrieval_strategy = st.selectbox(
        "Retrieval Strategy",
        ["dense", "mmr", "hybrid"]
    )

    top_k = st.slider("Top-K Chunks", 1, 10, 5)


# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────

st.title("🧠 RAG Knowledge Assistant")
st.markdown("Chat with company documents · HR policies · Technical documentation")

tabs = st.tabs([
    "💬 Chat",
    "🎭 Hallucination Demo",
    "⚡ Vector DB Compare",
    "🔬 System Internals"
])


# ─────────────────────────────────────────────
# TAB 1 — CHAT
# ─────────────────────────────────────────────

with tabs[0]:

    st.subheader("Ask anything from the knowledge base")

    from corpus import DEMO_QUESTIONS

    suggestions = DEMO_QUESTIONS["hr"][:3] + DEMO_QUESTIONS["technical"][:3]

    cols = st.columns(3)

    clicked_q = None

    for i, q in enumerate(suggestions):
        if cols[i % 3].button(q):
            clicked_q = q

    user_input = st.text_input(
        "Your question",
        value=clicked_q or ""
    )

    ask_btn = st.button("Ask")

    if ask_btn and user_input:

        with st.spinner("Retrieving answer..."):

            result = system.ask(
                user_input,
                store=store_choice,
                strategy=retrieval_strategy,
                top_k=top_k
            )

        st.markdown("### Answer")

        st.markdown(
            f'<div class="rag-answer">{result.answer}</div>',
            unsafe_allow_html=True
        )

        st.markdown("### Sources")

        for s in result.sources:
            st.markdown(
                f"- **{s['title']}** ({s['category']}) score={s['score']:.3f}"
            )


# ─────────────────────────────────────────────
# TAB 2 — HALLUCINATION DEMO
# ─────────────────────────────────────────────

with tabs[1]:

    st.subheader("Hallucination Demo")

    from corpus import DEMO_QUESTIONS

    trap_questions = DEMO_QUESTIONS["hallucination_traps"]

    q = st.selectbox(
        "Choose test question",
        trap_questions
    )

    if st.button("Run test"):

        with st.spinner("Running comparison..."):

            comparison = system.compare(q, store=store_choice)

        rag = comparison["rag"]
        van = comparison["vanilla"]

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### RAG Answer")

            st.markdown(
                f'<div class="rag-answer">{rag.answer}</div>',
                unsafe_allow_html=True
            )

        with col2:

            st.markdown("### Vanilla LLM")

            st.markdown(
                f'<div class="vanilla-answer">{van.answer}</div>',
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────
# TAB 3 — VECTOR DB COMPARISON
# ─────────────────────────────────────────────

with tabs[2]:

    st.subheader("Vector Store Comparison")

    query = st.text_input(
        "Query",
        "What is the annual leave policy?"
    )

    if st.button("Benchmark stores"):

        bench = system.benchmark_stores(query)

        st.write("FAISS p50:", bench["FAISS"]["p50_ms"])
        st.write("ChromaDB p50:", bench["ChromaDB"]["p50_ms"])


# ─────────────────────────────────────────────
# TAB 4 — SYSTEM INTERNALS
# ─────────────────────────────────────────────

with tabs[3]:

    st.subheader("Chunking Analysis")

    from corpus import ALL_DOCUMENTS
    from chunker import DocumentChunker

    doc_names = [d["title"] for d in ALL_DOCUMENTS]

    selected_doc = st.selectbox(
        "Select document",
        doc_names
    )

    doc = next(d for d in ALL_DOCUMENTS if d["title"] == selected_doc)

    chunker = DocumentChunker()

    comparison = chunker.compare_strategies(doc)

    rows = []

    for strat, info in comparison.items():

        rows.append({
            "Strategy": strat,
            "Chunks": info["n_chunks"],
            "Avg Words": info["avg_words"]
        })

    st.dataframe(pd.DataFrame(rows))
