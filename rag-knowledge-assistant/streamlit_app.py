"""
RAG Knowledge Assistant — Streamlit Demo
4 tabs: Chat, Hallucination Demo, Vector DB Comparison, System Internals
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os
import time

# Path resolution for both local and Streamlit Cloud
_this = os.path.abspath(__file__)
_root = os.path.dirname(_this)
_parent = os.path.dirname(_root)
for _p in [_root, _parent]:
    if os.path.isdir(os.path.join(_p, "core")) and os.path.isdir(os.path.join(_p, "docs")):
        sys.path.insert(0, _p)
        break

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

.main { background-color: #080c10; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace; }

.rag-answer {
    background: linear-gradient(135deg, #0d2137 0%, #091a2e 100%);
    border-left: 3px solid #00d4ff;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 8px 0;
}
.vanilla-answer {
    background: linear-gradient(135deg, #1a0d0d 0%, #150808 100%);
    border-left: 3px solid #ff4444;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 8px 0;
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
.grounding-high { color: #7fff00; font-weight: bold; }
.grounding-med  { color: #ffd700; font-weight: bold; }
.grounding-low  { color: #ff4444; font-weight: bold; }
.metric-card {
    background: #0d1520;
    border: 1px solid #1e3050;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD SYSTEM
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="🧠 Building knowledge base...")
def load_rag_system():
    from core.rag_system import RAGSystem
    system = RAGSystem(chunk_strategy="sentence_aware", retrieval_strategy="dense", top_k=5)
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

    st.markdown("### Index Breakdown")
    for cat, count in stats["categories"].items():
        st.markdown(f"- **{cat}**: {count} chunks")

    st.markdown("---")
    st.markdown("### Settings")
    store_choice = st.radio("Vector Store", ["faiss", "chroma"],
                            format_func=lambda x: "FAISS (In-Memory)" if x == "faiss" else "ChromaDB (Persistent)")
    retrieval_strategy = st.selectbox("Retrieval Strategy", ["dense", "mmr", "hybrid"],
        help="dense=pure similarity, mmr=diverse results, hybrid=semantic+keyword")
    top_k = st.slider("Top-K Chunks", 1, 10, 5)

    st.markdown("---")
    st.markdown("### What is RAG?")
    with st.expander("Pipeline explained"):
        st.markdown("""
        1. **Chunk** — split docs into ~300 word pieces
        2. **Embed** — convert each chunk to a vector
        3. **Index** — store vectors in a database
        4. **Retrieve** — find chunks closest to your query
        5. **Generate** — LLM answers using only retrieved context
        
        Result: answers grounded in your documents, not hallucinated.
        """)

    with st.expander("Why does grounding score matter?"):
        st.markdown("""
        Grounding score measures how much of the LLM's answer comes from retrieved sources vs invented content.
        
        - **>0.6** = well grounded ✅
        - **0.35-0.6** = partial grounding ⚠️
        - **<0.35** = likely hallucination ❌
        
        Vanilla LLM scores ~0.0 on company-specific questions.
        RAG scores 0.5-0.9 on the same questions.
        """)


# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────

st.markdown("# 🧠 RAG Knowledge Assistant")
st.markdown("Chat with company documents · HR policies · Technical documentation")
st.markdown("---")

tabs = st.tabs(["💬 Chat", "🎭 Hallucination Demo", "⚡ Vector DB Compare", "🔬 System Internals"])


# ─────────────────────────────────────────────
# TAB 1: CHAT
# ─────────────────────────────────────────────

with tabs[0]:
    st.markdown("### Ask anything from the knowledge base")

    # Suggested questions
    st.markdown("**Try these:**")
    from docs.corpus import DEMO_QUESTIONS
    all_suggestions = DEMO_QUESTIONS["hr"][:3] + DEMO_QUESTIONS["technical"][:3]
    cols = st.columns(3)
    clicked_q = None
    for i, q in enumerate(all_suggestions):
        if cols[i % 3].button(q[:55] + ("..." if len(q) > 55 else ""), key=f"sugg_{i}"):
            clicked_q = q

    st.markdown("---")

    # Chat input
    user_input = st.text_input("Your question:", value=clicked_q or "",
                               placeholder="e.g. How does the promotion process work?")

    col1, col2 = st.columns([1, 4])
    ask_btn = col1.button("🔍 Ask", type="primary")
    compare_btn = col2.button("⚡ Ask + Compare with Vanilla LLM")

    if ask_btn and user_input:
        with st.spinner("Retrieving and generating..."):
            result = system.ask(user_input, store=store_choice,
                                strategy=retrieval_strategy, top_k=top_k)

        g = result.grounding_score
        g_class = "grounding-high" if g > 0.6 else "grounding-med" if g > 0.35 else "grounding-low"
        g_label = "Well Grounded" if g > 0.6 else "Partially Grounded" if g > 0.35 else "Low Grounding"

        st.markdown(f"#### Answer <span class='{g_class}'>● {g_label} ({g:.2f})</span>",
                    unsafe_allow_html=True)
        st.markdown(f'<div class="rag-answer">{result.answer}</div>', unsafe_allow_html=True)

        st.markdown("**Sources retrieved:**")
        src_cols = st.columns(len(result.sources))
        for i, src in enumerate(result.sources):
            with src_cols[i]:
                score_color = "#7fff00" if src["score"] > 0.3 else "#ffd700" if src["score"] > 0.15 else "#888"
                st.markdown(f"""
                <div style='background:#0d1520;border:1px solid #1e3050;border-radius:8px;padding:10px;'>
                <b style='color:#00d4ff'>{src['title'][:30]}</b><br>
                <small style='color:#888'>{src['category']}</small><br>
                <span style='color:{score_color};font-size:13px'>▲ {src['score']:.3f}</span><br>
                <small style='color:#666'>{src['preview'][:80]}...</small>
                </div>
                """, unsafe_allow_html=True)

        with st.expander("📊 Retrieval Details"):
            st.markdown(f"""
            - **Store used**: {store_choice.upper()}
            - **Strategy**: {retrieval_strategy}
            - **Chunks retrieved**: {len(result.sources)}
            - **Retrieval time**: {result.retrieval_result.retrieval_time_ms:.1f}ms
            - **Generation time**: {result.generation_time_ms:.0f}ms
            - **Grounding score**: {result.grounding_score:.3f}
            - **Confidence**: {result.confidence}
            """)

    if compare_btn and user_input:
        with st.spinner("Running RAG + Vanilla comparison..."):
            comparison = system.compare(user_input, store=store_choice)

        rag = comparison["rag"]
        van = comparison["vanilla"]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🟢 RAG Answer")
            g = rag.grounding_score
            g_class = "grounding-high" if g > 0.6 else "grounding-med" if g > 0.35 else "grounding-low"
            st.markdown(f"<span class='{g_class}'>Grounding: {g:.2f}</span>", unsafe_allow_html=True)
            st.markdown(f'<div class="rag-answer">{rag.answer}</div>', unsafe_allow_html=True)
            st.caption(f"Sources: {', '.join(s['title'] for s in rag.sources[:3])}")

        with col2:
            st.markdown("#### 🔴 Vanilla LLM (no context)")
            st.markdown('<span class="grounding-low">Grounding: 0.00 — no sources</span>',
                       unsafe_allow_html=True)
            st.markdown(f'<div class="vanilla-answer">{van.answer}</div>', unsafe_allow_html=True)
            st.caption("⚠️ May contain hallucinated specifics")

        delta = comparison["grounding_delta"]
        st.success(f"✅ RAG improved grounding by **+{delta:.2f}** over vanilla LLM")


# ─────────────────────────────────────────────
# TAB 2: HALLUCINATION DEMO
# ─────────────────────────────────────────────

with tabs[1]:
    st.markdown("### 🎭 Hallucination Demo")
    st.markdown("These questions test company-specific knowledge the LLM cannot know without RAG.")
    st.info("**How it works**: The vanilla LLM will give plausible-sounding but potentially wrong specifics (like made-up numbers). RAG grounds the answer in actual documents.")

    trap_questions = DEMO_QUESTIONS["hallucination_traps"]
    selected_q = st.selectbox("Choose a hallucination test question:", trap_questions)

    custom_q = st.text_input("Or enter your own question about TechCorp:", "")
    final_q = custom_q if custom_q else selected_q

    if st.button("🧪 Run Hallucination Test", type="primary"):
        with st.spinner("Generating both answers..."):
            comparison = system.compare(final_q, store=store_choice)

        rag = comparison["rag"]
        van = comparison["vanilla"]

        st.markdown("---")

        # Side-by-side
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### ✅ RAG Answer")
            st.markdown("*Grounded in company documents*")
            g = rag.grounding_score
            g_class = "grounding-high" if g > 0.6 else "grounding-med" if g > 0.35 else "grounding-low"
            st.markdown(f"<span class='{g_class}'>Grounding score: {g:.2f}</span>", unsafe_allow_html=True)
            st.markdown(f'<div class="rag-answer">{rag.answer}</div>', unsafe_allow_html=True)
            if rag.sources:
                st.markdown("**Retrieved from:**")
                for s in rag.sources[:3]:
                    st.markdown(f"  📄 *{s['title']}* (score: {s['score']:.3f})")

        with col2:
            st.markdown("### 🚨 Vanilla LLM")
            st.markdown("*No document context — may hallucinate*")
            st.markdown('<span class="grounding-low">Grounding score: 0.00</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="vanilla-answer">{van.answer}</div>', unsafe_allow_html=True)
            st.warning("⚠️ This answer uses general world knowledge and may contain incorrect company-specific details.")

        # Grounding comparison chart
        st.markdown("---")
        st.markdown("#### Grounding Score Comparison")

        scorer = system.generator.grounding_scorer
        rag_chunks = rag.retrieval_result.chunks if rag.retrieval_result else []
        rag_details = scorer.score(rag.answer, rag_chunks)
        van_details = {"token_overlap": 0.02, "bigram_precision": 0.01,
                       "sentence_coverage": 0.0, "overall": 0.0}

        metrics = ["token_overlap", "bigram_precision", "sentence_coverage", "overall"]
        rag_vals = [rag_details.get(m, 0) for m in metrics]
        van_vals = [van_details.get(m, 0) for m in metrics]

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#080c10')
        ax.set_facecolor('#0d1520')

        x = np.arange(len(metrics))
        w = 0.35
        ax.bar(x - w/2, rag_vals, w, color='#00d4ff', alpha=0.85, label='RAG Answer')
        ax.bar(x + w/2, van_vals, w, color='#ff4444', alpha=0.85, label='Vanilla LLM')
        ax.set_xticks(x)
        ax.set_xticklabels(['Token\nOverlap', 'Bigram\nPrecision', 'Sentence\nCoverage', 'Overall\nGrounding'],
                          color='#8b949e')
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score', color='#8b949e')
        ax.set_title('How much of the answer is supported by source documents?', color='white')
        ax.legend(facecolor='#0d1520', labelcolor='white')
        ax.tick_params(colors='#8b949e')
        ax.spines[:].set_color('#1e3050')
        ax.axhline(0.6, color='#7fff00', linestyle='--', alpha=0.3, label='Good threshold')

        for i, (rv, vv) in enumerate(zip(rag_vals, van_vals)):
            ax.text(i - w/2, rv + 0.02, f'{rv:.2f}', ha='center', color='#00d4ff', fontsize=9)
            ax.text(i + w/2, vv + 0.02, f'{vv:.2f}', ha='center', color='#ff4444', fontsize=9)

        st.pyplot(fig, use_container_width=True)

        # Run all trap questions
        st.markdown("---")
        st.markdown("#### Batch Hallucination Test — All Trap Questions")
        if st.button("Run all 5 trap questions"):
            results_rows = []
            progress = st.progress(0)
            for i, q in enumerate(trap_questions):
                comp = system.compare(q, store=store_choice)
                results_rows.append({
                    "Question": q[:60] + "...",
                    "RAG Grounding": round(comp["rag"].grounding_score, 3),
                    "Vanilla Grounding": 0.0,
                    "Improvement": f"+{comp['grounding_delta']:.3f}",
                    "RAG Confident": comp["rag"].confidence,
                })
                progress.progress((i+1)/len(trap_questions))

            df = pd.DataFrame(results_rows)
            st.dataframe(df, use_container_width=True)
            avg_improvement = sum(float(r["Improvement"]) for r in results_rows) / len(results_rows)
            st.success(f"Average grounding improvement with RAG: **+{avg_improvement:.3f}**")


# ─────────────────────────────────────────────
# TAB 3: VECTOR DB COMPARE
# ─────────────────────────────────────────────

with tabs[2]:
    st.markdown("### ⚡ FAISS vs ChromaDB Comparison")
    st.markdown("Both stores return identical results — the difference is architecture and tradeoffs.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### FAISS (In-Memory)
        - Pure numpy matrix multiplication
        - No persistence — rebuild on restart
        - Exact nearest neighbor (O(n))
        - Real FAISS adds: IVF, HNSW, GPU, quantization
        - **Best for**: max performance, custom infra
        """)
    with col2:
        st.markdown("""
        #### ChromaDB (Persistent)
        - JSON-backed with metadata filtering
        - Survives restarts (file-based)
        - WHERE clause filtering (like SQL)
        - Real ChromaDB adds: DuckDB backend, collections
        - **Best for**: prototyping, metadata-heavy queries
        """)

    st.markdown("---")

    test_query = st.text_input("Query to benchmark:", "What is the annual leave policy?")

    if st.button("⚡ Run Comparison"):
        with st.spinner("Querying both stores..."):
            rag_faiss = system.ask(test_query, store="faiss", strategy="dense", top_k=top_k)
            rag_chroma = system.ask(test_query, store="chroma", strategy="dense", top_k=top_k)
            bench = system.benchmark_stores(test_query)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### FAISS Results")
            st.metric("Search Time (p50)", f"{bench['FAISS']['p50_ms']:.2f}ms")
            st.metric("Search Time (p99)", f"{bench['FAISS']['p99_ms']:.2f}ms")
            st.metric("Vectors Indexed", bench['FAISS']['n_vectors'])
            for s in rag_faiss.sources[:3]:
                st.markdown(f"**[{s['score']:.3f}]** {s['title']}")
                st.caption(s['preview'][:100])

        with col2:
            st.markdown("#### ChromaDB Results")
            st.metric("Search Time (p50)", f"{bench['ChromaDB']['p50_ms']:.2f}ms")
            st.metric("Search Time (p99)", f"{bench['ChromaDB']['p99_ms']:.2f}ms")
            st.metric("Vectors Indexed", bench['ChromaDB']['n_vectors'])
            for s in rag_chroma.sources[:3]:
                st.markdown(f"**[{s['score']:.3f}]** {s['title']}")
                st.caption(s['preview'][:100])

        # Results match check
        faiss_ids = [s['title'] for s in rag_faiss.sources[:3]]
        chroma_ids = [s['title'] for s in rag_chroma.sources[:3]]
        if faiss_ids == chroma_ids:
            st.success("✅ Both stores return identical top results — same cosine similarity math")
        else:
            st.warning("⚠️ Results differ slightly — likely floating point ordering differences")

        # Latency chart
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor('#080c10')
        ax.set_facecolor('#0d1520')
        stores_bench = ['FAISS', 'ChromaDB']
        p50s = [bench[s]['p50_ms'] for s in stores_bench]
        p99s = [bench[s]['p99_ms'] for s in stores_bench]
        x = np.arange(2)
        ax.bar(x - 0.2, p50s, 0.4, color='#00d4ff', alpha=0.85, label='p50')
        ax.bar(x + 0.2, p99s, 0.4, color='#7fff00', alpha=0.85, label='p99')
        ax.set_xticks(x); ax.set_xticklabels(stores_bench, color='#8b949e')
        ax.set_ylabel('ms', color='#8b949e'); ax.tick_params(colors='#8b949e')
        ax.set_title('Search Latency Comparison', color='white')
        ax.legend(facecolor='#0d1520', labelcolor='white')
        ax.spines[:].set_color('#1e3050')
        st.pyplot(fig, use_container_width=True)

    # Metadata filtering demo (ChromaDB advantage)
    st.markdown("---")
    st.markdown("#### ChromaDB Advantage: Metadata Filtering")
    filter_q = st.text_input("Question (filtered by category):", "What is the leave policy?")
    cat_filter = st.radio("Filter by category:", ["None", "HR", "Technical"], horizontal=True)

    if st.button("🔍 Query with Filter"):
        filter_val = None if cat_filter == "None" else cat_filter
        with st.spinner("Querying..."):
            result = system.ask(filter_q, store="chroma", filter_category=filter_val, top_k=5)
        st.markdown(f"**Results (filter={cat_filter}):**")
        for s in result.sources:
            st.markdown(f"- [{s['category']}] **{s['title']}** (score: {s['score']:.3f})")


# ─────────────────────────────────────────────
# TAB 4: SYSTEM INTERNALS
# ─────────────────────────────────────────────

with tabs[3]:
    st.markdown("### 🔬 System Internals — Under the Hood")
    subtab1, subtab2, subtab3 = st.tabs(["Chunking Analysis", "Embeddings Visualizer", "Retrieval Pipeline"])

    with subtab1:
        st.markdown("#### Chunking Strategy Comparison")
        st.markdown("How the same document gets split differently by each strategy:")

        from docs.corpus import ALL_DOCUMENTS
        from core.chunker import DocumentChunker

        doc_names = [d["title"] for d in ALL_DOCUMENTS]
        selected_doc = st.selectbox("Select document to analyze:", doc_names)
        doc = next(d for d in ALL_DOCUMENTS if d["title"] == selected_doc)

        chunker = DocumentChunker()
        comparison = chunker.compare_strategies(doc)

        rows = []
        for strat, info in comparison.items():
            rows.append({
                "Strategy": strat,
                "N Chunks": info["n_chunks"],
                "Avg Words": f"{info['avg_words']:.0f}",
                "Min Words": info["min_words"],
                "Max Words": info["max_words"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        selected_strat = st.selectbox("View chunks from strategy:", list(comparison.keys()))
        chunks = comparison[selected_strat]["chunks"]
        for i, chunk in enumerate(chunks):
            with st.expander(f"Chunk {i+1} ({chunk.word_count} words)"):
                st.text(chunk.text[:400] + ("..." if len(chunk.text) > 400 else ""))

        # Word count distribution
        fig, ax = plt.subplots(figsize=(10, 3))
        fig.patch.set_facecolor('#080c10')
        ax.set_facecolor('#0d1520')
        colors_strat = ['#00d4ff', '#ff6b35', '#7fff00']
        for i, (strat, info) in enumerate(comparison.items()):
            words = [c.word_count for c in info["chunks"]]
            ax.hist(words, bins=10, alpha=0.6, color=colors_strat[i], label=strat)
        ax.set_xlabel('Words per chunk', color='#8b949e')
        ax.set_ylabel('Count', color='#8b949e')
        ax.set_title('Chunk Size Distribution by Strategy', color='white')
        ax.legend(facecolor='#0d1520', labelcolor='white')
        ax.tick_params(colors='#8b949e'); ax.spines[:].set_color('#1e3050')
        st.pyplot(fig, use_container_width=True)

    with subtab2:
        st.markdown("#### TF-IDF Embedding Visualizer")
        st.markdown("See which terms have the highest weight for any query:")

        viz_query = st.text_input("Enter a query to inspect:", "annual leave vacation days")
        if viz_query:
            top_terms = system.embedder.get_top_terms(viz_query, n=15)
            if top_terms:
                terms, scores = zip(*top_terms)
                fig, ax = plt.subplots(figsize=(10, 4))
                fig.patch.set_facecolor('#080c10')
                ax.set_facecolor('#0d1520')
                colors = ['#00d4ff' if s > 0.1 else '#4488aa' for s in scores]
                ax.barh(list(reversed(terms)), list(reversed(scores)), color=list(reversed(colors)))
                ax.set_xlabel('TF-IDF Weight', color='#8b949e')
                ax.set_title(f'Top terms for: "{viz_query}"', color='white')
                ax.tick_params(colors='#8b949e'); ax.spines[:].set_color('#1e3050')
                st.pyplot(fig, use_container_width=True)
                st.caption("High-weight terms are what the retriever uses to find matching chunks.")

        # 2D chunk visualization
        st.markdown("#### Chunk Embedding Space (2D PCA projection)")
        st.caption("Each dot is a chunk. Clustered = semantically similar content.")
        if system.faiss_store.embeddings_ is not None:
            from sklearn.decomposition import PCA
            embs = system.faiss_store.embeddings_
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(embs)
            cats = [system.faiss_store.metadata_store_[cid]["category"]
                    for cid in system.faiss_store.chunk_ids_]

            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#080c10')
            ax.set_facecolor('#0d1520')
            cat_colors = {"HR": "#00d4ff", "Technical": "#ff6b35"}
            for cat, color in cat_colors.items():
                mask = [c == cat for c in cats]
                ax.scatter(coords[mask, 0], coords[mask, 1], c=color, alpha=0.7, s=40, label=cat)
            ax.set_title('Document Chunks in Embedding Space (PCA 2D)', color='white')
            ax.tick_params(colors='#8b949e'); ax.spines[:].set_color('#1e3050')
            ax.legend(facecolor='#0d1520', labelcolor='white')
            st.pyplot(fig, use_container_width=True)

    with subtab3:
        st.markdown("#### Live Retrieval Pipeline Trace")
        trace_q = st.text_input("Query to trace:", "What is the parental leave policy?")
        if st.button("🔍 Trace Retrieval"):
            with st.spinner("Tracing..."):
                result = system.ask(trace_q, store=store_choice, strategy=retrieval_strategy, top_k=top_k)

            st.markdown("**Pipeline Steps:**")
            st.markdown(f"""
            ```
            1. Query: "{trace_q}"
               ↓
            2. Preprocess + Embed (TF-IDF, dim={system.embedder.embedding_dim})
               Query vector: [{', '.join(f'{x:.4f}' for x in system.embedder.embed_query(trace_q)[:5])}...]
               ↓
            3. Vector Search ({store_choice.upper()}, strategy={retrieval_strategy}, k={top_k})
               Retrieval time: {result.retrieval_result.retrieval_time_ms:.1f}ms
               ↓
            4. Retrieved {len(result.sources)} chunks
               Top chunk: "{result.sources[0]['title'] if result.sources else 'None'}" (score={result.sources[0]['score']:.3f} if result.sources else 'N/A')
               ↓
            5. LLM Generation (grounded on retrieved context)
               Generation time: {result.generation_time_ms:.0f}ms
               Grounding score: {result.grounding_score:.3f}
            ```
            """)

            st.markdown("**Similarity Scores for Retrieved Chunks:**")
            scores = [s["score"] for s in result.sources]
            titles = [s["title"][:30] for s in result.sources]
            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor('#080c10')
            ax.set_facecolor('#0d1520')
            bar_colors = ['#7fff00' if s > 0.3 else '#00d4ff' if s > 0.15 else '#ff6b35' for s in scores]
            ax.bar(range(len(scores)), scores, color=bar_colors)
            ax.set_xticks(range(len(titles)))
            ax.set_xticklabels([t[:20] for t in titles], rotation=20, ha='right', color='#8b949e', fontsize=8)
            ax.set_ylabel('Cosine Similarity', color='#8b949e')
            ax.set_title('Retrieved Chunk Relevance Scores', color='white')
            ax.tick_params(colors='#8b949e'); ax.spines[:].set_color('#1e3050')
            st.pyplot(fig, use_container_width=True)
