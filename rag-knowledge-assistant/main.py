"""
RAG Knowledge Assistant — Main Runner
Run this to test the full pipeline without Streamlit.
"""

import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def main():
    print("\n" + "="*60)
    print("  🧠  RAG KNOWLEDGE ASSISTANT — Pipeline Test")
    print("="*60 + "\n")

    from core.rag_system import RAGSystem
    from docs.corpus import DEMO_QUESTIONS

    system = RAGSystem(chunk_strategy="sentence_aware", top_k=5)
    stats = system.index()

    print(f"\n📚 Index built:")
    print(f"   Documents : {stats['n_documents']}")
    print(f"   Chunks    : {stats['n_chunks']}")
    print(f"   Emb Dim   : {stats['embedding_dim']}")
    print(f"   Avg words : {stats['avg_chunk_words']:.0f}")

    print("\n" + "="*60)
    print("RAG vs Vanilla Hallucination Test")
    print("="*60)

    test_qs = DEMO_QUESTIONS["hallucination_traps"][:3]
    for q in test_qs:
        comp = system.compare(q)
        rag = comp["rag"]
        van = comp["vanilla"]
        print(f"\nQ: {q}")
        print(f"  RAG   [grounding={rag.grounding_score:.2f}] {rag.answer[:100]}...")
        print(f"  VANILLA [grounding=0.00] {van.answer[:100]}...")
        print(f"  Improvement: +{comp['grounding_delta']:.2f}")

    print("\n" + "="*60)
    print("Store Benchmark")
    print("="*60)
    bench = system.benchmark_stores("What is the parental leave policy?")
    for store, metrics in bench.items():
        print(f"  {store:12s}: p50={metrics['p50_ms']:.2f}ms p99={metrics['p99_ms']:.2f}ms")

    print(f"""
{"="*60}
✅ Pipeline complete!

Next steps:
  Streamlit demo : streamlit run streamlit_app.py
  Set API key    : export ANTHROPIC_API_KEY=sk-...
                   (enables real LLM generation; mock used without it)

CV Talking Points:
  ✅ Built full RAG pipeline from scratch (no LangChain)
  ✅ Implemented TF-IDF embeddings + cosine similarity retrieval
  ✅ Built two vector stores: FAISS-style + ChromaDB-style
  ✅ Quantified hallucination reduction with grounding scores
  ✅ Implemented MMR (diverse retrieval) and hybrid search
  ✅ Production patterns: chunking strategies, metadata filtering
{"="*60}
    """)

if __name__ == "__main__":
    main()
