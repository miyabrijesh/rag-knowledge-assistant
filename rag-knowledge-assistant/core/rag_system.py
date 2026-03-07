"""
RAG Knowledge Assistant — System Orchestrator

Ties together: chunking → embedding → indexing → retrieval → generation
One class to rule them all.
"""

import os
import sys
import time
import logging
import numpy as np
from typing import List, Dict, Optional

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from docs.corpus import ALL_DOCUMENTS, DEMO_QUESTIONS
from core.chunker import DocumentChunker
from core.embeddings import TFIDFEmbedder
from core.retriever import Retriever, LLMGenerator, RetrievalResult, GenerationResult
from vectordb.stores import FAISSVectorStore, ChromaVectorStore, benchmark_stores

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger("RAGSystem")


class RAGSystem:
    """
    End-to-end RAG system.

    Maintains two vector stores (FAISS + ChromaDB-style) simultaneously
    for comparison. Uses TF-IDF embeddings with neural embedding interface
    available as a drop-in upgrade.
    """

    def __init__(
        self,
        chunk_strategy: str = "sentence_aware",
        retrieval_strategy: str = "dense",
        top_k: int = 5,
    ):
        self.chunk_strategy = chunk_strategy
        self.retrieval_strategy = retrieval_strategy
        self.top_k = top_k

        # Components
        self.chunker = DocumentChunker(strategy=chunk_strategy)
        self.embedder = TFIDFEmbedder(max_features=8000)
        self.faiss_store = FAISSVectorStore()
        self.chroma_store = ChromaVectorStore()
        self.generator = LLMGenerator()

        self.chunks_: List = []
        self.indexed_ = False
        self.stats_: Dict = {}

    def index(self, documents: Optional[List[Dict]] = None) -> Dict:
        """
        Full indexing pipeline:
          documents → chunks → embeddings → vector store
        """
        if documents is None:
            documents = ALL_DOCUMENTS

        logger.info(f"{'='*55}")
        logger.info(f"Indexing {len(documents)} documents...")

        t0 = time.perf_counter()

        # ── Step 1: Chunk ──────────────────────────────────────
        self.chunks_ = self.chunker.chunk_corpus(documents)
        chunk_texts = [c.text for c in self.chunks_]
        logger.info(f"Step 1 ✓ Chunks: {len(self.chunks_)}")

        # ── Step 2: Embed ──────────────────────────────────────
        embeddings = self.embedder.fit_transform(chunk_texts)
        logger.info(f"Step 2 ✓ Embeddings: {embeddings.shape} | Vocab: {self.embedder.embedding_dim}")

        # ── Step 3: Index into both stores ─────────────────────
        chunk_ids = [c.chunk_id for c in self.chunks_]
        metadatas = [{
            "doc_id": c.doc_id,
            "doc_title": c.doc_title,
            "category": c.category,
            "text": c.text,
            "chunk_index": c.chunk_index,
            "strategy": c.strategy,
        } for c in self.chunks_]

        self.faiss_store.add_batch(chunk_ids, embeddings, metadatas)
        self.chroma_store.add_batch(chunk_ids, embeddings, metadatas)

        elapsed = time.perf_counter() - t0

        self.stats_ = {
            "n_documents": len(documents),
            "n_chunks": len(self.chunks_),
            "embedding_dim": self.embedder.embedding_dim,
            "avg_chunk_words": sum(c.word_count for c in self.chunks_) / len(self.chunks_),
            "index_time_s": round(elapsed, 2),
            "chunk_strategy": self.chunk_strategy,
            "categories": {cat: sum(1 for c in self.chunks_ if c.category == cat)
                           for cat in set(c.category for c in self.chunks_)},
        }

        logger.info(f"Step 3 ✓ FAISS: {self.faiss_store.count()} | Chroma: {self.chroma_store.count()}")
        logger.info(f"Indexing complete in {elapsed:.2f}s")
        self.indexed_ = True
        return self.stats_

    def ask(
        self,
        query: str,
        store: str = "faiss",       # "faiss" or "chroma"
        strategy: Optional[str] = None,
        top_k: Optional[int] = None,
        filter_category: Optional[str] = None,
    ) -> GenerationResult:
        """
        Ask a question — retrieves relevant chunks and generates a grounded answer.
        """
        if not self.indexed_:
            self.index()

        active_store = self.faiss_store if store == "faiss" else self.chroma_store
        retriever = Retriever(active_store, self.embedder)

        retrieval = retriever.retrieve(
            query=query,
            k=top_k or self.top_k,
            strategy=strategy or self.retrieval_strategy,
            filter_category=filter_category,
        )

        result = self.generator.generate_rag_answer(query, retrieval)
        return result

    def ask_vanilla(self, query: str) -> GenerationResult:
        """Ask the same question WITHOUT retrieval — for hallucination comparison"""
        return self.generator.generate_vanilla_answer(query)

    def compare(self, query: str, store: str = "faiss") -> Dict:
        """
        Side-by-side comparison: RAG vs Vanilla LLM.
        Returns both answers with grounding scores.
        """
        rag_result = self.ask(query, store=store)
        vanilla_result = self.ask_vanilla(query)

        return {
            "query": query,
            "rag": rag_result,
            "vanilla": vanilla_result,
            "grounding_delta": rag_result.grounding_score - vanilla_result.grounding_score,
            "rag_wins": rag_result.grounding_score > vanilla_result.grounding_score,
        }

    def benchmark_stores(self, query: str) -> Dict:
        """Compare FAISS vs ChromaDB search speed"""
        q_emb = self.embedder.embed_query(query)
        return benchmark_stores(self.faiss_store, self.chroma_store, q_emb)

    def get_chunk_stats(self) -> Dict:
        """Statistics about the indexed corpus"""
        if not self.chunks_:
            return {}
        word_counts = [c.word_count for c in self.chunks_]
        return {
            "total_chunks": len(self.chunks_),
            "avg_words": np.mean(word_counts),
            "min_words": min(word_counts),
            "max_words": max(word_counts),
            "std_words": np.std(word_counts),
            "by_category": {
                cat: [c.word_count for c in self.chunks_ if c.category == cat]
                for cat in set(c.category for c in self.chunks_)
            }
        }


if __name__ == "__main__":
    system = RAGSystem()
    stats = system.index()
    print(f"\n📚 Index Stats: {stats}")

    test_questions = [
        "How many days of annual leave do employees get?",
        "What are the main components of a RAG pipeline?",
        "What is TechCorp's parental leave policy?",
    ]

    print("\n" + "="*55)
    print("RAG vs Vanilla Comparison")
    print("="*55)

    for q in test_questions:
        comp = system.compare(q)
        print(f"\nQ: {q}")
        print(f"RAG  [{comp['rag'].grounding_score:.2f}]: {comp['rag'].answer[:120]}...")
        print(f"VANILLA [0.00]: {comp['vanilla'].answer[:120]}...")
        print(f"Grounding delta: +{comp['grounding_delta']:.2f} with RAG")
