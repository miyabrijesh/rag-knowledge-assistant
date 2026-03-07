"""
RAG Knowledge Assistant — Vector Database Implementations

Two vector stores built from first principles:
  1. FAISSVectorStore  — in-memory numpy matrix, mirrors FAISS IndexFlatIP
  2. ChromaVectorStore — persistent JSON-backed store, mirrors ChromaDB API

Both implement the same VectorStore interface so they're drop-in replaceable.
This lets you compare them directly in the demo.
"""

import numpy as np
import json
import os
import pickle
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger("VectorDB")


@dataclass
class SearchResult:
    """Single retrieval result"""
    chunk_id: str
    doc_id: str
    doc_title: str
    category: str
    text: str
    score: float            # cosine similarity [0, 1]
    rank: int
    metadata: Dict


# ─────────────────────────────────────────────
# BASE INTERFACE
# ─────────────────────────────────────────────

class VectorStore:
    """Abstract interface — both stores implement this"""

    def add(self, chunk_id: str, embedding: np.ndarray, metadata: Dict): ...
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[SearchResult]: ...
    def delete(self, chunk_id: str): ...
    def count(self) -> int: ...
    def save(self, path: str): ...
    def load(self, path: str): ...


# ─────────────────────────────────────────────
# 1. FAISS-STYLE VECTOR STORE
# ─────────────────────────────────────────────

class FAISSVectorStore(VectorStore):
    """
    In-memory vector store mirroring FAISS IndexFlatIP behavior.

    Architecture:
      - Embeddings stored as numpy matrix [n_vectors × dim]
      - Search = matrix-vector multiplication → cosine similarities
      - O(n) exact search — no approximation

    Real FAISS adds:
      - IVF (Inverted File): clusters vectors, searches only nearby clusters
        → sub-linear search time at the cost of slight recall degradation
      - HNSW: graph-based ANN, best speed/recall tradeoff in practice
      - GPU support: massive parallelization for billion-scale corpora
    
    When to use real FAISS:
      > 100k vectors: IVFFlat gives ~10-50x speedup
      > 1M vectors: HNSW or IVF+PQ (quantization) for memory efficiency
      Embedding-only use case: no need for metadata filtering
    """

    def __init__(self):
        self.embeddings_: Optional[np.ndarray] = None   # [n × dim] matrix
        self.chunk_ids_: List[str] = []
        self.metadata_store_: Dict[str, Dict] = {}
        self._index_map: Dict[str, int] = {}            # chunk_id → row index
        self.name = "FAISS (In-Memory)"

    def add(self, chunk_id: str, embedding: np.ndarray, metadata: Dict):
        """Add a single embedding. In production: use add_batch for efficiency."""
        if chunk_id in self._index_map:
            return  # dedup

        embedding = np.array(embedding, dtype=np.float32)

        if self.embeddings_ is None:
            self.embeddings_ = embedding.reshape(1, -1)
        else:
            self.embeddings_ = np.vstack([self.embeddings_, embedding.reshape(1, -1)])

        idx = len(self.chunk_ids_)
        self.chunk_ids_.append(chunk_id)
        self._index_map[chunk_id] = idx
        self.metadata_store_[chunk_id] = metadata

    def add_batch(self, chunk_ids: List[str], embeddings: np.ndarray, metadatas: List[Dict]):
        """Efficient batch insert — preferred for indexing"""
        for cid, emb, meta in zip(chunk_ids, embeddings, metadatas):
            self.add(cid, emb, meta)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_category: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Nearest neighbor search.

        Since embeddings are L2-normalized, dot product = cosine similarity.
        This is the core operation: query_vec @ corpus_matrix.T
        """
        if self.embeddings_ is None or len(self.chunk_ids_) == 0:
            return []

        start = time.perf_counter()

        # Matrix-vector multiply: [n × dim] @ [dim] → [n] similarity scores
        scores = self.embeddings_ @ query_embedding.astype(np.float32)

        # Optional metadata filter (post-filtering — simple but slightly wasteful)
        if filter_category:
            for i, cid in enumerate(self.chunk_ids_):
                if self.metadata_store_[cid].get("category") != filter_category:
                    scores[i] = -1.0

        # Get top-k indices
        k_actual = min(k, len(scores))
        top_indices = np.argpartition(scores, -k_actual)[-k_actual:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        elapsed_ms = (time.perf_counter() - start) * 1000

        results = []
        for rank, idx in enumerate(top_indices):
            cid = self.chunk_ids_[idx]
            meta = self.metadata_store_[cid]
            results.append(SearchResult(
                chunk_id=cid,
                doc_id=meta.get("doc_id", ""),
                doc_title=meta.get("doc_title", ""),
                category=meta.get("category", ""),
                text=meta.get("text", ""),
                score=float(scores[idx]),
                rank=rank + 1,
                metadata={**meta, "_search_time_ms": elapsed_ms}
            ))

        return results

    def delete(self, chunk_id: str):
        """Mark as deleted (real FAISS requires rebuild — this is a soft delete)"""
        if chunk_id in self._index_map:
            idx = self._index_map[chunk_id]
            self.embeddings_[idx] = 0  # zero out (won't be returned)
            del self._index_map[chunk_id]
            del self.metadata_store_[chunk_id]

    def count(self) -> int:
        return len(self._index_map)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings_,
                'chunk_ids': self.chunk_ids_,
                'metadata_store': self.metadata_store_,
                'index_map': self._index_map,
            }, f)
        logger.info(f"FAISS store saved: {path} ({self.count()} vectors)")

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.embeddings_ = data['embeddings']
        self.chunk_ids_ = data['chunk_ids']
        self.metadata_store_ = data['metadata_store']
        self._index_map = data['index_map']
        logger.info(f"FAISS store loaded: {self.count()} vectors")


# ─────────────────────────────────────────────
# 2. CHROMADB-STYLE VECTOR STORE
# ─────────────────────────────────────────────

class ChromaVectorStore(VectorStore):
    """
    Persistent JSON-backed vector store mirroring ChromaDB's API design.

    Architecture:
      - Documents + embeddings + metadata stored in JSON files (persistence)
      - Search uses same cosine similarity but with explicit metadata support
      - Simulates ChromaDB's collection concept

    Real ChromaDB adds:
      - DuckDB backend for efficient metadata querying + vector search
      - Automatic embedding with built-in embedding functions
      - WHERE clause filtering (like SQL but for vector search)
      - Multiple collections with separate namespaces
      - ClickHouse backend for production scale

    Key difference from FAISS:
      FAISS: Pure vector math, you manage everything else
      ChromaDB: Full document store — metadata, IDs, persistence, filtering
    """

    def __init__(self, collection_name: str = "knowledge_base", persist_dir: str = "./chroma_store"):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self._documents: Dict[str, Dict] = {}   # chunk_id → {text, metadata, embedding}
        self.name = "ChromaDB (Persistent)"
        self._loaded = False

    def add(self, chunk_id: str, embedding: np.ndarray, metadata: Dict):
        """Add document with embedding and metadata"""
        self._documents[chunk_id] = {
            "embedding": embedding.tolist(),
            "metadata": metadata,
            "text": metadata.get("text", ""),
        }

    def add_batch(self, chunk_ids: List[str], embeddings: np.ndarray, metadatas: List[Dict]):
        for cid, emb, meta in zip(chunk_ids, embeddings, metadatas):
            self.add(cid, emb, meta)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        where: Optional[Dict] = None,     # metadata filter e.g. {"category": "HR"}
    ) -> List[SearchResult]:
        """
        Metadata-filtered similarity search.

        ChromaDB's killer feature: combine vector search WITH metadata filters.
        e.g., "find similar chunks WHERE category='HR' AND doc_id='hr_001'"
        This is called "filtered ANN" — crucial for multi-tenant or access-controlled RAG.
        """
        if not self._documents:
            return []

        start = time.perf_counter()

        # Apply WHERE filter first (like ChromaDB's where parameter)
        candidates = {}
        for cid, doc in self._documents.items():
            if where:
                match = all(doc["metadata"].get(k) == v for k, v in where.items())
                if not match:
                    continue
            candidates[cid] = doc

        if not candidates:
            return []

        # Stack embeddings and compute similarities
        cids = list(candidates.keys())
        emb_matrix = np.array([candidates[c]["embedding"] for c in cids], dtype=np.float32)
        scores = emb_matrix @ query_embedding.astype(np.float32)

        k_actual = min(k, len(scores))
        top_indices = np.argpartition(scores, -k_actual)[-k_actual:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        elapsed_ms = (time.perf_counter() - start) * 1000

        results = []
        for rank, idx in enumerate(top_indices):
            cid = cids[idx]
            meta = candidates[cid]["metadata"]
            results.append(SearchResult(
                chunk_id=cid,
                doc_id=meta.get("doc_id", ""),
                doc_title=meta.get("doc_title", ""),
                category=meta.get("category", ""),
                text=candidates[cid]["text"],
                score=float(scores[idx]),
                rank=rank + 1,
                metadata={**meta, "_search_time_ms": elapsed_ms}
            ))

        return results

    def get_by_id(self, chunk_id: str) -> Optional[Dict]:
        return self._documents.get(chunk_id)

    def delete(self, chunk_id: str):
        self._documents.pop(chunk_id, None)

    def count(self) -> int:
        return len(self._documents)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                "collection": self.collection_name,
                "documents": self._documents
            }, f, indent=2)
        logger.info(f"Chroma store saved: {path} ({self.count()} vectors)")

    def load(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        self._documents = {
            cid: {**doc, "embedding": np.array(doc["embedding"])}
            for cid, doc in data["documents"].items()
        }
        logger.info(f"Chroma store loaded: {self.count()} vectors")
        self._loaded = True


# ─────────────────────────────────────────────
# COMPARISON UTILITY
# ─────────────────────────────────────────────

def benchmark_stores(
    faiss_store: FAISSVectorStore,
    chroma_store: ChromaVectorStore,
    query_embedding: np.ndarray,
    k: int = 5,
    n_trials: int = 50,
) -> Dict:
    """Benchmark both stores on the same query"""
    results = {}

    for name, store in [("FAISS", faiss_store), ("ChromaDB", chroma_store)]:
        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            store.search(query_embedding, k=k)
            times.append((time.perf_counter() - t0) * 1000)

        results[name] = {
            "mean_ms": np.mean(times),
            "p50_ms": np.percentile(times, 50),
            "p99_ms": np.percentile(times, 99),
            "n_vectors": store.count(),
        }

    return results
