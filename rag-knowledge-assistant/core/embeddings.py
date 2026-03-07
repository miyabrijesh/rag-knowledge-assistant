"""
RAG Knowledge Assistant — Embeddings Engine

Implements TF-IDF embeddings (no external dependencies) as the core engine.
Also provides a drop-in interface for neural embeddings (sentence-transformers)
when available. This lets you understand what embeddings DO before treating
them as a black box.

TF-IDF is surprisingly competitive for domain-specific corpora like HR docs.
"""

import numpy as np
import re
import math
import pickle
import os
from typing import List, Dict, Tuple, Optional
from collections import Counter
import logging

logger = logging.getLogger("Embeddings")


# ─────────────────────────────────────────────
# TEXT PREPROCESSING
# ─────────────────────────────────────────────

STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'shall', 'can', 'this',
    'that', 'these', 'those', 'it', 'its', 'also', 'which', 'who', 'all',
    'not', 'no', 'more', 'than', 'into', 'about', 'up', 'out', 'if',
    'they', 'their', 'them', 'we', 'our', 'you', 'your', 'he', 'she',
    'his', 'her', 'i', 'my', 'me', 'each', 'per', 'any', 'both',
}


def preprocess(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenize and normalize text"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


# ─────────────────────────────────────────────
# TF-IDF EMBEDDER
# ─────────────────────────────────────────────

class TFIDFEmbedder:
    """
    TF-IDF vectorizer built from scratch.

    How it works:
      TF(t, d) = count of term t in document d / total terms in d
      IDF(t)   = log(N / (1 + df(t)))   where df = documents containing t
      TF-IDF(t, d) = TF(t, d) × IDF(t)

    The result is a sparse vector where:
      - Common words (appear in many docs) get LOW weight
      - Rare, distinctive words get HIGH weight
      - This naturally surfaces the most informative terms

    For retrieval: embed both query and chunks, find highest cosine similarity.
    """

    def __init__(self, max_features: int = 8000, min_df: int = 1):
        self.max_features = max_features
        self.min_df = min_df
        self.vocab_: Dict[str, int] = {}        # word → column index
        self.idf_: np.ndarray = None
        self.fitted_ = False
        self.n_docs_ = 0

    def fit(self, texts: List[str]) -> 'TFIDFEmbedder':
        """Build vocabulary and IDF from corpus"""
        logger.info(f"Fitting TF-IDF on {len(texts)} texts...")

        self.n_docs_ = len(texts)
        tokenized = [preprocess(t) for t in texts]

        # Count document frequency for each term
        df = Counter()
        for tokens in tokenized:
            df.update(set(tokens))  # set: count each term once per doc

        # Filter by min_df, sort by df descending, take top max_features
        valid_terms = [(term, count) for term, count in df.items() if count >= self.min_df]
        valid_terms.sort(key=lambda x: -x[1])
        selected = valid_terms[:self.max_features]

        self.vocab_ = {term: idx for idx, (term, _) in enumerate(selected)}

        # Compute IDF for each term
        self.idf_ = np.zeros(len(self.vocab_))
        for term, idx in self.vocab_.items():
            self.idf_[idx] = math.log((1 + self.n_docs_) / (1 + df[term])) + 1  # smoothed

        self.fitted_ = True
        logger.info(f"Vocabulary size: {len(self.vocab_)} terms")
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """Convert texts to TF-IDF matrix [n_texts × vocab_size]"""
        if not self.fitted_:
            raise RuntimeError("Call fit() first")

        matrix = np.zeros((len(texts), len(self.vocab_)))

        for i, text in enumerate(texts):
            tokens = preprocess(text)
            if not tokens:
                continue
            tf_counts = Counter(tokens)
            for term, count in tf_counts.items():
                if term in self.vocab_:
                    j = self.vocab_[term]
                    tf = count / len(tokens)
                    matrix[i, j] = tf * self.idf_[j]

        # L2 normalize each row → cosine similarity = dot product
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # avoid div by zero
        return matrix / norms

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string"""
        return self.transform([query])[0]

    @property
    def embedding_dim(self) -> int:
        return len(self.vocab_)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'TFIDFEmbedder':
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get_top_terms(self, text: str, n: int = 10) -> List[Tuple[str, float]]:
        """Get top TF-IDF weighted terms for a text — great for explainability"""
        vec = self.embed_query(text)
        top_indices = np.argsort(vec)[-n:][::-1]
        idx_to_term = {v: k for k, v in self.vocab_.items()}
        return [(idx_to_term[i], vec[i]) for i in top_indices if vec[i] > 0]


# ─────────────────────────────────────────────
# NEURAL EMBEDDER (drop-in when available)
# ─────────────────────────────────────────────

class NeuralEmbedder:
    """
    Wrapper for sentence-transformers (when installed).
    Drop-in replacement for TFIDFEmbedder.

    In production: use 'BAAI/bge-small-en-v1.5' or 'all-MiniLM-L6-v2'
    These models produce 384-768 dim dense vectors that capture semantics
    far beyond TF-IDF's lexical matching.

    Example (requires: pip install sentence-transformers):
      embedder = NeuralEmbedder('all-MiniLM-L6-v2')
      embedder.fit(texts)  # no-op, model is pre-trained
      vecs = embedder.transform(texts)
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self._available = True
            logger.info(f"Neural embedder loaded: {model_name}")
        except ImportError:
            self._available = False
            logger.warning("sentence-transformers not installed. Use TFIDFEmbedder instead.")

    def fit(self, texts: List[str]) -> 'NeuralEmbedder':
        return self  # pre-trained, no fitting needed

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self._available:
            raise RuntimeError("sentence-transformers not installed")
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)

    def embed_query(self, query: str) -> np.ndarray:
        return self.transform([query])[0]


# ─────────────────────────────────────────────
# SIMILARITY UTILITIES
# ─────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_matrix(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between one query vector and all corpus vectors.
    Much faster than looping over cosine_similarity().

    Since both query and corpus are L2-normalized in TFIDFEmbedder,
    this reduces to a simple dot product: cos(q, d) = q · d
    """
    return corpus @ query  # [n_docs] similarity scores


def get_best_embedding_model():
    """Return best available embedder"""
    try:
        ne = NeuralEmbedder()
        if ne._available:
            return ne, "neural"
    except Exception:
        pass
    return TFIDFEmbedder(), "tfidf"


if __name__ == "__main__":
    texts = [
        "Employees are entitled to 25 days annual leave",
        "Self-attention computes query key value vectors",
        "The parental leave policy provides 16 weeks",
        "FAISS supports multiple index types for vector search",
    ]

    embedder = TFIDFEmbedder()
    vecs = embedder.fit_transform(texts)

    query = "how much vacation time do employees get"
    q_vec = embedder.embed_query(query)

    sims = cosine_similarity_matrix(q_vec, vecs)
    ranked = np.argsort(sims)[::-1]

    print(f"Query: '{query}'")
    print("Ranked results:")
    for i in ranked:
        print(f"  [{sims[i]:.3f}] {texts[i]}")

    print(f"\nTop TF-IDF terms for query:")
    for term, score in embedder.get_top_terms(query):
        print(f"  {term}: {score:.4f}")
