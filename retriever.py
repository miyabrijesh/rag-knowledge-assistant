"""
RAG Knowledge Assistant — Retrieval Logic & LLM Response Generation

Implements:
  - Retriever: query → relevant chunks
  - RAGGenerator: chunks + query → grounded answer (via Anthropic API)
  - VanillaGenerator: query → ungrounded answer (for hallucination comparison)
  - GroundingScorer: measures how well the answer is supported by sources
"""

import numpy as np
import re
import os
import json
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger("RAG")


@dataclass
class RetrievalResult:
    query: str
    chunks: List  # List[SearchResult]
    retrieval_time_ms: float
    strategy: str


@dataclass
class GenerationResult:
    query: str
    answer: str
    sources: List[Dict]
    grounding_score: float          # 0-1: how well answer is supported by sources
    is_rag: bool                    # True = RAG, False = vanilla LLM
    retrieval_result: Optional[RetrievalResult]
    generation_time_ms: float
    tokens_used: int
    confidence: str                 # HIGH / MEDIUM / LOW
    cited_chunks: List[str]         # chunk IDs referenced in answer


# ─────────────────────────────────────────────
# RETRIEVER
# ─────────────────────────────────────────────

class Retriever:
    """
    Retrieves relevant chunks from the vector store for a given query.

    Strategies implemented:
      - dense: pure vector similarity (default)
      - mmr: Maximal Marginal Relevance — diverse results, less redundancy
      - hybrid: combine dense scores with keyword overlap (simulated BM25)
    """

    def __init__(self, vector_store, embedder):
        self.store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        k: int = 5,
        strategy: str = "dense",
        filter_category: Optional[str] = None,
        mmr_lambda: float = 0.7,    # 1.0 = pure relevance, 0.0 = pure diversity
    ) -> RetrievalResult:
        """Retrieve top-k chunks for query"""
        t0 = time.perf_counter()

        query_embedding = self.embedder.embed_query(query)

        if strategy == "dense":
            results = self._dense_retrieve(query_embedding, k, filter_category)
        elif strategy == "mmr":
            results = self._mmr_retrieve(query_embedding, k, mmr_lambda, filter_category)
        elif strategy == "hybrid":
            results = self._hybrid_retrieve(query, query_embedding, k, filter_category)
        else:
            results = self._dense_retrieve(query_embedding, k, filter_category)

        elapsed = (time.perf_counter() - t0) * 1000

        return RetrievalResult(
            query=query,
            chunks=results,
            retrieval_time_ms=elapsed,
            strategy=strategy,
        )

    def _dense_retrieve(self, query_embedding, k, filter_category):
        """Pure cosine similarity search"""
        kwargs = {}
        if filter_category:
            # Handle both FAISS and Chroma interfaces
            if hasattr(self.store, 'search'):
                try:
                    return self.store.search(query_embedding, k=k, filter_category=filter_category)
                except TypeError:
                    return self.store.search(query_embedding, k=k,
                                             where={"category": filter_category})
        return self.store.search(query_embedding, k=k)

    def _mmr_retrieve(self, query_embedding, k, lambda_param, filter_category):
        """
        Maximal Marginal Relevance (MMR)

        Problem with pure similarity: top results are often near-duplicates
        (same content, slightly different wording).

        MMR selects results that are:
          - Relevant to the query (high similarity to query)
          - Diverse from already-selected results (low similarity to selected set)

        Score = λ × sim(chunk, query) - (1-λ) × max_sim(chunk, selected)

        λ=0.7 gives 70% weight to relevance, 30% to diversity.
        """
        # Get more candidates than needed
        candidates = self.store.search(query_embedding, k=k * 3)
        if not candidates:
            return []

        selected = []
        selected_embeddings = []

        for _ in range(min(k, len(candidates))):
            best_score = -np.inf
            best_chunk = None
            best_idx = -1

            for i, chunk in enumerate(candidates):
                if chunk in selected:
                    continue

                # Relevance: similarity to query
                relevance = chunk.score

                # Diversity: max similarity to already-selected chunks
                if selected_embeddings:
                    chunk_emb = self.embedder.embed_query(chunk.text)
                    sims_to_selected = [
                        float(chunk_emb @ sel_emb)
                        for sel_emb in selected_embeddings
                    ]
                    max_sim = max(sims_to_selected)
                else:
                    max_sim = 0.0

                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_chunk = chunk
                    best_idx = i

            if best_chunk:
                selected.append(best_chunk)
                selected_embeddings.append(self.embedder.embed_query(best_chunk.text))

        return selected

    def _hybrid_retrieve(self, query, query_embedding, k, filter_category):
        """
        Hybrid retrieval: combine dense (semantic) + sparse (keyword) scores.

        Real implementation uses BM25 for sparse scoring.
        Here we approximate with term overlap ratio.
        Final score = α × dense_score + (1-α) × sparse_score
        """
        alpha = 0.7  # weight for dense
        dense_results = self.store.search(query_embedding, k=k * 2)

        query_tokens = set(query.lower().split())

        rescored = []
        for result in dense_results:
            chunk_tokens = set(result.text.lower().split())
            overlap = len(query_tokens & chunk_tokens)
            sparse_score = overlap / max(len(query_tokens), 1)
            hybrid_score = alpha * result.score + (1 - alpha) * sparse_score
            result.metadata["dense_score"] = result.score
            result.metadata["sparse_score"] = sparse_score
            result.score = hybrid_score
            rescored.append(result)

        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:k]


# ─────────────────────────────────────────────
# GROUNDING SCORER
# ─────────────────────────────────────────────

class GroundingScorer:
    """
    Measures how well an LLM answer is supported by source documents.

    This is a key metric for RAG quality and hallucination detection.
    A grounding score close to 0 = likely hallucination.
    A grounding score close to 1 = well-supported by sources.

    Methods:
      token_overlap: What % of answer tokens appear in sources?
      ngram_precision: What % of answer n-grams appear in sources?
      sentence_coverage: What fraction of answer sentences have a source match?
    """

    def score(self, answer: str, source_chunks: List) -> Dict:
        if not source_chunks or not answer:
            return {"overall": 0.0, "method": "none", "details": {}}

        source_text = " ".join(chunk.text for chunk in source_chunks).lower()
        answer_lower = answer.lower()

        # Method 1: Token overlap
        answer_tokens = set(re.findall(r'\b\w+\b', answer_lower))
        source_tokens = set(re.findall(r'\b\w+\b', source_text))
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                     'it', 'this', 'that', 'i', 'you', 'we', 'they', 'based', 'according'}
        content_tokens = answer_tokens - stopwords
        if content_tokens:
            token_overlap = len(content_tokens & source_tokens) / len(content_tokens)
        else:
            token_overlap = 0.0

        # Method 2: Bigram precision
        def get_ngrams(text, n):
            words = re.findall(r'\b\w+\b', text.lower())
            return set(tuple(words[i:i+n]) for i in range(len(words)-n+1))

        answer_bigrams = get_ngrams(answer, 2)
        source_bigrams = get_ngrams(source_text, 2)
        bigram_precision = (len(answer_bigrams & source_bigrams) / len(answer_bigrams)
                           if answer_bigrams else 0.0)

        # Method 3: Sentence-level coverage
        sentences = [s.strip() for s in re.split(r'[.!?]+', answer) if len(s.strip()) > 20]
        covered = 0
        for sent in sentences:
            sent_tokens = set(re.findall(r'\b\w+\b', sent.lower())) - stopwords
            if sent_tokens:
                overlap = len(sent_tokens & source_tokens) / len(sent_tokens)
                if overlap > 0.4:  # 40% token overlap = "covered"
                    covered += 1
        sentence_coverage = covered / len(sentences) if sentences else 0.0

        # Weighted average
        overall = 0.4 * token_overlap + 0.3 * bigram_precision + 0.3 * sentence_coverage

        return {
            "overall": round(overall, 3),
            "token_overlap": round(token_overlap, 3),
            "bigram_precision": round(bigram_precision, 3),
            "sentence_coverage": round(sentence_coverage, 3),
            "n_sentences_checked": len(sentences),
            "n_covered_sentences": covered,
        }


# ─────────────────────────────────────────────
# LLM GENERATOR (Anthropic API)
# ─────────────────────────────────────────────

class LLMGenerator:
    """
    Wraps the Anthropic API for both RAG and vanilla generation.
    Falls back to a template-based mock if API key not available.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self.grounding_scorer = GroundingScorer()
        self._api_available = self._check_api()

    def _check_api(self) -> bool:
        # In Streamlit Cloud: set ANTHROPIC_API_KEY in secrets
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _call_api(self, system: str, user: str, max_tokens: int = 800) -> Tuple[str, int]:
        """Call Anthropic API"""
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}]
        }).encode()

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
            }
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            text = data["content"][0]["text"]
            tokens = data["usage"]["input_tokens"] + data["usage"]["output_tokens"]
            return text, tokens

    def generate_rag_answer(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        max_tokens: int = 600,
    ) -> GenerationResult:
        """Generate grounded answer using retrieved context"""
        t0 = time.perf_counter()
        chunks = retrieval_result.chunks

        # Build context block
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk.doc_title} | {chunk.category}]\n{chunk.text}"
            )
        context = "\n\n---\n\n".join(context_parts)

        system_prompt = """You are a knowledgeable assistant for TechCorp's internal knowledge base.
Answer questions ONLY using the provided source documents.
If the answer is not in the sources, say "I don't have that information in the available documents."
Always indicate which source(s) you're drawing from.
Be concise and precise. Do not add information beyond what's in the sources."""

        user_prompt = f"""RETRIEVED DOCUMENTS:
{context}

QUESTION: {query}

Answer based only on the documents above. Cite sources by name."""

        if self._api_available:
            try:
                answer, tokens = self._call_api(system_prompt, user_prompt, max_tokens)
            except Exception as e:
                logger.warning(f"API call failed: {e}, using mock")
                answer, tokens = self._mock_rag_answer(query, chunks), 0
        else:
            answer, tokens = self._mock_rag_answer(query, chunks), 0

        grounding = self.grounding_scorer.score(answer, chunks)
        elapsed = (time.perf_counter() - t0) * 1000

        sources = [{"title": c.doc_title, "category": c.category,
                    "score": c.score, "preview": c.text[:150] + "..."} for c in chunks]

        confidence = ("HIGH" if grounding["overall"] > 0.6
                      else "MEDIUM" if grounding["overall"] > 0.35 else "LOW")

        return GenerationResult(
            query=query, answer=answer, sources=sources,
            grounding_score=grounding["overall"],
            is_rag=True,
            retrieval_result=retrieval_result,
            generation_time_ms=elapsed,
            tokens_used=tokens,
            confidence=confidence,
            cited_chunks=[c.chunk_id for c in chunks],
        )

    def generate_vanilla_answer(self, query: str, max_tokens: int = 400) -> GenerationResult:
        """Generate answer WITHOUT any retrieved context — baseline for hallucination demo"""
        t0 = time.perf_counter()

        system_prompt = """You are a helpful assistant. Answer the user's question to the best of your ability.
Be helpful and specific, even if you're not 100% certain."""

        if self._api_available:
            try:
                answer, tokens = self._call_api(system_prompt, query, max_tokens)
            except Exception as e:
                answer, tokens = self._mock_vanilla_answer(query), 0
        else:
            answer, tokens = self._mock_vanilla_answer(query), 0

        elapsed = (time.perf_counter() - t0) * 1000

        return GenerationResult(
            query=query, answer=answer, sources=[],
            grounding_score=0.0,
            is_rag=False,
            retrieval_result=None,
            generation_time_ms=elapsed,
            tokens_used=tokens,
            confidence="LOW",
            cited_chunks=[],
        )

    def _mock_rag_answer(self, query: str, chunks: List) -> str:
        """Template-based RAG answer from retrieved chunks (no API needed)"""
        if not chunks:
            return "I couldn't find relevant information in the knowledge base for this question."

        top = chunks[0]
        # Extract most relevant sentences from top chunk
        sentences = [s.strip() for s in top.text.split('.') if len(s.strip()) > 30][:3]
        excerpt = '. '.join(sentences) + '.'

        return (
            f"Based on the **{top.doc_title}** document:\n\n"
            f"{excerpt}\n\n"
            f"*Source: {top.doc_title} (relevance score: {top.score:.2f})*"
        )

    def _mock_vanilla_answer(self, query: str) -> str:
        """Simulated vanilla LLM answer — may contain plausible-sounding but wrong specifics"""
        query_lower = query.lower()

        # Deliberately gives plausible-but-wrong company-specific answers
        # to demonstrate hallucination
        if "leave" in query_lower or "vacation" in query_lower or "annual" in query_lower:
            return ("Most companies typically offer between 15-20 days of annual leave for full-time employees, "
                    "though this varies by company and region. Many organizations also provide sick leave, "
                    "usually around 10 days per year. Some companies may offer unlimited PTO policies. "
                    "Parental leave is typically 12 weeks for primary caregivers in the US.")

        elif "remote" in query_lower or "home office" in query_lower or "hybrid" in query_lower:
            return ("Most tech companies have adopted hybrid work models following the pandemic. "
                    "Common setups allow 2-3 days remote per week. Home office stipends typically "
                    "range from $500-$1,000 as a one-time setup allowance. Internet reimbursement "
                    "of around $50/month is common practice at many companies.")

        elif "salary" in query_lower or "compensation" in query_lower or "pay" in query_lower:
            return ("Software engineer salaries vary widely by company and level. Junior engineers "
                    "typically earn $70,000-$90,000, mid-level $100,000-$140,000, and senior engineers "
                    "$150,000-$200,000+ in major tech hubs. Many companies also offer equity, bonuses, "
                    "and comprehensive benefits packages.")

        elif "transformer" in query_lower or "attention" in query_lower:
            return ("Transformers use self-attention mechanisms to process sequences in parallel. "
                    "The attention formula is: Attention(Q,K,V) = softmax(QK^T/√d_k)V. "
                    "This allows the model to weigh the importance of different tokens when encoding "
                    "each position. BERT uses bidirectional attention while GPT uses causal attention.")

        elif "rag" in query_lower or "retrieval" in query_lower:
            return ("RAG (Retrieval-Augmented Generation) combines a retrieval system with a language model. "
                    "Documents are chunked, embedded, and stored in a vector database. At query time, "
                    "relevant chunks are retrieved and provided to the LLM as context. This helps ground "
                    "responses in factual information and reduces hallucinations.")

        elif "hallucination" in query_lower:
            return ("LLM hallucinations occur when models generate confident but incorrect information. "
                    "Common causes include training data noise, knowledge cutoffs, and overconfident "
                    "training signals from RLHF. Mitigations include RAG, chain-of-thought prompting, "
                    "and self-consistency sampling.")

        elif "401k" in query_lower or "benefits" in query_lower or "health" in query_lower:
            return ("Most tech companies offer competitive benefits including health insurance (medical, dental, vision), "
                    "401k matching (typically 3-6% of salary), stock options or RSUs, and learning stipends. "
                    "Health insurance premiums are usually split between employer and employee, "
                    "with companies covering 70-90% for employees.")

        elif "faiss" in query_lower or "chroma" in query_lower or "vector" in query_lower:
            return ("FAISS (Facebook AI Similarity Search) and ChromaDB are both vector databases used "
                    "for semantic search. FAISS is optimized for high-performance ANN search and supports "
                    "multiple index types. ChromaDB is more user-friendly with built-in persistence and "
                    "metadata filtering. For production at scale, Pinecone or Weaviate are often preferred.")

        else:
            return (f"This is an interesting question about '{query}'. Based on general knowledge, "
                    "I can provide some context, though for specific company policies or proprietary "
                    "information, you would need to consult the relevant documentation directly. "
                    "The answer may vary depending on your specific situation and organizational context.")
