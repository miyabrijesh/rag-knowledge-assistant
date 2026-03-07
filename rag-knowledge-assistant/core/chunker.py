"""
RAG Knowledge Assistant — Document Chunking Pipeline

Implements multiple chunking strategies with overlap.
This is one of the most critical (and underappreciated) parts of RAG quality.
Bad chunking = bad retrieval = bad answers, regardless of model quality.
"""

import re
import hashlib
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("Chunker")


@dataclass
class Chunk:
    """A single text chunk ready for embedding"""
    chunk_id: str               # deterministic hash of content
    doc_id: str                 # parent document ID
    doc_title: str              # parent document title
    category: str               # HR / Technical
    text: str                   # the actual chunk text
    chunk_index: int            # position within document
    total_chunks: int           # total chunks in this document
    start_char: int             # character offset in original document
    end_char: int
    word_count: int
    strategy: str               # which chunking strategy was used
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "category": self.category,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "word_count": self.word_count,
            "strategy": self.strategy,
        }


def _make_chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
    """Deterministic chunk ID from content hash"""
    content = f"{doc_id}_{chunk_index}_{text[:50]}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def _clean_text(text: str) -> str:
    """Normalize whitespace, remove excessive blank lines"""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n ', '\n', text)
    return text.strip()


# ─────────────────────────────────────────────
# CHUNKING STRATEGIES
# ─────────────────────────────────────────────

def chunk_fixed_size(
    doc: Dict,
    chunk_size: int = 400,      # target words per chunk
    overlap: int = 80,          # overlap words between chunks
) -> List[Chunk]:
    """
    Fixed-size word chunking with overlap.

    Simplest strategy. Split every N words with M word overlap.
    Overlap prevents losing context at chunk boundaries.

    Example: chunk_size=400, overlap=80
      Chunk 1: words 0-399
      Chunk 2: words 320-719   (80 word overlap with chunk 1)
      Chunk 3: words 640-1039
    
    Pros: Simple, predictable chunk sizes
    Cons: May split mid-sentence, mid-concept
    """
    text = _clean_text(doc["content"])
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    i = 0
    chunk_idx = 0

    while i < len(words):
        chunk_words = words[i: i + chunk_size]
        chunk_text = " ".join(chunk_words)
        start_char = len(" ".join(words[:i]))
        end_char = start_char + len(chunk_text)

        chunks.append(Chunk(
            chunk_id=_make_chunk_id(doc["id"], chunk_idx, chunk_text),
            doc_id=doc["id"],
            doc_title=doc["title"],
            category=doc["category"],
            text=chunk_text,
            chunk_index=chunk_idx,
            total_chunks=-1,       # filled in below
            start_char=start_char,
            end_char=end_char,
            word_count=len(chunk_words),
            strategy="fixed_size",
        ))
        i += step
        chunk_idx += 1

    for c in chunks:
        c.total_chunks = len(chunks)

    return chunks


def chunk_sentence_aware(
    doc: Dict,
    max_chunk_words: int = 350,
    overlap_sentences: int = 2,
) -> List[Chunk]:
    """
    Sentence-aware chunking.

    Split at sentence boundaries, accumulate until size limit,
    then start a new chunk with the last N sentences as overlap.

    Pros: Never splits mid-sentence, reads naturally
    Cons: Chunk sizes vary more
    Best for: Prose documents like HR policies
    """
    text = _clean_text(doc["content"])
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    chunk_idx = 0
    current_sentences = []
    current_words = 0

    for sent in sentences:
        sent_words = len(sent.split())

        if current_words + sent_words > max_chunk_words and current_sentences:
            # Flush current chunk
            chunk_text = " ".join(current_sentences)
            chunks.append(Chunk(
                chunk_id=_make_chunk_id(doc["id"], chunk_idx, chunk_text),
                doc_id=doc["id"],
                doc_title=doc["title"],
                category=doc["category"],
                text=chunk_text,
                chunk_index=chunk_idx,
                total_chunks=-1,
                start_char=0,
                end_char=len(chunk_text),
                word_count=current_words,
                strategy="sentence_aware",
            ))
            chunk_idx += 1
            # Keep last N sentences as overlap
            current_sentences = current_sentences[-overlap_sentences:]
            current_words = sum(len(s.split()) for s in current_sentences)

        current_sentences.append(sent)
        current_words += sent_words

    # Flush remaining
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append(Chunk(
            chunk_id=_make_chunk_id(doc["id"], chunk_idx, chunk_text),
            doc_id=doc["id"],
            doc_title=doc["title"],
            category=doc["category"],
            text=chunk_text,
            chunk_index=chunk_idx,
            total_chunks=-1,
            start_char=0,
            end_char=len(chunk_text),
            word_count=current_words,
            strategy="sentence_aware",
        ))

    for c in chunks:
        c.total_chunks = len(chunks)

    return chunks


def chunk_by_section(
    doc: Dict,
    max_section_words: int = 500,
    overlap: int = 50,
) -> List[Chunk]:
    """
    Section-aware chunking (best for structured documents).

    Detects section headers (ALL CAPS lines, lines ending with ':')
    and splits at section boundaries. Long sections are sub-chunked.

    Pros: Preserves document structure, semantically coherent
    Cons: Requires structured documents
    Best for: Policy documents, technical documentation
    """
    text = _clean_text(doc["content"])
    lines = text.split('\n')

    # Detect section headers
    sections = []
    current_header = doc["title"]
    current_lines = []

    for line in lines:
        stripped = line.strip()
        is_header = (
            stripped and
            (stripped.isupper() or
             (stripped.endswith(':') and len(stripped) < 80 and len(stripped.split()) < 10))
        )
        if is_header and current_lines:
            sections.append((current_header, "\n".join(current_lines)))
            current_header = stripped
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_header, "\n".join(current_lines)))

    chunks = []
    chunk_idx = 0

    for header, section_text in sections:
        words = section_text.split()
        if not words:
            continue

        if len(words) <= max_section_words:
            combined = f"{header}\n{section_text}".strip()
            chunks.append(Chunk(
                chunk_id=_make_chunk_id(doc["id"], chunk_idx, combined),
                doc_id=doc["id"],
                doc_title=doc["title"],
                category=doc["category"],
                text=combined,
                chunk_index=chunk_idx,
                total_chunks=-1,
                start_char=0,
                end_char=len(combined),
                word_count=len(combined.split()),
                strategy="section_aware",
                metadata={"section_header": header}
            ))
            chunk_idx += 1
        else:
            # Sub-chunk long sections
            step = max_section_words - overlap
            for i in range(0, len(words), step):
                sub_words = words[i: i + max_section_words]
                sub_text = f"{header} (continued)\n" + " ".join(sub_words)
                chunks.append(Chunk(
                    chunk_id=_make_chunk_id(doc["id"], chunk_idx, sub_text),
                    doc_id=doc["id"],
                    doc_title=doc["title"],
                    category=doc["category"],
                    text=sub_text,
                    chunk_index=chunk_idx,
                    total_chunks=-1,
                    start_char=0,
                    end_char=len(sub_text),
                    word_count=len(sub_words),
                    strategy="section_aware",
                    metadata={"section_header": header}
                ))
                chunk_idx += 1

    for c in chunks:
        c.total_chunks = len(chunks)

    return chunks


# ─────────────────────────────────────────────
# MAIN CHUNKER CLASS
# ─────────────────────────────────────────────

class DocumentChunker:
    """
    Orchestrates document chunking with configurable strategy.
    In production: also handles PDF extraction, DOCX parsing, HTML cleaning.
    """

    STRATEGIES = {
        "fixed_size": chunk_fixed_size,
        "sentence_aware": chunk_sentence_aware,
        "section_aware": chunk_by_section,
    }

    def __init__(self, strategy: str = "sentence_aware", **kwargs):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(self.STRATEGIES)}")
        self.strategy = strategy
        self.kwargs = kwargs
        self.chunks_: List[Chunk] = []

    def chunk_document(self, doc: Dict) -> List[Chunk]:
        fn = self.STRATEGIES[self.strategy]
        chunks = fn(doc, **self.kwargs)
        logger.debug(f"{doc['id']}: {len(chunks)} chunks ({self.strategy})")
        return chunks

    def chunk_corpus(self, documents: List[Dict]) -> List[Chunk]:
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        self.chunks_ = all_chunks
        logger.info(
            f"Chunked {len(documents)} documents → {len(all_chunks)} chunks "
            f"| Strategy: {self.strategy} "
            f"| Avg words/chunk: {sum(c.word_count for c in all_chunks)/len(all_chunks):.0f}"
        )
        return all_chunks

    def compare_strategies(self, doc: Dict) -> Dict:
        """Compare all strategies on a single document — useful for analysis"""
        results = {}
        for name, fn in self.STRATEGIES.items():
            chunks = fn(doc)
            results[name] = {
                "n_chunks": len(chunks),
                "avg_words": sum(c.word_count for c in chunks) / len(chunks),
                "min_words": min(c.word_count for c in chunks),
                "max_words": max(c.word_count for c in chunks),
                "chunks": chunks,
            }
        return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/claude/rag_assistant')
    from docs.corpus import ALL_DOCUMENTS

    chunker = DocumentChunker(strategy="sentence_aware")
    chunks = chunker.chunk_corpus(ALL_DOCUMENTS)

    print(f"\nSample chunk:")
    print(f"  ID: {chunks[0].chunk_id}")
    print(f"  Doc: {chunks[0].doc_title}")
    print(f"  Words: {chunks[0].word_count}")
    print(f"  Text preview: {chunks[0].text[:120]}...")

    # Compare strategies on one doc
    print("\nStrategy comparison on HR Leave Policy:")
    results = chunker.compare_strategies(ALL_DOCUMENTS[0])
    for strat, info in results.items():
        print(f"  {strat:20s}: {info['n_chunks']} chunks, avg {info['avg_words']:.0f} words")
