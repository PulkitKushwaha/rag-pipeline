from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
 
 
@dataclass
class Chunk:
    """
    Represents a single chunk of text ready for embedding.
 
    Attributes:
        content     : The chunk text
        metadata    : Inherited from parent document + chunk-level fields
        chunk_id    : Unique identifier (doc_id + chunk index)
        chunk_index : Position of this chunk within the document
    """
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    chunk_index: int
 
 
class BaseChunker(ABC):
    """
    Abstract base class for all chunking strategies.
 
    Every chunker takes document text and metadata and returns
    a list of Chunk objects. Completely interchangeable from
    the pipeline's perspective.
    """
 
    @abstractmethod
    def chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
        doc_id: str
    ) -> List[Chunk]:
        raise NotImplementedError
 
    def _build_chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
        doc_id: str,
        index: int
    ) -> Chunk:
        return Chunk(
            content=text.strip(),
            metadata={
                **metadata,
                "chunk_index": index,
                "chunk_size": len(text),
                "chunking_strategy": self.__class__.__name__,
            },
            chunk_id=f"{doc_id}_chunk_{index}",
            chunk_index=index
        )
 
 
class FixedSizeChunker(BaseChunker):
    """
    Splits text into fixed-size chunks with optional overlap.
 
    When to use: Uniform structured documents, baseline benchmarking.
    When NOT to use: Narrative text, documents with natural paragraphs.
    """
 
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
 
    def chunk(self, text: str, metadata: Dict[str, Any], doc_id: str) -> List[Chunk]:
        chunks = []
        start = 0
        index = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append(self._build_chunk(chunk_text, metadata, doc_id, index))
                index += 1
            start += self.chunk_size - self.chunk_overlap
        return chunks
 
 
class RecursiveChunker(BaseChunker):
    """
    Splits text using a hierarchy of separators.
 
    Tries paragraph splits first, then sentences, then words.
    Best general-purpose chunking strategy for most documents.
 
    When to use: General-purpose RAG, documents with natural structure.
    When NOT to use: Tables, code, highly structured data.
    """
 
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
 
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
 
    def chunk(self, text: str, metadata: Dict[str, Any], doc_id: str) -> List[Chunk]:
        raw_chunks = self._split_recursive(text, self.separators)
        merged = self._merge_chunks(raw_chunks)
        return [
            self._build_chunk(chunk_text, metadata, doc_id, i)
            for i, chunk_text in enumerate(merged)
            if chunk_text.strip()
        ]
 
    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text]
        separator = separators[0]
        remaining = separators[1:]
        if separator == "":
            return [
                text[i:i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
            ]
        splits = text.split(separator)
        result = []
        for split in splits:
            if len(split) <= self.chunk_size:
                result.append(split)
            else:
                result.extend(self._split_recursive(split, remaining))
        return result
 
    def _merge_chunks(self, splits: List[str]) -> List[str]:
        merged = []
        current_chunk = ""
        for split in splits:
            if not split.strip():
                continue
            candidate = current_chunk + " " + split if current_chunk else split
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    merged.append(current_chunk)
                overlap_text = current_chunk[-self.chunk_overlap:] if current_chunk else ""
                current_chunk = overlap_text + " " + split if overlap_text else split
        if current_chunk.strip():
            merged.append(current_chunk)
        return merged
 
 
class SemanticChunker(BaseChunker):
    """
    Splits text based on semantic similarity between sentences.
 
    Instead of splitting by character count or separators,
    semantic chunking groups sentences that are semantically
    related — measured by embedding similarity. When similarity
    drops significantly between adjacent sentences, a new chunk
    starts.
 
    How it works:
        1. Split text into sentences
        2. Embed each sentence
        3. Compute cosine similarity between adjacent sentences
        4. Split where similarity drops below threshold
        5. Merge small chunks to meet minimum size
 
    Why this matters:
        Fixed and recursive chunking can split related sentences
        across chunk boundaries — the first half of an explanation
        in one chunk, the second half in another. Semantic chunking
        keeps related sentences together, improving retrieval quality.
 
    When to use:
        - Long-form documents with varied topics
        - When context coherence matters more than chunk size consistency
        - High-precision retrieval where noise is costly
 
    When NOT to use:
        - Short documents where all content is related
        - When embedding API costs are a concern (requires N embed calls)
        - When chunk size consistency is required
 
    Args:
        embedder           : Embedding model with embed_documents() method
        similarity_threshold: Cosine similarity below which to split (default: 0.5)
        min_chunk_size     : Minimum characters per chunk (default: 200)
        max_chunk_size     : Maximum characters per chunk (default: 2000)
    """
 
    def __init__(
        self,
        embedder=None,
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 200,
        max_chunk_size: int = 2000
    ):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
 
    def chunk(self, text: str, metadata: Dict[str, Any], doc_id: str) -> List[Chunk]:
        # Step 1 — split into sentences
        sentences = self._split_into_sentences(text)
 
        if len(sentences) <= 1:
            return [self._build_chunk(text, metadata, doc_id, 0)]
 
        # Step 2 — embed sentences (or use fallback)
        embeddings = self._embed_sentences(sentences)
 
        # Step 3 — find split points based on similarity drops
        split_points = self._find_split_points(embeddings)
 
        # Step 4 — group sentences into chunks at split points
        raw_chunks = self._group_sentences(sentences, split_points)
 
        # Step 5 — merge small chunks, split oversized ones
        final_chunks = self._normalize_chunks(raw_chunks)
 
        return [
            self._build_chunk(chunk_text, metadata, doc_id, i)
            for i, chunk_text in enumerate(final_chunks)
            if chunk_text.strip()
        ]
 
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple punctuation rules."""
        import re
        sentence_endings = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]
 
    def _embed_sentences(self, sentences: List[str]) -> List[List[float]]:
        """Embed sentences — falls back to random if no embedder."""
        if self.embedder is None:
            import random
            return [[random.random() for _ in range(384)] for _ in sentences]
        return self.embedder.embed_documents(sentences)
 
    def _find_split_points(self, embeddings: List[List[float]]) -> List[int]:
        """
        Find positions where cosine similarity between adjacent
        sentence embeddings drops below the threshold.
        These are natural topic boundaries.
        """
        import numpy as np
        split_points = []
 
        for i in range(len(embeddings) - 1):
            vec1 = np.array(embeddings[i])
            vec2 = np.array(embeddings[i + 1])
            similarity = np.dot(vec1, vec2) / (
                np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8
            )
            if similarity < self.similarity_threshold:
                split_points.append(i + 1)
 
        return split_points
 
    def _group_sentences(
        self,
        sentences: List[str],
        split_points: List[int]
    ) -> List[str]:
        """Group sentences into chunks at split boundaries."""
        chunks = []
        start = 0
 
        for split_point in split_points:
            chunk_sentences = sentences[start:split_point]
            chunks.append(" ".join(chunk_sentences))
            start = split_point
 
        # Last chunk
        if start < len(sentences):
            chunks.append(" ".join(sentences[start:]))
 
        return chunks
 
    def _normalize_chunks(self, chunks: List[str]) -> List[str]:
        """Merge chunks below min_size, split chunks above max_size."""
        normalized = []
        buffer = ""
 
        for chunk in chunks:
            if len(buffer) + len(chunk) < self.min_chunk_size:
                buffer = buffer + " " + chunk if buffer else chunk
            else:
                if buffer:
                    normalized.append(buffer)
                buffer = chunk
 
        if buffer:
            normalized.append(buffer)
 
        # Split any chunks that are still too large
        final = []
        for chunk in normalized:
            if len(chunk) > self.max_chunk_size:
                # Fall back to recursive splitting for oversized chunks
                words = chunk.split()
                current = ""
                for word in words:
                    if len(current) + len(word) < self.max_chunk_size:
                        current = current + " " + word if current else word
                    else:
                        if current:
                            final.append(current)
                        current = word
                if current:
                    final.append(current)
            else:
                final.append(chunk)
 
        return final
 
 
class SentenceWindowChunker(BaseChunker):
    """
    Creates overlapping chunks centered around individual sentences.
 
    Each chunk contains a target sentence plus a window of
    surrounding sentences for context. At retrieval time, the
    surrounding context helps the LLM understand the sentence
    in isolation.
 
    How it works:
        For each sentence in the document, create a chunk
        containing that sentence plus window_size sentences
        on each side.
 
        Sentence 3 with window=2:
            [Sentence 1, Sentence 2, *Sentence 3*, Sentence 4, Sentence 5]
 
    Why this matters:
        Individual sentences are often too short to carry meaning
        alone — "It was approved in 1998." doesn't tell you what
        was approved. The sentence window provides the context
        needed to make the sentence retrievable and useful.
 
    Best paired with:
        A reranker that scores retrieved chunks by the target
        sentence relevance — not the full window relevance.
 
    When to use:
        - Documents where individual sentences are the retrieval unit
        - Q&A over dense factual documents (legal, medical, technical)
        - When used with cross-encoder reranking
 
    When NOT to use:
        - Long documents with high redundancy (too many overlapping chunks)
        - When storage/memory is a constraint
 
    Args:
        window_size   : Number of sentences on each side (default: 2)
        min_sentences : Minimum sentences in document to use this strategy
    """
 
    def __init__(self, window_size: int = 2, min_sentences: int = 5):
        self.window_size = window_size
        self.min_sentences = min_sentences
 
    def chunk(self, text: str, metadata: Dict[str, Any], doc_id: str) -> List[Chunk]:
        sentences = self._split_into_sentences(text)
 
        if len(sentences) < self.min_sentences:
            # Document too short — return as single chunk
            return [self._build_chunk(text, metadata, doc_id, 0)]
 
        chunks = []
 
        for i, sentence in enumerate(sentences):
            # Define window boundaries
            start = max(0, i - self.window_size)
            end = min(len(sentences), i + self.window_size + 1)
 
            # Window content
            window_sentences = sentences[start:end]
            window_text = " ".join(window_sentences)
 
            # Build chunk with extra metadata about the target sentence
            chunk_metadata = {
                **metadata,
                "target_sentence": sentence,
                "target_sentence_index": i,
                "window_start": start,
                "window_end": end,
                "window_size": self.window_size,
            }
 
            chunks.append(Chunk(
                content=window_text,
                metadata={
                    **chunk_metadata,
                    "chunk_index": i,
                    "chunk_size": len(window_text),
                    "chunking_strategy": self.__class__.__name__,
                },
                chunk_id=f"{doc_id}_chunk_{i}",
                chunk_index=i
            ))
 
        return chunks
 
    def _split_into_sentences(self, text: str) -> List[str]:
        import re
        sentence_endings = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]
 
 
class ChunkerFactory:
    """
    Factory for creating chunkers by strategy name.
 
    Allows pipeline config files to specify chunking strategy
    as a string rather than importing classes directly.
 
    Usage:
        chunker = ChunkerFactory.create("semantic", similarity_threshold=0.4)
        chunks = chunker.chunk(text, metadata, doc_id)
    """
 
    _registry = {
        "fixed": FixedSizeChunker,
        "recursive": RecursiveChunker,
        "semantic": SemanticChunker,
        "sentence_window": SentenceWindowChunker,
    }
 
    @classmethod
    def create(cls, strategy: str, **kwargs) -> BaseChunker:
        if strategy not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown chunking strategy: '{strategy}'. "
                f"Available: {available}"
            )
        return cls._registry[strategy](**kwargs)
 
    @classmethod
    def register(cls, name: str, chunker_class: type) -> None:
        if not issubclass(chunker_class, BaseChunker):
            raise TypeError(f"{chunker_class.__name__} must inherit from BaseChunker")
        cls._registry[name] = chunker_class
 
    @classmethod
    def available_strategies(cls) -> List[str]:
        return list(cls._registry.keys())
