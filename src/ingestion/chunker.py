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
 
    Every chunker takes a document's text and metadata and
    returns a list of Chunk objects. The chunking strategy
    is completely interchangeable — the retrieval layer only
    ever sees Chunk objects, never raw text.
    """
 
    @abstractmethod
    def chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
        doc_id: str
    ) -> List[Chunk]:
        """
        Split text into chunks.
 
        Args:
            text     : Full document text to chunk
            metadata : Document metadata to inherit into each chunk
            doc_id   : Parent document ID for generating chunk IDs
 
        Returns:
            List of Chunk objects
        """
        raise NotImplementedError
 
    def _build_chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
        doc_id: str,
        index: int
    ) -> Chunk:
        """Helper to build a Chunk with consistent ID format."""
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
 
    The simplest chunking strategy: splits by character count.
    Fast and predictable, but ignores semantic boundaries.
    Can split sentences or paragraphs mid-way, which loses context.
 
    When to use:
        - Uniform, structured documents (tables, logs, code)
        - When chunk size consistency matters more than coherence
        - As a baseline to benchmark other strategies against
 
    When NOT to use:
        - Narrative text where sentences carry the meaning
        - Documents where paragraphs are the natural unit of retrieval
 
    Args:
        chunk_size    : Number of characters per chunk (default: 1000)
        chunk_overlap : Number of characters to overlap between chunks (default: 200)
    """
 
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
 
    def chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
        doc_id: str
    ) -> List[Chunk]:
        """Split text into fixed-size chunks with overlap."""
        chunks = []
        start = 0
        index = 0
 
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
 
            if chunk_text.strip():
                chunks.append(
                    self._build_chunk(chunk_text, metadata, doc_id, index)
                )
                index += 1
 
            # Move forward by chunk_size minus overlap
            start += self.chunk_size - self.chunk_overlap
 
        return chunks
 
 
class RecursiveChunker(BaseChunker):
    """
    Splits text using a hierarchy of separators: paragraphs first,
    then sentences, then words, then characters.
 
    This is the most practical general-purpose chunking strategy.
    It tries to keep semantically related text together by respecting
    natural document structure before falling back to arbitrary splits.
 
    LangChain's RecursiveCharacterTextSplitter uses this approach.
    This is a clean reimplementation without the LangChain dependency.
 
    When to use:
        - Most general-purpose RAG use cases
        - Documents with natural paragraph/sentence structure
        - When you want sensible defaults without tuning
 
    When NOT to use:
        - Highly structured documents (tables, code). In such cases, use FixedSizeChunker
        - When semantic coherence matters most. In such cases, use SemanticChunker
 
    Args:
        chunk_size    : Target chunk size in characters (default: 1000)
        chunk_overlap : Overlap between chunks in characters (default: 200)
        separators    : Ordered list of separators to try (default: paragraph → sentence → word → char)
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
 
    def chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
        doc_id: str
    ) -> List[Chunk]:
        """Split text using recursive separator hierarchy."""
        raw_chunks = self._split_recursive(text, self.separators)
        merged = self._merge_chunks(raw_chunks)
 
        return [
            self._build_chunk(chunk_text, metadata, doc_id, i)
            for i, chunk_text in enumerate(merged)
            if chunk_text.strip()
        ]
 
    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using the separator hierarchy.
 
        Tries each separator in order. If a split produces chunks
        that are still too large, recursively splits those chunks
        with the next separator in the hierarchy.
        """
        if not separators:
            return [text]
 
        separator = separators[0]
        remaining_separators = separators[1:]
 
        if separator == "":
            # Last resort — split by character
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
                # This split is still too large — recurse with next separator
                result.extend(
                    self._split_recursive(split, remaining_separators)
                )
 
        return result
 
    def _merge_chunks(self, splits: List[str]) -> List[str]:
        """
        Merge small splits back together up to chunk_size,
        with overlap between merged chunks.
        """
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
                # Start new chunk with overlap from end of previous
                overlap_text = current_chunk[-self.chunk_overlap:] if current_chunk else ""
                current_chunk = overlap_text + " " + split if overlap_text else split
 
        if current_chunk.strip():
            merged.append(current_chunk)
 
        return merged
 
 
class ChunkerFactory:
    """
    Factory for creating chunkers by name.
 
    Allows pipeline configuration files to specify chunking
    strategy as a string rather than importing classes directly.
 
    Usage:
        chunker = ChunkerFactory.create("recursive", chunk_size=800)
        chunks = chunker.chunk(text, metadata, doc_id)
    """
 
    _registry = {
        "fixed": FixedSizeChunker,
        "recursive": RecursiveChunker,
    }
 
    @classmethod
    def create(cls, strategy: str, **kwargs) -> BaseChunker:
        """
        Create a chunker by strategy name.
 
        Args:
            strategy : Name of the chunking strategy
            **kwargs : Arguments passed to the chunker constructor
 
        Returns:
            BaseChunker instance
 
        Raises:
            ValueError: If strategy name is not registered
        """
        if strategy not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown chunking strategy: '{strategy}'. "
                f"Available: {available}"
            )
        return cls._registry[strategy](**kwargs)
 
    @classmethod
    def register(cls, name: str, chunker_class: type) -> None:
        """
        Register a custom chunker class.
 
        Args:
            name          : Strategy name to register under
            chunker_class : Class inheriting from BaseChunker
        """
        if not issubclass(chunker_class, BaseChunker):
            raise TypeError(
                f"{chunker_class.__name__} must inherit from BaseChunker"
            )
        cls._registry[name] = chunker_class
        print(f"Registered chunker: '{name}'")
