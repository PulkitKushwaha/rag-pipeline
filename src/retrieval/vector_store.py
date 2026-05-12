import os
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
 
from src.ingestion.chunker import Chunk
 
 
class FAISSVectorStore:
    """
    A clean wrapper around FAISS for storing and searching
    embedded chunks.
 
    Design decision: We keep the vector store separate from the
    retriever. The vector store is responsible for storage and
    raw similarity search. The retriever is responsible for
    retrieval strategy (HyDE, reranking, filtering etc.).
    This separation makes it easy to swap vector stores without
    touching retrieval logic.
 
    Usage:
        store = FAISSVectorStore(embedding_dim=1536)
        store.add_chunks(chunks, embeddings)
        results = store.search(query_embedding, k=5)
        store.save("data/vector_store")
        store.load("data/vector_store")
    """
 
    def __init__(self, embedding_dim: int = 1536):
        """
        Initialize the vector store.
 
        Args:
            embedding_dim: Dimension of the embedding vectors.
                           OpenAI text-embedding-3-small = 1536
                           HuggingFace all-MiniLM-L6-v2 = 384
        """
        try:
            import faiss
            import numpy as np
        except ImportError:
            raise ImportError(
                "faiss-cpu is required. Install with: pip install faiss-cpu"
            )
 
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.chunks: List[Chunk] = []
        self._np = np
 
    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]]
    ) -> None:
        """
        Add chunks and their embeddings to the vector store.
 
        Args:
            chunks     : List of Chunk objects to store
            embeddings : Corresponding embedding vectors (one per chunk)
 
        Raises:
            ValueError: If chunks and embeddings counts don't match
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                f"must have the same length"
            )
 
        import numpy as np
        vectors = np.array(embeddings, dtype="float32")
        self.index.add(vectors)
        self.chunks.extend(chunks)
 
        print(f"Added {len(chunks)} chunks. Total: {len(self.chunks)}")
 
    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for the k most similar chunks.
 
        Args:
            query_embedding : Embedding of the query
            k               : Number of results to return
            metadata_filter : Optional dict of metadata key-value pairs
                              to filter results. Applied post-retrieval.
 
        Returns:
            List of (Chunk, similarity_score) tuples,
            sorted by similarity descending.
        """
        import numpy as np
 
        if len(self.chunks) == 0:
            return []
 
        query_vector = np.array([query_embedding], dtype="float32")
 
        # Over-fetch if filtering — we'll need more candidates
        fetch_k = k * 3 if metadata_filter else k
        fetch_k = min(fetch_k, len(self.chunks))
 
        distances, indices = self.index.search(query_vector, fetch_k)
 
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
 
            chunk = self.chunks[idx]
 
            # Apply metadata filter if specified
            if metadata_filter:
                if not self._passes_filter(chunk.metadata, metadata_filter):
                    continue
 
            # Convert L2 distance to similarity score (0-1, higher = more similar)
            similarity = 1 / (1 + dist)
            results.append((chunk, float(similarity)))
 
            if len(results) >= k:
                break
 
        return results
 
    def _passes_filter(
        self,
        metadata: Dict[str, Any],
        filter_dict: Dict[str, Any]
    ) -> bool:
        """Check if chunk metadata matches all filter criteria."""
        for key, value in filter_dict.items():
            chunk_value = metadata.get(key)
            if isinstance(value, list):
                if chunk_value not in value:
                    return False
            else:
                if chunk_value != value:
                    return False
        return True
 
    def save(self, directory: str) -> None:
        """
        Save the vector store to disk.
 
        Saves the FAISS index and chunk metadata separately.
 
        Args:
            directory: Path to save directory (created if not exists)
        """
        import faiss
 
        save_path = Path(directory)
        save_path.mkdir(parents=True, exist_ok=True)
 
        faiss.write_index(self.index, str(save_path / "index.faiss"))
 
        with open(save_path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
 
        config = {
            "embedding_dim": self.embedding_dim,
            "num_chunks": len(self.chunks)
        }
        with open(save_path / "config.json", "w") as f:
            json.dump(config, f)
 
        print(f"Vector store saved to {directory} ({len(self.chunks)} chunks)")
 
    def load(self, directory: str) -> None:
        """
        Load a vector store from disk.
 
        Args:
            directory: Path to directory containing saved vector store.
 
        Raises:
            FileNotFoundError: If the directory or required files don't exist.
        """
        import faiss
 
        load_path = Path(directory)
 
        if not load_path.exists():
            raise FileNotFoundError(f"Vector store directory not found: {directory}")
 
        self.index = faiss.read_index(str(load_path / "index.faiss"))
 
        with open(load_path / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
 
        print(f"Vector store loaded from {directory} ({len(self.chunks)} chunks)")
 
    @property
    def size(self) -> int:
        """Number of chunks in the vector store."""
        return len(self.chunks)
