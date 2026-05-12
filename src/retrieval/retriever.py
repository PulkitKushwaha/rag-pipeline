from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
 
from src.ingestion.chunker import Chunk
from src.retrieval.vector_store import FAISSVectorStore
 
 
class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval strategies.
 
    Every retriever takes a query string and returns a list of
    relevant chunks. The retrieval strategy (naive similarity,
    HyDE, reranking) is completely interchangeable from the
    pipeline's perspective.
    """
 
    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        raise NotImplementedError
 
 
class SimilarityRetriever(BaseRetriever):
    """
    Baseline retriever using naive cosine similarity search.
 
    Embeds the query directly and retrieves the k most similar
    chunks from the vector store. No reranking, no query expansion.
 
    This is the baseline every other retrieval strategy is
    benchmarked against. Faster and cheaper than alternatives —
    but often retrieves chunks that are semantically adjacent
    rather than truly relevant.
 
    When to use:
        - Baseline benchmarking
        - Low-latency requirements
        - Short, well-formed queries that closely match document language
 
    When NOT to use:
        - Complex or indirect queries where query language differs
          from document language (use HyDE instead)
        - When precision matters more than speed (use reranking)
 
    Args:
        vector_store : FAISSVectorStore instance with indexed chunks
        embedder     : Embedding model instance with embed_query() method
    """
 
    def __init__(self, vector_store: FAISSVectorStore, embedder):
        self.vector_store = vector_store
        self.embedder = embedder
 
    def retrieve(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Retrieve k most similar chunks for a query.
 
        Args:
            query           : User's query string
            k               : Number of chunks to retrieve
            metadata_filter : Optional metadata filter dict
 
        Returns:
            List of Chunk objects sorted by similarity descending
        """
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.search(
            query_embedding,
            k=k,
            metadata_filter=metadata_filter
        )
        return [chunk for chunk, score in results]
