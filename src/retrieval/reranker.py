"""
Cross-Encoder Reranking
 
The problem with bi-encoder retrieval (what FAISS does):
    Bi-encoders embed query and document independently and compare
    their embeddings. This is fast. You can precompute all document
    embeddings and search in milliseconds. But it's approximate,
    the model never sees query and document together, so it can miss
    subtle relevance signals.
 
    Example: "What are the side effects of ibuprofen?"
    The bi-encoder compares query embedding to chunk embeddings.
    It retrieves chunks about ibuprofen but might rank a general
    "about ibuprofen" chunk above a specific "side effects" chunk
    because the embeddings are similar overall.
 
How cross-encoders fix this:
    A cross-encoder takes the query AND document as input together
    and produces a single relevance score. It can see the full
    interaction between query and document, hence, it is much more accurate.
 
    The tradeoff: you can't precompute scores. Every query requires
    running the model on every (query, chunk) pair. Too slow for
    initial retrieval but perfect for reranking a small candidate set.
 
The two-stage retrieval pattern (industry standard):
    Stage 1: Bi-encoder retrieval (fast, approximate):
        Retrieve top-K candidates from FAISS (e.g. K=20)
    Stage 2: Cross-encoder reranking (slow, precise):
        Rerank the K candidates, return top-k (e.g. k=5)
 
    This gives you the speed of bi-encoders for the full corpus
    and the precision of cross-encoders for the final result set.
 
Reference:
    Sentence-Transformers cross-encoder models:
    https://www.sbert.net/docs/pretrained_cross-encoders.html
"""
 
from typing import List, Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod
 
from src.ingestion.chunker import Chunk
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.retriever import BaseRetriever, SimilarityRetriever
 
 
class BaseReranker(ABC):
    """
    Abstract base class for reranking strategies.
 
    A reranker takes a query and a list of candidate chunks
    and returns them reordered by relevance score.
    """
 
    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: List[Chunk],
        top_k: Optional[int] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Rerank a list of chunks by relevance to the query.
 
        Args:
            query   : User's original query
            chunks  : Candidate chunks from initial retrieval
            top_k   : Return only top_k results. If None, return all.
 
        Returns:
            List of (Chunk, score) tuples sorted by score descending.
        """
        raise NotImplementedError
 
 
class CrossEncoderReranker(BaseReranker):
    """
    Reranker using a cross-encoder model from sentence-transformers.
 
    Takes query and chunk together as input. It is much more accurate
    than bi-encoder similarity but slower. Used as Stage 2 in
    the two-stage retrieval pattern.
 
    Recommended models (accuracy vs speed tradeoff):
        - cross-encoder/ms-marco-MiniLM-L-6-v2   (fast, good)
        - cross-encoder/ms-marco-MiniLM-L-12-v2  (slower, better)
        - cross-encoder/ms-marco-electra-base     (slowest, best)
 
    Args:
        model_name  : HuggingFace cross-encoder model name
        batch_size  : Batch size for scoring (default: 32)
        max_length  : Max token length for query+chunk (default: 512)
    """
 
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
 
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        max_length: int = 512
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
 
    def _load_model(self):
        """Lazy model loading: only load when first used."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(
                    self.model_name,
                    max_length=self.max_length
                )
                print(f"[CrossEncoderReranker] Loaded model: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for cross-encoder reranking. "
                    "Install with: pip install sentence-transformers"
                )
 
    def rerank(
        self,
        query: str,
        chunks: List[Chunk],
        top_k: Optional[int] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Rerank chunks using cross-encoder scoring.
 
        Args:
            query   : User's original query
            chunks  : Candidate chunks to rerank
            top_k   : Number of top results to return
 
        Returns:
            List of (Chunk, relevance_score) tuples, sorted descending.
        """
        if not chunks:
            return []
 
        self._load_model()
 
        # Build (query, chunk_content) pairs for cross-encoder
        pairs = [(query, chunk.content) for chunk in chunks]
 
        # Score all pairs
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False
        )
 
        # Combine chunks with scores and sort
        chunk_scores = list(zip(chunks, scores.tolist()))
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
 
        if top_k is not None:
            chunk_scores = chunk_scores[:top_k]
 
        return chunk_scores
 
 
class MockReranker(BaseReranker):
    """
    Mock reranker for testing without loading a real model.
 
    Returns chunks in reverse order (last retrieved first):
    simple enough to be predictable in tests, different enough
    from the original order to verify reranking happened.
    """
 
    def rerank(
        self,
        query: str,
        chunks: List[Chunk],
        top_k: Optional[int] = None
    ) -> List[Tuple[Chunk, float]]:
        # Assign mock scores in reverse order
        n = len(chunks)
        scored = [
            (chunk, float(n - i) / n)
            for i, chunk in enumerate(chunks)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
 
        if top_k is not None:
            scored = scored[:top_k]
 
        return scored
 
 
class TwoStageRetriever(BaseRetriever):
    """
    Two-stage retrieval: bi-encoder retrieval + cross-encoder reranking.
 
    Stage 1: Retrieve a large candidate set using fast bi-encoder
             similarity search (FAISS).
    Stage 2: Rerank candidates using precise cross-encoder scoring.
 
    This is the industry-standard approach for high-precision RAG.
    The initial retrieval fetches more candidates than needed
    (retrieval_multiplier × k) to give the reranker good material
    to work with.
 
    Usage:
        retriever = TwoStageRetriever(
            vector_store=vector_store,
            embedder=embedder,
            reranker=CrossEncoderReranker(),
            retrieval_multiplier=4  # fetch 4x candidates for reranking
        )
        chunks = retriever.retrieve("What is the return policy?", k=5)
 
    Args:
        vector_store          : FAISSVectorStore with indexed chunks
        embedder              : Embedding model for Stage 1
        reranker              : Reranker for Stage 2
        retrieval_multiplier  : How many extra candidates to fetch for
                                reranking. k * multiplier candidates
                                fetched in Stage 1, top-k returned
                                after Stage 2. (default: 4)
    """
 
    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder,
        reranker: Optional[BaseReranker] = None,
        retrieval_multiplier: int = 4
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker or MockReranker()
        self.retrieval_multiplier = retrieval_multiplier
 
    def retrieve(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Two-stage retrieval with reranking.
 
        Stage 1: Fetch k * retrieval_multiplier candidates
        Stage 2: Rerank and return top k
 
        Args:
            query           : User's query
            k               : Final number of chunks to return
            metadata_filter : Optional metadata filter applied in Stage 1
 
        Returns:
            Top-k chunks after reranking, sorted by relevance.
        """
        # Stage 1 — bi-encoder retrieval (over-fetch for reranker)
        candidate_k = k * self.retrieval_multiplier
        query_embedding = self.embedder.embed_query(query)
 
        candidates_with_scores = self.vector_store.search(
            query_embedding,
            k=candidate_k,
            metadata_filter=metadata_filter
        )
 
        candidates = [chunk for chunk, _ in candidates_with_scores]
 
        if not candidates:
            return []
 
        # Stage 2 — cross-encoder reranking
        reranked = self.reranker.rerank(query, candidates, top_k=k)
 
        return [chunk for chunk, score in reranked]
 
    def retrieve_with_scores(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Two-stage retrieval returning chunks with reranking scores.
 
        Useful for debugging: lets you see the reranking scores
        alongside the retrieved chunks.
 
        Returns:
            List of (Chunk, reranking_score) tuples.
        """
        candidate_k = k * self.retrieval_multiplier
        query_embedding = self.embedder.embed_query(query)
 
        candidates_with_scores = self.vector_store.search(
            query_embedding,
            k=candidate_k,
            metadata_filter=metadata_filter
        )
 
        candidates = [chunk for chunk, _ in candidates_with_scores]
 
        if not candidates:
            return []
 
        return self.reranker.rerank(query, candidates, top_k=k)
