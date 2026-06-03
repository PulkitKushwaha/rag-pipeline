"""
Multi-Document RAG Retriever
 
Standard RAG retrieves from a single unified vector store.
Multi-doc RAG adds structure — documents are organized into
collections, and retrieval is aware of which collections
are relevant to a query.
 
Why this matters in enterprise:
    A knowledge base typically contains documents from multiple
    sources with different characteristics:
        - HR policies (confidential, rarely changes)
        - Product documentation (internal, updated frequently)
        - Legal contracts (restricted, highly sensitive)
        - FAQ documents (public, updated often)
 
    Treating all of these identically in retrieval leads to:
        - HR policies surfaced for product questions
        - Public FAQs mixed with restricted legal content
        - Outdated docs retrieved alongside current ones
 
    Multi-doc RAG solves this through:
        1. Collection-aware retrieval — query relevant collections only
        2. Source weighting — trust different sources differently
        3. Metadata-filtered retrieval — respect access controls
        4. Cross-collection deduplication — avoid redundant chunks
"""
 
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
 
from src.ingestion.chunker import Chunk
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.retriever import BaseRetriever
from src.retrieval.metadata_filter import MetadataFilter, AccessLevelFilter
 
 
@dataclass
class DocumentCollection:
    """
    A named collection of documents with shared metadata.
 
    Collections are the organizational unit for multi-doc RAG.
    Each collection has its own vector store, access level,
    and retrieval weight.
 
    Args:
        name         : Unique collection identifier
        vector_store : FAISS index for this collection
        access_level : Minimum access level to retrieve from this collection
        weight       : Retrieval weight — higher weight = more chunks from this collection
        description  : What this collection contains (used for collection routing)
        metadata     : Additional collection-level metadata
    """
    name: str
    vector_store: FAISSVectorStore
    access_level: str = "internal"
    weight: float = 1.0
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
 
 
class MultiDocRetriever(BaseRetriever):
    """
    Retrieves from multiple document collections with
    access control, source weighting, and deduplication.
 
    Usage:
        retriever = MultiDocRetriever(embedder=embedder)
 
        # Register collections
        retriever.add_collection(DocumentCollection(
            name="hr_policies",
            vector_store=hr_store,
            access_level="internal",
            weight=1.0
        ))
        retriever.add_collection(DocumentCollection(
            name="legal_contracts",
            vector_store=legal_store,
            access_level="restricted",
            weight=1.2
        ))
 
        # Retrieve with access control
        chunks = retriever.retrieve(
            query="What is the parental leave policy?",
            k=5,
            user_access_level="internal"
        )
    """
 
    def __init__(self, embedder=None):
        self.embedder = embedder
        self.collections: Dict[str, DocumentCollection] = {}
 
    def add_collection(self, collection: DocumentCollection) -> None:
        """Register a document collection for retrieval."""
        self.collections[collection.name] = collection
        print(f"[MultiDocRetriever] Added collection: '{collection.name}' "
              f"(access: {collection.access_level}, weight: {collection.weight})")
 
    def retrieve(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        user_access_level: str = "internal",
        collection_names: Optional[List[str]] = None
    ) -> List[Chunk]:
        """
        Retrieve from multiple collections with access control.
 
        Args:
            query              : User's query
            k                  : Total chunks to return across all collections
            metadata_filter    : Additional metadata filter applied to all collections
            user_access_level  : User's access level — only collections at or
                                 below this level are searched
            collection_names   : If provided, only search these collections.
                                 If None, search all accessible collections.
 
        Returns:
            Top-k chunks from all accessible collections, weighted and deduplicated
        """
        if not self.collections:
            return []
 
        # Determine which collections to search
        accessible = self._get_accessible_collections(
            user_access_level, collection_names
        )
 
        if not accessible:
            print(f"[MultiDocRetriever] No accessible collections for level: {user_access_level}")
            return []
 
        # Retrieve from each collection
        all_chunks_with_scores: List[Tuple[Chunk, float, str]] = []
 
        # Calculate per-collection k based on weight
        total_weight = sum(c.weight for c in accessible)
 
        for collection in accessible:
            collection_k = max(1, int(k * 2 * (collection.weight / total_weight)))
            chunks_with_scores = self._retrieve_from_collection(
                query=query,
                collection=collection,
                k=collection_k,
                metadata_filter=metadata_filter
            )
            for chunk, score in chunks_with_scores:
                # Apply collection weight to score
                weighted_score = score * collection.weight
                all_chunks_with_scores.append((chunk, weighted_score, collection.name))
 
        # Sort by weighted score and deduplicate
        all_chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
        deduplicated = self._deduplicate(all_chunks_with_scores)
 
        # Add collection name to chunk metadata
        final_chunks = []
        for chunk, score, collection_name in deduplicated[:k]:
            chunk.metadata["collection"] = collection_name
            chunk.metadata["retrieval_score"] = round(score, 4)
            final_chunks.append(chunk)
 
        return final_chunks
 
    def retrieve_with_sources(
        self,
        query: str,
        k: int = 5,
        user_access_level: str = "internal"
    ) -> Dict[str, Any]:
        """
        Retrieve chunks and return a structured result with source breakdown.
 
        Useful for debugging and for building citation-aware responses.
 
        Returns:
            Dict with chunks, source breakdown, and collection metadata
        """
        chunks = self.retrieve(query, k=k, user_access_level=user_access_level)
 
        # Group by collection
        by_collection: Dict[str, List[Chunk]] = {}
        for chunk in chunks:
            col = chunk.metadata.get("collection", "unknown")
            by_collection.setdefault(col, []).append(chunk)
 
        return {
            "chunks": chunks,
            "total_retrieved": len(chunks),
            "by_collection": {
                col: {
                    "count": len(col_chunks),
                    "sources": list({c.metadata.get("filename", "") for c in col_chunks})
                }
                for col, col_chunks in by_collection.items()
            },
            "query": query,
            "user_access_level": user_access_level
        }
 
    def _get_accessible_collections(
        self,
        user_access_level: str,
        collection_names: Optional[List[str]]
    ) -> List[DocumentCollection]:
        """Filter collections by user access level."""
        access_hierarchy = {
            "public": 0, "internal": 1,
            "confidential": 2, "restricted": 3
        }
        user_level = access_hierarchy.get(user_access_level, 0)
 
        collections = list(self.collections.values())
 
        if collection_names:
            collections = [c for c in collections if c.name in collection_names]
 
        return [
            c for c in collections
            if access_hierarchy.get(c.access_level, 0) <= user_level
        ]
 
    def _retrieve_from_collection(
        self,
        query: str,
        collection: DocumentCollection,
        k: int,
        metadata_filter: Optional[Dict[str, Any]]
    ) -> List[Tuple[Chunk, float]]:
        """Retrieve from a single collection."""
        if self.embedder is None:
            return []
 
        try:
            query_embedding = self.embedder.embed_query(query)
            results = collection.vector_store.search(
                query_embedding,
                k=k,
                metadata_filter=metadata_filter
            )
            return results
        except Exception as e:
            print(f"[MultiDocRetriever] Collection '{collection.name}' failed: {e}")
            return []
 
    def _deduplicate(
        self,
        chunks_with_scores: List[Tuple[Chunk, float, str]]
    ) -> List[Tuple[Chunk, float, str]]:
        """Remove duplicate chunks based on content similarity."""
        seen_content = set()
        deduplicated = []
 
        for chunk, score, collection_name in chunks_with_scores:
            # Use first 100 chars as deduplication key
            content_key = chunk.content[:100].strip()
            if content_key not in seen_content:
                seen_content.add(content_key)
                deduplicated.append((chunk, score, collection_name))
 
        return deduplicated
 
    @property
    def collection_names(self) -> List[str]:
        return list(self.collections.keys())
 
    def __repr__(self) -> str:
        return (
            f"MultiDocRetriever("
            f"collections={self.collection_names}, "
            f"embedder={'configured' if self.embedder else 'none'})"
        )
