"""
GraphRAG-lite: Entity-Aware Retrieval with Graph Traversal
 
Standard RAG retrieves by semantic similarity. It finds chunks
that are semantically close to the query. But similarity doesn't
capture relationships between entities across documents.
 
Example problem:
    Document 1: "CEO John Smith approved the acquisition."
    Document 2: "The acquisition target is Acme Corp."
    Document 3: "Acme Corp's primary product is an inventory system."
 
    Query: "What product did the CEO approve acquiring?"
 
    Standard RAG: May retrieve Document 1 and Document 3 but miss
    the connection — "CEO approved acquisition" is in Doc 1, and
    "Acme Corp product" is in Doc 3, but connecting them requires
    knowing that "the acquisition" in Doc 1 refers to Acme Corp.
 
    GraphRAG: Extracts entities (CEO, John Smith, Acme Corp,
    inventory system) and relationships (approved, target, produces).
    Graph traversal connects CEO → approved → acquisition → Acme Corp
    → produces → inventory system. All three documents are surfaced.
 
This "lite" implementation uses:
    - Simple regex-based entity extraction (no NLP model required)
    - NetworkX for graph storage and traversal
    - Hybrid retrieval: vector similarity + graph neighbors
 
For production GraphRAG, see Microsoft's full implementation:
    https://github.com/microsoft/graphrag
"""
 
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
 
from src.ingestion.chunker import Chunk
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.retriever import BaseRetriever
 
 
@dataclass
class Entity:
    """An entity extracted from a document chunk."""
    name: str
    entity_type: str  # PERSON, ORG, PRODUCT, CONCEPT, etc.
    chunk_id: str
    mentions: List[str] = field(default_factory=list)
 
 
@dataclass
class Relationship:
    """A relationship between two entities."""
    source: str
    target: str
    relation: str
    chunk_id: str
    confidence: float = 1.0
 
 
class EntityExtractor:
    """
    Simple regex-based entity extraction.
 
    Extracts named entities using pattern matching.
    Not as accurate as NLP-based NER but requires no model loading.
 
    For better accuracy, swap this for spaCy or Presidio NER.
    """
 
    # Simple patterns for common entity types
    PATTERNS = {
        "PERSON": [
            r"\b(?:Mr\.|Mrs\.|Dr\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*",
            r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b(?=\s+(?:said|stated|reported|approved|founded))"
        ],
        "ORG": [
            r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:Corp|Inc|LLC|Ltd|Company|Co\.)\b",
            r"\b(?:the\s+)?[A-Z][a-zA-Z]+\s+(?:Group|Holdings|Partners|Associates)\b"
        ],
        "PRODUCT": [
            r"\b(?:product|system|platform|solution|service|tool)\s+(?:called|named|known as)\s+[A-Z][a-zA-Z]+\b"
        ]
    }
 
    def extract(self, text: str, chunk_id: str) -> List[Entity]:
        """Extract entities from text."""
        import re
        entities = []
        seen = set()
 
        for entity_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    name = match.group().strip()
                    if name not in seen and len(name) > 2:
                        seen.add(name)
                        entities.append(Entity(
                            name=name,
                            entity_type=entity_type,
                            chunk_id=chunk_id
                        ))
 
        return entities
 
 
class GraphRetriever(BaseRetriever):
    """
    Hybrid retriever combining vector similarity with graph traversal.
 
    Stage 1: Standard vector search to find seed chunks
    Stage 2: Entity extraction from seed chunks
    Stage 3: Graph traversal to find related chunks via entity links
    Stage 4: Combine and rerank all results
 
    Args:
        vector_store    : FAISS vector store with indexed chunks
        embedder        : Embedding model for vector search
        max_hops        : Graph traversal depth (default: 2)
        graph_weight    : Weight for graph-retrieved chunks vs vector chunks
    """
 
    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder=None,
        max_hops: int = 2,
        graph_weight: float = 0.7
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.max_hops = max_hops
        self.graph_weight = graph_weight
        self.extractor = EntityExtractor()
 
        # Entity graph: entity_name → set of chunk_ids that mention it
        self._entity_index: Dict[str, Set[str]] = {}
        # Chunk graph: chunk_id → set of related chunk_ids via shared entities
        self._chunk_graph: Dict[str, Set[str]] = {}
        self._is_indexed = False
 
    def build_graph(self, chunks: List[Chunk]) -> None:
        """
        Build the entity graph from a list of chunks.
 
        Call this after ingesting documents into the vector store.
        Creates a graph where chunks are connected if they share entities.
 
        Args:
            chunks: All chunks in the knowledge base
        """
        print(f"[GraphRetriever] Building entity graph from {len(chunks)} chunks...")
 
        for chunk in chunks:
            entities = self.extractor.extract(chunk.content, chunk.chunk_id)
 
            for entity in entities:
                entity_key = entity.name.lower()
 
                # Add to entity index
                if entity_key not in self._entity_index:
                    self._entity_index[entity_key] = set()
                self._entity_index[entity_key].add(chunk.chunk_id)
 
                # Connect this chunk to all other chunks sharing this entity
                for other_chunk_id in self._entity_index[entity_key]:
                    if other_chunk_id != chunk.chunk_id:
                        # Add bidirectional edge
                        if chunk.chunk_id not in self._chunk_graph:
                            self._chunk_graph[chunk.chunk_id] = set()
                        self._chunk_graph[chunk.chunk_id].add(other_chunk_id)
 
                        if other_chunk_id not in self._chunk_graph:
                            self._chunk_graph[other_chunk_id] = set()
                        self._chunk_graph[other_chunk_id].add(chunk.chunk_id)
 
        self._is_indexed = True
        entity_count = len(self._entity_index)
        edge_count = sum(len(v) for v in self._chunk_graph.values()) // 2
        print(f"[GraphRetriever] Graph built: {entity_count} entities, {edge_count} edges")
 
    def retrieve(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Hybrid retrieval: vector search + graph traversal.
 
        Args:
            query           : User's query
            k               : Total chunks to return
            metadata_filter : Optional metadata filter
 
        Returns:
            Top-k chunks from vector + graph retrieval combined
        """
        if self.embedder is None:
            return []
 
        # Stage 1: Vector search for seed chunks
        query_embedding = self.embedder.embed_query(query)
        seed_results = self.vector_store.search(
            query_embedding,
            k=max(3, k // 2),
            metadata_filter=metadata_filter
        )
        seed_chunks = {chunk.chunk_id: chunk for chunk, _ in seed_results}
        seed_scores = {chunk.chunk_id: score for chunk, score in seed_results}
 
        if not self._is_indexed:
            # Graph not built — return vector results only
            return list(seed_chunks.values())[:k]
 
        # Stage 2 & 3: Graph traversal from seed chunks
        graph_chunk_ids = self._traverse(
            seed_chunk_ids=set(seed_chunks.keys()),
            max_hops=self.max_hops
        )
 
        # Stage 4: Retrieve graph chunks from vector store
        all_chunks = dict(seed_chunks)
        all_scores = dict(seed_scores)
 
        for chunk in self.vector_store.chunks:
            if chunk.chunk_id in graph_chunk_ids and chunk.chunk_id not in all_chunks:
                all_chunks[chunk.chunk_id] = chunk
                all_scores[chunk.chunk_id] = self.graph_weight
 
        # Sort by score and return top-k
        sorted_chunk_ids = sorted(
            all_scores.keys(),
            key=lambda cid: all_scores[cid],
            reverse=True
        )
 
        return [all_chunks[cid] for cid in sorted_chunk_ids[:k] if cid in all_chunks]
 
    def _traverse(
        self,
        seed_chunk_ids: Set[str],
        max_hops: int
    ) -> Set[str]:
        """
        BFS traversal of the chunk graph from seed nodes.
 
        Args:
            seed_chunk_ids : Starting chunk IDs
            max_hops       : Maximum traversal depth
 
        Returns:
            Set of chunk IDs reachable within max_hops
        """
        visited = set(seed_chunk_ids)
        frontier = set(seed_chunk_ids)
 
        for hop in range(max_hops):
            next_frontier = set()
            for chunk_id in frontier:
                neighbors = self._chunk_graph.get(chunk_id, set())
                new_neighbors = neighbors - visited
                next_frontier.update(new_neighbors)
                visited.update(new_neighbors)
 
            if not next_frontier:
                break
            frontier = next_frontier
 
        # Return only graph-traversal results (not seeds)
        return visited - seed_chunk_ids
