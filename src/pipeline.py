# End-to-end RAG pipeline orchestration
# Composes ingestion → embedding → retrieval → generation
# into a single configurable pipeline class

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
 
from src.ingestion.loader import DocumentLoader, Document
from src.ingestion.chunker import BaseChunker, RecursiveChunker, Chunk
from src.ingestion.metadata import enrich_metadata
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.retriever import SimilarityRetriever
from src.generation.generator import RAGGenerator, GenerationResult
 
 
@dataclass
class PipelineConfig:
    """
    Configuration for the RAG pipeline.
 
    Centralizes all pipeline settings in one place:
    makes it easy to swap strategies and track experiments.
    """
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunking_strategy: str = "recursive"
    embedding_dim: int = 1536
    retrieval_k: int = 5
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.0
    vector_store_path: Optional[str] = None
 
 
class RAGPipeline:
    """
    End-to-end RAG pipeline orchestration.
 
    Composes all pipeline components:
        DocumentLoader → Chunker → Embedder → VectorStore
        → Retriever → Generator
 
    This is the main entry point for the pipeline. It wires
    together all components and exposes a simple interface:
        pipeline.ingest(documents)
        pipeline.query(question)
 
    Design decision: The pipeline accepts pre-instantiated
    components rather than creating them internally. This
    makes it easy to swap components (e.g. different chunkers,
    retrievers) and test each component in isolation.
 
    Usage:
        pipeline = RAGPipeline(
            chunker=RecursiveChunker(chunk_size=800),
            embedder=my_embedder,
            llm_client=my_openai_client
        )
        pipeline.ingest_directory("data/sample_docs/")
        result = pipeline.query("What is the return policy?")
        print(result.answer)
    """
 
    def __init__(
        self,
        chunker: Optional[BaseChunker] = None,
        embedder=None,
        llm_client=None,
        config: Optional[PipelineConfig] = None
    ):
        self.config = config or PipelineConfig()
        self.chunker = chunker or RecursiveChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        self.embedder = embedder
        self.loader = DocumentLoader()
        self.vector_store = FAISSVectorStore(
            embedding_dim=self.config.embedding_dim
        )
        self.retriever = SimilarityRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder
        )
        self.generator = RAGGenerator(
            llm_client=llm_client,
            model=self.config.llm_model,
            temperature=self.config.llm_temperature
        )
        self._ingested_docs = 0
        self._ingested_chunks = 0
 
    def ingest_directory(
        self,
        directory_path: str,
        access_level: str = "internal",
        department: str = "general"
    ) -> None:
        """
        Load and index all documents in a directory.
 
        Args:
            directory_path : Path to directory containing documents
            access_level   : Access control level for all documents
            department     : Department tag for all documents
        """
        documents = self.loader.load_directory(directory_path)
        self.ingest_documents(documents, access_level, department)
 
    def ingest_documents(
        self,
        documents: List[Document],
        access_level: str = "internal",
        department: str = "general"
    ) -> None:
        """
        Chunk, embed, and index a list of Documents.
 
        Args:
            documents    : List of loaded Document objects
            access_level : Access control level
            department   : Department tag
        """
        all_chunks = []
 
        for doc in documents:
            enriched_metadata = enrich_metadata(
                doc.metadata,
                access_level=access_level,
                department=department
            )
            chunks = self.chunker.chunk(
                text=doc.content,
                metadata=enriched_metadata,
                doc_id=doc.doc_id
            )
            all_chunks.extend(chunks)
 
        if not all_chunks:
            print("No chunks produced — check document content")
            return
 
        embeddings = self._embed_chunks(all_chunks)
        self.vector_store.add_chunks(all_chunks, embeddings)
 
        self._ingested_docs += len(documents)
        self._ingested_chunks += len(all_chunks)
 
        print(
            f"Ingested {len(documents)} document(s) → "
            f"{len(all_chunks)} chunks → indexed"
        )
 
    def query(
        self,
        question: str,
        k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> GenerationResult:
        """
        Query the RAG pipeline with a question.
 
        Retrieves relevant chunks and generates an answer.
 
        Args:
            question        : User's question
            k               : Number of chunks to retrieve
            metadata_filter : Optional metadata filter for retrieval
 
        Returns:
            GenerationResult with answer and metadata
        """
        if self.vector_store.size == 0:
            raise RuntimeError(
                "Vector store is empty. Call ingest_directory() or "
                "ingest_documents() before querying."
            )
 
        retrieval_k = k or self.config.retrieval_k
 
        chunks = self.retriever.retrieve(
            query=question,
            k=retrieval_k,
            metadata_filter=metadata_filter
        )
 
        if not chunks:
            return GenerationResult(
                answer="No relevant documents found for this question.",
                question=question,
                context_chunks=[],
                model=self.config.llm_model
            )
 
        context_texts = [chunk.content for chunk in chunks]
        context_sources = [
            chunk.metadata.get("filename", chunk.chunk_id)
            for chunk in chunks
        ]
 
        result = self.generator.generate(
            question=question,
            context_chunks=context_texts,
            context_sources=context_sources
        )
 
        return result
 
    def save(self, path: str) -> None:
        """Save the vector store to disk."""
        self.vector_store.save(path)
 
    def load(self, path: str) -> None:
        """Load the vector store from disk."""
        self.vector_store.load(path)
 
    def _embed_chunks(self, chunks: List[Chunk]) -> List[List[float]]:
        """
        Embed a list of chunks using the configured embedder.
 
        Falls back to random vectors if no embedder configured —
        useful for testing pipeline structure without API keys.
        """
        if self.embedder is None:
            import random
            return [
                [random.random() for _ in range(self.config.embedding_dim)]
                for _ in chunks
            ]
 
        texts = [chunk.content for chunk in chunks]
        return self.embedder.embed_documents(texts)
 
    @property
    def stats(self) -> Dict[str, Any]:
        """Return pipeline ingestion statistics."""
        return {
            "ingested_documents": self._ingested_docs,
            "ingested_chunks": self._ingested_chunks,
            "vector_store_size": self.vector_store.size,
            "chunking_strategy": self.chunker.__class__.__name__,
            "retrieval_k": self.config.retrieval_k
        }
