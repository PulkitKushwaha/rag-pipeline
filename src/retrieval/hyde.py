"""
HyDE — Hypothetical Document Embeddings
 
HyDE is a retrieval technique that improves on naive similarity
search by bridging the semantic gap between queries and documents.
 
The problem with naive similarity search:
    A user asks: "What are the consequences of late payment?"
    The document says: "Failure to pay by the due date results
    in a 5% penalty fee applied to the outstanding balance."
 
    These sentences are semantically distant in embedding space
    even though they contain the same information. The query
    uses "consequences" and "late payment": the document uses
    "failure to pay", "due date", and "penalty fee". Cosine
    similarity between their embeddings may be low enough to
    miss the retrieval entirely.
 
How HyDE solves this:
    Step 1: Ask the LLM to generate a hypothetical answer to
            the query — one that looks like it came from a document.
    Step 2: Embed the hypothetical answer instead of the query.
    Step 3: Use the hypothetical answer's embedding to search
            the vector store.
 
    The LLM generates: "Late payment typically results in penalty
    fees applied to the outstanding balance, often ranging from
    2-5% of the amount due."
 
    This hypothetical answer uses document-like language and is
    semantically much closer to the actual document in embedding
    space — improving retrieval significantly.
 
When HyDE helps:
    - Complex or indirect queries
    - When query language differs from document language
    - Technical documents queried in plain language
    - Multi-hop questions requiring synthesis
 
When HyDE doesn't help (or hurts):
    - Short factual queries ("What is the CEO's name?")
    - When the LLM generates a confidently wrong hypothesis
    - High-latency requirements: adds one LLM call per query
    - Very small knowledge bases where all content is relevant
 
Reference:
    Gao et al. (2022) "Precise Zero-Shot Dense Retrieval without
    Relevance Labels" — https://arxiv.org/abs/2212.10496
"""
 
from typing import List, Optional, Dict, Any
from src.retrieval.retriever import BaseRetriever
from src.retrieval.vector_store import FAISSVectorStore
from src.ingestion.chunker import Chunk
 
 
class HyDERetriever(BaseRetriever):
    """
    Retriever using Hypothetical Document Embeddings (HyDE).
 
    Generates a hypothetical answer to the query using an LLM,
    then uses that answer's embedding for vector search instead
    of the original query embedding.
 
    Args:
        vector_store  : FAISSVectorStore with indexed chunks
        embedder      : Embedding model with embed_query() method
        llm_client    : OpenAI or Azure OpenAI client for hypothesis generation
        model         : LLM model for generating hypothetical answers
        n_hypotheses  : Number of hypothetical answers to generate (default: 1)
                        Using multiple hypotheses and averaging their embeddings
                        can improve robustness.
    """
 
    HYDE_PROMPT = """Write a short passage from a document that directly
answers the following question. Write it as if it is an excerpt from
an official document, policy, or knowledge base article.
Be specific and factual. Do not say "According to the document",
just write the passage directly.
 
Question: {question}
 
Passage:"""
 
    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder,
        llm_client=None,
        model: str = "gpt-4",
        n_hypotheses: int = 1
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm_client = llm_client
        self.model = model
        self.n_hypotheses = n_hypotheses
 
    def retrieve(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Retrieve chunks using HyDE.
 
        Generates a hypothetical answer, embeds it, and uses
        the embedding for vector search.
 
        Args:
            query           : User's original query
            k               : Number of chunks to retrieve
            metadata_filter : Optional metadata filter
 
        Returns:
            List of Chunk objects sorted by relevance
        """
        # Step 1 — generate hypothetical answer(s)
        hypotheses = self._generate_hypotheses(query)
 
        if not hypotheses:
            # Fallback to standard similarity search if generation fails
            print(f"[HyDE] Hypothesis generation failed — falling back to standard retrieval")
            query_embedding = self.embedder.embed_query(query)
            results = self.vector_store.search(query_embedding, k=k, metadata_filter=metadata_filter)
            return [chunk for chunk, score in results]
 
        # Step 2 — embed the hypothesis/hypotheses
        if len(hypotheses) == 1:
            search_embedding = self.embedder.embed_query(hypotheses[0])
        else:
            # Average embeddings across multiple hypotheses
            search_embedding = self._average_embeddings(hypotheses)
 
        # Step 3 — search with hypothesis embedding
        results = self.vector_store.search(
            search_embedding,
            k=k,
            metadata_filter=metadata_filter
        )
 
        return [chunk for chunk, score in results]
 
    def retrieve_with_hypothesis(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> tuple:
        """
        Retrieve chunks and return the hypothesis for transparency.
 
        Useful for debugging and evaluation — lets you see what
        hypothetical answer the LLM generated for the query.
 
        Returns:
            Tuple of (List[Chunk], hypothesis_text)
        """
        hypotheses = self._generate_hypotheses(query)
 
        if not hypotheses:
            query_embedding = self.embedder.embed_query(query)
            results = self.vector_store.search(query_embedding, k=k)
            return [chunk for chunk, _ in results], "Generation failed — used direct query"
 
        if len(hypotheses) == 1:
            search_embedding = self.embedder.embed_query(hypotheses[0])
        else:
            search_embedding = self._average_embeddings(hypotheses)
 
        results = self.vector_store.search(
            search_embedding,
            k=k,
            metadata_filter=metadata_filter
        )
 
        hypothesis_text = "\n---\n".join(hypotheses)
        return [chunk for chunk, _ in results], hypothesis_text
 
    def _generate_hypotheses(self, query: str) -> List[str]:
        """
        Generate N hypothetical document passages for the query.
 
        Each call to the LLM generates one hypothetical passage.
        With n_hypotheses > 1, multiple passages are generated
        and their embeddings are averaged for more robust retrieval.
        """
        hypotheses = []
 
        for _ in range(self.n_hypotheses):
            hypothesis = self._call_llm(query)
            if hypothesis:
                hypotheses.append(hypothesis)
 
        return hypotheses
 
    def _average_embeddings(self, texts: List[str]) -> List[float]:
        """
        Embed multiple texts and return the average embedding.
 
        Averaging embeddings across multiple hypothetical answers
        produces a more robust search vector — less sensitive to
        any single hypothesis being off-target.
        """
        import numpy as np
 
        embeddings = [self.embedder.embed_query(text) for text in texts]
        avg = np.mean(embeddings, axis=0)
        return avg.tolist()
 
    def _call_llm(self, query: str) -> str:
        """Generate a hypothetical document passage for the query."""
        prompt = self.HYDE_PROMPT.format(question=query)
 
        if self.llm_client is None:
            # Mock hypothesis for testing
            return (
                f"The answer to '{query}' involves the following key points: "
                f"this is a mock hypothetical document generated for testing "
                f"purposes without a real LLM client configured."
            )
 
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[HyDE] LLM call failed: {e}")
            return ""
