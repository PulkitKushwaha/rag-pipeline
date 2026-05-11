# rag-pipeline
 
This repo is a production-focused RAG pipeline implementation covering the full spectrum of retrieval-augmented generation, right from basic PDF ingestion to advanced retrieval strategies like HyDE, cross-encoder reranking, metadata filtering, and GraphRAG.
 
I have built this as an attempt to go beyond the tutorial. Every design decision is documented, every retrieval strategy is benchmarked, and the entire pipeline is evaluated using the [llm-eval-framework](https://github.com/pulkitkushwaha/llm-eval-framework).

> RAG is not just "embed + retrieve + generate". The difference between
> a prototype and a production system is everything in between:
> chunking strategy, retrieval quality, reranking, metadata filtering,
> and continuous evaluation. This repo covers all of it.
 
---
 
## What this covers
 
| Area | Topics |
|---|---|
| Document ingestion | PDF, TXT, DOCX loading, metadata extraction |
| Chunking strategies | Fixed, recursive, semantic, sentence-window |
| Embeddings | OpenAI, Azure OpenAI, HuggingFace sentence-transformers |
| Vector stores | FAISS, with hybrid search support |
| Retrieval strategies | Naive similarity, HyDE, cross-encoder reranking |
| Advanced retrieval | Metadata filtering, multi-doc RAG, GraphRAG-lite |
| Generation | Prompt templates, structured outputs, streaming |
| Evaluation | Integrated with llm-eval-framework (RAGAS metrics) |
 
---
 
## Why this repo exists
 
Most RAG tutorials stop at the happy path: load a PDF, embed it, ask
a question, get an answer. Production RAG is a different problem entirely.
 
Real systems fail because of:
- **Chunking strategy**: chunks too large lose precision, too small
  lose context. The right strategy depends on document structure.
- **Retrieval quality**: cosine similarity retrieves what's
  semantically close, not always what's relevant. Reranking fixes this.
- **Context window management**: flooding the LLM with irrelevant
  chunks increases cost and degrades answer quality.
- **Evaluation**: without measuring faithfulness, relevancy, and
  recall, you don't know when your pipeline breaks.
This repo is a systematic exploration of these problems, with
implementations, benchmarks, and honest failure analysis.
 
---
 
## Architecture
 
```
Documents (PDF, TXT, DOCX)
        ↓
[ Ingestion Layer ]
  - Document loading and parsing
  - Metadata extraction (source, date, author)
        ↓
[ Chunking Layer ]
  - Fixed size chunking
  - Recursive character chunking
  - Semantic chunking
  - Sentence-window chunking
        ↓
[ Embedding Layer ]
  - OpenAI text-embedding-3-small
  - Azure OpenAI embeddings
  - HuggingFace sentence-transformers
        ↓
[ Vector Store ]
  - FAISS (primary)
  - Hybrid search support
        ↓
[ Retrieval Layer ]
  - Naive similarity search (baseline)
  - HyDE (Hypothetical Document Embeddings)
  - Cross-encoder reranking
  - Metadata filtering
  - GraphRAG-lite (entity + graph traversal)
        ↓
[ Generation Layer ]
  - Prompt templates (structured, grounded)
  - Azure OpenAI GPT-4
  - Structured output with Pydantic
        ↓
[ Evaluation Layer ]
  - llm-eval-framework integration
  - RAGAS metrics (faithfulness, relevancy, precision, recall)
  - Before/after benchmarks per retrieval strategy
```
 
---
 
## Repository structure
 
```
rag-pipeline/
├── src/
│   ├── ingestion/
│   │   ├── loader.py          # Document loading (PDF, TXT, DOCX)
│   │   ├── chunker.py         # All chunking strategies
│   │   └── metadata.py        # Metadata extraction and tagging
│   ├── embeddings/
│   │   ├── openai_embedder.py     # OpenAI / Azure OpenAI embeddings
│   │   └── hf_embedder.py         # HuggingFace sentence-transformers
│   ├── retrieval/
│   │   ├── vector_store.py        # FAISS vector store wrapper
│   │   ├── retriever.py           # Base retriever + similarity search
│   │   ├── hyde.py                # HyDE retrieval strategy
│   │   ├── reranker.py            # Cross-encoder reranking
│   │   └── metadata_filter.py     # Metadata-based filtering
│   ├── generation/
│   │   ├── prompts.py             # Prompt templates
│   │   ├── generator.py           # LLM generation wrapper
│   │   └── output_parser.py       # Structured output with Pydantic
│   └── pipeline.py                # End-to-end pipeline orchestration
├── data/
│   └── sample_docs/               # Sample PDFs for testing
├── notebooks/
│   ├── chunking_comparison.ipynb  # Chunking strategy benchmarks
│   └── retrieval_comparison.ipynb # Retrieval strategy benchmarks
├── tests/
├── configs/
│   └── pipeline_config.yaml       # Pipeline configuration
├── .env.example
└── requirements.txt
```
 
---
 
## Design decisions
 
**Why FAISS over Chroma or Pinecone?**
FAISS is a local, dependency-light vector store that runs without
any external service. For a portfolio project, this means anyone can
clone and run without API keys or cloud accounts beyond the LLM.
In production, the retrieval layer is abstracted so swapping to
Pinecone or Azure AI Search is a config change, not a code change.
 
**Why evaluate every retrieval strategy?**
Intuition about RAG performance is almost always wrong. HyDE sounds
clever but doesn't always outperform naive similarity. Reranking
adds latency, sometimes the precision gain isn't worth it. The
benchmarks in this repo show the actual tradeoffs, not the theoretical ones.
 
**Why Azure OpenAI as the primary LLM?**
Matches production experience. Azure OpenAI provides enterprise-grade
security, RBAC, and compliance controls that matter in real deployments.
All LLM calls are abstracted behind a generator interface, swapping
to OpenAI directly is a one-line config change.
 
**Why structured outputs?**
Free-form LLM responses are not suitable for production systems that
need to parse, route, or act on the output. Every generation in this
pipeline uses Pydantic schemas to enforce output structure and catch
malformed responses before they reach downstream systems.
 
---
 
## Evaluation strategy
 
Every retrieval strategy is evaluated using
[llm-eval-framework](https://github.com/pulkitkushwaha/llm-eval-framework)
across four metrics:
 
| Metric | What it measures |
|---|---|
| Faithfulness | Are answers grounded in retrieved context? |
| Answer Relevancy | Do answers address the actual question? |
| Context Precision | Are retrieved chunks relevant to the query? |
| Context Recall | Does retrieval find all necessary information? |
 
Benchmark results are committed alongside the code, so you can see
exactly how each optimization affected the metrics.
 
---
 
## Status
 
| Component | Status |
|---|---|
| Document ingestion (PDF, TXT) | In Progress |
| Fixed and recursive chunking | In Progress |
| Semantic and sentence-window chunking | Coming soon |
| FAISS vector store setup | Coming soon |
| Basic similarity retrieval | Coming soon |
| HyDE retrieval | Coming soon |
| Cross-encoder reranking | Coming soon |
| Metadata filtering | Coming soon |
| GraphRAG-lite | Coming soon |
| RAGAS evaluation integration | Coming soon |
 
---
 
*Part of the [ai-engineering-portfolio](https://github.com/pulkitkushwaha/ai-engineering-portfolio)
— built by [Pulkit Kushwaha](https://linkedin.com/in/pulkit-kushwaha)*
