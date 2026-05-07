# rag-pipeline
 
This repo is a production-focused RAG pipeline implementation covering the full spectrum of retrieval-augmented generation, right from basic PDF ingestion to advanced retrieval strategies like HyDE, cross-encoder reranking, metadata filtering, and GraphRAG.
 
I have built this as an attempt to go beyond the tutorial. Every design decision is documented, every retrieval strategy is benchmarked, and the entire pipeline is evaluated using the [llm-eval-framework](https://github.com/pulkitkushwaha/llm-eval-framework).
 
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
 
## Structure
 
```
rag-pipeline/
├── src/
│   ├── ingestion/        # Document loading and chunking
│   ├── embeddings/       # Embedding model wrappers
│   ├── retrieval/        # Vector store and retrieval strategies
│   ├── generation/       # LLM generation and prompt templates
│   └── pipeline.py       # End-to-end pipeline orchestration
├── data/
│   └── sample_docs/      # Sample PDFs for testing
├── notebooks/            # Experimentation and comparison notebooks
├── tests/                # Unit tests
├── configs/              # Pipeline configuration files
├── .env.example
└── requirements.txt
```
 
---
 
## Status
 
| Component | Status |
|---|---|
| Document ingestion (PDF, TXT) | Coming soon |
| Fixed and recursive chunking | Coming soon |
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
