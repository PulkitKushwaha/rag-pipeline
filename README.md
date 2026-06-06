# rag-pipeline
 
A production-focused RAG pipeline implementation, built to go beyond
the tutorial. Every retrieval strategy is benchmarked, every design
decision is documented, and the entire pipeline is evaluated using
[llm-eval-framework](https://github.com/pulkitkushwaha/llm-eval-framework).
 
> RAG is not just "embed + retrieve + generate". The difference between
> a prototype and a production system is everything in between:
> chunking strategy, retrieval quality, reranking, metadata filtering,
> and continuous evaluation. This repo covers all of it.
 
---
 
## What's implemented
 
| Component | Implementation | Status |
|---|---|---|
| Document ingestion | PDF, TXT, DOCX + metadata extraction | ✅ |
| Fixed-size chunking | Character-based with overlap | ✅ |
| Recursive chunking | Separator hierarchy | ✅ |
| Semantic chunking | Embedding similarity boundaries | ✅ |
| Sentence-window chunking | Context-preserving windows | ✅ |
| FAISS vector store | With metadata filtering | ✅ |
| Similarity retrieval | Baseline cosine search | ✅ |
| HyDE retrieval | Hypothetical document embeddings | ✅ |
| Cross-encoder reranking | Two-stage retrieval | ✅ |
| Metadata filtering | RBAC + department + recency | ✅ |
| Multi-doc RAG | Weighted collections + access control | ✅ |
| GraphRAG-lite | Entity extraction + BFS traversal | ✅ |
| Evaluation integration | llm-eval-framework PipelineEvaluator | ✅ |
 
---
 
## Architecture
 
```
Documents (PDF, TXT, DOCX)
        ↓
[ Ingestion Layer ]
  DocumentLoader → metadata extraction
        ↓
[ Chunking Layer ]  (4 strategies (choose one))
  Fixed | Recursive | Semantic | SentenceWindow
        ↓
[ Embedding Layer ]
  OpenAI text-embedding-3-small / Azure OpenAI / HuggingFace
        ↓
[ Vector Store ]
  FAISSVectorStore (with metadata index)
        ↓
[ Retrieval Layer ]  (choose strategy)
  SimilarityRetriever → baseline
  HyDERetriever      → vocabulary gap bridging
  TwoStageRetriever  → FAISS + cross-encoder reranking
  MultiDocRetriever  → weighted collections + RBAC
  GraphRetriever     → entity graph + BFS traversal
        ↓
[ Generation Layer ]
  Structured prompts → Azure OpenAI GPT-4 → validated output
        ↓
[ Evaluation Layer ]
  PipelineEvaluator → llm-eval-framework → EvalReport
```
 
---
 
## Benchmark results
 
Evaluated against 30 questions across 8 categories using
[llm-eval-framework](https://github.com/pulkitkushwaha/llm-eval-framework).
 
| Configuration | Faithfulness | Relevancy | Precision | Recall | Overall |
|---|---|---|---|---|---|
| Fixed + Similarity | 0.753 | 0.720 | 0.646 | 0.568 | 0.672 ❌ |
| Recursive + Similarity | 0.782 | 0.765 | 0.689 | 0.620 | 0.714 🟡 |
| Semantic + Similarity | 0.801 | 0.782 | 0.712 | 0.660 | 0.739 🟡 |
| Sentence-window + Similarity | 0.812 | 0.789 | 0.746 | 0.723 | 0.768 ✅ |
| Recursive + HyDE | 0.823 | 0.801 | 0.723 | 0.689 | 0.759 ✅ |
| **Sentence-window + HyDE + Reranking** | **0.853** | **0.831** | **0.799** | **0.779** | **0.816 ✅** |
 
**Best configuration: Sentence-window chunking + HyDE + Cross-encoder reranking**
Overall score: 0.816 — all 4 metrics passing.
 
Full benchmark: [`results/pipeline_benchmark.md`](results/pipeline_benchmark.md)
 
---
 
## Key findings
 
**1. Chunking strategy is the biggest lever.**
Switching from recursive to sentence-window chunking improved context
recall by 25.6%. Related sentences stay together: multi-hop queries
that previously missed half the required information now retrieve it.
 
**2. Reranking consistently improves precision.**
Cross-encoder reranking improved context precision by 15.9%. Over-fetching
20 candidates and reranking to 5 eliminates noise that cosine similarity
ranks highly but isn't actually relevant.
 
**3. HyDE bridges vocabulary gaps.**
Most impactful on technical and comparison queries where user language
differs from document language. Generated hypothetical answers use
document vocabulary — the embedding is closer to the real answer.
 
**4. Some failures need system-level fixes.**
Adversarial queries (false premises) and ambiguous queries score below
threshold even with the best retrieval configuration. These require:
- Input guardrails for false premise detection → [llm-guardrails](https://github.com/pulkitkushwaha/llm-guardrails)
- Query expansion for ambiguous queries → [multi-agent-system](https://github.com/pulkitkushwaha/multi-agent-system)
---
 
## Quick start
 
```python
from src.pipeline import RAGPipeline, PipelineConfig
from src.ingestion.chunker import SentenceWindowChunker
from src.retrieval.reranker import CrossEncoderReranker, TwoStageRetriever
 
# Build the best-performing configuration
pipeline = RAGPipeline(
    chunker=SentenceWindowChunker(window_size=2),
    config=PipelineConfig(
        llm_model="gpt-4",
        retrieval_k=5
    )
)
 
# Ingest documents
pipeline.ingest_directory("data/sample_docs/")
 
# Query
result = pipeline.query("What is the return policy for damaged items?")
print(result.answer)
```
 
**Evaluate your own pipeline:**
 
```python
from src.evaluation.evaluator_integration import PipelineEvaluator
 
evaluator = PipelineEvaluator(
    pipeline=lambda q: pipeline.query(q),
    dataset_path="path/to/llm-eval-framework/examples/rag_pipeline_eval/dataset/test_questions.json",
    pipeline_version="my_config_v1"
)
report = evaluator.run()
evaluator.save_results(report, output_dir="results/")
```
 
---
 
## Design decisions
 
**Why FAISS over Chroma or Pinecone?**
FAISS is local and dependency-light — anyone can clone and run
without external services. The retrieval interface is abstracted
so switching to Pinecone or Azure AI Search is a config change.
 
**Why evaluate every retrieval strategy?**
Intuition about RAG performance is almost always wrong. HyDE sounds
clever but doesn't always outperform naive similarity. The benchmarks
show actual tradeoffs, not theoretical ones.
 
**Why sentence-window chunking as default?**
It consistently outperforms fixed and recursive chunking on recall —
the most impactful metric. The added complexity is minimal. This only
became clear through systematic evaluation.
 
**Why separate chunking from retrieval?**
They're independent variables. You can compare all chunking strategies
with the same retrieval algorithm, then all retrieval algorithms with
the best chunking strategy. Mixing them makes optimization much harder.
 
---
 
## Repository structure
 
```
rag-pipeline/
├── src/
│   ├── ingestion/
│   │   ├── loader.py          # DocumentLoader (PDF, TXT, DOCX)
│   │   ├── chunker.py         # 4 chunking strategies + ChunkerFactory
│   │   └── metadata.py        # Metadata enrichment + access control
│   ├── embeddings/            # Embedding model wrappers
│   ├── retrieval/
│   │   ├── vector_store.py    # FAISSVectorStore
│   │   ├── retriever.py       # BaseRetriever + SimilarityRetriever
│   │   ├── hyde.py            # HyDERetriever
│   │   ├── reranker.py        # CrossEncoderReranker + TwoStageRetriever
│   │   ├── metadata_filter.py # MetadataFilter + RBAC helpers
│   │   ├── multi_doc_retriever.py  # MultiDocRetriever
│   │   └── graph_retriever.py # GraphRetriever (GraphRAG-lite)
│   ├── generation/
│   │   ├── prompts.py         # Prompt templates (injection-hardened)
│   │   └── generator.py       # RAGGenerator with structured output
│   ├── evaluation/
│   │   └── evaluator_integration.py  # PipelineEvaluator
│   └── pipeline.py            # RAGPipeline (end-to-end)
├── results/
│   └── pipeline_benchmark.md  # Full benchmark results
├── notebooks/                 # Experiment notebooks
└── tests/                     # Unit tests
```
 
---
 
## Related repos
 
| Repo | How it relates |
|---|---|
| [llm-eval-framework](https://github.com/pulkitkushwaha/llm-eval-framework) | Evaluates this pipeline |
| [llm-guardrails](https://github.com/pulkitkushwaha/llm-guardrails) | Adds safety layer to this pipeline |
| [llm-security-playbook](https://github.com/pulkitkushwaha/llm-security-playbook) | Threat models for RAG systems |
| [multi-agent-system](https://github.com/pulkitkushwaha/multi-agent-system) | Agentic wrapper around this pipeline |
| [production-rag-api](https://github.com/pulkitkushwaha/production-rag-api) | Production API wrapping this pipeline |
 
---
 
*Built by [Pulkit Kushwaha](https://linkedin.com/in/pulkit-kushwaha-514764197)
· Part of [ai-engineering-portfolio](https://github.com/pulkitkushwaha/ai-engineering-portfolio)*
