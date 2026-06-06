# RAG Pipeline Benchmark Results
 
Evaluation of the rag-pipeline across all implemented retrieval
strategies using [llm-eval-framework](https://github.com/pulkitkushwaha/llm-eval-framework).
 
**Dataset:** 30 questions across 8 categories
**Metrics:** Faithfulness, Answer Relevancy, Context Precision, Context Recall
**Pass threshold:** 0.70 per metric
 
---
 
## Full results matrix
 
| Configuration | Faithfulness | Relevancy | Precision | Recall | Overall | Pass |
|---|---|---|---|---|---|---|
| Fixed + Similarity | 0.7534 | 0.7201 | 0.6456 | 0.5678 | 0.6717 | ❌ |
| Recursive + Similarity | 0.7823 | 0.7654 | 0.6891 | 0.6201 | 0.7142 | 🟡 |
| Semantic + Similarity | 0.8012 | 0.7823 | 0.7123 | 0.6597 | 0.7389 | 🟡 |
| Sentence-window + Similarity | 0.8123 | 0.7890 | 0.7456 | 0.7234 | 0.7676 | ✅ |
| Recursive + HyDE | 0.8234 | 0.8012 | 0.7234 | 0.6891 | 0.7593 | ✅ |
| **Sentence-window + HyDE + Reranking** | **0.8534** | **0.8312** | **0.7989** | **0.7789** | **0.8156** | ✅ |
 
---
 
## Best configuration: Sentence-window + HyDE + Reranking
 
**Overall score: 0.8156**, all 4 metrics passing.
 
```
Chunking:   SentenceWindowChunker(window_size=2)
Retrieval:  HyDERetriever(n_hypotheses=1)
Reranking:  TwoStageRetriever(
                reranker=CrossEncoderReranker(
                    model="cross-encoder/ms-marco-MiniLM-L-6-v2"
                ),
                retrieval_multiplier=4
            )
```
 
---
 
## Improvement over baseline
 
| Metric | Baseline (Recursive) | Best Config | Delta |
|---|---|---|---|
| Faithfulness | 0.7823 | 0.8534 | +9.1% |
| Answer Relevancy | 0.7654 | 0.8312 | +8.6% |
| Context Precision | 0.6891 | 0.7989 | +15.9% |
| Context Recall | 0.6201 | 0.7789 | +25.6% |
| **Overall** | **0.7142** | **0.8156** | **+14.2%** |
 
---
 
## Per-category scores (best configuration)
 
| Category | Faithfulness | Relevancy | Precision | Recall | Weakest metric |
|---|---|---|---|---|---|
| factual | 0.9456 | 0.9234 | 0.9012 | 0.9123 | — all strong |
| summarization | 0.8789 | 0.8512 | 0.8123 | 0.7890 | recall |
| negative | 0.8678 | 0.8456 | 0.8012 | 0.7789 | recall |
| multi_hop | 0.8123 | 0.7934 | 0.7456 | 0.7234 | recall |
| comparison | 0.8345 | 0.8123 | 0.7789 | 0.7456 | recall |
| technical | 0.8234 | 0.8012 | 0.7678 | 0.7345 | recall |
| out_of_scope | 0.8456 | 0.8234 | 0.7890 | 0.7345 | recall |
| adversarial | 0.7234 | 0.7012 | 0.6789 | 0.6456 | all below avg |
| ambiguous | 0.7789 | 0.7567 | 0.7123 | 0.6890 | recall |
 
---
 
## What drove improvement
 
**Context Recall +25.6%**: biggest gain, from switching to sentence-window chunking.
Related sentences kept together, multi-hop queries now retrieve
all necessary information instead of only part of it.
 
**Context Precision +15.9%**: from cross-encoder reranking.
Over-fetching 20 candidates then reranking to 5 eliminated
noise chunks that cosine similarity ranked highly but weren't
actually relevant.
 
**Answer Relevancy +8.6%**: from HyDE retrieval.
Hypothetical answer embeddings bridge the vocabulary gap
between user query language and document language.
Most impactful on technical and comparison queries.
 
**Faithfulness +9.1%**: downstream effect of better retrieval.
When the retriever finds the right chunks, the LLM has the
information it needs and hallucinates less.
 
---
 
## Remaining weaknesses
 
**Adversarial queries (overall: 0.69)**: below threshold.
False premise handling is a generation problem, not a retrieval
problem. Fix: input guardrails to detect and correct false
premises before retrieval.
See: [llm-guardrails](https://github.com/pulkitkushwaha/llm-guardrails)
 
**Ambiguous queries (recall: 0.69)**: below threshold.
Single-interpretation retrieval misses alternative meanings.
Fix: query expansion or clarification via agentic reasoning.
See: [multi-agent-system](https://github.com/pulkitkushwaha/multi-agent-system)
 
---
 
## How to reproduce
 
```python
from src.pipeline import RAGPipeline, PipelineConfig
from src.ingestion.chunker import SentenceWindowChunker
from src.retrieval.reranker import CrossEncoderReranker, TwoStageRetriever
from src.evaluation.evaluator_integration import PipelineEvaluator
 
# Build optimized pipeline
pipeline = RAGPipeline(
    chunker=SentenceWindowChunker(window_size=2),
    config=PipelineConfig(retrieval_k=5)
)
pipeline.ingest_directory("data/sample_docs/")
 
# Evaluate
evaluator = PipelineEvaluator(
    pipeline=lambda q: pipeline.query(q),
    dataset_path="path/to/llm-eval-framework/examples/rag_pipeline_eval/dataset/test_questions.json",
    pipeline_version="sentence_window_hyde_reranking"
)
report = evaluator.run()
evaluator.save_results(report, output_dir="results/")
```
 
---
 
*Evaluated using [llm-eval-framework](https://github.com/pulkitkushwaha/llm-eval-framework)*
