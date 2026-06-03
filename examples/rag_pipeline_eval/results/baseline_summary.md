# Baseline Evaluation Results
 
**Pipeline:** Recursive chunking + similarity retrieval (no reranking)
**Date:** 2024-03-15
**Dataset:** 30 questions across 8 categories
 
---
 
## Summary
 
| Metric | Score | Status | Bar |
|---|---|---|---|
| Faithfulness | `0.7823` | ✅ Good | ████████░░ |
| Answer Relevancy | `0.7654` | ✅ Good | ███████░░░ |
| Context Precision | `0.6891` | 🟡 Moderate | ██████░░░░ |
| Context Recall | `0.6201` | 🟡 Moderate | ██████░░░░ |
| **Overall** | **`0.7142`** | 🟡 **Good** | ███████░░░ |
 
---
 
## What this tells us
 
**Faithfulness (0.78) and Answer Relevancy (0.77) are acceptable.**
When the pipeline retrieves the right content, the LLM generates
faithful, on-topic answers. The generation layer is working well.
 
**Context Precision (0.69) is below threshold.**
The retriever is returning irrelevant chunks alongside relevant ones
— particularly for multi-hop and adversarial queries. This suggests
the embedding model struggles when query language differs significantly
from document language.
 
**Context Recall (0.62) is the weakest metric.**
The retriever is missing chunks that contain information needed to
answer questions fully. This is the most impactful failure — it
means the LLM never had the information needed to give a complete
answer, regardless of how good the generation is.
 
---
 
## Per-category breakdown
 
| Category | Faithfulness | Relevancy | Precision | Recall | Weakest |
|---|---|---|---|---|---|
| factual | 0.91 | 0.89 | 0.82 | 0.85 | — |
| summarization | 0.81 | 0.79 | 0.72 | 0.68 | recall |
| negative | 0.80 | 0.78 | 0.70 | 0.62 | recall |
| multi_hop | 0.72 | 0.70 | 0.59 | 0.49 | recall |
| comparison | 0.76 | 0.72 | 0.65 | 0.57 | recall |
| technical | 0.75 | 0.72 | 0.62 | 0.57 | recall |
| out_of_scope | 0.79 | 0.72 | 0.68 | 0.59 | recall |
| adversarial | 0.62 | 0.60 | 0.52 | 0.46 | all |
| ambiguous | 0.68 | 0.65 | 0.57 | 0.49 | all |
 
**Factual queries score well — everything else reveals gaps.**
 
---
 
## Key failure patterns
 
**1. Multi-hop retrieval failure**
Questions requiring information from multiple document sections
consistently score low on recall (avg 0.49). The recursive chunker
splits related content across chunk boundaries. Retrieving k=5
chunks isn't enough to cover all required sections.
 
**2. Adversarial query handling**
Questions with false premises cause partial faithfulness failures —
the LLM sometimes affirms the false claim rather than correcting it.
Recall is also low because the retriever matches the false premise
to unrelated chunks.
 
**3. Ambiguous query under-retrieval**
Underspecified queries retrieve only one interpretation of possible
answers. Recall suffers because the retriever commits to one semantic
direction early.
 
---
 
## Root cause diagnosis
 
| Issue | Root cause | Fix |
|---|---|---|
| Low context recall | chunk_size=1000 splits related content | Try sentence-window chunking |
| Low context precision | Similarity retrieval over-fetches noise | Add cross-encoder reranking |
| Adversarial failures | No false-premise detection | Add input guardrails |
| Multi-hop failures | k=5 insufficient for complex queries | Increase k + add reranking |
 
---
 
## Next step
 
See `hyde_reranking_results.json` for scores after adding
HyDE retrieval and cross-encoder reranking.
 
---
 
*Evaluated using [llm-eval-framework](https://github.com/pulkitkushwaha/llm-eval-framework)*
