"""
llm-eval-framework Integration for rag-pipeline
 
This module connects the rag-pipeline to the llm-eval-framework
evaluation library. It provides a clean interface to:
 
    1. Run a RAG pipeline against a test dataset
    2. Collect (question, answer, contexts, ground_truth) tuples
    3. Pass them to the Evaluator for scoring
    4. Save results in JSON and Markdown formats
 
Why this integration matters:
    Without evaluation, RAG pipeline changes are made by intuition.
    With this integration, every change to chunking strategy,
    retrieval algorithm, or prompt template produces a new eval
    report in minutes, making optimization systematic.
 
This is also the module that produced the benchmark results
documented in llm-eval-framework's reference implementation:
    - Baseline: 0.7142 overall
    - Optimized: 0.8156 overall (+14.2%)
 
Usage:
    evaluator = PipelineEvaluator(
        pipeline=my_rag_pipeline,
        dataset_path="path/to/test_questions.json"
    )
    report = evaluator.run()
    evaluator.save_results(report, output_dir="results/")
"""
 
import json
import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
 
 
@dataclass
class EvalQuestion:
    """A single question from the evaluation dataset."""
    id: str
    question: str
    ground_truth: str
    category: str
    notes: str = ""
 
 
@dataclass
class PipelineOutput:
    """Output from a RAG pipeline for a single question."""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str
    metadata: Dict[str, Any] = None
 
 
class PipelineEvaluator:
    """
    Evaluates a RAG pipeline using llm-eval-framework.
 
    Runs the pipeline against a test dataset, collects outputs,
    and produces an EvalReport with metric scores.
 
    Args:
        pipeline        : RAG pipeline callable (query: str) -> (answer: str, contexts: list)
        dataset_path    : Path to test_questions.json
        metrics         : List of metric names to evaluate (default: all 4 core metrics)
        pipeline_version: Version tag for tracking results over time
        model_name      : LLM model used (for metadata)
        verbose         : Print progress (default: True)
    """
 
    DEFAULT_METRICS = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall"
    ]
 
    def __init__(
        self,
        pipeline,
        dataset_path: str,
        metrics: Optional[List[str]] = None,
        pipeline_version: str = "unknown",
        model_name: str = "unknown",
        verbose: bool = True
    ):
        self.pipeline = pipeline
        self.dataset_path = dataset_path
        self.metrics = metrics or self.DEFAULT_METRICS
        self.pipeline_version = pipeline_version
        self.model_name = model_name
        self.verbose = verbose
 
    def run(self, max_samples: Optional[int] = None):
        """
        Run full evaluation pipeline:
            Load dataset → Run pipeline → Score → Return report
 
        Args:
            max_samples: Limit evaluation to first N samples (useful for quick checks)
 
        Returns:
            EvalReport from llm-eval-framework
        """
        try:
            from llm_eval import Evaluator
            from llm_eval.models import EvalSample
            from llm_eval.metrics import (
                Faithfulness, AnswerRelevancy,
                ContextPrecision, ContextRecall
            )
        except ImportError:
            raise ImportError(
                "llm-eval-framework is required for pipeline evaluation.\n"
                "Install from: https://github.com/pulkitkushwaha/llm-eval-framework\n"
                "Or: pip install -e path/to/llm-eval-framework"
            )
 
        # Load dataset
        questions = self._load_dataset(max_samples)
        if self.verbose:
            print(f"[PipelineEvaluator] Loaded {len(questions)} questions")
 
        # Run pipeline on each question
        samples = self._collect_outputs(questions)
        if self.verbose:
            print(f"[PipelineEvaluator] Collected {len(samples)} pipeline outputs")
 
        # Build metric instances
        metric_map = {
            "faithfulness": Faithfulness,
            "answer_relevancy": AnswerRelevancy,
            "context_precision": ContextPrecision,
            "context_recall": ContextRecall
        }
 
        metric_instances = []
        for metric_name in self.metrics:
            if metric_name in metric_map:
                metric_instances.append(metric_map[metric_name]())
            else:
                print(f"[PipelineEvaluator] Unknown metric: {metric_name} — skipping")
 
        # Run evaluation
        evaluator = Evaluator(
            metrics=metric_instances,
            metadata={
                "pipeline_version": self.pipeline_version,
                "model": self.model_name,
                "dataset": self.dataset_path,
                "num_questions": len(questions)
            },
            verbose=self.verbose
        )
 
        report = evaluator.evaluate(samples)
        return report
 
    def run_by_category(self) -> Dict[str, Any]:
        """
        Run evaluation separately for each question category.
 
        Returns per-category scores — useful for identifying
        which query types the pipeline handles poorly.
 
        Returns:
            Dict mapping category name to EvalReport
        """
        questions = self._load_dataset()
        categories = list({q.category for q in questions})
        results = {}
 
        for category in categories:
            category_questions = [q for q in questions if q.category == category]
            if self.verbose:
                print(f"\n[PipelineEvaluator] Evaluating category: {category} ({len(category_questions)} questions)")
 
            samples = self._collect_outputs(category_questions)
            try:
                from llm_eval import Evaluator
                from llm_eval.models import EvalSample
                from llm_eval.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
 
                evaluator = Evaluator(
                    metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
                    verbose=False
                )
                results[category] = evaluator.evaluate(samples)
            except Exception as e:
                print(f"[PipelineEvaluator] Failed to evaluate category {category}: {e}")
 
        return results
 
    def save_results(
        self,
        report,
        output_dir: str = "results",
        run_name: str = "eval"
    ) -> None:
        """
        Save evaluation report in JSON and Markdown formats.
 
        Args:
            report     : EvalReport from run()
            output_dir : Directory to save results
            run_name   : Name prefix for result files
        """
        try:
            from llm_eval.reporters import JSONReporter, MarkdownReporter
        except ImportError:
            print("[PipelineEvaluator] llm-eval-framework not installed — cannot save")
            return
 
        os.makedirs(output_dir, exist_ok=True)
 
        json_path = os.path.join(output_dir, f"{run_name}_results.json")
        md_path = os.path.join(output_dir, f"{run_name}_summary.md")
 
        JSONReporter().save(report, json_path)
        MarkdownReporter(include_per_sample=True).save(report, md_path)
 
        print(f"\n[PipelineEvaluator] Results saved:")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")
        print(f"\n{report.summary()}")
 
    def _load_dataset(
        self,
        max_samples: Optional[int] = None
    ) -> List[EvalQuestion]:
        """Load questions from the test dataset JSON file."""
        path = Path(self.dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
 
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
 
        questions = [
            EvalQuestion(
                id=item.get("id", f"q{i:03d}"),
                question=item["question"],
                ground_truth=item.get("ground_truth", ""),
                category=item.get("category", "general"),
                notes=item.get("notes", "")
            )
            for i, item in enumerate(data)
            if item.get("question")
        ]
 
        if max_samples:
            questions = questions[:max_samples]
 
        return questions
 
    def _collect_outputs(self, questions: List[EvalQuestion]):
        """
        Run the RAG pipeline on each question and collect outputs.
 
        The pipeline callable must return either:
            - A string (just the answer) — contexts will be empty
            - A tuple of (answer: str, contexts: list) — preferred
            - A dict with 'answer' and 'contexts' keys
        """
        try:
            from llm_eval.models import EvalSample
        except ImportError:
            raise ImportError("llm-eval-framework required")
 
        samples = []
 
        for i, question in enumerate(questions):
            if self.verbose and i % 5 == 0:
                print(f"[PipelineEvaluator] Running question {i+1}/{len(questions)}...")
 
            try:
                raw_output = self.pipeline(question.question)
 
                # Parse pipeline output
                if isinstance(raw_output, tuple) and len(raw_output) == 2:
                    answer, contexts = raw_output
                elif isinstance(raw_output, dict):
                    answer = raw_output.get("answer", "")
                    contexts = raw_output.get("contexts", [])
                else:
                    answer = str(raw_output)
                    contexts = []
 
                # Ensure contexts is a list of strings
                if contexts and not isinstance(contexts[0], str):
                    contexts = [
                        c.page_content if hasattr(c, 'page_content') else str(c)
                        for c in contexts
                    ]
 
                samples.append(EvalSample(
                    question=question.question,
                    answer=answer,
                    contexts=contexts if contexts else ["No context retrieved"],
                    ground_truth=question.ground_truth or None
                ))
 
            except Exception as e:
                print(f"[PipelineEvaluator] Pipeline failed on question {i+1}: {e}")
                samples.append(EvalSample(
                    question=question.question,
                    answer=f"Pipeline error: {str(e)}",
                    contexts=["Pipeline execution failed"],
                    ground_truth=question.ground_truth or None
                ))
 
        return samples
