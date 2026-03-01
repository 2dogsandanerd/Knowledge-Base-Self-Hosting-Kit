"""
RAG Evaluator Service using RAGAS Framework.

This module provides comprehensive quality evaluation for RAG responses using:
- Faithfulness (hallucination detection)
- Answer Relevancy (answer quality)
- Context Precision (relevant chunks ranked high)
- Context Recall (all relevant chunks found)
- Context Relevancy (retrieved context relevant)
"""

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevancy
)
from typing import List, Dict, Optional
import asyncio
import logging
from datasets import Dataset

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluates RAG responses using RAGAS metrics."""

    def __init__(self):
        self.metrics = [
            faithfulness,        # 0-1: Hallucination detection
            answer_relevancy,    # 0-1: Answer quality
            context_precision,   # 0-1: Relevant chunks ranked high?
            context_recall,      # 0-1: All relevant chunks found?
            context_relevancy    # 0-1: Retrieved context relevant?
        ]

    async def evaluate_response(
        self,
        query: str,
        response: str,
        context: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single RAG response.

        Args:
            query: User query
            response: LLM-generated answer
            context: List of retrieved context chunks
            ground_truth: Optional expected answer

        Returns:
            Dict with metric scores (0-1)
        """
        # Prepare dataset in RAGAS format
        data = {
            "question": [query],
            "answer": [response],
            "contexts": [context],
        }

        if ground_truth:
            data["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data)

        # Run evaluation (synchronous, so run in executor)
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: evaluate(
                    dataset=dataset,
                    metrics=self.metrics
                )
            )

            # Extract scores
            scores = {
                "faithfulness": float(result.get("faithfulness", 0.0)),
                "answer_relevancy": float(result.get("answer_relevancy", 0.0)),
                "context_precision": float(result.get("context_precision", 0.0)),
                "context_recall": float(result.get("context_recall", 0.0)) if ground_truth else 0.0,
                "context_relevancy": float(result.get("context_relevancy", 0.0))
            }

            logger.info(f"Evaluation scores: {scores}")
            return scores

        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            # Return default scores on failure
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "context_relevancy": 0.0
            }

    def is_passing(
        self,
        scores: Dict[str, float],
        thresholds: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Check if scores meet quality thresholds.

        Args:
            scores: Metric scores from evaluation
            thresholds: Optional custom thresholds (defaults to strict values)

        Returns:
            True if all thresholds are met
        """
        if thresholds is None:
            thresholds = {
                "faithfulness": 0.9,      # Very strict on hallucinations
                "answer_relevancy": 0.7,  # Moderate on relevancy
                "context_precision": 0.6,
                "context_recall": 0.6,
                "context_relevancy": 0.7
            }

        return all(
            scores.get(metric, 0.0) >= threshold
            for metric, threshold in thresholds.items()
        )

    def get_overall_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate weighted overall score.

        Weights are based on importance for production quality:
        - Faithfulness is most critical (no hallucinations)
        - Answer relevancy is second most important
        - Context metrics are supporting indicators

        Args:
            scores: Metric scores from evaluation

        Returns:
            Weighted overall score (0-1)
        """
        weights = {
            "faithfulness": 0.3,      # Most important
            "answer_relevancy": 0.25,
            "context_precision": 0.15,
            "context_recall": 0.15,
            "context_relevancy": 0.15
        }

        overall = sum(
            scores.get(metric, 0.0) * weight
            for metric, weight in weights.items()
        )

        return overall

    def get_failure_reasons(
        self,
        scores: Dict[str, float],
        thresholds: Optional[Dict[str, float]] = None
    ) -> List[str]:
        """
        Get list of metrics that failed to meet thresholds.

        Args:
            scores: Metric scores from evaluation
            thresholds: Optional custom thresholds

        Returns:
            List of failure reasons
        """
        if thresholds is None:
            thresholds = {
                "faithfulness": 0.9,
                "answer_relevancy": 0.7,
                "context_precision": 0.6,
                "context_recall": 0.6,
                "context_relevancy": 0.7
            }

        failures = []
        for metric, threshold in thresholds.items():
            score = scores.get(metric, 0.0)
            if score < threshold:
                failures.append(
                    f"{metric}: {score:.2f} < {threshold:.2f}"
                )

        return failures
