"""
Live evaluation service for production queries.

This service evaluates a sample of production queries in the background
to monitor quality without impacting response times.
"""

import asyncio
import random
from typing import List, Dict, Any, Optional
import logging
import time

# Import RAGEvaluator only if available (optional dependency)
try:
    from src.core.evaluation.rag_evaluator import RAGEvaluator
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logging.warning("RAGAS not available - live evaluation disabled")

from src.core.observability.metrics import (
    evaluation_faithfulness,
    evaluation_answer_relevancy,
    evaluation_failures_total,
    evaluation_requests_total,
    evaluation_latency_seconds
)

logger = logging.getLogger(__name__)


class LiveEvaluator:
    """Evaluates production queries in background."""

    def __init__(self, sample_rate: float = 0.1, enabled: bool = True):
        """
        Initialize live evaluator.

        Args:
            sample_rate: Percentage of queries to evaluate (0.0-1.0)
            enabled: Whether evaluation is enabled
        """
        self.sample_rate = sample_rate
        self.enabled = enabled and RAGAS_AVAILABLE
        
        if self.enabled:
            self.evaluator = RAGEvaluator()
            logger.info(f"Live evaluator initialized with {sample_rate*100:.0f}% sampling")
        else:
            self.evaluator = None
            if not RAGAS_AVAILABLE:
                logger.warning("Live evaluator disabled - RAGAS not available")
            else:
                logger.info("Live evaluator disabled")

    def should_evaluate(self) -> bool:
        """Determine if this query should be evaluated."""
        if not self.enabled:
            return False
        return random.random() < self.sample_rate

    async def evaluate_and_log(
        self,
        query: str,
        response: str,
        context: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        query_id: Optional[str] = None
    ):
        """
        Evaluate a query response and log metrics.

        This runs in background and does not block the response.

        Args:
            query: User query text
            response: LLM-generated response
            context: Retrieved context chunks
            user_id: Optional user ID for tracking
            query_id: Optional query ID for correlation
        """
        if not self.should_evaluate():
            return

        evaluation_requests_total.inc()
        start_time = time.time()

        try:
            # Extract context content
            context_texts = [c.get('content', '') for c in context if isinstance(c, dict)]
            
            # Fallback if context is list of strings
            if not context_texts and context:
                context_texts = [str(c) for c in context]

            # Evaluate (no ground truth in production)
            scores = await self.evaluator.evaluate_response(
                query=query,
                response=response,
                context=context_texts,
                ground_truth=None
            )

            # Log to Prometheus
            if scores.get('faithfulness') is not None:
                evaluation_faithfulness.observe(scores['faithfulness'])
            
            if scores.get('answer_relevancy') is not None:
                evaluation_answer_relevancy.observe(scores['answer_relevancy'])

            # Calculate overall score
            overall = self.evaluator.get_overall_score(scores)

            # Check thresholds and log failures
            if not self.evaluator.is_passing(scores):
                failures = self.evaluator.get_failure_reasons(scores)
                
                for failure in failures:
                    metric = failure.split(':')[0].strip()
                    evaluation_failures_total.labels(metric=metric).inc()

                logger.warning(
                    "Low quality response detected",
                    extra={
                        "query_id": query_id,
                        "query": query[:100],
                        "scores": scores,
                        "overall_score": overall,
                        "failures": failures,
                        "user_id": user_id
                    }
                )
            else:
                logger.info(
                    f"Query evaluated: overall_score={overall:.2f}",
                    extra={
                        "query_id": query_id,
                        "scores": scores
                    }
                )

            # Log duration
            duration = time.time() - start_time
            evaluation_latency_seconds.observe(duration)

        except Exception as e:
            logger.error(
                f"Live evaluation failed: {e}",
                exc_info=True,
                extra={
                    "query_id": query_id,
                    "user_id": user_id
                }
            )


# Global instance
_live_evaluator = None


def get_live_evaluator(sample_rate: float = 0.1, enabled: bool = True) -> LiveEvaluator:
    """
    Get or create the global live evaluator instance.

    Args:
        sample_rate: Percentage of queries to evaluate (0.0-1.0)
        enabled: Whether evaluation is enabled

    Returns:
        LiveEvaluator instance
    """
    global _live_evaluator
    
    if _live_evaluator is None:
        _live_evaluator = LiveEvaluator(sample_rate=sample_rate, enabled=enabled)
    
    return _live_evaluator
