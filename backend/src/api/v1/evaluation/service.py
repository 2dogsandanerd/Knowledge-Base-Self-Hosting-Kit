import asyncio
from typing import List

from ragas import evaluate  # type: ignore

from .models import EvaluationRequest, EvaluationResult

import time
from opentelemetry import trace
from src.core.observability.metrics import (
    evaluation_requests_total,
    evaluation_latency_seconds,
    evaluation_faithfulness,
    evaluation_answer_relevancy
)

tracer = trace.get_tracer(__name__)

async def evaluate_ragas(request: EvaluationRequest) -> EvaluationResult:
    """Run RAGAS evaluation in a thread to avoid blocking the event loop.
    Instrumented with Prometheus metrics and OpenTelemetry tracing.
    """
    evaluation_requests_total.inc()
    start_time = time.time()

    # Defensive validation
    if len(request.generated) != len(request.reference):
        raise ValueError("generated and reference lists must have the same length")

    with tracer.start_as_current_span("ragas_evaluation"):
        # ``ragas.evaluate`` is synchronous, so we offload it to a thread.
        ragas_result = await asyncio.to_thread(
            evaluate,
            generated_texts=request.generated,
            reference_texts=request.reference,
        )

    duration = time.time() - start_time
    evaluation_latency_seconds.observe(duration)

    # Extract scores
    scores = {
        "faithfulness": getattr(ragas_result, "faithfulness", None),
        "relevance": getattr(ragas_result, "relevance", None),
        "answer_correctness": getattr(ragas_result, "answer_correctness", None),
        "context_precision": getattr(ragas_result, "context_precision", None),
        "context_recall": getattr(ragas_result, "context_recall", None),
    }
    
    # Record metric observations
    if scores.get("faithfulness") is not None:
        evaluation_faithfulness.observe(scores["faithfulness"])
    if scores.get("relevance") is not None:
        evaluation_answer_relevancy.observe(scores["relevance"]) # Mapping relevance to answer_relevancy metric

    # Remove ``None`` entries
    scores = {k: v for k, v in scores.items() if v is not None}

    return EvaluationResult(scores=scores)
