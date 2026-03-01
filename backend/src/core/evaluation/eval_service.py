"""
Evaluation Service - RAG Quality Metrics using RAGAS

This service evaluates RAG query quality using the RAGAS framework:
- context_precision: Are retrieved chunks relevant?
- context_recall: Did we retrieve all necessary information?
- faithfulness: Is the answer derived from the context?
- answer_relevancy: Does the answer address the query?
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger

try:
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("RAGAS not installed. Evaluation metrics will be disabled.")


class EvaluationService:
    """
    Service for evaluating RAG query quality.
    
    Uses RAGAS framework to measure:
    - Context Precision
    - Context Recall  
    - Faithfulness
    - Answer Relevancy
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize evaluation service.
        
        Args:
            log_dir: Directory to store evaluation logs
        """
        self.logger = logger.bind(component="EvaluationService")
        
        if not RAGAS_AVAILABLE:
            self.logger.warning("RAGAS not available - evaluation disabled")
            self.enabled = False
            return
        
        self.enabled = True
        
        # Setup log directory
        if log_dir is None:
            log_dir = Path(__file__).parent.parent.parent.parent / "data" / "logs"
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "evaluations.jsonl"
        
        self.logger.info(f"Evaluation service initialized. Logs: {self.log_file}")
    
    async def evaluate_query(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        query_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single RAG query using RAGAS metrics.
        
        Args:
            query: User query
            answer: Generated answer
            contexts: Retrieved context chunks
            ground_truth: Optional ground truth answer (for recall)
            query_id: Optional query identifier
            
        Returns:
            Dict with evaluation metrics
        """
        if not self.enabled:
            return {"error": "RAGAS not available"}
        
        start_time = time.time()
        
        try:
            # Prepare data in RAGAS format
            data = {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts],
            }
            
            # Add ground truth if available (needed for context_recall)
            if ground_truth:
                data["ground_truth"] = [ground_truth]
            
            # Create dataset
            dataset = Dataset.from_dict(data)
            
            # Select metrics based on available data
            metrics = [
                context_precision,
                faithfulness,
                answer_relevancy
            ]
            
            # Only add context_recall if we have ground truth
            if ground_truth:
                metrics.append(context_recall)
            
            # Run evaluation in a separate thread to avoid nested event loop issues with RAGAS
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: evaluate(
                    dataset,
                    metrics=metrics
                )
            )
            
            # Extract scores
            scores = {
                "context_precision": float(result["context_precision"]) if "context_precision" in result else None,
                "context_recall": float(result["context_recall"]) if "context_recall" in result else None,
                "faithfulness": float(result["faithfulness"]) if "faithfulness" in result else None,
                "answer_relevancy": float(result["answer_relevancy"]) if "answer_relevancy" in result else None,
            }
            
            # Calculate average score
            valid_scores = [v for v in scores.values() if v is not None]
            avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            
            evaluation_time = time.time() - start_time
            
            result_dict = {
                "query_id": query_id or f"eval_{int(time.time() * 1000)}",
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "answer": answer[:200] + "..." if len(answer) > 200 else answer,  # Truncate for logs
                "num_contexts": len(contexts),
                "metrics": scores,
                "average_score": avg_score,
                "evaluation_time_ms": int(evaluation_time * 1000)
            }
            
            # Log asynchronously
            asyncio.create_task(self._log_metrics(result_dict))
            
            self.logger.info(f"Evaluation complete: avg_score={avg_score:.3f}, time={evaluation_time:.2f}s")
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "query_id": query_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _log_metrics(self, metrics: Dict[str, Any]):
        """
        Log evaluation metrics to JSONL file.
        
        Args:
            metrics: Evaluation metrics dict
        """
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(metrics) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {e}")
    
    async def get_stats(self, limit: int = 100) -> Dict[str, Any]:
        """
        Get aggregated evaluation statistics.
        
        Args:
            limit: Number of recent evaluations to analyze
            
        Returns:
            Dict with aggregated stats
        """
        if not self.log_file.exists():
            return {
                "total_evaluations": 0,
                "message": "No evaluations logged yet"
            }
        
        try:
            # Read recent evaluations
            evaluations = []
            with open(self.log_file, "r") as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    try:
                        evaluations.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            
            if not evaluations:
                return {
                    "total_evaluations": 0,
                    "message": "No valid evaluations found"
                }
            
            # Calculate averages
            metrics_sums = {
                "context_precision": [],
                "context_recall": [],
                "faithfulness": [],
                "answer_relevancy": [],
                "average_score": []
            }
            
            for eval_data in evaluations:
                if "metrics" in eval_data:
                    for metric, value in eval_data["metrics"].items():
                        if value is not None:
                            metrics_sums[metric].append(value)
                
                if "average_score" in eval_data:
                    metrics_sums["average_score"].append(eval_data["average_score"])
            
            # Calculate averages
            averages = {}
            for metric, values in metrics_sums.items():
                if values:
                    averages[f"avg_{metric}"] = sum(values) / len(values)
                    averages[f"min_{metric}"] = min(values)
                    averages[f"max_{metric}"] = max(values)
            
            return {
                "total_evaluations": len(evaluations),
                "recent_evaluations_analyzed": limit,
                "averages": averages,
                "latest_evaluation": evaluations[-1] if evaluations else None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}


# Singleton instance
_evaluation_service = None

def get_evaluation_service() -> EvaluationService:
    """Get or create evaluation service singleton."""
    global _evaluation_service
    if _evaluation_service is None:
        _evaluation_service = EvaluationService()
    return _evaluation_service
