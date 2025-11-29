"""
Performance Monitoring for RAG System.

Tracks key metrics and provides performance insights.
"""

import time
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import asyncio
from loguru import logger


@dataclass
class PerformanceMetrics:
    """Data class for performance metrics."""
    operation: str
    duration: float
    timestamp: datetime
    tags: Dict[str, Any] = None


class PerformanceMonitor:
    """
    Performance monitoring system for RAG operations.

    Features:
    - Operation timing
    - Resource tracking
    - Performance history
    - Alerting for slow operations
    """

    def __init__(self):
        self.metrics: list[PerformanceMetrics] = []
        self.slow_operation_threshold = 1.0  # 1 second
        self.logger = logger.bind(component="PerformanceMonitor")

    @contextmanager
    def track_operation(self, operation_name: str, tags: Optional[Dict] = None):
        """
        Context manager to track operation performance.

        Usage:
        with perf_monitor.track_operation("query", {"collection": "docs", "k": 5}):
            result = await query_service.query(...)
        """
        start_time = time.time()
        tags = tags or {}

        try:
            yield
        finally:
            duration = time.time() - start_time
            metric = PerformanceMetrics(
                operation=operation_name,
                duration=duration,
                timestamp=datetime.now(),
                tags=tags
            )

            self.metrics.append(metric)

            # Log slow operations
            if duration > self.slow_operation_threshold:
                self.logger.warning(
                    f"SLOW OPERATION: {operation_name} took {duration:.2f}s",
                    extra=tags
                )

            self.logger.debug(
                f"Operation {operation_name} took {duration:.3f}s",
                extra=tags
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.metrics:
            return {"message": "No metrics collected yet"}

        operations = {}
        for metric in self.metrics:
            if metric.operation not in operations:
                operations[metric.operation] = []
            operations[metric.operation].append(metric.duration)

        stats = {}
        for op_name, durations in operations.items():
            stats[op_name] = {
                "count": len(durations),
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "total_duration": sum(durations)
            }

        return stats

    def clear_metrics(self):
        """Clear all collected metrics."""
        self.metrics.clear()


# Global singleton
perf_monitor = PerformanceMonitor()
