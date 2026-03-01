"""
Background Metrics Collectors for System Health.

This module provides background tasks that periodically update system metrics
like active collections count and total document count.
"""

import asyncio
from loguru import logger
from typing import Optional

from src.core.observability.metrics import (
    active_collections,
    total_documents
)


class MetricsCollector:
    """Background task to collect and update system metrics."""

    def __init__(self, chroma_manager, interval_seconds: int = 60):
        """
        Initialize metrics collector.

        Args:
            chroma_manager: ChromaManager instance
            interval_seconds: Collection interval in seconds (default: 60)
        """
        self.chroma_manager = chroma_manager
        self.interval = interval_seconds
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self.logger = logger.bind(component="MetricsCollector")

    async def collect_system_metrics(self):
        """Collect and update system metrics once."""
        try:
            client = self.chroma_manager.get_client()
            if not client:
                self.logger.warning("ChromaDB client not available, skipping metrics collection")
                return

            # Get all collections
            collections = client.list_collections()
            collection_count = len(collections)

            # Update active collections gauge
            active_collections.set(collection_count)

            # Count total documents across all collections
            total_docs = 0
            for collection in collections:
                try:
                    count = collection.count()
                    total_docs += count
                except Exception as e:
                    self.logger.warning(f"Failed to get count for collection '{collection.name}': {e}")
                    continue

            # Update total documents gauge
            total_documents.set(total_docs)

            self.logger.debug(
                f"System metrics updated: {collection_count} collections, {total_docs} documents"
            )

        except Exception as e:
            self.logger.error(f"Metrics collection failed: {e}")

    async def _collection_loop(self):
        """Background loop that collects metrics periodically."""
        self.logger.info(f"Metrics collector started (interval: {self.interval}s)")

        while self._running:
            await self.collect_system_metrics()
            await asyncio.sleep(self.interval)

        self.logger.info("Metrics collector stopped")

    def start(self):
        """Start the background metrics collection task."""
        if self._running:
            self.logger.warning("Metrics collector already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._collection_loop())
        self.logger.info("Metrics collector task created")

    async def stop(self):
        """Stop the background metrics collection task."""
        if not self._running:
            return

        self._running = False

        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.warning("Metrics collector task did not stop gracefully, cancelling")
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass

        self.logger.info("Metrics collector stopped")


# Global instance
_metrics_collector: Optional[MetricsCollector] = None


def initialize_metrics_collector(chroma_manager, interval_seconds: int = 60) -> MetricsCollector:
    """
    Initialize the global metrics collector.

    Args:
        chroma_manager: ChromaManager instance
        interval_seconds: Collection interval in seconds

    Returns:
        MetricsCollector instance
    """
    global _metrics_collector

    if _metrics_collector is not None:
        logger.warning("Metrics collector already initialized")
        return _metrics_collector

    _metrics_collector = MetricsCollector(chroma_manager, interval_seconds)
    return _metrics_collector


def get_metrics_collector() -> MetricsCollector:
    """
    Get the global metrics collector instance.

    Returns:
        MetricsCollector instance

    Raises:
        RuntimeError: If metrics collector has not been initialized
    """
    if _metrics_collector is None:
        raise RuntimeError("Metrics collector has not been initialized. Call initialize_metrics_collector() first.")

    return _metrics_collector
