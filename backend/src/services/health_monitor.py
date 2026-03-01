"""Background health monitoring service with event emission.

This module provides real-time health monitoring for backend services
(Ollama, ChromaDB) with event-based notifications instead of polling.

Features:
- Background task checking services every N seconds (configurable)
- Status change detection (only emits events on changes)
- Event debouncing to prevent flapping services from spamming
- Subscriber limit to prevent memory leaks
- Graceful shutdown with task cancellation
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from loguru import logger


class HealthMonitor:
    """
    Monitors service health in background and emits events on status changes.

    Features:
    - Runs background task checking services every N seconds
    - Compares with last known status
    - Emits events only on changes (not continuous)
    - Event debouncing prevents flapping services from spamming
    - MAX_SUBSCRIBERS limit prevents memory leaks
    - Proper cleanup prevents semaphore leaks
    """

    def __init__(self, service_manager, config):
        """Initialize health monitor.

        Args:
            service_manager: ServiceManager instance for health checks
            config: HealthMonitorConfig with check_interval, event_debounce, max_subscribers
        """
        self.service_manager = service_manager
        self.config = config

        # Background task
        self._task: Optional[asyncio.Task] = None
        self._running = False

        # Status tracking
        self._last_status: Dict[str, Any] = {}

        # Event subscribers
        self._subscribers: List[Callable] = []

        # Event debouncing (anti-flapping)
        self._pending_events: Dict[str, Dict[str, Any]] = {}  # service_name -> event
        self._event_timers: Dict[str, asyncio.Task] = {}  # service_name -> timer task

    def can_subscribe(self) -> bool:
        """Check if a new subscriber can be added.

        Returns:
            True if subscriber count is below MAX_SUBSCRIBERS
        """
        return len(self._subscribers) < self.config.max_subscribers

    def get_subscriber_count(self) -> int:
        """Get current number of subscribers.

        Returns:
            Number of active subscribers
        """
        return len(self._subscribers)

    def subscribe(self, callback: Callable):
        """Subscribe to health change events.

        Args:
            callback: Async or sync function to call on health change events

        Raises:
            RuntimeError: If MAX_SUBSCRIBERS limit is reached
        """
        if len(self._subscribers) >= self.config.max_subscribers:
            logger.warning(
                f"Max subscribers ({self.config.max_subscribers}) reached, "
                f"rejecting new subscription"
            )
            raise RuntimeError(
                f"Maximum number of subscribers ({self.config.max_subscribers}) reached"
            )

        if callback not in self._subscribers:
            self._subscribers.append(callback)
            logger.debug(
                f"New subscriber added, total: {len(self._subscribers)}/"
                f"{self.config.max_subscribers}"
            )

    def unsubscribe(self, callback: Callable):
        """Unsubscribe from health change events.

        Args:
            callback: Previously subscribed callback function
        """
        if callback in self._subscribers:
            self._subscribers.remove(callback)
            logger.debug(f"Subscriber removed, total: {len(self._subscribers)}")

    def clear_all_subscribers(self):
        """Clear all subscribers (emergency cleanup for leaked connections).

        This should only be used for debugging or emergency recovery.
        """
        count = len(self._subscribers)
        self._subscribers.clear()
        logger.warning(f"Cleared all subscribers ({count} total) - emergency cleanup")

    async def _emit_event(self, event: Dict[str, Any]):
        """Emit event to all subscribers.

        Args:
            event: Event dictionary with service, old_status, new_status, timestamp
        """
        # Create a copy of the subscribers list to iterate over, to avoid issues when modifying the list during iteration
        subscribers_to_call = list(self._subscribers)
        
        for callback in subscribers_to_call:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in event subscriber: {e}", exc_info=True)
                # If there's an error during callback execution, it might be due to a closed WebSocket
                # Remove the callback to prevent repeated errors
                if callback in self._subscribers:
                    self._subscribers.remove(callback)
                    logger.debug(f"Removed problematic callback from subscribers, total: {len(self._subscribers)}")

    async def _debounced_emit(self, service_name: str, event: Dict[str, Any]):
        """Debounce events to prevent flapping services from spamming.

        If a service status changes multiple times rapidly, only the final
        state after debounce_delay seconds will be emitted.

        Args:
            service_name: Name of service (e.g., "ollama", "chroma")
            event: Event dictionary to emit after debounce delay
        """
        # Cancel existing timer for this service
        if service_name in self._event_timers:
            self._event_timers[service_name].cancel()
            logger.debug(f"Cancelled previous debounce timer for {service_name}")

        # Store pending event
        self._pending_events[service_name] = event

        # Create new timer
        async def emit_after_delay():
            try:
                await asyncio.sleep(self.config.event_debounce)
                if service_name in self._pending_events:
                    await self._emit_event(self._pending_events[service_name])
                    del self._pending_events[service_name]
                    if service_name in self._event_timers:
                        del self._event_timers[service_name]
            except asyncio.CancelledError:
                # Timer was cancelled (new event came in)
                pass

        self._event_timers[service_name] = asyncio.create_task(emit_after_delay())

        logger.debug(
            f"Event for {service_name} debounced for {self.config.event_debounce}s"
        )

    async def _check_and_emit(self, user_config: Dict[str, Any]):
        """Check current status and emit events if changed.

        Args:
            user_config: User configuration dict (for service manager)
        """
        try:
            current_status = await self.service_manager.get_status(user_config)

            # Check for changes in each service
            for service_name in ["ollama", "chroma"]:
                current = current_status.get(service_name, {})
                last = self._last_status.get(service_name, {})

                current_state = current.get("running", False)
                last_state = last.get("running", False)

                if current_state != last_state:
                    # Status changed - create event
                    event = {
                        "service": service_name,
                        "old_status": "running" if last_state else "stopped",
                        "new_status": "running" if current_state else "stopped",
                        "timestamp": datetime.now().isoformat(),
                        "details": current
                    }

                    logger.info(
                        f"Service status changed: {service_name} "
                        f"{event['old_status']} → {event['new_status']}"
                    )

                    # Use debounced emit to prevent flapping spam
                    await self._debounced_emit(service_name, event)

            # Update last known status
            self._last_status = current_status

        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)

    async def _monitor_loop(self, user_config: Dict[str, Any]):
        """Main monitoring loop.

        Args:
            user_config: User configuration dict
        """
        logger.info("Health monitor started")

        while self._running:
            try:
                await self._check_and_emit(user_config)
                await asyncio.sleep(self.config.check_interval)
            except asyncio.CancelledError:
                logger.info("Health monitor cancelled")
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}", exc_info=True)
                await asyncio.sleep(self.config.check_interval)

        logger.info("Health monitor stopped")

    def start(self, user_config: Dict[str, Any]):
        """Start background monitoring.

        Args:
            user_config: User configuration dict for service checks
        """
        if self._running:
            logger.warning("Health monitor already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop(user_config))
        logger.info(
            f"Health monitor task created (check_interval={self.config.check_interval}s, "
            f"debounce={self.config.event_debounce}s)"
        )

    async def stop(self):
        """Stop background monitoring and cleanup resources."""
        if not self._running:
            return

        self._running = False

        # Cancel pending debounce timers
        for service_name, timer in self._event_timers.items():
            timer.cancel()
            logger.debug(f"Cancelled debounce timer for {service_name}")

        self._event_timers.clear()
        self._pending_events.clear()

        # Cancel main monitoring task
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("Health monitor stopped and cleaned up")


# Global instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance.

    Returns:
        HealthMonitor singleton instance

    Note:
        Must call initialize_health_monitor() first during app startup
    """
    global _health_monitor
    if _health_monitor is None:
        raise RuntimeError(
            "HealthMonitor not initialized. Call initialize_health_monitor() first."
        )
    return _health_monitor


def initialize_health_monitor(service_manager, config) -> HealthMonitor:
    """Initialize global health monitor instance.

    Args:
        service_manager: ServiceManager instance
        config: HealthMonitorConfig instance

    Returns:
        Initialized HealthMonitor instance
    """
    global _health_monitor
    if _health_monitor is not None:
        logger.warning("HealthMonitor already initialized, returning existing instance")
        return _health_monitor

    _health_monitor = HealthMonitor(service_manager, config)
    logger.info("HealthMonitor initialized successfully")
    return _health_monitor
