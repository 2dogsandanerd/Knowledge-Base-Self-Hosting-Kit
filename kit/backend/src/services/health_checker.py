"""
Health Checker Service.

Manages health checks for external services (Ollama, ChromaDB) with:
- Intelligent caching (3-second TTL)
- Concurrent check prevention
- Timeout handling
"""

import asyncio
import httpx
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from loguru import logger
from enum import Enum
from contextlib import asynccontextmanager


class ServiceStatus(str, Enum):
    """Service status states"""
    STOPPED = "stopped"
    CONNECTING = "connecting"
    RUNNING = "running"
    ERROR = "error"


class HealthChecker:
    """
    Health check service with caching and concurrency control.

    Features:
    - Status caching (3-second TTL)
    - Prevents concurrent health checks
    - Parallel execution of multiple services
    - Graceful timeout handling

    Usage:
        checker = HealthChecker()

        # Check single service
        is_healthy = await checker.check_service("ollama", "http://localhost:11434", "/api/tags")

        # Check all services
        status = await checker.get_all_status(service_configs, config)
    """

    def __init__(self):
        # Status caching
        self._status_cache: Optional[Dict[str, Any]] = None
        self._status_cache_time: Optional[datetime] = None
        self._status_cache_ttl = timedelta(seconds=3)

        # Concurrency control
        self._health_check_semaphore = asyncio.Semaphore(1)
        self._health_check_in_progress = False

        self.logger = logger.bind(component="HealthChecker")

    async def check_service(
        self,
        service_name: str,
        base_url: str,
        health_path: str,
        timeout: float = 5.0
    ) -> bool:
        """
        Check if service is healthy via HTTP.

        Args:
            service_name: Service identifier (e.g., "ollama")
            base_url: Base URL (e.g., "http://localhost:11434")
            health_path: Health check endpoint (e.g., "/api/tags")
            timeout: HTTP timeout in seconds (default: 5.0)

        Returns:
            True if service is healthy (HTTP 200), False otherwise
        """
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                max_redirects=0,
                follow_redirects=False
            ) as client:
                response = await client.get(f"{base_url}{health_path}")
                is_healthy = response.status_code == 200

                if is_healthy:
                    self.logger.debug(f"{service_name} health check: OK")
                else:
                    self.logger.debug(f"{service_name} health check: FAIL (status={response.status_code})")

                return is_healthy

        except Exception as e:
            self.logger.debug(f"{service_name} health check failed: {e}")
            return False

    async def get_all_status(
        self,
        service_configs: Dict[str, Dict[str, Any]],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get status of all services with caching.

        Implements 3-level guard system:
        1. Return cached if check in progress
        2. Return cached if TTL not expired
        3. Perform actual health checks

        Args:
            service_configs: Dict of service configurations
            config: System configuration

        Returns:
            Dict with status for each service
        """
        # GUARD 1: Return cached status if check is already in progress
        if self._health_check_in_progress:
            if self._status_cache:
                self.logger.debug("Health check in progress, returning cached status")
                return self._status_cache
            else:
                self.logger.debug("Health check in progress, no cache available")
                return self._get_default_status()

        # GUARD 2: Check if cache is still valid
        if self._is_cache_valid():
            self.logger.debug("Returning cached status (TTL not expired)")
            return self._status_cache

        # GUARD 3: Perform actual health check with semaphore
        try:
            async with self._acquire_health_check_lock(timeout=1.0):
                self._health_check_in_progress = True

                try:
                    # Get service URLs
                    ollama_url = (
                        service_configs.get("ollama", {}).get("url")
                        or config.get("OLLAMA_HOST", "http://localhost:11434")
                    )
                    chroma_url = (
                        service_configs.get("chroma", {}).get("url")
                        or config.get("CHROMA_HOST", "http://localhost:8000")
                    )

                    # Perform health checks in parallel
                    ollama_healthy, chroma_healthy = await asyncio.gather(
                        self.check_service("ollama", ollama_url, "/api/tags"),
                        self.check_service("chroma", chroma_url, "/api/v2/heartbeat"),
                        return_exceptions=True
                    )

                    # Handle exceptions from gather
                    if isinstance(ollama_healthy, Exception):
                        self.logger.warning(f"Ollama health check exception: {ollama_healthy}")
                        ollama_healthy = False
                    if isinstance(chroma_healthy, Exception):
                        self.logger.warning(f"Chroma health check exception: {chroma_healthy}")
                        chroma_healthy = False

                    # Build status response
                    status_dict = {
                        "ollama": {
                            "running": ollama_healthy,
                            "status": ServiceStatus.RUNNING if ollama_healthy else ServiceStatus.ERROR,
                            "url": ollama_url
                        },
                        "chroma": {
                            "running": chroma_healthy,
                            "status": ServiceStatus.RUNNING if chroma_healthy else ServiceStatus.ERROR,
                            "url": chroma_url
                        }
                    }

                    # Update cache
                    self._status_cache = status_dict
                    self._status_cache_time = datetime.now()

                    self.logger.debug("Health check completed, cache updated")
                    return status_dict

                finally:
                    self._health_check_in_progress = False

        except asyncio.TimeoutError:
            self.logger.debug("Health check lock timeout, returning cached/default")
            return self._status_cache or self._get_default_status()

    @asynccontextmanager
    async def _acquire_health_check_lock(self, timeout: float = 1.0):
        """
        Context manager for health check lock with timeout.

        Args:
            timeout: Lock acquisition timeout in seconds

        Yields:
            Semaphore context
        """
        try:
            async with asyncio.timeout(timeout):
                async with self._health_check_semaphore:
                    yield
        except asyncio.TimeoutError:
            self.logger.warning("Health check lock timeout - skipping check")
            raise

    def _is_cache_valid(self) -> bool:
        """
        Check if status cache is still valid.

        Returns:
            True if cache exists and TTL not expired
        """
        if not self._status_cache or not self._status_cache_time:
            return False

        age = datetime.now() - self._status_cache_time
        is_valid = age < self._status_cache_ttl

        if not is_valid:
            self.logger.debug(f"Cache expired (age: {age.total_seconds():.1f}s)")

        return is_valid

    def invalidate_cache(self):
        """Invalidate status cache (call when service state changes)."""
        self._status_cache = None
        self._status_cache_time = None
        self.logger.debug("Status cache invalidated")

    def _get_default_status(self) -> Dict[str, Any]:
        """
        Return default status when services are unknown.

        Used when:
        - No cache available
        - Health check in progress
        - Health check timed out
        """
        return {
            "ollama": {
                "running": False,
                "status": "stopped",
                "message": "Status check in progress",
                "host": "localhost",
                "port": 11434
            },
            "chroma": {
                "running": False,
                "status": "stopped",
                "message": "Status check in progress",
                "host": "localhost",
                "port": 8000
            }
        }


# Global singleton instance
health_checker = HealthChecker()
