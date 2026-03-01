"""
Service Manager for Ollama and ChromaDB connection management.

This module manages connections to external services (Ollama and ChromaDB) that
run as systemd services. It NO LONGER starts/stops processes via subprocess.

Services are expected to be running via systemd before the application starts.
The manager only checks connectivity and maintains connection state.

Systemd services:
- ollama.service (Port 11434)
- chroma.service (Port 8000)

For service management, use:
  sudo systemctl start/stop/restart ollama
  sudo systemctl start/stop/restart chroma
"""

import asyncio
import httpx
from typing import Dict, Any, Optional, List
from loguru import logger
from enum import Enum
from datetime import datetime, timedelta
from contextlib import asynccontextmanager


class ServiceStatus(str, Enum):
    """Service status states"""
    STOPPED = "stopped"
    CONNECTING = "connecting"
    RUNNING = "running"
    ERROR = "error"


class ServiceManager:
    """Manages connections to Ollama and ChromaDB systemd services."""

    def __init__(self):
        self.service_configs: Dict[str, Dict[str, Any]] = {}
        self.status: Dict[str, ServiceStatus] = {
            "ollama": ServiceStatus.STOPPED,
            "chroma": ServiceStatus.STOPPED
        }
        self.error_messages: Dict[str, str] = {}
        self.logger = logger.bind(component="ServiceManager")

        # Status caching to prevent excessive health checks
        self._status_cache: Optional[Dict[str, Any]] = None
        self._status_cache_time: Optional[datetime] = None
        self._status_cache_ttl = timedelta(seconds=3)  # Cache for 3 seconds

        # Health check semaphore to prevent concurrent checks
        self._health_check_semaphore = asyncio.Semaphore(1)
        self._health_check_in_progress = False

    @asynccontextmanager
    async def _acquire_health_check_lock(self, timeout: float = 1.0):
        """Context manager for health check lock with timeout."""
        try:
            async with asyncio.timeout(timeout):
                async with self._health_check_semaphore:
                    yield
        except asyncio.TimeoutError:
            self.logger.warning("Health check lock timeout - skipping check")
            raise

    def _invalidate_status_cache(self):
        """Invalidate status cache (call when service state changes)."""
        self._status_cache = None
        self._status_cache_time = None

    async def start_ollama(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Connect to Ollama systemd service (does NOT start subprocess).

        Expects Ollama to be running via: systemctl start ollama

        Args:
            config: Configuration dict with optional OLLAMA_HOST override

        Returns:
            Dict with success status and message
        """
        return await self.connect_to_ollama(config)

    async def connect_to_ollama(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Connect to running Ollama systemd service.

        Standard port: 11434
        Health check endpoint: /api/tags
        """
        # Get configured host or use default
        config_ollama_host = config.get("OLLAMA_HOST", "http://localhost:11434")

        self.status["ollama"] = ServiceStatus.CONNECTING
        self.error_messages.pop("ollama", None)

        try:
            # Parse URL
            host_url = config_ollama_host
            if not host_url.startswith("http"):
                host_url = f"http://{host_url}"

            # Check if service is running
            if await self._check_service_health("ollama", host_url, "/api/tags"):
                # Extract host and port for config
                import re
                match = re.match(r'https?://([^:]+):?(\d+)?', host_url)
                if match:
                    host = match.group(1)
                    port = int(match.group(2)) if match.group(2) else 11434
                else:
                    host = "localhost"
                    port = 11434

                self.status["ollama"] = ServiceStatus.RUNNING
                self.service_configs["ollama"] = {
                    "host": host,
                    "port": port,
                    "url": host_url
                }
                self._invalidate_status_cache()

                self.logger.success(f"Connected to Ollama service at {host_url}")
                return {
                    "success": True,
                    "message": f"Connected to Ollama at {host_url}",
                    "status": ServiceStatus.RUNNING
                }
            else:
                # Service not running
                self.status["ollama"] = ServiceStatus.ERROR
                error_msg = (
                    f"Ollama not running at {host_url}. "
                    f"Please start via: sudo systemctl start ollama"
                )
                self.error_messages["ollama"] = error_msg
                self.logger.error(error_msg)

                return {
                    "success": False,
                    "message": error_msg,
                    "status": ServiceStatus.ERROR
                }

        except Exception as e:
            self.status["ollama"] = ServiceStatus.ERROR
            self.error_messages["ollama"] = str(e)
            self.logger.error(f"Failed to connect to Ollama: {e}")
            return {
                "success": False,
                "message": f"Failed to connect to Ollama: {e}",
                "status": ServiceStatus.ERROR
            }

    async def start_chroma(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Connect to ChromaDB systemd service (does NOT start subprocess).

        Expects ChromaDB to be running via: systemctl start chroma

        Args:
            config: Configuration dict with optional CHROMA_HOST override

        Returns:
            Dict with success status and message
        """
        return await self.connect_to_chroma(config)

    async def connect_to_chroma(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Connect to running ChromaDB systemd service.

        Standard port: 8000
        Health check endpoint: /api/v1/heartbeat
        """
        # Get configured host or use default
        config_chroma_host = config.get("CHROMA_HOST", "http://localhost:8000")

        self.status["chroma"] = ServiceStatus.CONNECTING
        self.error_messages.pop("chroma", None)

        try:
            # Parse URL
            host_url = config_chroma_host
            if not host_url.startswith("http"):
                host_url = f"http://{host_url}"

            # Check if service is running
            if await self._check_service_health("chroma", host_url, "/api/v1/heartbeat"):
                # Extract host and port for config
                import re
                match = re.match(r'https?://([^:]+):?(\d+)?', host_url)
                if match:
                    host = match.group(1)
                    port = int(match.group(2)) if match.group(2) else 8000
                else:
                    host = "localhost"
                    port = 8000

                self.status["chroma"] = ServiceStatus.RUNNING
                self.service_configs["chroma"] = {
                    "host": host,
                    "port": port,
                    "url": host_url
                }
                self._invalidate_status_cache()

                self.logger.success(f"Connected to ChromaDB service at {host_url}")
                return {
                    "success": True,
                    "message": f"Connected to ChromaDB at {host_url}",
                    "status": ServiceStatus.RUNNING
                }
            else:
                # Service not running
                self.status["chroma"] = ServiceStatus.ERROR
                error_msg = (
                    f"ChromaDB not running at {host_url}. "
                    f"Please start via: sudo systemctl start chroma"
                )
                self.error_messages["chroma"] = error_msg
                self.logger.error(error_msg)

                return {
                    "success": False,
                    "message": error_msg,
                    "status": ServiceStatus.ERROR
                }

        except Exception as e:
            self.status["chroma"] = ServiceStatus.ERROR
            self.error_messages["chroma"] = str(e)
            self.logger.error(f"Failed to connect to ChromaDB: {e}")
            return {
                "success": False,
                "message": f"Failed to connect to ChromaDB: {e}",
                "status": ServiceStatus.ERROR
            }

    async def stop_ollama(self) -> Dict[str, Any]:
        """
        Disconnect from Ollama service.

        NOTE: This does NOT stop the systemd service!
        Use: sudo systemctl stop ollama
        """
        if "ollama" not in self.service_configs:
            return {
                "success": False,
                "message": "Ollama is not connected",
                "status": self.status.get("ollama", ServiceStatus.STOPPED)
            }

        self.service_configs.pop("ollama", None)
        self.status["ollama"] = ServiceStatus.STOPPED
        self.error_messages.pop("ollama", None)
        self._invalidate_status_cache()

        self.logger.info("Disconnected from Ollama service")
        return {
            "success": True,
            "message": "Disconnected from Ollama (service still running via systemd)",
            "status": ServiceStatus.STOPPED
        }

    async def stop_chroma(self) -> Dict[str, Any]:
        """
        Disconnect from ChromaDB service.

        NOTE: This does NOT stop the systemd service!
        Use: sudo systemctl stop chroma
        """
        if "chroma" not in self.service_configs:
            return {
                "success": False,
                "message": "ChromaDB is not connected",
                "status": self.status.get("chroma", ServiceStatus.STOPPED)
            }

        self.service_configs.pop("chroma", None)
        self.status["chroma"] = ServiceStatus.STOPPED
        self.error_messages.pop("chroma", None)
        self._invalidate_status_cache()

        self.logger.info("Disconnected from ChromaDB service")
        return {
            "success": True,
            "message": "Disconnected from ChromaDB (service still running via systemd)",
            "status": ServiceStatus.STOPPED
        }

    async def get_status(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get status of all services with caching to prevent excessive health checks."""

        # GUARD 1: Return cached status if check is already in progress
        if self._health_check_in_progress:
            if self._status_cache:
                self.logger.debug("Health check in progress, returning cached status")
                return self._status_cache
            else:
                self.logger.debug("Health check in progress, no cache available, returning default")
                return self._get_default_status()

        # GUARD 2: Check if cache is still valid
        if self._is_cache_valid():
            self.logger.debug("Returning cached status")
            return self._status_cache

        # GUARD 3: Perform actual health check with semaphore
        try:
            async with self._acquire_health_check_lock(timeout=1.0):
                self._health_check_in_progress = True
                try:
                    # Cache expired or not present, do actual health checks
                    self.logger.debug("Cache miss or expired, performing health checks")

                    ollama_config = self.service_configs.get("ollama")
                    chroma_config = self.service_configs.get("chroma")

                    ollama_url = ollama_config["url"] if ollama_config else config.get("OLLAMA_HOST", "http://localhost:11434")
                    chroma_url = chroma_config["url"] if chroma_config else config.get("CHROMA_HOST", "http://localhost:8000")

                    # Perform health checks in parallel for better performance
                    ollama_healthy, chroma_healthy = await asyncio.gather(
                        self._check_service_health("ollama", ollama_url, "/api/tags"),
                        self._check_service_health("chroma", chroma_url, "/api/v1/heartbeat"),
                        return_exceptions=True
                    )

                    # Handle exceptions from gather
                    if isinstance(ollama_healthy, Exception):
                        self.logger.warning(f"Ollama health check exception: {ollama_healthy}")
                        ollama_healthy = False
                    if isinstance(chroma_healthy, Exception):
                        self.logger.warning(f"Chroma health check exception: {chroma_healthy}")
                        chroma_healthy = False

                    # Update running status based on health check
                    if ollama_healthy and self.status["ollama"] != ServiceStatus.RUNNING:
                        self.status["ollama"] = ServiceStatus.RUNNING
                    elif not ollama_healthy and self.status["ollama"] == ServiceStatus.RUNNING:
                        self.status["ollama"] = ServiceStatus.ERROR
                        self.error_messages["ollama"] = "Service unreachable"

                    if chroma_healthy and self.status["chroma"] != ServiceStatus.RUNNING:
                        self.status["chroma"] = ServiceStatus.RUNNING
                    elif not chroma_healthy and self.status["chroma"] == ServiceStatus.RUNNING:
                        self.status["chroma"] = ServiceStatus.ERROR
                        self.error_messages["chroma"] = "Service unreachable"

                    # Build status response
                    status_dict = {
                        "ollama": {
                            "running": ollama_healthy,
                            "status": self.status["ollama"],
                            "host": ollama_config.get("host") if ollama_config else None,
                            "port": ollama_config.get("port") if ollama_config else None,
                            "error": self.error_messages.get("ollama")
                        },
                        "chroma": {
                            "running": chroma_healthy,
                            "status": self.status["chroma"],
                            "host": chroma_config.get("host") if chroma_config else None,
                            "port": chroma_config.get("port") if chroma_config else None,
                            "error": self.error_messages.get("chroma")
                        }
                    }

                    # Update cache
                    self._status_cache = status_dict
                    self._status_cache_time = datetime.now()

                    return status_dict

                finally:
                    self._health_check_in_progress = False
        except asyncio.TimeoutError:
            # Lock timeout - return cached or default status
            self.logger.debug("Health check lock timeout, returning cached/default status")
            return self._status_cache or self._get_default_status()

    async def test_service(self, service_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Test service connectivity."""
        service_config = self.service_configs.get(service_name)

        if service_name == "ollama":
            host_url = service_config["url"] if service_config else config.get("OLLAMA_HOST", "http://localhost:11434")
            path = "/api/tags"
        elif service_name == "chroma":
            host_url = service_config["url"] if service_config else config.get("CHROMA_HOST", "http://localhost:8000")
            path = "/api/v1/heartbeat"
        else:
            return {"success": False, "message": f"Unknown service: {service_name}"}

        import time
        start_time = time.time()
        healthy = await self._check_service_health(service_name, host_url, path)
        duration = time.time() - start_time

        return {
            "success": healthy,
            "service": service_name,
            "message": f"{service_name} is {'reachable' if healthy else 'not reachable'} at {host_url}",
            "duration": round(duration, 2)
        }

    async def _check_service_health(self, service_name: str, base_url: str, health_path: str) -> bool:
        """Check if service is healthy via HTTP."""
        try:
            async with httpx.AsyncClient(
                timeout=5.0,
                max_redirects=0,
                follow_redirects=False
            ) as client:
                response = await client.get(f"{base_url}{health_path}")
                return response.status_code == 200
        except Exception as e:
            self.logger.debug(f"{service_name} health check failed: {e}")
            return False

    async def get_ollama_models(self, config: Dict[str, Any]) -> List[str]:
        """Get list of available Ollama models."""
        try:
            ollama_config = self.service_configs.get("ollama")
            if ollama_config and ollama_config.get("url"):
                ollama_host = ollama_config["url"]
            else:
                ollama_host = config.get("OLLAMA_HOST", "http://localhost:11434")

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{ollama_host}/api/tags")
                response.raise_for_status()
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            self.logger.error(f"Failed to get Ollama models: {e}")
            return []

    def _is_cache_valid(self) -> bool:
        """Check if status cache is still valid."""
        if not self._status_cache or not self._status_cache_time:
            return False
        return datetime.now() - self._status_cache_time < self._status_cache_ttl

    def _get_default_status(self) -> Dict[str, Any]:
        """Return default status when services are unknown."""
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


# Global singleton
service_manager = ServiceManager()
