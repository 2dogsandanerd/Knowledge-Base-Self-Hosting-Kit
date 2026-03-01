"""
ChromaDB Client Manager with singleton pattern for centralized connection management.

This module implements centralized ChromaDB client management to solve the issue of
unclosed connections during application shutdown. The manager ensures that only one
ChromaDB client instance exists throughout the application lifecycle and that this
connection is properly closed during shutdown.

This is part of the new startup/shutdown architecture that uses FastAPI's lifespan
context manager to coordinate the startup and shutdown of all services in the correct
order:
1. During startup: Services start first, then client connections are established
2. During shutdown: Client connections close first, then services stop

This sequence prevents CLOSE_WAIT connections and port blocking issues.
"""

import os

# Disable telemetry to prevent PostHog errors. This must be set before any
# chromadb imports to ensure it's effective on initialization.
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

import asyncio
import httpx
from datetime import datetime, timedelta
import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings
from loguru import logger
from typing import Optional

# Import resilience patterns
from src.core.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from src.core.retry_policy import RetryPolicy, DEFAULT_RETRY

# Import crash detector for connection tracking
try:
    from src.utils.crash_detector import crash_detector
    CRASH_DETECTOR_AVAILABLE = True
except ImportError:
    CRASH_DETECTOR_AVAILABLE = False

# Global singleton instance (initialized lazily)
_chroma_manager_instance: Optional['ChromaManager'] = None

def get_chroma_manager() -> 'ChromaManager':
    global _chroma_manager_instance
    if _chroma_manager_instance is None:
        _chroma_manager_instance = ChromaManager()
    return _chroma_manager_instance

# The global instance will now be accessed via get_chroma_manager()
# chroma_manager = ChromaManager() # Commented out to avoid direct instantiation


class ChromaManager:
    """Singleton manager for ChromaDB client connections.

    Enhanced with async methods, connection pooling, health check caching,
    exponential backoff retry, and thread-safety (Phase 1, Task 1.3).
    """

    def __init__(self):
        print("DEBUG: ChromaManager __init__ called. get_client_async should be available.") # Added debug print
        # Existing sync client
        self._client: Optional[ClientAPI] = None
        self._host: Optional[str] = None
        self._port: Optional[int] = None
        self._in_memory: bool = False # New attribute for in-memory mode
        self._configured: bool = False
        self.logger = logger.bind(component="ChromaManager")

        # New async enhancements (Phase 1, Task 1.3)
        self._connection_pool: Optional[httpx.AsyncClient] = None
        self._last_health_check: Optional[datetime] = None
        self._health_check_interval = timedelta(seconds=30)  # Cache health checks for 30s
        self._connection_lock = asyncio.Lock()  # Thread-safety for async operations
        self._retry_delays = [1, 2, 4]  # Exponential backoff: 1s, 2s, 4s
        
        # Resilience patterns (Phase 2)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            success_threshold=2,
            recovery_timeout=30,
            name="chroma_circuit"
        )
        self.retry_policy = DEFAULT_RETRY
    
    def configure(self, host: Optional[str] = None, port: Optional[int] = None, in_memory: bool = False):
        """Configure the ChromaDB connection parameters."""
        self._in_memory = in_memory
        if in_memory:
            self._host = None
            self._port = None
            self.logger.info("ChromaDB client configured for in-memory mode.")
        else:
            if not host or not port:
                self.logger.error("Host and port must be provided for non-in-memory ChromaDB configuration.")
                self._configured = False
                return
            self._host = host
            self._port = port
            self.logger.info(f"ChromaDB client configured to connect to {host}:{port}")
        self._configured = True
    
    def get_client(self) -> Optional[ClientAPI]:
        """Get the active ChromaDB client instance, creating or reconnecting if necessary."""
        if not self._configured:
            self.logger.warning("ChromaDB not configured yet. Call configure() first.")
            return None
        
        # Check if we have a client and if it's still connected
        if self._client is not None:
            try:
                # For in-memory client, heartbeat always succeeds or is not needed
                if not self._in_memory:
                    self._client.heartbeat()
                return self._client
            except Exception as e:
                self.logger.warning(f"Existing client connection is not responsive: {e}")
                # Connection is not working, we'll create a new one
                self._client = None

        if self._client is None:
            try:
                if self._in_memory:
                    self.logger.info("Creating new in-memory ChromaDB client.")
                    self._client = chromadb.Client()
                else:
                    self.logger.info(f"Creating new ChromaDB client connection to {self._host}:{self._port}")
                    settings = Settings(anonymized_telemetry=False)
                    self._client = chromadb.HttpClient(
                        host=self._host,
                        port=self._port,
                        settings=settings
                    )

                # Test the connection with heartbeat (only for HttpClient)
                if not self._in_memory:
                    heartbeat = self._client.heartbeat()
                    self.logger.success(f"ChromaDB client successfully connected to {self._host}:{self._port}, heartbeat: {heartbeat}")
                else:
                    self.logger.success("In-memory ChromaDB client created successfully.")

                # Track connection
                if CRASH_DETECTOR_AVAILABLE:
                    crash_detector.track_connection("chroma", +1)
            except Exception as e:
                self.logger.error(f"Failed to create ChromaDB client connection: {e}")
                return None
        
        return self._client
    
    def is_connected(self) -> bool:
        """Check if the client is connected to ChromaDB."""
        if self._client is None:
            return False
        
        if self._in_memory:
            return True # In-memory client is always "connected"
        
        try:
            self._client.heartbeat()
            return True
        except Exception:
            return False
    
    def close_client(self):
        """Close the ChromaDB client connection and reset internal state."""
        if self._client is not None:
            self.logger.info("Closing ChromaDB client connection...")
            try:
                # Track connection close
                if CRASH_DETECTOR_AVAILABLE:
                    crash_detector.track_connection("chroma", -1)

                # No explicit close method in ChromaDB HttpClient, but we'll reset our reference
                # For in-memory, we just clear the reference
                self._client = None
                self.logger.success("ChromaDB client connection closed")
            except Exception as e:
                self.logger.error(f"Error while closing ChromaDB client: {e}")
            except AttributeError: # In-memory client might not have a close method
                self._client = None
                self.logger.success("In-memory ChromaDB client reference cleared.")
        else:
            self.logger.info("ChromaDB client was not connected")

    async def close(self):
        """Close all connections and cleanup resources (async).

        FIX BUG #4: Properly close httpx.AsyncClient to prevent connection leaks.
        Must be called during application shutdown (FastAPI lifespan).
        """
        self.logger.info("Closing ChromaManager resources...")

        # Close async HTTP connection pool
        if self._connection_pool:
            try:
                await self._connection_pool.aclose()
                self._connection_pool = None
                self.logger.info("HTTP connection pool closed")
            except Exception as e:
                self.logger.error(f"Failed to close HTTP connection pool: {e}")

        # Close sync ChromaDB client
        self.close_client()

        self.logger.info("ChromaManager cleanup complete")

    def list_collections(self):
        """List all collections in ChromaDB.

        Wrapper method for CollectionManager compatibility.
        Returns the raw ChromaDB collections list.
        """
        client = self.get_client()
        if not client:
            raise ConnectionError("ChromaDB client not available")
        return client.list_collections()

    def get_collection(self, collection_name: str):
        """Get a collection from ChromaDB.

        Wrapper method for API compatibility.
        Returns the ChromaDB collection object.
        """
        client = self.get_client()
        if not client:
            raise ConnectionError("ChromaDB client not available")
        return client.get_collection(collection_name)

    def delete_collection(self, collection_name: str):
        """Delete a collection from ChromaDB.

        Wrapper method for API compatibility.
        """
        client = self.get_client()
        if not client:
            raise ConnectionError("ChromaDB client not available")
        return client.delete_collection(collection_name)

    # ==================== ASYNC METHODS (Phase 1, Task 1.3) ====================

    async def health_check_async(self, force: bool = False) -> bool:
        """Async health check with caching (30s TTL).

        Args:
            force: If True, bypass cache and force fresh health check

        Returns:
            True if ChromaDB is healthy, False otherwise
        """
        if not self._configured:
            self.logger.warning("ChromaDB not configured yet. Call configure() first.")
            return False

        if self._in_memory:
            return True # In-memory client is always healthy

        # Check cache unless force=True
        if not force and self._last_health_check:
            time_since_check = datetime.now() - self._last_health_check
            if time_since_check < self._health_check_interval:
                self.logger.debug(f"Using cached health check (age: {time_since_check.total_seconds():.1f}s)")
                return True

        # Perform fresh health check
        try:
            if self._connection_pool is None:
                self._connection_pool = httpx.AsyncClient(timeout=10.0)

            url = f"http://{self._host}:{self._port}/api/v2/heartbeat"
            response = await self._connection_pool.get(url)

            if response.status_code == 200:
                self._last_health_check = datetime.now()
                self.logger.debug(f"Health check passed: {self._host}:{self._port}")
                return True
            else:
                self.logger.warning(f"Health check failed with status {response.status_code}")
                return False
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return False

    async def ensure_connected_async(self, max_retries: int = 3) -> bool:
        """Ensure ChromaDB connection with exponential backoff retry.

        Args:
            max_retries: Maximum number of retry attempts (default: 3)

        Returns:
            True if connected successfully, False otherwise
        """
        if not self._configured:
            self.logger.warning("ChromaDB not configured yet. Call configure() first.")
            return False

        if self._in_memory:
            if self._client is None:
                self.logger.info("Creating new in-memory ChromaDB client for async connection.")
                self._client = chromadb.Client()
                self.logger.success("In-memory ChromaDB client created successfully for async connection.")
            return True

        async with self._connection_lock:  # Thread-safety
            # Check if already connected
            if await self.health_check_async():
                return True

            # Retry with exponential backoff
            for attempt in range(max_retries):
                try:
                    self.logger.info(f"Connection attempt {attempt + 1}/{max_retries} to {self._host}:{self._port}")

                    # Create new client
                    settings = Settings(anonymized_telemetry=False)
                    self._client = chromadb.HttpClient(
                        host=self._host,
                        port=self._port,
                        settings=settings
                    )

                    # Test connection
                    heartbeat = self._client.heartbeat()
                    self.logger.success(f"Connected to ChromaDB: {self._host}:{self._port}, heartbeat: {heartbeat}")

                    # Update health check cache
                    self._last_health_check = datetime.now()

                    # Track connection
                    if CRASH_DETECTOR_AVAILABLE:
                        crash_detector.track_connection("chroma", +1)

                    return True

                except Exception as e:
                    self.logger.warning(f"Connection attempt {attempt + 1} failed: {e}")

                    # Exponential backoff (except on last attempt)
                    if attempt < max_retries - 1:
                        delay = self._retry_delays[min(attempt, len(self._retry_delays) - 1)]
                        self.logger.info(f"Retrying in {delay}s...")
                        await asyncio.sleep(delay)

            self.logger.error(f"Failed to connect to ChromaDB after {max_retries} attempts")
            return False

    async def get_client_async(self) -> Optional[ClientAPI]:
        """Async method to get ChromaDB client with auto-reconnect.

        Returns:
            ChromaDB client or None if connection fails
        """
        if not self._configured:
            self.logger.warning("ChromaDB not configured yet. Call configure() first.")
            return None

        if self._in_memory:
            if self._client is None:
                self.logger.info("Creating new in-memory ChromaDB client for async get_client_async.")
                self._client = chromadb.Client()
                self.logger.success("In-memory ChromaDB client created successfully for async get_client_async.")
            return self._client

        # Try health check first (uses cache)
        if await self.health_check_async():
            return self._client

        # If health check failed, try to reconnect
        if await self.ensure_connected_async():
            return self._client

        return None

    async def get_collection_with_resilience(self, collection_name: str):
        """
        Get a collection with circuit breaker and retry logic.
        """
        async def _get_collection():
            client = await self.get_client_async()
            if not client:
                raise ConnectionError("ChromaDB client not available")
            return client.get_collection(collection_name)

        try:
            # Execute with circuit breaker and retry policy
            return await self.circuit_breaker.call(
                self.retry_policy.execute,
                _get_collection
            )
        except CircuitBreakerOpenError as e:
            self.logger.error(f"Circuit breaker open: {e}")
            # Re-raise as a more specific exception or handle it
            raise ConnectionError("ChromaDB service is currently unavailable due to repeated failures.") from e
        except Exception as e:
            self.logger.error(f"Failed to get collection '{collection_name}' after retries: {e}")
            raise

    def get_health_status(self) -> dict:
        """Get health status including circuit breaker state"""
        return {
            "connected": self.is_connected(),
            "last_check": self._last_health_check,
            "circuit_breaker": self.circuit_breaker.get_state()
        }

    async def close_async(self):
        """Async close method to clean up connection pool and client."""
        # Close connection pool
        if self._connection_pool is not None and not self._in_memory:
            await self._connection_pool.aclose()
            self._connection_pool = None
            self.logger.info("Connection pool closed")

        # Close sync client
        self.close_client()

    