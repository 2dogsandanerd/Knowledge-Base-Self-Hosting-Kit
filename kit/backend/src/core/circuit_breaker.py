from enum import Enum
from typing import Callable, Any, Optional
from datetime import datetime, timedelta
import asyncio
from loguru import logger

class CircuitState(str, Enum):
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    """
    Circuit Breaker pattern to prevent cascading failures.

    States:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Service is failing, reject requests immediately
    - HALF_OPEN: Testing recovery, allow limited requests

    Transitions:
    - CLOSED -> OPEN: After failure_threshold consecutive failures
    - OPEN -> HALF_OPEN: After recovery_timeout seconds
    - HALF_OPEN -> CLOSED: After success_threshold consecutive successes
    - HALF_OPEN -> OPEN: If any request fails
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        recovery_timeout: int = 30,
        name: str = "circuit_breaker"
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.recovery_timeout = timedelta(seconds=recovery_timeout)
        self.name = name

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.

        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        async with self._lock:
            # Check if we should try recovery
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    logger.info(f"Circuit breaker {self.name}: Attempting recovery (HALF_OPEN)")
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker {self.name} is OPEN. "
                        f"Try again in {self._time_until_recovery()}s"
                    )

        # Execute function
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise

    async def _on_success(self):
        """Handle successful request"""
        async with self._lock:
            self.failure_count = 0

            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    logger.info(f"Circuit breaker {self.name}: Recovery successful (CLOSED)")
                    self.state = CircuitState.CLOSED
                    self.success_count = 0

    async def _on_failure(self):
        """Handle failed request"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.state == CircuitState.HALF_OPEN:
                logger.warning(f"Circuit breaker {self.name}: Recovery failed (OPEN)")
                self.state = CircuitState.OPEN
                self.success_count = 0
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    logger.error(
                        f"Circuit breaker {self.name}: Threshold reached "
                        f"({self.failure_count} failures) - Opening circuit"
                    )
                    self.state = CircuitState.OPEN

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery"""
        if not self.last_failure_time:
            return True
        return datetime.now() - self.last_failure_time >= self.recovery_timeout

    def _time_until_recovery(self) -> float:
        """Get seconds until recovery attempt"""
        if not self.last_failure_time:
            return 0
        delta = self.recovery_timeout - (datetime.now() - self.last_failure_time)
        return max(0, delta.total_seconds())

    def get_state(self) -> dict:
        """Get current circuit breaker state for monitoring"""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "time_until_recovery": self._time_until_recovery() if self.state == CircuitState.OPEN else None
        }

    async def reset(self):
        """Manually reset circuit breaker to CLOSED state"""
        async with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            logger.info(f"Circuit breaker {self.name}: Manually reset")


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and rejects request"""
    pass

class RAGOperationStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    SERVICE_UNAVAILABLE = "service_unavailable"
    CIRCUIT_OPEN = "circuit_open"

class RAGResponse:
    """Standardized response for all RAG operations

    Unified response format supporting both Phase 3 (ok/fail API) and Phase 5 (status enum).
    """

    def __init__(
        self,
        status: RAGOperationStatus,
        data: Optional[Any] = None,
        error: Optional[str] = None,
        metadata: Optional[dict] = None,
        message: str = ""
    ):
        self.status = status
        self.data = data
        self.error = error
        self.metadata = metadata or {}
        self.message = message

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
            "message": self.message,
            "success": self.is_success  # Backward compatibility
        }

    @property
    def is_success(self) -> bool:
        return self.status in [RAGOperationStatus.SUCCESS, RAGOperationStatus.PARTIAL_SUCCESS]

    @property
    def success(self) -> bool:
        """Backward compatibility with Phase 3 API"""
        return self.is_success

    @classmethod
    def ok(cls, data: Any = None, message: str = "Operation successful", metadata: Optional[dict] = None) -> "RAGResponse":
        """Create successful response (Phase 3 compatibility helper)"""
        return cls(
            status=RAGOperationStatus.SUCCESS,
            data=data,
            message=message,
            error=None,
            metadata=metadata
        )

    @classmethod
    def fail(cls, error: str, message: str = "Operation failed", metadata: Optional[dict] = None) -> "RAGResponse":
        """Create failed response (Phase 3 compatibility helper)"""
        return cls(
            status=RAGOperationStatus.FAILURE,
            data=None,
            message=message,
            error=error,
            metadata=metadata
        )