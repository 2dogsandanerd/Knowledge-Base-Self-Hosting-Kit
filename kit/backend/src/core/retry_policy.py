"""
Retry policies for ingestion and RAG operations.
"""
import asyncio
import random
from typing import Dict, Any, Callable
from loguru import logger

class RetryPolicy:
    """
    A flexible retry policy class that executes a function with retries.
    """
    def __init__(self, config: Dict[str, Any]):
        self.max_retries = config.get("max_retries", 3)
        self.interval_start = config.get("interval_start", 1)
        self.interval_step = config.get("interval_step", 2)
        self.interval_max = config.get("interval_max", 10)
        # We need to handle exceptions as strings from config
        self.retry_on_exceptions = tuple(
            getattr(__builtins__, exc_name, Exception) if isinstance(exc_name, str) else exc_name
            for exc_name in config.get("retry_on_exceptions", (Exception,))
        )

    async def execute(self, func: Callable, *args, **kwargs):
        """
        Execute an async function with the configured retry policy.
        """
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except self.retry_on_exceptions as e:
                last_exception = e
                delay = min(self.interval_start + (self.interval_step * attempt), self.interval_max)
                # Add jitter to avoid thundering herd problem
                delay *= (1 + random.random())
                
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed for '{func.__name__}' "
                    f"due to {type(e).__name__}. Retrying in {delay:.2f} seconds."
                )
                await asyncio.sleep(delay)
        
        logger.error(
            f"Function '{func.__name__}' failed after {self.max_retries} retries."
        )
        raise last_exception


# Default retry policy configuration, suitable for most network operations.
DEFAULT_RETRY_CONFIG: Dict[str, Any] = {
    "max_retries": 3,
    "interval_start": 1,
    "interval_step": 2,
    "interval_max": 10,
    "retry_on_exceptions": (
        "ConnectionError",
        "TimeoutError",
        "ServiceUnavailableError",
    )
}

# Standard retry policy for ingestion tasks
INGESTION_RETRY_CONFIG: Dict[str, Any] = {
    "max_retries": 3,
    "interval_start": 2,  # Start with 2 seconds wait
    "interval_step": 2,   # Add 2 seconds each retry
    "interval_max": 10,   # Max wait 10 seconds
    "retry_on_exceptions": (
        "ConnectionError",
        "TimeoutError",
        "ServiceUnavailableError",
        "RateLimitError",
        "APIConnectionError",  # OpenAI/Anthropic
        "APITimeoutError",     # OpenAI/Anthropic
        "InternalServerError"  # OpenAI/Anthropic
    )
}

# Pre-configured instances for easy use
DEFAULT_RETRY = RetryPolicy(DEFAULT_RETRY_CONFIG)
INGESTION_RETRY = RetryPolicy(INGESTION_RETRY_CONFIG)

# Rate limits for external services (calls per minute)
RATE_LIMITS = {
    "openai": 60,      # Conservative default
    "anthropic": 60,
    "ollama": 1000,    # Local is fast
    "chroma": 1000
}