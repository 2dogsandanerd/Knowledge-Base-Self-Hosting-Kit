"""
Context Propagation Utilities for OpenTelemetry.

This module provides utilities for propagating trace context across async tasks
and background jobs.
"""

from opentelemetry import context, trace
from typing import Any, Callable, TypeVar, Coroutine
import asyncio
from functools import wraps

T = TypeVar('T')


async def trace_async_task(
    task_name: str,
    func: Callable[..., Coroutine[Any, Any, T]],
    *args,
    **kwargs
) -> T:
    """
    Execute async task with trace context propagation.
    
    Args:
        task_name: Name for the span
        func: Async function to execute
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func
        
    Returns:
        Result from func
    """
    tracer = trace.get_tracer(__name__)
    
    with tracer.start_as_current_span(task_name):
        result = await func(*args, **kwargs)
        return result


def traced_async(span_name: str = None):
    """
    Decorator to automatically trace async functions.
    
    Args:
        span_name: Optional custom span name (defaults to function name)
        
    Example:
        @traced_async("my_operation")
        async def my_function():
            ...
    """
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            tracer = trace.get_tracer(func.__module__)
            name = span_name or func.__name__
            
            with tracer.start_as_current_span(name):
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def add_span_attributes(**attributes):
    """
    Add attributes to the current span.
    
    Args:
        **attributes: Key-value pairs to add as span attributes
    """
    span = trace.get_current_span()
    if span.is_recording():
        for key, value in attributes.items():
            span.set_attribute(key, value)


def add_span_event(name: str, attributes: dict = None):
    """
    Add an event to the current span.
    
    Args:
        name: Event name
        attributes: Optional event attributes
    """
    span = trace.get_current_span()
    if span.is_recording():
        span.add_event(name, attributes=attributes or {})


def record_exception(exception: Exception):
    """
    Record an exception in the current span.
    
    Args:
        exception: Exception to record
    """
    span = trace.get_current_span()
    if span.is_recording():
        span.record_exception(exception)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))
