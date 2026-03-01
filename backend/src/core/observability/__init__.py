"""
Observability Module - OpenTelemetry Tracing Setup.

This module initializes OpenTelemetry tracing for the Mail Modul Alpha application.
It provides distributed tracing capabilities to track requests through the RAG pipeline.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased, ParentBased
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from loguru import logger
import os


def setup_tracing(service_name: str = "knowledge_base_self_hosting_kit") -> trace.Tracer:
    """
    Initialize OpenTelemetry tracing.
    
    Args:
        service_name: Name of the service for trace identification
        
    Returns:
        Configured tracer instance
    """
    try:
        # Create resource with service name
        resource = Resource(attributes={
            SERVICE_NAME: service_name
        })
        
        # Environment-based sampling
        sample_rate = float(os.getenv("OTEL_SAMPLE_RATE", "1.0"))  # 100% in dev
        logger.info(f"OpenTelemetry sampling rate: {sample_rate * 100}%")
        
        sampler = ParentBased(
            root=TraceIdRatioBased(sample_rate)
        )
        
        # Create tracer provider
        provider = TracerProvider(
            resource=resource,
            sampler=sampler
        )
        
        # Configure exporters
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        
        try:
            # OTLP Exporter (to Jaeger/Tempo)
            otlp_exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                insecure=True
            )
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info(f"OpenTelemetry OTLP exporter configured: {otlp_endpoint}")
        except Exception as e:
            logger.warning(f"Could not configure OTLP exporter: {e}")
            logger.info("Falling back to console exporter for development")
            # Fallback to console exporter for development
            console_exporter = ConsoleSpanExporter()
            provider.add_span_processor(BatchSpanProcessor(console_exporter))
        
        # Set global tracer provider
        trace.set_tracer_provider(provider)
        
        logger.info(f"OpenTelemetry tracing initialized for service: {service_name}")
        
        return trace.get_tracer(__name__)
        
    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry tracing: {e}")
        # Return a no-op tracer if initialization fails
        return trace.get_tracer(__name__)


def get_tracer(name: str = __name__) -> trace.Tracer:
    """
    Get a tracer instance.
    
    Args:
        name: Name for the tracer (usually __name__)
        
    Returns:
        Tracer instance
    """
    return trace.get_tracer(name)
