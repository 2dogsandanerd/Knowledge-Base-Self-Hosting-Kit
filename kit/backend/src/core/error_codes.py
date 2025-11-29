"""
Error Codes for RAG System.

Provides structured error codes for:
- Client errors (4xx)
- Server errors (5xx)
- Service-specific errors
"""

from enum import Enum


class ErrorCode(str, Enum):
    """Base error codes."""

    # === Client Errors (4xx) ===
    INVALID_INPUT = "INVALID_INPUT"
    INVALID_FILE_TYPE = "INVALID_FILE_TYPE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    COLLECTION_NOT_FOUND = "COLLECTION_NOT_FOUND"
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    TASK_NOT_FOUND = "TASK_NOT_FOUND"
    DUPLICATE_FILE = "DUPLICATE_FILE"
    INVALID_COLLECTION_NAME = "INVALID_COLLECTION_NAME"

    # === Server Errors (5xx) ===
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    CHROMADB_ERROR = "CHROMADB_ERROR"
    OLLAMA_ERROR = "OLLAMA_ERROR"
    LLM_ERROR = "LLM_ERROR"
    EMBEDDING_ERROR = "EMBEDDING_ERROR"
    INGESTION_ERROR = "INGESTION_ERROR"
    EXTRACTION_ERROR = "EXTRACTION_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"

    # === Operational Errors ===
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    RETRY_LATER = "RETRY_LATER"


class ServiceError(str, Enum):
    """Service-specific error codes."""

    CHROMADB_NOT_RUNNING = "CHROMADB_NOT_RUNNING"
    OLLAMA_NOT_RUNNING = "OLLAMA_NOT_RUNNING"
    EMBEDDING_MODEL_NOT_LOADED = "EMBEDDING_MODEL_NOT_LOADED"
    COLLECTION_EMBEDDING_MISMATCH = "COLLECTION_EMBEDDING_MISMATCH"


# Error Code to HTTP Status mapping
ERROR_STATUS_MAP = {
    # 400 - Bad Request
    ErrorCode.INVALID_INPUT: 400,
    ErrorCode.INVALID_FILE_TYPE: 400,
    ErrorCode.FILE_TOO_LARGE: 400,
    ErrorCode.DUPLICATE_FILE: 400,
    ErrorCode.INVALID_COLLECTION_NAME: 400,

    # 404 - Not Found
    ErrorCode.FILE_NOT_FOUND: 404,
    ErrorCode.COLLECTION_NOT_FOUND: 404,
    ErrorCode.DOCUMENT_NOT_FOUND: 404,
    ErrorCode.TASK_NOT_FOUND: 404,

    # 500 - Internal Server Error
    ErrorCode.INTERNAL_ERROR: 500,
    ErrorCode.DATABASE_ERROR: 500,
    ErrorCode.INGESTION_ERROR: 500,
    ErrorCode.EXTRACTION_ERROR: 500,

    # 503 - Service Unavailable
    ErrorCode.SERVICE_UNAVAILABLE: 503,
    ErrorCode.CHROMADB_ERROR: 503,
    ErrorCode.OLLAMA_ERROR: 503,
    ErrorCode.LLM_ERROR: 503,
    ErrorCode.EMBEDDING_ERROR: 503,
    ServiceError.CHROMADB_NOT_RUNNING: 503,
    ServiceError.OLLAMA_NOT_RUNNING: 503,

    # 504 - Gateway Timeout
    ErrorCode.TIMEOUT_ERROR: 504,

    # 429 - Too Many Requests
    ErrorCode.RATE_LIMIT_EXCEEDED: 429,
}
