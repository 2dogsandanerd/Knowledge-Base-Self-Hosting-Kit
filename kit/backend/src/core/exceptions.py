"""
Custom Exceptions for RAG System.

All custom exceptions inherit from BaseRAGException for consistent handling.
"""

from typing import Optional, Dict, Any
from .error_codes import ErrorCode, ServiceError, ERROR_STATUS_MAP


class BaseRAGException(Exception):
    """
    Base exception for all RAG errors.

    Attributes:
        message: Human-readable error message
        code: Structured error code (ErrorCode enum)
        details: Additional context (dict)
        retryable: Whether operation can be retried
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode | ServiceError,
        details: Optional[Dict[str, Any]] = None,
        retryable: bool = False
    ):
        self.message = message
        self.code = code
        self.details = details or {}
        self.retryable = retryable
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dict for API response."""
        return {
            "error": {
                "message": self.message,
                "code": self.code.value,
                "details": self.details,
                "retryable": self.retryable
            }
        }

    @property
    def status_code(self) -> int:
        """Get HTTP status code for this error."""
        return ERROR_STATUS_MAP.get(self.code, 500)


# === Client Errors ===

class ValidationError(BaseRAGException):
    """Validation error (400)."""

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            message=message,
            code=ErrorCode.INVALID_INPUT,
            details=details,
            retryable=False
        )


class RAGFileNotFoundError(BaseRAGException):
    """File not found error (404)."""

    def __init__(self, file_path: str):
        super().__init__(
            message=f"File not found: {file_path}",
            code=ErrorCode.FILE_NOT_FOUND,
            details={"file_path": file_path},
            retryable=False
        )


class CollectionNotFoundError(BaseRAGException):
    """Collection not found error (404)."""

    def __init__(self, collection_name: str):
        super().__init__(
            message=f"Collection not found: {collection_name}",
            code=ErrorCode.COLLECTION_NOT_FOUND,
            details={"collection": collection_name},
            retryable=False
        )


class DocumentNotFoundError(BaseRAGException):
    """Document not found error (404)."""

    def __init__(self, document_id: str, collection: str = ""):
        details = {"document_id": document_id}
        if collection:
            details["collection"] = collection

        super().__init__(
            message=f"Document not found: {document_id}",
            code=ErrorCode.DOCUMENT_NOT_FOUND,
            details=details,
            retryable=False
        )


class InvalidFileTypeError(BaseRAGException):
    """Invalid file type error (400)."""

    def __init__(self, file_type: str, allowed_types: list):
        super().__init__(
            message=f"Invalid file type: {file_type}",
            code=ErrorCode.INVALID_FILE_TYPE,
            details={
                "provided_type": file_type,
                "allowed_types": allowed_types
            },
            retryable=False
        )


class DuplicateFileError(BaseRAGException):
    """Duplicate file error (400)."""

    def __init__(self, file_path: str, existing_id: str = ""):
        details = {"file_path": file_path}
        if existing_id:
            details["existing_id"] = existing_id

        super().__init__(
            message=f"File already exists: {file_path}",
            code=ErrorCode.DUPLICATE_FILE,
            details=details,
            retryable=False
        )


# === Server Errors ===

class ServiceUnavailableError(BaseRAGException):
    """Service unavailable error (503)."""

    def __init__(self, service_name: str, reason: str = ""):
        super().__init__(
            message=f"Service unavailable: {service_name}. {reason}",
            code=ErrorCode.SERVICE_UNAVAILABLE,
            details={"service": service_name, "reason": reason},
            retryable=True
        )


class ChromaDBError(BaseRAGException):
    """ChromaDB specific error (503)."""

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            message=f"ChromaDB error: {message}",
            code=ErrorCode.CHROMADB_ERROR,
            details=details,
            retryable=True
        )


class IngestionError(BaseRAGException):
    """Ingestion processing error (500)."""

    def __init__(self, message: str, file_path: str = "", details: Optional[Dict] = None):
        full_details = details or {}
        if file_path:
            full_details["file_path"] = file_path

        super().__init__(
            message=f"Ingestion failed: {message}",
            code=ErrorCode.INGESTION_ERROR,
            details=full_details,
            retryable=False
        )


class LLMError(BaseRAGException):
    """LLM processing error (503)."""

    def __init__(self, message: str, model: str = "", details: Optional[Dict] = None):
        full_details = details or {}
        if model:
            full_details["model"] = model

        super().__init__(
            message=f"LLM error: {message}",
            code=ErrorCode.LLM_ERROR,
            details=full_details,
            retryable=True
        )


class TaskNotFoundError(BaseRAGException):
    """Task not found error (404)."""

    def __init__(self, task_id: str):
        super().__init__(
            message=f"Task not found: {task_id}",
            code=ErrorCode.TASK_NOT_FOUND,
            details={"task_id": task_id},
            retryable=False
        )


class ExtractionError(BaseRAGException):
    """Generic extraction processing error (500)."""

    def __init__(self, message: str, file_path: str = "", details: Optional[Dict] = None):
        full_details = details or {}
        if file_path:
            full_details["file_path"] = file_path

        super().__init__(
            message=f"Extraction failed: {message}",
            code=ErrorCode.EXTRACTION_ERROR,
            details=full_details,
            retryable=False
        )
