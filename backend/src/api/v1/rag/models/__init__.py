"""
Pydantic Models for RAG API.

All models are organized by domain:
- query: Query requests and responses
- collection: Collection management
- document: Document operations
- ingestion: Ingestion and upload
"""

# Query models
from .query import QueryRequest, QueryResponse

# Collection models
from .collection import (
    CollectionCreate,
    CollectionResponse,
    CollectionListResponse
)

# Document models
from .document import (
    DocumentResponse,
    DocumentListResponse,
    UploadResponse,
    IndexRequest
)

# Ingestion models
from .ingestion import (
    FilePreview,
    AnalyzeFilesRequest,
    AnalyzeFilesResponse,
    FileAssignment,
    ScanFolderRequest,
    ScanFolderResponse,
    IngestBatchRequest,
    IngestionResponse
)

__all__ = [
    # Query
    "QueryRequest",
    "QueryResponse",
    # Collection
    "CollectionCreate",
    "CollectionResponse",
    "CollectionListResponse",
    # Document
    "DocumentResponse",
    "DocumentListResponse",
    "UploadResponse",
    "IndexRequest",
    # Ingestion
    "FilePreview",
    "AnalyzeFilesRequest",
    "AnalyzeFilesResponse",
    "FileAssignment",
    "ScanFolderRequest",
    "ScanFolderResponse",
    "IngestBatchRequest",
    "IngestionResponse",
]
