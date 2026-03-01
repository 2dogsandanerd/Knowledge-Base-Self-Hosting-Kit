"""
Pydantic models for RAG API endpoints.

This module contains all request/response models for:
- Query operations
- Collection management
- Document operations
- Docling ingestion (Phase 4)
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# ===== QUERY MODELS =====

class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str = Field(..., description="The query text to search for")
    collection: str = Field(description="Collection to query")
    k: int = Field(default=5, ge=1, le=50, description="Number of results to return")


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Query results")
    collection: str = Field(..., description="Collection that was queried")
    query: str = Field(..., description="Original query text")
    count: int = Field(..., description="Number of results returned")


# ===== COLLECTION MODELS =====

class CollectionCreate(BaseModel):
    """Request model for creating a collection."""
    name: str = Field(..., min_length=1, max_length=63, description="Collection name (ChromaDB compatible)")
    description: Optional[str] = Field(None, description="Optional collection description")


class CollectionResponse(BaseModel):
    """Response model for collection info."""
    name: str
    document_count: int = 0
    metadata: Optional[Dict[str, Any]] = None


class CollectionListResponse(BaseModel):
    """Response model for listing collections."""
    collections: List[CollectionResponse]
    total: int


# ===== DOCUMENT MODELS =====

class DocumentResponse(BaseModel):
    """Response model for a single document."""
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    collection: str


class DocumentListResponse(BaseModel):
    """Response model for listing documents."""
    documents: List[DocumentResponse]
    total: int
    collection: str
    offset: int = 0
    limit: int = 100


class UploadResponse(BaseModel):
    """Response model for document upload."""
    success: bool
    message: str
    document_count: int = 0
    collection: str
    errors: List[str] = Field(default_factory=list)


# ===== INDEX MODELS =====

class IndexRequest(BaseModel):
    """Request model for indexing documents."""
    text: str = Field(..., description="Text content to index")
    collection: str = Field(description="Target collection")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata")


# ===== INGESTION MODELS (Docling Phase 4) =====

class FilePreview(BaseModel):
    """Preview info for uploaded file before ingestion."""
    filename: str
    size_bytes: int
    mime_type: str
    preview_text: str = Field(..., max_length=500, description="First 500 chars of text")
    page_count: Optional[int] = None
    detected_language: Optional[str] = None


class AnalyzeFilesRequest(BaseModel):
    """Request to analyze files before ingestion."""
    files: List[str] = Field(..., description="List of file paths or base64-encoded content")


class AnalyzeFilesResponse(BaseModel):
    """Response with file analysis results."""
    previews: List[FilePreview]
    total_size_bytes: int
    estimated_chunks: int


class FileAssignment(BaseModel):
    """Assignment of a file to a collection."""
    file: str = Field(..., description="Full file path to process")
    collection: str = Field(..., description="Target ChromaDB collection name")
    process_options: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {"chunk_size": 1000, "chunk_overlap": 200}
    )


class ScanFolderRequest(BaseModel):
    """Request to scan a folder for Docling-compatible files."""
    folder_path: str
    recursive: bool = Field(default=True, description="Scan subdirectories recursively")
    max_depth: int = Field(default=10, ge=1, le=20, description="Maximum recursion depth")


class ScanFolderResponse(BaseModel):
    """Response with list of files found in folder scan."""
    files: List[Dict[str, Any]]
    total_files: int
    total_size: int
    summary: Dict[str, int]  # Count by extension


class IngestBatchRequest(BaseModel):
    """Request to ingest multiple files with collection assignments."""
    assignments: List[FileAssignment]
    async_mode: bool = Field(default=True, description="Process asynchronously")


class IngestionResponse(BaseModel):
    """Response for batch ingestion."""
    success: bool
    processed_files: int = 0
    failed_files: int = 0
    details: Dict[str, Any] = Field(default_factory=dict)
    task_id: Optional[str] = None  # For async processing
