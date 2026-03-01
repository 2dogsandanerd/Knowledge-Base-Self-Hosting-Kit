"""Ingestion and upload models."""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from pathlib import Path


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

    file: str = Field(
        ...,
        description="Full file path to process"
    )
    collection: str = Field(
        ...,
        description="Target ChromaDB collection name"
    )
    process_options: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {"chunk_size": 1000, "chunk_overlap": 200}
    )

    @field_validator("file")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """Validate that file path exists and is readable."""
        path = Path(v)

        if not path.exists():
            raise ValueError(f"File does not exist: {v}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {v}")

        # Check file extension
        allowed_extensions = {
            '.pdf', '.docx', '.pptx', '.xlsx',
            '.html', '.md', '.csv', '.txt',
            '.eml', '.mbox', '.py', '.js', '.java'
        }

        if path.suffix.lower() not in allowed_extensions:
            raise ValueError(
                f"Unsupported file type: {path.suffix}. "
                f"Allowed: {allowed_extensions}"
            )

        return v

    @field_validator("process_options")
    @classmethod
    def validate_process_options(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate chunking parameters."""
        chunk_size = v.get("chunk_size", 1000)
        chunk_overlap = v.get("chunk_overlap", 200)

        if chunk_size < 100:
            raise ValueError("chunk_size must be at least 100")
        if chunk_size > 5000:
            raise ValueError("chunk_size must not exceed 5000")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        return v


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

    assignments: List[FileAssignment] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="File-to-collection assignments (max 100 files per batch)"
    )
    async_mode: bool = Field(
        default=True,
        description="Process asynchronously in background"
    )

    @field_validator("assignments")
    @classmethod
    def validate_batch_size(cls, v: List[FileAssignment]) -> List[FileAssignment]:
        """Ensure batch is not empty and not too large."""
        if len(v) == 0:
            raise ValueError("At least one file assignment is required")
        if len(v) > 100:
            raise ValueError("Maximum 100 files per batch allowed")
        return v


class IngestionResponse(BaseModel):
    """Response for batch ingestion."""
    success: bool
    processed_files: int = 0
    failed_files: int = 0
    details: Dict[str, Any] = Field(default_factory=dict)
    task_id: Optional[str] = None  # For async processing
