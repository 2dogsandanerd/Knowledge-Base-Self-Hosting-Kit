"""Document operations models."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


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


class IndexRequest(BaseModel):
    """Request model for indexing documents."""
    text: str = Field(..., description="Text content to index")
    collection: str = Field(description="Target collection")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata")
