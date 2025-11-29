"""
RAG Document Management API Package.

This package provides endpoints for document lifecycle management:
- Upload: File upload and ingestion
- Query: Document retrieval and testing
- Management: Metadata and deletion
"""

from fastapi import APIRouter

from .upload import router as upload_router
from .query import router as query_router
from .management import router as management_router

# Main documents router that combines all sub-routers
router = APIRouter()

# Include all sub-routers with tags
router.include_router(upload_router, tags=["RAG Documents - Upload"])
router.include_router(query_router, tags=["RAG Documents - Query"])
router.include_router(management_router, tags=["RAG Documents - Management"])

__all__ = ["router"]
