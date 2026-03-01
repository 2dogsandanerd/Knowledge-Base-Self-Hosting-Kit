"""
RAG Ingestion API Package.

This package provides endpoints for document ingestion into ChromaDB:
- Analysis: AI-powered file analysis and classification
- Scanning: Folder scanning for compatible files
- Status: Task status polling
"""

from fastapi import APIRouter

from .analysis import router as analysis_router
from .batch import router as batch_router
from .scanning import router as scanning_router
from .status import router as status_router
# NOTE: llm_tasks removed - requires dependencies not in Community Edition

# Main ingestion router that combines all sub-routers
router = APIRouter()

# Include all sub-routers with tags
router.include_router(analysis_router, tags=["RAG Ingestion - Analysis"])
router.include_router(batch_router, tags=["RAG Ingestion - Batch"])
router.include_router(scanning_router, tags=["RAG Ingestion - Scanning"])
router.include_router(status_router, tags=["RAG Ingestion - Status"])

from .ingest_folder import router as ingest_folder_router
router.include_router(ingest_folder_router, tags=["RAG Ingestion - Folder"])

__all__ = ["router"]
