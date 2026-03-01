"""
RAG API Router Package.

This package combines all RAG-related endpoints:
- Query: RAG knowledge base queries
- Collections: Collection management (CRUD)
- Documents: Document upload, retrieval, deletion
- Ingestion: Advanced Docling-based ingestion (Phase 4)
"""

from fastapi import APIRouter

from .collections import router as collections_router
from .documents import router as documents_router
from .query import router as query_router
from .ingestion import router as ingestion_router
from .cockpit import router as cockpit_router # Import cockpit router

# Main RAG router that combines all sub-routers
router = APIRouter()

# Include all sub-routers
router.include_router(query_router, tags=["RAG Query"])
router.include_router(cockpit_router, tags=["RAG Cockpit"]) # Include cockpit router
router.include_router(collections_router, tags=["RAG Collections"])
router.include_router(documents_router, tags=["RAG Documents"])
router.include_router(ingestion_router, tags=["RAG Ingestion (Phase 4)"])

__all__ = ["router"]
