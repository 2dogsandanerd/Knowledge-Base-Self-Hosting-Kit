"""
Knowledge Base Self-Hosting Kit - Main FastAPI Application

Simplified RAG-focused application extracted from Mail Modul Alpha.
Provides document ingestion, querying, and collection management for ChromaDB.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys
import os

from src.api.v1.rag import router as rag_router
from src.core.chroma_manager import get_chroma_manager
from src.core.config import get_config

# Configure logger
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    logger.info("🚀 Knowledge Base Self-Hosting Kit API starting...")

    try:
        # Configure ChromaDB connection
        # use_hot_reload=False to read from ENV vars instead of .env file
        config = get_config(use_hot_reload=False)
        chroma_host = os.getenv("CHROMA_HOST", "localhost")
        chroma_port = int(os.getenv("CHROMA_PORT", 8000))

        chroma_manager = get_chroma_manager()
        chroma_manager.configure(chroma_host, chroma_port)

        # Test connection
        client = chroma_manager.get_client()
        if client:
            logger.success(f"✅ ChromaDB connected at {chroma_host}:{chroma_port}")
            logger.info(f"📚 API Documentation: http://localhost:8080/docs")
        else:
            logger.warning("⚠️  Failed to connect to ChromaDB")

        yield  # Application runs here

    except Exception as startup_error:
        logger.error(f"❌ Startup error: {startup_error}")
        raise
    finally:
        # Shutdown sequence
        logger.info("🛑 Shutting down...")
        try:
            chroma_manager = get_chroma_manager()
            if chroma_manager._client:
                # Close ChromaDB client connection properly
                logger.info("Closing ChromaDB connection...")
                # ChromaDB client cleanup happens here if needed
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

        logger.info("👋 Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Knowledge Base Self-Hosting Kit",
    description="RAG-powered knowledge base with ChromaDB backend",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware (only enabled in DEBUG mode for development)
# In production with nginx reverse proxy, CORS is not needed
if os.getenv("DEBUG", "false").lower() == "true":
    logger.info("🔓 CORS enabled (DEBUG mode)")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:8080"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    logger.info("🔒 CORS disabled (Production mode - using nginx reverse proxy)")

# Include RAG router
app.include_router(rag_router, prefix="/api/v1/rag", tags=["RAG"])


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "name": "Knowledge Base Self-Hosting Kit",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "source": "https://github.com/2dogsandanerd/Knowledge-Base-Self-Hosting-Kit"
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    try:
        chroma_manager = get_chroma_manager()
        client = chroma_manager.get_client()

        if client:
            # Try to list collections as a health check
            collections = client.list_collections()
            return {
                "status": "healthy",
                "chromadb": "connected",
                "collections_count": len(collections)
            }
        else:
            return {
                "status": "degraded",
                "chromadb": "disconnected"
            }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
