"""
Document Management Endpoints.

Handles document metadata and lifecycle management:
- Embedding configuration info
- Document deletion
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from src.core.exceptions import ChromaDBError, ValidationError, CollectionNotFoundError, DocumentNotFoundError
import logging

from src.api.v1.dependencies import get_rag_client
from src.services.auth_service import get_current_user
from src.database.models import User

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/collections/{collection_name}/embedding-info")
async def get_collection_embedding_info(
    collection_name: str,
    rag_client=Depends(get_rag_client),
    current_user: User = Depends(get_current_user)
):
    """Get embedding configuration for a collection and check compatibility."""
    logger.debug(f"Getting embedding info for collection: {collection_name}")

    try:
        metadata = await rag_client.get_collection_metadata(collection_name)

        if not metadata:
            logger.warning(f"No metadata found for collection '{collection_name}'")
            raise HTTPException(
                status_code=404,
                detail=f"No metadata found for collection '{collection_name}'. This might be an old collection."
            )

        from src.services.config_service import config_service
        config = config_service.load_configuration()

        current_embedding_model = config.get("EMBEDDING_MODEL", "nomic-embed-text:latest")
        current_embedding_provider = config.get("EMBEDDING_PROVIDER", "ollama")

        try:
            current_dimensions = await rag_client.get_embedding_dimensions(current_embedding_model)
        except Exception:
            current_dimensions = 768

        current_embedding = {
            "model": current_embedding_model,
            "provider": current_embedding_provider,
            "dimensions": current_dimensions
        }

        collection_embedding = {
            "model": metadata.get("embedding_model"),
            "provider": metadata.get("embedding_provider"),
            "dimensions": metadata.get("embedding_dimensions")
        }

        compatible = (
            current_embedding["model"] == collection_embedding["model"] and
            current_embedding["dimensions"] == collection_embedding["dimensions"]
        )

        logger.debug(f"Embedding compatibility check for '{collection_name}': {compatible}")

        return {
            "collection_embedding": collection_embedding,
            "current_settings": current_embedding,
            "compatible": compatible,
            "created_at": metadata.get("created_at"),
            "description": metadata.get("description", "")
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get embedding info: {e}", exc_info=True)
        raise ChromaDBError(str(e))


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    collection_name: str = Query(...),
    rag_client=Depends(get_rag_client),
    current_user: User = Depends(get_current_user)
):
    """Delete document from collection."""
    logger.debug(f"Deleting document '{document_id}' from collection '{collection_name}'")

    try:
        response = await rag_client.delete_document(
            doc_id=document_id,
            collection_name=collection_name
        )

        if not response.is_success:
            logger.error(f"Failed to delete document '{document_id}': {response.error}")
            raise ChromaDBError(response.error)

        logger.info(f"Document '{document_id}' deleted successfully")
        return {
            "success": True,
            "deleted_id": document_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document deletion failed: {e}", exc_info=True)
        raise ChromaDBError(str(e))
