"""
API endpoint for the RAG Analysis Cockpit.

This endpoint provides a more detailed, unmapped response from the QueryService,
suitable for a rich analysis UI.
"""

from fastapi import APIRouter, Depends, HTTPException
from src.core.exceptions import ChromaDBError
import logging

from src.api.v1.dependencies import get_query_service
from src.services.auth_service import get_current_user
from src.database.models import User
from .models import QueryRequest # Re-use the same request model
from src.core.services.query_service import QueryService

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/cockpit/query")
async def cockpit_query(
    request: QueryRequest,
    query_service: QueryService = Depends(get_query_service),
    current_user: User = Depends(get_current_user)
):
    """
    Query RAG and get the full, detailed response from the QueryService.
    This is intended for the RAG Analysis Cockpit UI.
    """
    logger.debug(f"Cockpit query request: collections={request.collections}, k={request.k}, query_len={len(request.query)}")

    try:
        # Use system_prompt from request, or default if not provided
        system_context = request.system_prompt or "You are a helpful assistant. Answer the user's query based on the provided context."

        # Use the new centralized answer_query method
        result = await query_service.answer_query(
            query_text=request.query,
            collection_names=request.collections,
            final_k=request.k,
            system_prompt=system_context,
            temperature=request.temperature, # Pass temperature
            llm_model=request.llm_model, # Pass llm_model
            use_reranker=request.use_reranker or False # Pass use_reranker
        )

        if not result["metadata"]["success"]:
            error_message = result["metadata"].get("error", "Unknown error during query.")
            logger.error(f"Cockpit query failed: {error_message}")
            # Return the full error response for the cockpit to display
            return result

        logger.info(f"Cockpit query successful, returning full detailed response.")

        # Return the full, unmapped response from the QueryService
        return result

    except Exception as e:
        logger.error(f"Cockpit query failed with an exception: {e}", exc_info=True)
        raise ChromaDBError(str(e))
