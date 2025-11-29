"""
RAG Query endpoints.

Handles querying the vector database for relevant context.
"""

from fastapi import APIRouter, Depends, HTTPException
from src.core.exceptions import ChromaDBError, ValidationError
from typing import Dict, Any
import logging

from src.api.v1.dependencies import get_query_service, get_rag_client # Import get_query_service and get_rag_client
from src.services.auth_service import get_current_user
from src.database.models import User
from .models import QueryRequest, IndexRequest
from src.core.services.query_service import QueryService # Import QueryService for type hinting

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/query")
async def query_rag(
    request: QueryRequest,
    query_service: QueryService = Depends(get_query_service), # Use QueryService dependency
    current_user: User = Depends(get_current_user)
):
    """Query RAG knowledge base and get a synthesized answer from the LLM."""
    logger.debug(f"Query request: collections={request.collections}, k={request.k}, query_len={len(request.query)}")


    try:
        # Handle both 'collection' (singular) and 'collections' (plural) from frontend
        collection_names = request.collections
        if not collection_names and request.collection:
            collection_names = [request.collection]
        elif not collection_names:
            raise ValidationError("Either 'collection' or 'collections' must be provided")
        
        # Define a default system context
        system_context = "You are a helpful assistant. Answer the user's query based on the provided context."

        # Include user ID for experiment tracking
        result = await query_service.answer_query(
            query_text=request.query,
            collection_names=collection_names,
            final_k=request.k,
            system_prompt=system_context,
            temperature=request.temperature or 0.1,
            use_reranker=request.use_reranker,
            # rerank_top_k removed - not a valid parameter for QueryService.answer_query()
            user_id=current_user.id if hasattr(current_user, 'id') else str(current_user.email) if hasattr(current_user, 'email') else "unknown_user"
        )

        if not result["metadata"]["success"]:
            error_message = result["metadata"].get("error", "Unknown error during query.")
            logger.error(f"Query with context failed: {error_message}")
            raise ChromaDBError(error_message)

        logger.info(f"Query successful, returning synthesized LLM answer.")

        # Return the actual LLM response and the context used in the frontend-compatible format
        return {
            "llm_response": result["response"],
            "context_chunks": result["context"], # Frontend expects context_chunks
            "query": request.query
        }

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise ChromaDBError(str(e))


@router.post("/search")
async def search_documents(
    request: QueryRequest,
    query_service: QueryService = Depends(get_query_service),
    current_user: User = Depends(get_current_user)
):
    """Search for documents without LLM generation - instant results!"""
    logger.debug(f"Search request: collections={request.collections}, k={request.k}, query_len={len(request.query)}")

    try:
        # Handle both 'collection' (singular) and 'collections' (plural) from frontend
        collection_names = request.collections
        if not collection_names and request.collection:
            collection_names = [request.collection]
        elif not collection_names:
            raise ValidationError("Either 'collection' or 'collections' must be provided")

        # Retrieve and rerank documents WITHOUT calling LLM
        ranked_nodes, total_candidates = await query_service.retrieve_and_rerank(
            query_text=request.query,
            collection_names=collection_names,
            initial_k_per_collection=request.k * 2,  # Get more candidates for reranking
            final_k=request.k,
            use_reranker=request.use_reranker or False,
            user_id=current_user.id if hasattr(current_user, 'id') else str(current_user.email) if hasattr(current_user, 'email') else "unknown_user"
        )

        logger.info(f"Search successful, returning {len(ranked_nodes)} documents from {total_candidates} candidates.")

        # Convert RankedNode objects to dicts for JSON serialization
        documents = []
        for node in ranked_nodes:
            documents.append({
                "content": node.content,
                "source_collection": node.source_collection,
                "relevance_score": node.relevance_score,
                "metadata": node.metadata,
                "distance": node.distance,
                "source": node.source,
                "page_number": node.page_number
            })

        # Return documents in a format compatible with frontend
        return {
            "documents": documents,
            "total_found": len(documents),
            "total_candidates": total_candidates,
            "query": request.query
        }

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise ChromaDBError(str(e))


@router.post("/index")
async def index_documents(
    request: IndexRequest,
    rag_client=Depends(get_rag_client),
    current_user: User = Depends(get_current_user)
):
    """Index documents into RAG knowledge base"""
    logger.debug(f"Index request: collection={request.collection}, docs_path={request.docs_path}")

    try:
        success = await rag_client.index_documents(
            docs_path=request.docs_path,
            collection_name=request.collection
        )

        logger.info(f"Indexing {'successful' if success else 'failed'}: {request.docs_path} → {request.collection}")

        return {
            "success": success,
            "collection": request.collection,
            "docs_path": request.docs_path
        }

    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        raise ChromaDBError(str(e))


@router.get("/stats")
async def get_rag_stats(
    collection: str = "project_knowledge_base",
    rag_client=Depends(get_rag_client),
    current_user: User = Depends(get_current_user)
):
    """Get RAG collection statistics"""
    logger.debug(f"Stats request for collection: {collection}")

    try:
        response = await rag_client.collection_manager.get_collection_stats(collection)
        if not response.is_success:
            logger.error(f"Failed to get stats for '{collection}': {response.error}")
            raise ChromaDBError(response.error)

        logger.debug(f"Stats retrieved for {collection}: {response.data}")
        return response.data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}", exc_info=True)
        raise ChromaDBError(str(e))
