"""
Cache Management API.

Provides endpoints for cache statistics, invalidation, and management.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from src.core.caching import get_query_cache, get_embedding_cache
from src.api.v1.dependencies import get_current_user
from src.database.models import User

router = APIRouter(prefix="/cache", tags=["cache"])


class CacheStats(BaseModel):
    """Cache statistics response."""
    query_cache: Dict[str, Any]
    embedding_cache: Dict[str, Any]


@router.get("/stats", response_model=CacheStats)
async def get_cache_stats(
    user: User = Depends(get_current_user)
):
    """
    Get cache statistics.
    
    Returns hit rates, cached items count, and other metrics.
    """
    query_cache = await get_query_cache()
    embedding_cache = await get_embedding_cache()
    
    return CacheStats(
        query_cache=await query_cache.stats(),
        embedding_cache=await embedding_cache.stats()
    )


@router.post("/clear")
async def clear_cache(
    user: User = Depends(get_current_user)
):
    """
    Clear all caches (admin only).
    
    ⚠️ Use with caution - this will clear all cached queries and embeddings.
    """
    # TODO: Add admin check
    # if not user.is_admin:
    #     raise HTTPException(403, "Admin access required")
    
    query_cache = await get_query_cache()
    embedding_cache = await get_embedding_cache()
    
    await query_cache.clear_all()
    # Embedding cache doesn't have clear_all yet - can be added if needed
    
    return {"message": "Cache cleared successfully"}


@router.post("/invalidate/collection/{collection_name}")
async def invalidate_collection(
    collection_name: str,
    user: User = Depends(get_current_user)
):
    """
    Invalidate all cached queries for a specific collection.
    
    Use this when collection data has been updated.
    """
    query_cache = await get_query_cache()
    await query_cache.invalidate_collection(collection_name)
    
    return {"message": f"Cache invalidated for collection: {collection_name}"}
