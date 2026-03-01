"""
Query response cache using Redis.

Provides fast caching of RAG query responses to reduce latency
and API costs. Supports TTL, selective invalidation, and metrics.
"""

import redis.asyncio as redis
import hashlib
import json
from typing import Optional, Dict, Any, List
import logging
import os

logger = logging.getLogger(__name__)


class QueryCache:
    """Redis-based cache for query responses."""

    def __init__(
        self,
        redis_url: str = None,
        ttl: int = 3600,  # 1 hour default
        enabled: bool = True
    ):
        """
        Initialize query cache.

        Args:
            redis_url: Redis connection URL (default: from env or localhost)
            ttl: Time-to-live in seconds (default: 1 hour)
            enabled: Enable/disable cache globally
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.ttl = ttl
        self.enabled = enabled
        self._redis = None

    async def connect(self):
        """Connect to Redis."""
        if not self.enabled:
            logger.info("Query cache disabled")
            return

        try:
            self._redis = await redis.from_url(
                self.redis_url,
                decode_responses=True,
                encoding="utf-8"
            )
            await self._redis.ping()
            logger.info(f"✅ Query cache connected to Redis: {self.redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            logger.warning("Query cache disabled - continuing without cache")
            self.enabled = False

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            logger.info("Query cache connection closed")

    def _cache_key(
        self,
        query: str,
        collections: List[str],
        k: int,
        use_reranker: bool = False
    ) -> str:
        """
        Generate cache key from query parameters.

        Args:
            query: Query text
            collections: Collection names
            k: Number of results
            use_reranker: Reranker enabled

        Returns:
            Cache key (MD5 hash)
        """
        # Normalize inputs
        collections_sorted = ','.join(sorted(collections))
        key_data = f"{query}:{collections_sorted}:{k}:{use_reranker}"

        # Hash for compact key
        key_hash = hashlib.md5(key_data.encode()).hexdigest()

        return f"rag:query:{key_hash}"

    async def get(
        self,
        query: str,
        collections: List[str],
        k: int,
        use_reranker: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached query response.

        Returns:
            Cached result or None if not found
        """
        if not self.enabled or not self._redis:
            return None

        try:
            key = self._cache_key(query, collections, k, use_reranker)
            cached = await self._redis.get(key)

            if cached:
                logger.info(f"✅ Cache HIT for query: {query[:50]}...")
                return json.loads(cached)

            logger.debug(f"Cache MISS for query: {query[:50]}...")
            return None

        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    async def set(
        self,
        query: str,
        collections: List[str],
        k: int,
        result: Dict[str, Any],
        use_reranker: bool = False,
        ttl: Optional[int] = None
    ):
        """
        Cache query response.

        Args:
            query: Query text
            collections: Collection names
            k: Number of results
            result: Query result to cache
            use_reranker: Reranker enabled
            ttl: Override default TTL
        """
        if not self.enabled or not self._redis:
            return

        try:
            key = self._cache_key(query, collections, k, use_reranker)
            ttl = ttl or self.ttl

            # Serialize result
            cached_value = json.dumps(result, ensure_ascii=False)

            # Store with TTL
            await self._redis.setex(key, ttl, cached_value)

            logger.debug(f"Cached result for query: {query[:50]}... (TTL: {ttl}s)")

        except Exception as e:
            logger.error(f"Cache set error: {e}")

    async def invalidate_collection(self, collection_name: str):
        """
        Invalidate all cached queries for a collection.

        Args:
            collection_name: Collection to invalidate
        """
        if not self.enabled or not self._redis:
            return

        try:
            # Scan for all query keys
            pattern = "rag:query:*"
            count = 0

            async for key in self._redis.scan_iter(match=pattern, count=100):
                # Delete all query cache (simple approach)
                # Better: Store collection → key mapping
                await self._redis.delete(key)
                count += 1

            logger.info(f"Invalidated {count} cache entries for collection: {collection_name}")

        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")

    async def clear_all(self):
        """Clear all cached queries (development only)."""
        if not self.enabled or not self._redis:
            return

        try:
            pattern = "rag:query:*"
            count = 0

            async for key in self._redis.scan_iter(match=pattern):
                await self._redis.delete(key)
                count += 1

            logger.info(f"Cleared {count} cache entries")

        except Exception as e:
            logger.error(f"Cache clear error: {e}")

    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.enabled or not self._redis:
            return {"enabled": False}

        try:
            info = await self._redis.info("stats")
            
            # Count cached queries
            query_keys = 0
            async for key in self._redis.scan_iter(match="rag:query:*", count=1000):
                query_keys += 1

            hits = info.get("keyspace_hits", 0)
            misses = info.get("keyspace_misses", 0)
            total = hits + misses
            hit_rate = hits / total if total > 0 else 0.0

            return {
                "enabled": True,
                "total_connections": info.get("total_connections_received", 0),
                "total_commands": info.get("total_commands_processed", 0),
                "keyspace_hits": hits,
                "keyspace_misses": misses,
                "hit_rate": hit_rate,
                "cached_queries": query_keys
            }

        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {"enabled": True, "error": str(e)}


# Global instance
_query_cache = None


async def get_query_cache() -> QueryCache:
    """Get global query cache instance (singleton)."""
    global _query_cache
    
    if _query_cache is None:
        _query_cache = QueryCache()
        await _query_cache.connect()
    
    return _query_cache
