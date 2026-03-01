"""
Query Result Caching for ChromaDB.

Caches query results to avoid repeated ChromaDB calls.
"""

from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import json
from loguru import logger


class QueryCache:
    """
    LRU cache for ChromaDB query results.

    Features:
    - TTL-based expiration (default: 5 minutes)
    - LRU eviction when size limit reached
    - Cache key based on (collection, query, k, filters, hybrid_search)
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 300  # 5 minutes
    ):
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._access_order: list = []
        self.logger = logger.bind(component="QueryCache")

        self._hits = 0
        self._misses = 0

    def _make_key(
        self,
        collection_name: str,
        query: str,
        k: int,
        filter_metadata: Optional[Dict] = None,
        hybrid_search: bool = False
    ) -> str:
        """Generate cache key from query parameters."""
        key_data = {
            "collection": collection_name,
            "query": query,
            "k": k,
            "filter": filter_metadata or {},
            "hybrid": hybrid_search
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(
        self,
        collection_name: str,
        query: str,
        k: int,
        filter_metadata: Optional[Dict] = None,
        hybrid_search: bool = False
    ) -> Optional[Any]:
        """Get cached result if available and not expired."""
        key = self._make_key(collection_name, query, k, filter_metadata, hybrid_search)

        if key not in self._cache:
            self._misses += 1
            return None

        result, timestamp = self._cache[key]

        # Check TTL
        if datetime.now() - timestamp > self.ttl:
            self.logger.debug(f"Cache expired: {key[:8]}...")
            del self._cache[key]
            self._access_order.remove(key)
            self._misses += 1
            return None

        # Update LRU
        self._access_order.remove(key)
        self._access_order.append(key)

        self._hits += 1
        self.logger.debug(f"Cache hit: {key[:8]}...")
        return result

    def set(
        self,
        collection_name: str,
        query: str,
        k: int,
        result: Any,
        filter_metadata: Optional[Dict] = None,
        hybrid_search: bool = False
    ):
        """Cache query result."""
        key = self._make_key(collection_name, query, k, filter_metadata, hybrid_search)

        # Evict LRU if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            lru_key = self._access_order.pop(0)
            del self._cache[lru_key]
            self.logger.debug(f"Evicted LRU: {lru_key[:8]}...")

        self._cache[key] = (result, datetime.now())

        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

        self.logger.debug(f"Cached: {key[:8]}...")

    def invalidate_collection(self, collection_name: str):
        """Invalidate all queries for a collection."""
        keys_to_remove = [
            key for key in self._cache.keys()
            if collection_name in str(key)
        ]

        for key in keys_to_remove:
            del self._cache[key]
            self._access_order.remove(key)

        self.logger.info(f"Invalidated {len(keys_to_remove)} cached queries for {collection_name}")

    def clear(self):
        """Clear entire cache."""
        count = len(self._cache)
        self._cache.clear()
        self._access_order.clear()
        self.logger.info(f"Cleared {count} cached queries")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests
        }


# Global singleton
query_cache = QueryCache(max_size=1000, ttl_seconds=300)
