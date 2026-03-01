"""
Embedding vector cache using Redis.

Caches embedding vectors to reduce API calls and costs.
Supports batch operations and binary storage for efficiency.
"""

import redis.asyncio as redis
import hashlib
import numpy as np
from typing import Optional, Dict, List
import logging
import os

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Redis-based cache for embedding vectors."""

    def __init__(
        self,
        redis_url: str = None,
        ttl: int = 86400,  # 24 hours default
        enabled: bool = True
    ):
        """
        Initialize embedding cache.

        Args:
            redis_url: Redis connection URL
            ttl: Time-to-live in seconds (default: 24 hours)
            enabled: Enable/disable cache
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.ttl = ttl
        self.enabled = enabled
        self._redis = None

    async def connect(self):
        """Connect to Redis."""
        if not self.enabled:
            logger.info("Embedding cache disabled")
            return

        try:
            self._redis = await redis.from_url(
                self.redis_url,
                decode_responses=False  # Binary mode for vectors
            )
            await self._redis.ping()
            logger.info("✅ Embedding cache connected to Redis")
        except Exception as e:
            logger.error(f"Embedding cache connection failed: {e}")
            logger.warning("Embedding cache disabled - continuing without cache")
            self.enabled = False

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            logger.info("Embedding cache connection closed")

    def _cache_key(self, text: str) -> str:
        """
        Generate cache key from text.

        Args:
            text: Input text

        Returns:
            Cache key
        """
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"rag:embedding:{text_hash}"

    async def get(self, text: str) -> Optional[np.ndarray]:
        """
        Get cached embedding.

        Args:
            text: Input text

        Returns:
            Embedding vector or None if not cached
        """
        if not self.enabled or not self._redis:
            return None

        try:
            key = self._cache_key(text)
            cached = await self._redis.get(key)

            if cached:
                # Deserialize numpy array
                embedding = np.frombuffer(cached, dtype=np.float32)
                logger.debug(f"✅ Embedding cache HIT: {text[:50]}...")
                return embedding

            logger.debug(f"Embedding cache MISS: {text[:50]}...")
            return None

        except Exception as e:
            logger.error(f"Embedding cache get error: {e}")
            return None

    async def set(self, text: str, embedding: np.ndarray):
        """
        Cache embedding vector.

        Args:
            text: Input text
            embedding: Embedding vector (numpy array)
        """
        if not self.enabled or not self._redis:
            return

        try:
            key = self._cache_key(text)

            # Serialize numpy array to bytes
            embedding_bytes = embedding.astype(np.float32).tobytes()

            # Store with TTL
            await self._redis.setex(key, self.ttl, embedding_bytes)

            logger.debug(f"Cached embedding: {text[:50]}...")

        except Exception as e:
            logger.error(f"Embedding cache set error: {e}")

    async def batch_get(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Get multiple embeddings at once.

        Args:
            texts: List of input texts

        Returns:
            Dict mapping text to embedding (only cached ones)
        """
        if not self.enabled or not self._redis:
            return {}

        results = {}
        keys = [self._cache_key(text) for text in texts]

        try:
            cached_values = await self._redis.mget(keys)

            for text, cached in zip(texts, cached_values):
                if cached:
                    embedding = np.frombuffer(cached, dtype=np.float32)
                    results[text] = embedding

            if results:
                logger.info(f"✅ Embedding batch cache: {len(results)}/{len(texts)} hits")

            return results

        except Exception as e:
            logger.error(f"Embedding batch get error: {e}")
            return {}

    async def batch_set(self, embeddings: Dict[str, np.ndarray]):
        """
        Cache multiple embeddings at once.

        Args:
            embeddings: Dict mapping text to embedding
        """
        if not self.enabled or not self._redis:
            return

        try:
            pipe = self._redis.pipeline()

            for text, embedding in embeddings.items():
                key = self._cache_key(text)
                embedding_bytes = embedding.astype(np.float32).tobytes()
                pipe.setex(key, self.ttl, embedding_bytes)

            await pipe.execute()

            logger.debug(f"Cached {len(embeddings)} embeddings")

        except Exception as e:
            logger.error(f"Embedding batch set error: {e}")

    async def stats(self) -> Dict[str, any]:
        """Get embedding cache statistics."""
        if not self.enabled or not self._redis:
            return {"enabled": False}

        try:
            # Count cached embeddings
            embedding_keys = 0
            async for key in self._redis.scan_iter(match="rag:embedding:*", count=1000):
                embedding_keys += 1

            return {
                "enabled": True,
                "cached_embeddings": embedding_keys
            }

        except Exception as e:
            logger.error(f"Embedding stats error: {e}")
            return {"enabled": True, "error": str(e)}


# Global instance
_embedding_cache = None


async def get_embedding_cache() -> EmbeddingCache:
    """Get global embedding cache instance (singleton)."""
    global _embedding_cache
    
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
        await _embedding_cache.connect()
    
    return _embedding_cache
