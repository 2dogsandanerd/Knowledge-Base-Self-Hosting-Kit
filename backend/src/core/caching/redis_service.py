"""
Redis Cache Service (Phase I.1).

Provides a unified interface for caching RAG operations (Queries, Embeddings) using Redis.
"""

import json
import hashlib
import pickle
from typing import Optional, Any, Union
from datetime import timedelta
import redis
from loguru import logger
import os

class RedisCacheService:
    """
    Wrapper around Redis for caching RAG data.
    """
    
    def __init__(self, host: str = None, port: int = None, db: int = 0):
        self.logger = logger.bind(component="RedisCacheService")
        
        # Load config from env or args
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", 6379))
        self.db = db
        
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=False # We handle serialization manually for flexibility
            )
            self.client.ping()
            self.enabled = True
            self.logger.info(f"Connected to Redis at {self.host}:{self.port}/{self.db}")
        except Exception as e:
            self.logger.warning(f"Failed to connect to Redis: {e}. Caching disabled.")
            self.enabled = False

    def _generate_key(self, prefix: str, *args) -> str:
        """Generate a unique cache key based on prefix and arguments."""
        content = "".join(str(arg) for arg in args)
        hash_val = hashlib.sha256(content.encode()).hexdigest()
        return f"{prefix}:{hash_val}"

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache."""
        if not self.enabled:
            return None
            
        try:
            data = self.client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            self.logger.warning(f"Cache get failed for {key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache with TTL (seconds)."""
        if not self.enabled:
            return
            
        try:
            serialized = pickle.dumps(value)
            self.client.setex(key, timedelta(seconds=ttl), serialized)
        except Exception as e:
            self.logger.warning(f"Cache set failed for {key}: {e}")

    # Specific Helpers for RAG
    
    def get_query_result(self, query: str, params: dict) -> Optional[dict]:
        """Get cached query result."""
        # Sort params to ensure stable key
        sorted_params = json.dumps(params, sort_keys=True)
        key = self._generate_key("query_v1", query, sorted_params)
        return self.get(key)

    def set_query_result(self, query: str, params: dict, result: dict, ttl: int = 86400):
        """Cache query result (default 24h)."""
        sorted_params = json.dumps(params, sort_keys=True)
        key = self._generate_key("query_v1", query, sorted_params)
        self.set(key, result, ttl)

    def get_embedding(self, text: str, model: str) -> Optional[list]:
        """Get cached embedding."""
        key = self._generate_key(f"emb_{model}", text)
        return self.get(key)

    def set_embedding(self, text: str, model: str, embedding: list, ttl: int = 604800):
        """Cache embedding (default 7 days)."""
        key = self._generate_key(f"emb_{model}", text)
        self.set(key, embedding, ttl)

# Singleton instance
redis_cache = RedisCacheService()
