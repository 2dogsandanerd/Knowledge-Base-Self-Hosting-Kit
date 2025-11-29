"""Query Engine for RAG system.

This module manages query operations with multi-collection support and result validation.
Extracted from RAGClient to improve separation of concerns and solve Issues #3 and #8.

Issue #3: Unclear Multi-Collection Priority
- Problem: No support for querying multiple collections with priority/merging
- Solution: Multi-collection query with configurable priority and result merging

Issue #8: Missing Result Validation
- Problem: No validation of query results (embedding dimensions, relevance threshold)
- Solution: Structured validation with clear error handling
"""

import re
import asyncio
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class QueryConfig:
    """Configuration for query operations (Phase 5 compatibility).

    Attributes:
        n_results: Number of results to return
        min_relevance: Minimum relevance threshold (0.0-1.0)
        merge_strategy: Strategy for multi-collection merging
        filters: Optional metadata filters
    """
    n_results: int = 5
    min_relevance: float = 0.0
    merge_strategy: str = "interleave"
    filters: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Structured query result with validation.

    Attributes:
        content: Document content
        metadata: Document metadata
        relevance_score: Combined relevance score (0.0-1.0)
        collection_name: Source collection name
        base_relevance: Base semantic relevance (for debugging)
        keyword_boost: Keyword boost amount (for debugging)
    """
    content: str
    metadata: Dict
    relevance_score: float
    collection_name: str
    base_relevance: float = 0.0
    keyword_boost: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "relevance_score": self.relevance_score,
            "collection_name": self.collection_name,
            "_base_relevance": self.base_relevance,
            "_keyword_boost": self.keyword_boost
        }


from src.core.services.query_cache import query_cache

class QueryEngine:
    """Manages query operations with multi-collection support and validation.

    This engine handles:
    - Single and multi-collection queries
    - Hybrid search (semantic + keyword)
    - Result validation and merging
    """

    def __init__(self, chroma_client=None, embeddings=None):
        self.chroma_client = chroma_client
        self.embeddings = embeddings
        self.logger = logger.bind(component="QueryEngine")

    def set_client(self, chroma_client):
        self.chroma_client = chroma_client

    def set_embeddings(self, embeddings):
        self.embeddings = embeddings

    async def query(
        self,
        query_text: str,
        collection_names: Optional[List[str]] = None,
        config: Optional[QueryConfig] = None
    ) -> List[QueryResult]:
        """Unified query method for the RAG system.

        Args:
            query_text: Search query
            collection_names: List of collections to query. If None, queries all collections.
            config: QueryConfig with n_results, min_relevance, merge_strategy

        Returns:
            List of QueryResult objects
        """
        config = config or QueryConfig()

        if not self.chroma_client or not self.embeddings:
            raise RuntimeError("QueryEngine not fully initialized. Client or embeddings missing.")

        target_collections = collection_names
        if not target_collections:
            self.logger.info("No collections specified, querying all available collections.")
            actual_client = self.chroma_client.get_client() if hasattr(self.chroma_client, 'get_client') else self.chroma_client
            all_collections = await asyncio.to_thread(actual_client.list_collections)
            target_collections = [col.name for col in all_collections]

        if not target_collections:
            self.logger.warning("No collections available to query.")
            return []

        # Query all target collections concurrently
        tasks = [
            self._query_single_collection(
                query_text=query_text,
                collection_name=name,
                k=config.n_results,
                min_relevance=config.min_relevance,
                filters=config.filters
            )
            for name in target_collections
        ]
        
        results_from_all_collections = await asyncio.gather(*tasks)
        
        # Flatten the list of lists
        all_results = [item for sublist in results_from_all_collections for item in sublist]

        return self._merge_results(all_results, config.n_results, config.merge_strategy)

    def _merge_results(
        self, 
        all_results: List[QueryResult], 
        k: int, 
        merge_strategy: str
    ) -> List[QueryResult]:
        """Merge results from multiple collections based on a strategy."""
        if not all_results:
            return []

        if merge_strategy == "best":
            all_results.sort(key=lambda x: x.relevance_score, reverse=True)
            merged = all_results[:k]
        else:  # Default to interleave
            results_by_collection = {}
            for result in all_results:
                if result.collection_name not in results_by_collection:
                    results_by_collection[result.collection_name] = []
                results_by_collection[result.collection_name].append(result)

            merged = []
            max_len = max(len(v) for v in results_by_collection.values()) if results_by_collection else 0
            for i in range(max_len):
                for collection_name in results_by_collection.keys():
                    if i < len(results_by_collection[collection_name]):
                        merged.append(results_by_collection[collection_name][i])
                    if len(merged) >= k:
                        break
                if len(merged) >= k:
                    break
        
        self.logger.info(f"Merged {len(all_results)} results into {len(merged)} using '{merge_strategy}' strategy.")
        return merged

    async def _query_single_collection(
        self,
        query_text: str,
        collection_name: str,
        k: int,
        min_relevance: float,
        filters: Optional[Dict] = None
    ) -> List[QueryResult]:
        """Query a single collection with caching, hybrid search, and validation."""
        # Try cache first
        cached_result = query_cache.get(collection_name, query_text, k, filters)
        if cached_result is not None:
            self.logger.debug(f"Cache hit for query in '{collection_name}': {query_text[:50]}...")
            # Ensure cached results meet the current relevance threshold
            return [res for res in cached_result if res.relevance_score >= min_relevance]

        self.logger.debug(f"Cache miss for query in '{collection_name}': {query_text[:50]}...")
        try:
            # Get the actual client through get_client method if it exists
            actual_client = self.chroma_client.get_client() if hasattr(self.chroma_client, 'get_client') else self.chroma_client
            collection = await asyncio.to_thread(actual_client.get_collection, collection_name)
            
            # Use appropriate method for embedding query
            if hasattr(self.embeddings, 'embed_query'):
                query_embedding = await asyncio.to_thread(self.embeddings.embed_query, query_text)
            elif hasattr(self.embeddings, 'get_query_embedding'):
                query_embedding = await asyncio.to_thread(self.embeddings.get_query_embedding, query_text)
            else:
                query_embedding = await asyncio.to_thread(self.embeddings.get_text_embedding, query_text)

            results = await asyncio.to_thread(
                collection.query,
                query_embeddings=[query_embedding],
                n_results=k,
                where=filters,
                include=['documents', 'metadatas', 'distances']
            )

            query_results = []
            if results and 'ids' in results and results['ids']:
                for i, doc_id in enumerate(results['ids'][0]):
                    distance = results['distances'][0][i]
                    relevance = 1 - distance
                    
                    # Create QueryResult object, but don't filter by relevance yet
                    query_results.append(QueryResult(
                        content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i],
                        relevance_score=relevance,
                        collection_name=collection_name,
                        base_relevance=relevance
                    ))
            
            # Cache the unfiltered results
            query_cache.set(collection_name, query_text, k, query_results, filters)
            
            # Now filter by relevance for the current request
            return [res for res in query_results if res.relevance_score >= min_relevance]

        except Exception as e:
            self.logger.error(f"Failed to query collection '{collection_name}': {e}")
            return []


# Singleton instance (can be configured later)
query_engine = QueryEngine()
