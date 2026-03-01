"""
Local RAG Adapter implementation.

Wraps the existing QueryService to provide the standardized RAGAdapter interface.
"""

import time
from typing import Dict, Any
from src.core.adapters.rag_adapter import RAGAdapter, RAGResponse, SourceReference
from src.core.services.query_service import QueryService


class LocalRAGAdapter(RAGAdapter):
    """
    Local RAG adapter using ChromaDB + LlamaIndex.
    
    Wraps the existing QueryService to provide standardized interface.
    """
    
    def __init__(self, query_service: QueryService):
        """
        Initialize local adapter.
        
        Args:
            query_service: The QueryService instance to wrap
        """
        self.query_service = query_service
    
    async def query(
        self,
        question: str,
        context: Dict[str, Any]
    ) -> RAGResponse:
        """
        Query using local QueryService.
        
        Args:
            question: The user's question
            context: Parameters including:
                - collection_names: List of collections
                - n_results: Number of results (default: 5)
                - temperature: LLM temperature (default: 0.1)
                - use_reranker: Use reranking (default: False)
                - user_id: User identifier
        
        Returns:
            RAGResponse with standardized format
        """
        start_time = time.time()
        
        # Extract parameters from context
        collection_names = context.get("collection_names", ["default"])
        n_results = context.get("n_results", 5)
        temperature = context.get("temperature", 0.1)
        use_reranker = context.get("use_reranker", False)
        user_id = context.get("user_id", "unknown")
        
        # Call existing QueryService
        result = await self.query_service.answer_query(
            query_text=question,
            collection_names=collection_names,
            final_k=n_results,
            system_prompt="You are a helpful assistant. Answer the user's query based on the provided context.",
            temperature=temperature,
            use_reranker=use_reranker,
            user_id=user_id
        )
        
        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Convert context chunks to SourceReference format
        sources = []
        for chunk in result.get("context", []):
            source = SourceReference(
                content=chunk.get("content", ""),
                collection_name=chunk.get("source_collection"),
                score=chunk.get("relevance_score", 0.0),
                file=chunk.get("source") or chunk.get("metadata", {}).get("source"),
                page=chunk.get("page_number") or chunk.get("metadata", {}).get("page_number"),
                metadata=chunk.get("metadata", {})
            )
            sources.append(source)
        
        # Build RAGResponse
        answer_text = result.get("response", "")
        if answer_text is None:
            answer_text = "Keine Antwort erhalten. Möglicherweise ist das LLM-Modell nicht verfügbar."

        return RAGResponse(
            answer=answer_text,
            sources=sources,
            mode="local",
            confidence=None,  # Not available in local mode
            latency_ms=latency_ms
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of local RAG system.
        
        Returns:
            Health status dict
        """
        try:
            # Simple check: verify QueryService is accessible
            if self.query_service:
                return {
                    "status": "healthy",
                    "mode": "local",
                    "backend": "ChromaDB + LlamaIndex"
                }
            else:
                return {
                    "status": "unhealthy",
                    "mode": "local",
                    "error": "QueryService not initialized"
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "mode": "local",
                "error": str(e)
            }
