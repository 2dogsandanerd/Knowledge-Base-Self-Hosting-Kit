"""
RAG Adapter Interface and Response Models.

This module defines the abstract adapter interface for RAG backends
and standardized response models for consistent API output.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    """Standardized source reference for RAG responses."""
    
    content: str = Field(..., description="The content/text of the source chunk")
    collection_name: Optional[str] = Field(None, description="Collection this source came from")
    score: float = Field(..., description="Relevance score (0-1, higher is better)")
    file: Optional[str] = Field(None, description="Source file name")
    page: Optional[int] = Field(None, description="Page number in source document")
    chunk_id: Optional[str] = Field(None, description="Unique chunk identifier (Enterprise only)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RAGResponse(BaseModel):
    """Standardized RAG query response."""
    
    answer: str = Field(..., description="The synthesized answer from the LLM")
    sources: List[SourceReference] = Field(default_factory=list, description="Source chunks used")
    mode: str = Field(..., description="RAG mode used: 'local' or 'enterprise'")
    confidence: Optional[float] = Field(None, description="Confidence score (0-1, if available)")
    latency_ms: Optional[int] = Field(None, description="Query latency in milliseconds")


class RAGAdapter(ABC):
    """
    Abstract base class for RAG adapters.
    
    Implementations:
    - LocalRAGAdapter: Uses local ChromaDB + LlamaIndex
    - EnterpriseRAGAdapter: Calls external Enterprise RAG API
    """
    
    @abstractmethod
    async def query(
        self,
        question: str,
        context: Dict[str, Any]
    ) -> RAGResponse:
        """
        Query the RAG system.
        
        Args:
            question: The user's question
            context: Additional context parameters:
                - collection_names: List of collections to search
                - n_results: Number of results to return
                - temperature: LLM temperature
                - use_reranker: Whether to use reranking
                - user_id: User identifier (for tracking)
        
        Returns:
            RAGResponse with answer and sources
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of the RAG backend.
        
        Returns:
            Dict with status and details
        """
        pass
