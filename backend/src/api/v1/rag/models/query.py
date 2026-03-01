"""Query models for RAG API."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str = Field(..., description="The query text to search for")
    collections: List[str] = Field(default=[], description="List of collections to query")
    collection: Optional[str] = Field(default=None, description="Single collection to query (for backward compatibility)")
    k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    llm_model: Optional[str] = Field(default=None, description="The specific LLM model to use (e.g., 'llama3:latest')")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="The temperature for the LLM.")
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt for the LLM.")
    use_reranker: Optional[bool] = Field(default=False, description="Enable cross-encoder reranking for improved relevance.")
    rerank_top_k: Optional[int] = Field(default=10, ge=1, le=50, description="Number of results to rerank.")


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Query results")
    collection: str = Field(..., description="Collection that was queried")
    query: str = Field(..., description="Original query text")
    count: int = Field(..., description="Number of results returned")
