"""
Enterprise RAG Adapter (Stub).

Placeholder for future enterprise RAG API integration.
"""

from typing import Dict, Any
from src.core.adapters.rag_adapter import RAGAdapter, RAGResponse, SourceReference


class EnterpriseRAGAdapter(RAGAdapter):
    """
    Enterprise RAG adapter (stub for future implementation).
    
    Will call external Enterprise RAG API when implemented.
    """
    
    def __init__(self, api_url: str, api_key: str, timeout: int = 30):
        """
        Initialize enterprise adapter.
        
        Args:
            api_url: Enterprise API endpoint
            api_key: API authentication key
            timeout: Request timeout in seconds
        """
        self.api_url = api_url
        self.api_key = api_key
        self.timeout = timeout
    
    async def query(
        self,
        question: str,
        context: Dict[str, Any]
    ) -> RAGResponse:
        """
        Query enterprise RAG API.
        
        TODO: Implement actual API call
        """
        raise NotImplementedError(
            "EnterpriseRAGAdapter not yet implemented. "
            "Use RAG_MODE=local for now."
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of enterprise API.
        
        TODO: Implement actual health check
        """
        return {
            "status": "not_implemented",
            "mode": "enterprise",
            "message": "Enterprise adapter is a stub"
        }
