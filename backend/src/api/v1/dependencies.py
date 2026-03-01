"""
FastAPI dependencies for RAG service injection.
Community Edition - RAG only, no email/database features.
"""

import os
from fastapi import Depends
from loguru import logger

from src.core.rag_client import RAGClient
from src.core.services.query_service import QueryService
from src.core.adapters.rag_adapter import RAGAdapter
from src.core.adapters.local_adapter import LocalRAGAdapter
from src.core.adapters.enterprise_adapter import EnterpriseRAGAdapter
from src.services.config_service import config_service

# Singleton for RAGClient (prevents connection leak)
_rag_client = None
_rag_adapter = None

async def get_rag_client() -> RAGClient:
    """Get RAG Client (singleton to prevent connection leak)"""
    global _rag_client
    if _rag_client is None:
        config = config_service.load_configuration()
        _rag_client = RAGClient(config=config)
        logger.info("RAGClient singleton initialized")
    return _rag_client

async def get_query_service(
    rag_client: RAGClient = Depends(get_rag_client)
) -> QueryService:
    """Get QueryService from RAGClient"""
    return rag_client.query_service

async def get_rag_adapter(
    query_service: QueryService = Depends(get_query_service)
) -> RAGAdapter:
    """
    Get RAG Adapter based on RAG_MODE environment variable.
    
    Modes:
    - local: Use LocalRAGAdapter (ChromaDB + LlamaIndex)
    - enterprise: Use EnterpriseRAGAdapter (External API)
    
    Returns:
        RAGAdapter instance (singleton)
    """
    global _rag_adapter
    
    if _rag_adapter is None:
        rag_mode = os.getenv("RAG_MODE", "local").lower()
        
        if rag_mode == "local":
            logger.info("Initializing LocalRAGAdapter")
            _rag_adapter = LocalRAGAdapter(query_service=query_service)
        
        elif rag_mode == "enterprise":
            logger.info("Initializing EnterpriseRAGAdapter")
            api_url = os.getenv("ENTERPRISE_API_URL")
            api_key = os.getenv("ENTERPRISE_API_KEY")
            timeout = int(os.getenv("ENTERPRISE_TIMEOUT", "30"))
            
            if not api_url or not api_key:
                raise ValueError(
                    "ENTERPRISE_API_URL and ENTERPRISE_API_KEY must be set "
                    "when RAG_MODE=enterprise"
                )
            
            _rag_adapter = EnterpriseRAGAdapter(
                api_url=api_url,
                api_key=api_key,
                timeout=timeout
            )
        
        else:
            raise ValueError(
                f"Invalid RAG_MODE: {rag_mode}. Must be 'local' or 'enterprise'"
            )
    
    return _rag_adapter

