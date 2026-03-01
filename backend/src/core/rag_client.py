import asyncio
import os
from src.core.chroma_manager import get_chroma_manager
from src.core.embedding_manager import EmbeddingManager
from src.core.collection_manager import CollectionManager
from src.core.indexing_service import IndexingService, Document, ChunkConfig
from src.core.circuit_breaker import RAGResponse
from src.storage.document_store import DocumentStore
from src.core.services.query_service import QueryService
from src.core.services.embedding_service import EmbeddingService
from src.core.feature_limits import FeatureLimits, Edition
from typing import List, Dict, Optional, Any
from loguru import logger

class RAGClient:
    """
    Facade for RAG system.
    Orchestrates all RAG operations through specialized services.

    This is the main entry point for all RAG functionality.
    All complex logic has been moved to specialized services.
    """

    def __init__(self, config: dict):
        # Initialize core components
        self.chroma_manager = get_chroma_manager() # Use the getter function
        self.chroma_manager.configure(
            host=config.get("CHROMA_HOST") or os.getenv("CHROMA_HOST", "localhost"),
            port=int(config.get("CHROMA_PORT") or os.getenv("CHROMA_PORT", 8000)),
            in_memory=config.get("CHROMA_IN_MEMORY", False) # Pass in_memory flag
        )

        self.embedding_manager = EmbeddingManager(config_override=config)

        # Initialize services
        self.collection_manager = CollectionManager(
            self.chroma_manager,
            self.embedding_manager
        )

        self.indexing_service = IndexingService(
            self.chroma_manager,
            self.collection_manager,
            self.embedding_manager
        )

        # Initialize new services (Phase 4)
        self.query_service = QueryService(
            chroma_manager=self.chroma_manager,
            embedding_manager=self.embedding_manager,
            config=config
        )

        self.embedding_service = EmbeddingService(
            embedding_manager=self.embedding_manager,
            config=config
        )

        # Initialize edition-based limits
        edition_str = config.get("EDITION", "developer").lower()
        try:
            self.edition = Edition(edition_str)
        except ValueError:
            self.edition = Edition.DEVELOPER  # Default to developer edition

    # === Collection Operations ===

    async def create_collection(self, name: str, **kwargs) -> RAGResponse:
        """Create new collection"""
        # Check collection limit for current edition
        if self.edition == Edition.DEVELOPER:
            all_collections = await self.collection_manager.list_collections()
            if not all_collections.is_success:
                return RAGResponse.error("Failed to check collection limits")

            current_count = len(all_collections.data) if all_collections.data else 0
            if not FeatureLimits.check_collection_limit(current_count, self.edition):
                return RAGResponse.error(
                    "Collection limit exceeded for Developer Edition. "
                    f"Maximum {FeatureLimits.get_limit_value('max_collections', self.edition)} collection(s) allowed. "
                    "Upgrade to Team Edition for more collections."
                )

        # Get system-wide defaults from embedding_manager
        manager_config = self.embedding_manager._load_config()
        default_provider = manager_config.get("EMBEDDING_PROVIDER", "ollama")
        default_model = manager_config.get("EMBEDDING_MODEL", "nomic-embed-text:latest")

        embedding_config = {
            "provider": kwargs.get("embedding_provider", default_provider),
            "model": kwargs.get("embedding_model", default_model),
            "description": kwargs.get("description", "")
        }
        return await self.collection_manager.create_collection(name, embedding_config=embedding_config)

    async def list_collections(self) -> RAGResponse:
        """List all collections"""
        return await self.collection_manager.list_collections()

    async def delete_collection(self, name: str) -> RAGResponse:
        """Delete collection"""
        return await self.collection_manager.delete_collection(name)

    async def get_collection_metadata(self, collection_name: str) -> dict:
        """Get collection metadata"""
        return await self.collection_manager._get_collection_metadata(collection_name)

    async def get_embedding_dimensions(self, model_name: str) -> int:
        """Get embedding dimensions for a model (delegates to EmbeddingService)"""
        return await self.embedding_service.get_embedding_dimensions(model_name)

    async def get_documents(self, collection_name: str, limit: int = 100, offset: int = 0):
        """Get documents from collection with pagination"""
        response = await self.collection_manager.get_documents(collection_name, limit, offset)
        if response.is_success:
            return response.data
        return {"documents": [], "total": 0, "limit": limit, "offset": offset}

    # === Query Operations ===

    async def query(
        self,
        query_text: str,
        collection_names: Optional[List[str]] = None,
        n_results: int = 5,
        use_parent_retriever: bool = False,
        use_reranker: bool = False,
        rerank_top_k: int = 10,
        use_advanced_pipeline: bool = False,
        user_id: str = "default_user",
        **kwargs
    ) -> RAGResponse:
        """Query RAG system (delegates to QueryService)"""

        # Check if advanced features are enabled for current edition
        if use_advanced_pipeline and not FeatureLimits.is_feature_enabled("advanced_rag", self.edition):
            return RAGResponse.error(
                "Advanced RAG pipeline is not available in Developer Edition. "
                "Upgrade to Team Edition to use this feature."
            )

        if use_reranker and not FeatureLimits.is_feature_enabled("cross_encoder_reranking", self.edition):
            return RAGResponse.error(
                "Cross-encoder reranking is not available in Developer Edition. "
                "Upgrade to Team Edition to use this feature."
            )

        # For Developer Edition, only allow queries on one collection at a time
        if (collection_names and len(collection_names) > 1 and
            not FeatureLimits.is_feature_enabled("multi_collection_search", self.edition)):
            return RAGResponse.error(
                "Multi-collection search is not available in Developer Edition. "
                "Upgrade to Team Edition to search across multiple collections simultaneously."
            )

        return await self.query_service.query(
            query_text=query_text,
            collection_names=collection_names,
            n_results=n_results,
            use_parent_retriever=use_parent_retriever,
            use_reranker=use_reranker,
            rerank_top_k=rerank_top_k,
            use_advanced_pipeline=use_advanced_pipeline,
            user_id=user_id,
            **kwargs
        )

    async def get_query_engine(self, collection_name: str, use_parent_retriever: bool = False, use_reranker: bool = False, rerank_top_k: int = 10, user_id: str = "default_user"):
        """Get or create a LlamaIndex query engine (delegates to QueryService)"""
        return await self.query_service.get_query_engine(
            collection_name=collection_name,
            use_parent_retriever=use_parent_retriever,
            use_reranker=use_reranker,
            rerank_top_k=rerank_top_k,
            user_id=user_id
        )

    async def query_with_context(
        self,
        query_text: str,
        system_context: str,
        collection_names: Optional[List[str]] = None,
        n_results: int = 5
    ) -> dict:
        """Query with structured prompt for LLM (delegates to QueryService)"""
        return await self.query_service.query_with_context(
            query_text=query_text,
            system_context=system_context,
            collection_names=collection_names,
            n_results=n_results
        )

    def _format_structured_prompt(
        self,
        system_context: str,
        rag_context: List[dict],
        query: str
    ) -> str:
        """
        Format prompt with clear separation (Issue #6).

        Structure:
        1. System context (role, instructions)
        2. Retrieved context (RAG results)
        3. User query
        """
        prompt_parts = []

        # 1. System Context
        prompt_parts.append("=== SYSTEM CONTEXT ===")
        prompt_parts.append(system_context)
        prompt_parts.append("")

        # 2. Retrieved Context
        if rag_context:
            prompt_parts.append("=== RETRIEVED CONTEXT ===")
            prompt_parts.append(
                "The following information was retrieved from the knowledge base "
                "and may be relevant to answering the query:"
            )
            prompt_parts.append("")

            for i, result in enumerate(rag_context, 1):
                prompt_parts.append(f"[Source {i}] ({result['collection']})")
                prompt_parts.append(result['content'])
                prompt_parts.append("")

        # 3. User Query
        prompt_parts.append("=== USER QUERY ===")
        prompt_parts.append(query)
        prompt_parts.append("")
        prompt_parts.append("Please answer the query using the retrieved context when relevant.")

        return "\n".join(prompt_parts)

    def _translate_llm_error(self, error_msg: str) -> str:
        """Translate LlamaIndex-specific errors to user-friendly messages."""
        error_msg_lower = error_msg.lower()
        
        if "connection" in error_msg_lower or "timeout" in error_msg_lower:
            return "Verbindung zum LLM-Server konnte nicht hergestellt werden. Bitte überprüfen Sie Ihre Netzwerkverbindung und die LLM-Servereinstellungen."
        elif "rate limit" in error_msg_lower or "too many requests" in error_msg_lower:
            return "Maximale Anfragenrate erreicht. Bitte warten Sie kurz, bevor Sie eine neue Anfrage stellen."
        elif "invalid api key" in error_msg_lower or "authentication" in error_msg_lower:
            return "Ungültiger API-Schlüssel. Bitte überprüfen Sie Ihre LLM-Provider-Einstellungen."
        elif "not found" in error_msg_lower or "404" in error_msg_lower:
            return "Angeforderte Ressource nicht gefunden. Bitte überprüfen Sie Ihre Konfiguration."
        elif "dimension mismatch" in error_msg_lower or "embedding dimension" in error_msg_lower:
            return "Fehler bei der Abfrage: Embedding-Dimensionen stimmen nicht überein. Möglicherweise wurden unterschiedliche Embedding-Modelle verwendet."
        else:
            return f"Es ist ein Fehler bei der Verarbeitung der Anfrage aufgetreten: {error_msg}"

    def _validate_response_format(self, response_data: List[Dict[str, Any]], required_fields: List[str] = None) -> bool:
        """Validate that response contains all required fields for frontend compatibility."""
        if required_fields is None:
            required_fields = ['content', 'metadata', 'relevance_score', 'distance']
        
        if not isinstance(response_data, list):
            logger.error("Response data is not a list")
            return False
        
        for i, item in enumerate(response_data):
            if not isinstance(item, dict):
                logger.error(f"Response item at index {i} is not a dictionary: {type(item)}")
                return False
            
            missing_fields = []
            for field in required_fields:
                if field not in item:
                    missing_fields.append(field)
            
            if missing_fields:
                logger.error(f"Response item at index {i} is missing required fields: {missing_fields}")
                return False
        
        return True

    async def _query_with_advanced_pipeline(
        self,
        query_text: str,
        collection_names: List[str],
        n_results: int = 5,
        **kwargs
    ) -> RAGResponse:
        """Query using the advanced RAG pipeline."""
        try:
            all_results = []
            
            for collection_name in collection_names:
                # Create advanced pipeline for this collection
                from src.core.llm_singleton import get_llm, LLMServiceWrapper
                pipeline = AdvancedRAGPipelineFactory.create_pipeline(
                    collection_name=collection_name,
                    collection_registry=self.collection_manager,  # Using collection manager as registry
                    llm_service=LLMServiceWrapper(),  # Use the LLM service wrapper
                    chroma_client=self.chroma_manager.get_client(),
                    embeddings=self.embedding_manager.get_embeddings()
                )
                
                # Execute query with pipeline
                nodes = await pipeline.aquery(query_text, top_k=n_results)
                
                for node in nodes:
                    result = {
                        'content': node.node.text,
                        'document': node.node.text,  # Backward compatibility
                        'metadata': node.node.metadata,
                        'relevance_score': node.score if hasattr(node, 'score') else 1.0,
                        'distance': 1.0 - node.score if hasattr(node, 'score') else 0.0,
                        'collection_name': collection_name,
                        'source': node.node.metadata.get('source', '') if isinstance(node.node.metadata, dict) else '',
                        'page_number': node.node.metadata.get('page_label', node.node.metadata.get('page_num', 0)) if isinstance(node.node.metadata, dict) else 0
                    }
                    all_results.append(result)
            
            # Limit and sort results
            all_results = sorted(all_results, key=lambda x: x['relevance_score'], reverse=True)[:n_results]
            
            return RAGResponse.ok(
                data=all_results,
                metadata={
                    'collections_queried': collection_names,
                    'use_advanced_pipeline': True
                }
            )
        except Exception as e:
            logger.error(f"Advanced pipeline query failed: {e}")
            from src.core.circuit_breaker import RAGOperationStatus
            return RAGResponse(
                status=RAGOperationStatus.FAILURE,
                error=str(e)
            )

    def _format_prompt_no_context(self, query: str, system_context: str) -> str:
        """Format prompt when RAG context unavailable"""
        return f"""{system_context}

Note: Knowledge base context is currently unavailable.

User Query: {query}

Please answer based on your general knowledge."""

    # === Indexing Operations ===

    async def index_documents(
        self,
        documents: List[Document],
        collection_name: str,
        use_parent_child: bool = False,
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 200,
        child_chunk_overlap: int = 20,
        **kwargs
    ) -> RAGResponse:
        """Index documents with chunking"""

        # Check if advanced features are enabled for current edition
        if use_parent_child and not FeatureLimits.is_feature_enabled("advanced_rag", self.edition):
            return RAGResponse.error(
                "Parent-child document strategy is not available in Developer Edition. "
                "Upgrade to Team Edition to use this feature."
            )

        # Check document count limit for current collection
        if self.edition == Edition.DEVELOPER:
            # Get current document count in collection
            try:
                collection = await asyncio.to_thread(self.chroma_manager.get_collection, collection_name)
                if collection:
                    current_count = await asyncio.to_thread(collection.count)
                    new_count = current_count + len(documents)

                    if not FeatureLimits.check_document_limit(new_count, self.edition):
                        return RAGResponse.error(
                            "Document limit exceeded for Developer Edition. "
                            f"Maximum {FeatureLimits.get_limit_value('max_documents_per_collection', self.edition)} documents per collection allowed. "
                            "Upgrade to Team Edition for more documents."
                        )
            except Exception as e:
                logger.warning(f"Could not check document count: {e}")

        return await self.indexing_service.index_documents(
            documents,
            collection_name,
            use_parent_child=use_parent_child,
            parent_chunk_size=parent_chunk_size,
            child_chunk_size=child_chunk_size,
            child_chunk_overlap=child_chunk_overlap,
            **kwargs
        )

    async def delete_document(self, doc_id: str, collection_name: str) -> RAGResponse:
        """Delete document"""
        return await self.indexing_service.delete_document(doc_id, collection_name)

    # === Health & Status ===

    async def health_check(self) -> dict:
        """Get health status of all components"""
        return {
            "chroma": self.chroma_manager.get_health_status(),
            "embeddings": {
                "cache_size": len(self.embedding_manager._embeddings_cache)
            }
        }

    async def shutdown(self):
        """Graceful shutdown"""
        await self.chroma_manager.close_async()
        self.embedding_manager.clear_cache()
