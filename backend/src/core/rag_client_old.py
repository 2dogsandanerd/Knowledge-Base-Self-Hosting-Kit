from src.core.chroma_manager import get_chroma_manager
from src.core.embedding_manager import EmbeddingManager
from src.core.collection_manager import CollectionManager
from src.core.indexing_service import IndexingService, Document, ChunkConfig
from src.core.circuit_breaker import RAGResponse
from src.storage.document_store import DocumentStore
from src.core.retrievers.parent_retriever import ParentDocumentRetriever
from src.core.retrievers.advanced_pipeline import AdvancedRAGPipelineFactory
from src.core.postprocessors.custom_reranker import CustomReranker
from typing import List, Dict, Optional, Any
from loguru import logger

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import QueryType
from llama_index.vector_stores.chroma import ChromaVectorStore

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
            host=config.get("CHROMA_HOST", "127.0.0.1"),
            port=config.get("CHROMA_PORT", 33801),
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
        
        # LlamaIndex query components (will initialize on first use)
        self._vector_indices = {}
        self._query_engines = {}

    # === Collection Operations ===

    async def create_collection(self, name: str, **kwargs) -> RAGResponse:
        """Create new collection"""
        return await self.collection_manager.create_collection(name, **kwargs)

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
        """Get embedding dimensions for a model"""
        return self.embedding_manager.get_dimensions(
            provider="ollama",  # Default provider
            model=model_name
        )

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
        use_parent_retriever: bool = False,  # New parameter for Phase 2
        use_reranker: bool = False,  # New parameter for Phase 3
        rerank_top_k: int = 10,  # Top-k for reranking
        use_advanced_pipeline: bool = False,  # New parameter for Phase 5
        **kwargs
    ) -> RAGResponse:
        """Query RAG system using LlamaIndex"""
        try:
            # Default to a single collection if not specified
            if not collection_names:
                collection_names = ["default"]
            
            # Use advanced pipeline if requested
            if use_advanced_pipeline:
                return await self._query_with_advanced_pipeline(
                    query_text=query_text,
                    collection_names=collection_names,
                    n_results=n_results,
                    **kwargs
                )
            
            # Get results from all specified collections using basic approach
            all_results = []
            for collection_name in collection_names:
                query_engine = await self.get_query_engine(
                    collection_name, 
                    use_parent_retriever=use_parent_retriever,
                    use_reranker=use_reranker,
                    rerank_top_k=rerank_top_k
                )
                
                if query_engine:
                    # Execute the query using LlamaIndex
                    try:
                        response = await query_engine.aquery(query_text)
                    except Exception as query_error:
                        # Translate LlamaIndex-specific errors to user-friendly messages
                        error_msg = self._translate_llm_error(str(query_error))
                        logger.error(f"Query to collection '{collection_name}' failed: {error_msg}")
                        
                        # Continue with other collections instead of failing the entire query
                        continue
                    
                    # Convert LlamaIndex response to our format
                    if hasattr(response, 'source_nodes'):
                        for node in response.source_nodes:
                            result = {
                                'content': node.text,
                                'document': node.text,  # Backward compatibility
                                'metadata': node.metadata,
                                'relevance_score': 1.0 - node.score if hasattr(node, 'score') and node.score else 1.0,  # Convert similarity to relevance score
                                'distance': node.score if hasattr(node, 'score') and node.score else 0.0,
                                'collection_name': collection_name,
                                'source': node.metadata.get('source', '') if isinstance(node.metadata, dict) else '',
                                'page_number': node.metadata.get('page_label', node.metadata.get('page_num', 0)) if isinstance(node.metadata, dict) else 0
                            }
                            all_results.append(result)
                    elif hasattr(response, 'response'):
                        # Handle simple response
                        result = {
                            'content': response.response,
                            'document': response.response,
                            'metadata': {},
                            'relevance_score': 1.0,
                            'distance': 0.0,
                            'collection_name': collection_name,
                            'source': '',
                            'page_number': 0
                        }
                        all_results.append(result)
            
            # Limit results to n_results
            all_results = sorted(all_results, key=lambda x: x['relevance_score'], reverse=True)[:n_results]

            return RAGResponse.ok(
                data=all_results,
                metadata={
                    'collections_queried': collection_names,
                    'use_parent_retriever': use_parent_retriever,  # Add this to metadata
                    'use_reranker': use_reranker,  # Add this to metadata
                    'rerank_top_k': rerank_top_k,  # Add this to metadata
                    'use_advanced_pipeline': use_advanced_pipeline  # Add this to metadata
                }
            )
        except Exception as e:
            logger.error(f"Query failed: {e}")
            from src.core.circuit_breaker import RAGOperationStatus
            return RAGResponse(
                status=RAGOperationStatus.FAILURE,
                error=str(e)
            )

    async def get_query_engine(self, collection_name: str, use_parent_retriever: bool = False, use_reranker: bool = False, rerank_top_k: int = 10):
        """Get or create a LlamaIndex query engine for a collection."""
        # Create a unique key for caching when additional features are used
        cache_suffix = ""
        if use_parent_retriever:
            cache_suffix += "_parent"
        if use_reranker:
            cache_suffix += "_rerank"
        
        cache_key = f"{collection_name}{cache_suffix}" if cache_suffix else collection_name
        
        if cache_key in self._query_engines:
            return self._query_engines[cache_key]
        
        try:
            # Get the ChromaDB collection
            chroma_client = self.chroma_manager.get_client()
            chroma_collection = chroma_client.get_collection(collection_name)
            
            # Create LlamaIndex vector store
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            # Create embedding instance
            embeddings = self.embedding_manager.get_embeddings()
            if not embeddings:
                logger.error(f"Could not initialize embeddings for collection: {collection_name}")
                return None
            
            # Create index from vector store
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=embeddings
            )
            
            # --- START of Hybrid Retriever logic ---
            from src.core.retrievers.bm25_retriever import BM25Retriever
            from src.core.retrievers.enhanced_hybrid_retriever import EnhancedHybridRetriever

            # 1. Create the individual retrievers
            vector_retriever = index.as_retriever()
            bm25_retriever = BM25Retriever(collection_name=collection_name)

            # 2. Create the enhanced hybrid retriever with weighted combination
            hybrid_retriever = EnhancedHybridRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.7, 0.3]  # Slightly favor vector search but include BM25
            )
            # --- END of Hybrid Retriever logic ---
            
            # --- START of Parent Document Retriever logic (Phase 2) ---
            if use_parent_retriever:
                from src.storage.document_store import DocumentStore
                from src.core.retrievers.parent_retriever import ParentDocumentRetriever
                
                # Initialize DocumentStore
                document_store = DocumentStore()
                
                # Create parent document retriever that wraps the hybrid retriever
                hybrid_retriever_for_parent = ParentDocumentRetriever(
                    child_retriever=hybrid_retriever,
                    document_store=document_store
                )
            else:
                hybrid_retriever_for_parent = hybrid_retriever
            # --- END of Parent Document Retriever logic ---
            
            # --- START of Reranker logic (Phase 3) using node_postprocessors ---
            node_postprocessors = []
            if use_reranker:
                # Create the custom reranker as a node postprocessor
                reranker = CustomReranker(top_k=rerank_top_k)
                node_postprocessors.append(reranker)
            # --- END of Reranker logic ---
            
            # Create query engine with the appropriate retriever and postprocessors
            query_engine = RetrieverQueryEngine.from_args(
                retriever=hybrid_retriever_for_parent,
                node_postprocessors=node_postprocessors,
                # llm can be added here if needed
            )
            
            # Cache the query engine
            self._query_engines[cache_key] = query_engine
            
            return query_engine
        except Exception as e:
            logger.error(f"Failed to create query engine for collection '{collection_name}': {e}")
            return None

    async def query_with_context(
        self,
        query_text: str,
        system_context: str,
        collection_names: Optional[List[str]] = None,
        n_results: int = 5
    ) -> dict:
        """
        Query with structured prompt for LLM (Issue #6).

        Returns formatted prompt with clear separation:
        - System context
        - Retrieved context
        - Query
        """
        # Get RAG context
        response = await self.query(query_text, collection_names, n_results)

        if not response.is_success:
            return {
                "prompt": self._format_prompt_no_context(query_text, system_context),
                "rag_status": response.status,
                "error": response.error
            }

        # Format structured prompt (Issue #6)
        prompt = self._format_structured_prompt(
            system_context=system_context,
            rag_context=response.data,
            query=query_text
        )

        return {
            "prompt": prompt,
            "rag_status": response.status,
            "results_count": len(response.data),
            "collections_used": response.metadata.get("collections_queried", [])
        }

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
                pipeline = AdvancedRAGPipelineFactory.create_pipeline(
                    collection_name=collection_name,
                    collection_registry=self.collection_manager,  # Using collection manager as registry
                    llm_service=self.embedding_manager.llm_service if hasattr(self.embedding_manager, 'llm_service') else None,
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
        await self.chroma_manager.disconnect()
        self.embedding_manager.clear_cache()
