"""
Advanced RAG Pipeline Implementation for Phase 5.

This module orchestrates all advanced RAG components (query enhancement,
multi-source retrieval, fusion, reranking, parent document retrieval)
into a cohesive, configurable pipeline.
"""

from typing import List, Dict, Optional, Any
from loguru import logger
from llama_index.core.schema import NodeWithScore, QueryBundle
# from src.core.retrievers.hybrid_retriever import HybridRetriever  # Not used anymore
from src.core.retrievers.parent_retriever import ParentDocumentRetriever
from src.core.postprocessors.custom_reranker import CustomReranker
from src.core.retrievers.query_enhancement import QueryEnhancementRetriever, QueryEnhancer
from src.storage.document_store import DocumentStore


class AdvancedRAGPipeline:
    """
    A complete advanced RAG pipeline that orchestrates all components.
    
    The pipeline executes in this order (configurable):
    1. Query Enhancement (optional)
    2. Multi-Source Retrieval (Hybrid: Vector + BM25)
    3. Fusion (RRF)
    4. Reranking (optional)
    5. Parent Document Retrieval (optional)
    6. Contextual Compression (optional)
    """
    
    def __init__(
        self,
        collection_config: Dict[str, Any],
        llm_service=None,
        chroma_client=None,
        embeddings=None,
        document_store: Optional[DocumentStore] = None
    ):
        """
        Initialize the AdvancedRAGPipeline.

        Args:
            collection_config: Configuration for the collection
            llm_service: LLM service for query enhancement
            chroma_client: ChromaDB client
            embeddings: Embedding service
            document_store: Document store for parent documents
        """
        self.config = collection_config
        self.llm_service = llm_service
        self.chroma_client = chroma_client
        self.embeddings = embeddings
        self.document_store = document_store or DocumentStore()
        
        self.logger = logger.bind(component="AdvancedRAGPipeline")
        
        # Initialize pipeline components based on configuration
        self._init_components()

    def _init_components(self):
        """Initialize components based on configuration."""
        # Get collection-specific settings
        collection_name = self.config.get('collection_name', 'default')
        
        # Initialize basic retrievers
        self._init_basic_retrievers()
        
        # Initialize optional components based on config
        self.query_enhancer = None
        self.reranker_postprocessor = None  # Use as node postprocessor
        self.parent_retriever = None
        
        # Initialize query enhancer if enabled
        if self.config.get('query_enhancement', {}).get('enabled', False):
            if self.llm_service:
                from llama_index.core.llms import LLM
                llm_instance = self.llm_service.get_llm_instance()
                self.query_enhancer = QueryEnhancer(llm=llm_instance)
            else:
                self.logger.warning(f"Query enhancement enabled for {collection_name} but no LLM service provided")
        
        # Initialize reranker if enabled
        if self.config.get('reranking', {}).get('enabled', False):
            reranker_model = self.config.get('reranking', {}).get('model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
            top_k = self.config.get('reranking', {}).get('top_k', 10)
            self.reranker_postprocessor = CustomReranker(model_name=reranker_model, top_k=top_k)
        
        # Initialize parent document retriever if enabled
        if self.config.get('parent_document', {}).get('enabled', False):
            self.parent_retriever = ParentDocumentRetriever(
                child_retriever=self.hybrid_retriever,
                document_store=self.document_store
            )

    def _init_basic_retrievers(self):
        """Initialize the basic retrievers (hybrid vector + BM25)."""
        from src.core.retrievers.bm25_retriever import BM25Retriever
        from llama_index.core import VectorStoreIndex
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.core.query_engine import RetrieverQueryEngine
        
        collection_name = self.config.get('collection_name', 'default')
        
        # Create vector store and index for the collection
        actual_client = self.chroma_client.get_client() if hasattr(self.chroma_client, 'get_client') else self.chroma_client
        chroma_collection = actual_client.get_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=self.embeddings
        )
        
        # Create base retrievers
        vector_retriever = index.as_retriever()
        bm25_retriever = BM25Retriever(collection_name=collection_name)
        
        # Create enhanced hybrid retriever with weighted combination
        from src.core.retrievers.enhanced_hybrid_retriever import EnhancedHybridRetriever
        self.hybrid_retriever = EnhancedHybridRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3]  # Slightly favor vector search but include BM25
        )
        
        self.logger.info(f"Initialized basic retrievers for collection: {collection_name}")

    def query(self, query_text: str, top_k: int = 10) -> List[NodeWithScore]:
        """
        Execute the full advanced RAG pipeline.

        Args:
            query_text: The query text
            top_k: Number of results to return

        Returns:
            List of retrieved nodes with scores
        """
        self.logger.info(f"Starting advanced RAG pipeline for query: '{query_text[:50]}...'")
        
        # Create query bundle
        query_bundle = QueryBundle(query_str=query_text)
        
        try:
            # Stage 1: Query Enhancement (optional)
            enhanced_queries = [query_text]
            if self.query_enhancer:
                enhancement_config = self.config.get('query_enhancement', {})
                method = enhancement_config.get('method', 'expansion')
                
                if method == 'expansion':
                    num_expansions = enhancement_config.get('num_expansions', 3)
                    enhanced_queries = self.query_enhancer.expand_query(query_text, num_expansions)
                    self.logger.info(f"Query expanded to {len(enhanced_queries)} variants using expansion")
                elif method == 'hyde':
                    # For HyDE method, we use the hypothetical document
                    hyde_doc = self.query_enhancer.generate_hyde_document(query_text)
                    enhanced_queries = [hyde_doc] if hyde_doc else [query_text]
                    self.logger.info("Used HyDE method for query enhancement")
            
            # Stage 2: Multi-Source Retrieval with Fusion
            all_nodes = []
            for query_variant in enhanced_queries:
                if query_variant.strip():
                    variant_bundle = QueryBundle(query_str=query_variant)
                    # Perform retrieval using the hybrid retriever
                    query_engine_retriever = self.hybrid_retriever
                    variant_nodes = query_engine_retriever.retrieve(variant_bundle)
                    all_nodes.extend(variant_nodes)
            
            # Log retrieval stage
            self.logger.info(f"Retrieved {len(all_nodes)} nodes from multi-source retrieval")
            
            # Stage 3: Reranking (optional) using node postprocessor
            reranked_nodes = all_nodes
            if self.reranker_postprocessor:
                from llama_index.core.schema import QueryBundle
                
                # Create query bundle for the postprocessor
                query_bundle = QueryBundle(query_str=query_text)
                
                rerank_config = self.config.get('reranking', {})
                initial_retrieval_k = rerank_config.get('initial_retrieval_k', 100)
                
                # Take top_k_rerank from all_nodes for reranking
                nodes_to_rerank = all_nodes[:min(initial_retrieval_k, len(all_nodes))]
                
                # Use the postprocessor to rerank
                reranked_nodes = self.reranker_postprocessor._postprocess_nodes(
                    nodes=nodes_to_rerank,
                    query_bundle=query_bundle
                )
                self.logger.info(f"Reranked {len(reranked_nodes)} nodes using node postprocessor")
            
            # Stage 4: Parent Document Retrieval (optional)
            final_nodes = reranked_nodes
            if self.parent_retriever:
                # Use parent retriever if it's configured for this pipeline
                # For this implementation, we'll wrap the reranked results if parent retrieval is enabled
                # but in a real implementation, this would be integrated differently
                from llama_index.core.base.base_retriever import BaseRetriever
                from llama_index.core.query_engine import RetrieverQueryEngine
                
                # Create a temporary retriever using the reranked nodes to feed to parent retriever
                class NodeRetriever(BaseRetriever):
                    def __init__(self, nodes):
                        super().__init__()
                        self.nodes = nodes
                    
                    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
                        return self.nodes
                
                temp_retriever = NodeRetriever(reranked_nodes)
                self.parent_retriever.child_retriever = temp_retriever
                final_nodes = self.parent_retriever.retrieve(query_bundle)
                self.logger.info(f"Retrieved {len(final_nodes)} parent documents")
            
            # Stage 5: Contextual Compression (optional)
            compressed_nodes = final_nodes
            # In a full implementation, compression would go here
            # For now, we'll use the final nodes as-is
            
            # Limit to top_k results
            result_nodes = compressed_nodes[:top_k]
            self.logger.info(f"Advanced RAG pipeline completed, returning {len(result_nodes)} nodes")
            
            return result_nodes
            
        except Exception as e:
            self.logger.error(f"Advanced RAG pipeline failed: {e}")
            # Return empty list if pipeline fails completely
            return []

    async def aquery(self, query_text: str, top_k: int = 10) -> List[NodeWithScore]:
        """
        Asynchronous version of query method.
        """
        # For now, call the synchronous version
        # A full implementation would async-ify the individual stages
        return self.query(query_text, top_k)


class AdvancedRAGPipelineFactory:
    """
    Factory for creating AdvancedRAGPipeline instances based on collection configuration.
    
    This allows the QueryEngine to dispatch to the appropriate pipeline based
    on the collection's configuration.
    """
    
    @staticmethod
    def create_pipeline(
        collection_name: str,
        collection_registry,
        llm_service=None,
        chroma_client=None,
        embeddings=None
    ):
        """
        Create an AdvancedRAGPipeline for a specific collection.

        Args:
            collection_name: Name of the collection
            collection_registry: Collection registry service
            llm_service: LLM service
            chroma_client: ChromaDB client
            embeddings: Embedding service

        Returns:
            AdvancedRAGPipeline instance
        """
        # Get collection configuration from registry
        config = collection_registry.get_config(collection_name)
        if not config:
            # Use default configuration if not found
            config = {
                'collection_name': collection_name,
                'index_strategy': 'vector',  # Default
                'query_enhancement': {'enabled': False},
                'reranking': {'enabled': False},
                'parent_document': {'enabled': False}
            }
        else:
            # Convert to dictionary if it's a model object
            config = config.to_dict() if hasattr(config, 'to_dict') else {
                'collection_name': collection_name,
                'index_strategy': getattr(config, 'index_strategy', 'vector'),
                'query_enhancement': getattr(config, 'query_enhancement', {'enabled': False}),
                'reranking': getattr(config, 'reranking', {'enabled': False}),
                'parent_document': getattr(config, 'parent_document', {'enabled': False})
            }
        
        # Create pipeline instance
        pipeline = AdvancedRAGPipeline(
            collection_config=config,
            llm_service=llm_service,
            chroma_client=chroma_client,
            embeddings=embeddings
        )
        
        return pipeline