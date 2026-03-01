"""
Query Service - Handles all RAG query operations.

This service manages query execution against ChromaDB collections:
- Simple vector queries
- Hybrid queries (vector + BM25)
- Context-aware queries with LLM
- Advanced retrieval pipelines
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

# LlamaIndex imports
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore

# Local imports
from src.core.llm_singleton import get_llm
from src.core.models.rag_types import RankedNode
from src.core.experiments.experiment_service import experiment_service

# Observability imports
from opentelemetry import trace
from src.core.observability.metrics import (
    query_total,
    query_latency,
    query_context_chunks,
    reranker_enabled,
    llm_tokens_total,
    llm_cost_usd,
    llm_latency
)

# Get tracer for this module
tracer = trace.get_tracer(__name__)


class QueryService:
    """
    Service for RAG query operations.

    Responsibilities:
    - Execute queries against ChromaDB
    - Manage query engines (LlamaIndex)
    - Format prompts for LLM
    - Advanced retrieval pipelines
    """

    def __init__(self, chroma_manager, embedding_manager, config: dict):
        """
        Initialize query service.

        Args:
            chroma_manager: ChromaManager instance
            embedding_manager: EmbeddingManager instance
            config: Configuration dict
        """
        self.chroma_manager = chroma_manager
        self.embedding_manager = embedding_manager
        self.config = config
        self.logger = logger.bind(component="QueryService")

        # Cache for query engines
        self._query_engines = {}
        
        # Load RAG configuration (reranker, parent retriever, etc.)
        try:
            from src.core.config import get_rag_config
            self.rag_config = get_rag_config()
            self.logger.info(f"RAG Config loaded: reranker={self.rag_config.reranker_enabled}, parent={self.rag_config.use_parent_retriever}, advanced={self.rag_config.use_advanced_pipeline}")
        except Exception as e:
            self.logger.warning(f"Could not load RAG config: {e}")
            # Fallback to defaults
            from src.core.config import RAGConfig
            self.rag_config = RAGConfig()
        
        # Initialize query classifier if enabled
        self.query_classifier = None
        if config.get("QUERY_CLASSIFIER_ENABLED", "false").lower() == "true":
            try:
                from pathlib import Path
                import json
                from src.core.services.query_classifier import QueryClassifier
                
                domains_path = Path(__file__).parent.parent.parent / "core" / "rag_domains.json"
                with open(domains_path) as f:
                    domains_data = json.load(f)
                    self.query_classifier = QueryClassifier(domains_config=domains_data.get("domains", {}))
                    self.logger.info("Query classifier enabled")
            except Exception as e:
                self.logger.warning(f"Could not load query classifier: {e}")

    async def query(
        self,
        query_text: str,
        collection_names: Optional[List[str]] = None,
        n_results: int = 5,
        use_parent_retriever: Optional[bool] = None,
        use_reranker: Optional[bool] = None,
        rerank_top_k: Optional[int] = None,
        use_advanced_pipeline: Optional[bool] = None,
        user_id: str = "default_user",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a query against collections using LlamaIndex.

        Args:
            query_text: Search query
            collection_names: List of collections to query
            n_results: Number of results to return
            use_parent_retriever: Use parent document retriever (defaults to config)
            use_reranker: Enable reranking (defaults to config)
            rerank_top_k: Number of results to rerank (defaults to config)
            use_advanced_pipeline: Use advanced retrieval pipeline (defaults to config)

        Returns:
            Dict with query results and metadata
        """
        from src.core.circuit_breaker import RAGResponse, RAGOperationStatus
        
        # Apply config defaults if not explicitly provided
        if use_parent_retriever is None:
            use_parent_retriever = self.rag_config.use_parent_retriever
        if use_reranker is None:
            use_reranker = self.rag_config.reranker_enabled
        if rerank_top_k is None:
            rerank_top_k = self.rag_config.reranker_top_k
        if use_advanced_pipeline is None:
            use_advanced_pipeline = self.rag_config.use_advanced_pipeline

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
                    rerank_top_k=rerank_top_k,
                    user_id=user_id
                )

                if query_engine:
                    # Execute the query using LlamaIndex
                    try:
                        response = await query_engine.aquery(query_text)
                    except Exception as query_error:
                        # Translate LlamaIndex-specific errors to user-friendly messages
                        error_msg = self._translate_llm_error(str(query_error))
                        self.logger.error(f"Query to collection '{collection_name}' failed: {error_msg}")

                        # Continue with other collections instead of failing the entire query
                        continue

                    # Convert LlamaIndex response to our format
                    if hasattr(response, 'source_nodes'):
                        for node in response.source_nodes:
                            result = {
                                'content': node.text,
                                'document': node.text,  # Backward compatibility
                                'metadata': node.metadata,
                                'relevance_score': 1.0 - node.score if hasattr(node, 'score') and node.score else 1.0,
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
                    'use_parent_retriever': use_parent_retriever,
                    'use_reranker': use_reranker,
                    'rerank_top_k': rerank_top_k,
                    'use_advanced_pipeline': use_advanced_pipeline
                }
            )
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            return RAGResponse(
                status=RAGOperationStatus.FAILURE,
                error=str(e)
            )

    async def get_query_engine(
        self,
        collection_name: str,
        use_parent_retriever: bool = False,
        use_reranker: bool = False,
        rerank_top_k: int = 10,
        user_id: str = "default_user"
    ):
        """
        Get or create a LlamaIndex query engine for a collection.

        Args:
            collection_name: Target collection
            use_parent_retriever: Use parent document retriever
            use_reranker: Enable reranking
            rerank_top_k: Number of results to rerank

        Returns:
            LlamaIndex QueryEngine instance or None
        """
        # Create a unique key for caching
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
                self.logger.error(f"Could not initialize embeddings for collection: {collection_name}")
                return None

            # Create index from vector store
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=embeddings
            )

            # Create hybrid retriever (vector + BM25)
            from src.core.retrievers.bm25_retriever import BM25Retriever
            from src.core.retrievers.enhanced_hybrid_retriever import EnhancedHybridRetriever

            vector_retriever = index.as_retriever()
            bm25_retriever = BM25Retriever(collection_name=collection_name)
            hybrid_retriever = EnhancedHybridRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.7, 0.3]  # Slightly favor vector search but include BM25
            )

            # Optionally wrap with parent document retriever
            if use_parent_retriever:
                from src.storage.document_store import DocumentStore
                from src.core.retrievers.parent_retriever import ParentDocumentRetriever

                document_store = DocumentStore()
                hybrid_retriever = ParentDocumentRetriever(
                    child_retriever=hybrid_retriever,
                    document_store=document_store
                )

            # Optionally add reranker as postprocessor
            # Check if we should use reranker based on A/B test variant
            use_reranker_experiment = experiment_service.get_variant(user_id, "use_reranker_experiment")
            final_use_reranker = use_reranker  # Default to the passed parameter
            if use_reranker_experiment == "B":
                final_use_reranker = True  # Variant B always uses reranker
            elif use_reranker_experiment == "A":
                final_use_reranker = False  # Variant A never uses reranker
            # If experiment returns "control" or any other value, use the original parameter

            node_postprocessors = []
            if final_use_reranker:
                from src.core.postprocessors.custom_reranker import CustomReranker
                reranker = CustomReranker(
                    model_name=self.rag_config.reranker_model,
                    top_k=rerank_top_k
                )
                node_postprocessors.append(reranker)

            # Create query engine with the configured LLM
            llm_instance = get_llm()
            
            query_engine = RetrieverQueryEngine.from_args(
                retriever=hybrid_retriever,
                llm=llm_instance,
                node_postprocessors=node_postprocessors
            )

            # Cache the query engine
            self._query_engines[cache_key] = query_engine

            return query_engine
        except Exception as e:
            self.logger.error(f"Failed to create query engine for collection '{collection_name}': {e}")
            return None

    async def query_with_context(
        self,
        query_text: str,
        system_context: str,
        collection_names: Optional[List[str]] = None,
        n_results: int = 5
    ) -> dict:
        """
        Query with structured prompt for LLM and get a response.

        Args:
            query_text: User query
            system_context: System context for the LLM
            collection_names: Collections to query
            n_results: Number of results to retrieve

        Returns:
            Dict with the generated response and context.
        """
        from src.core.circuit_breaker import RAGResponse
        from src.core.llm_singleton import get_llm

        self.logger.info(f"Query with context: '{query_text}'")

        try:
            # Step 1: Retrieve context from collections
            retrieval_result = await self.query(
                query_text=query_text,
                collection_names=collection_names,
                n_results=n_results
            )

            if not retrieval_result.is_success:
                return {
                    "success": False,
                    "error": retrieval_result.error,
                    "response": "Failed to retrieve context.",
                    "context_chunks": []
                }

            chunks = retrieval_result.data

            # Step 2: Format structured prompt
            prompt = self._format_structured_prompt(query_text, chunks, system_context)

            # Step 3: Get LLM instance and generate response
            llm = get_llm()
            llm_response = await llm.acomplete(prompt)
            
            # Step 4: Return the actual response
            return {
                "success": True,
                "response": str(llm_response),
                "context_chunks": chunks,
                "query": query_text
            }

        except Exception as e:
            self.logger.error(f"Query with context failed: {e}")
            error_message = self._translate_llm_error(str(e))
            return {
                "success": False,
                "error": error_message,
                "response": None,
                "context_chunks": []
            }

    async def retrieve_and_rerank(
        self,
        query_text: str,
        collection_names: List[str],
        initial_k_per_collection: int = 10,
        final_k: int = 10,
        use_reranker: bool = False, # For future-proofing
        user_id: str = "default_user"
    ) -> Tuple[List[RankedNode], int]:
        """
        Performs multi-collection retrieval using Hybrid Search (Vector + BM25).
        Manually implemented to match the robust logic of /search endpoint.

        Args:
            query_text: The user's search query.
            collection_names: A list of collection names to search within.
            initial_k_per_collection: Number of results to retrieve from each collection.
            final_k: The total number of top-ranked results to return after global sorting.
            use_reranker: Flag to indicate if a more advanced reranker should be used.

        Returns:
            A tuple: (list of globally ranked document chunks, total number of candidates found).
        """
        self.logger.info(f"Retrieving and re-ranking for query: '{query_text}' across {collection_names}")
        all_results: List[RankedNode] = []
        total_candidates = 0

        # Import BM25 tools locally to avoid circular deps if any
        try:
            from src.core.bm25_index import BM25IndexManager, _tokenize_text
        except ImportError:
            self.logger.error("Could not import BM25IndexManager. BM25 search will be skipped.")
            BM25IndexManager = None

        # 1. Generate Query Embedding
        query_embedding = None
        try:
            embedding_instance = self.embedding_manager.get_embeddings()
            if hasattr(embedding_instance, 'aget_query_embedding'):
                query_embedding = await embedding_instance.aget_query_embedding(query_text)
            elif hasattr(embedding_instance, 'get_query_embedding'):
                query_embedding = await asyncio.to_thread(embedding_instance.get_query_embedding, query_text)
            elif hasattr(embedding_instance, 'aget_text_embedding'):
                query_embedding = await embedding_instance.aget_text_embedding(query_text)
            elif hasattr(embedding_instance, 'get_text_embedding'):
                query_embedding = await asyncio.to_thread(embedding_instance.get_text_embedding, query_text)
        except Exception as e:
            self.logger.error(f"Failed to generate query embedding: {e}")
            # If embedding fails, we might still proceed with BM25 only?
        
        # 2. Iterate Collections
        for collection_name in collection_names:
            try:
                collection_results = {} # Map doc_id -> dict result

                # A. Vector Search
                if query_embedding:
                    try:
                        chroma_client = self.chroma_manager.get_client()
                        collection = chroma_client.get_collection(collection_name) # Sync call usually
                        
                        # Chroma query
                        vector_results = await asyncio.to_thread(
                            collection.query,
                            query_embeddings=[query_embedding],
                            n_results=initial_k_per_collection * 2 # Get more for hybrid ranking
                        )

                        if vector_results and 'documents' in vector_results and vector_results['documents']:
                            docs = vector_results['documents'][0] if vector_results['documents'] else []
                            metadatas = vector_results['metadatas'][0] if 'metadatas' in vector_results and vector_results['metadatas'] else []
                            distances = vector_results['distances'][0] if 'distances' in vector_results and vector_results['distances'] else []
                            ids = vector_results['ids'][0] if 'ids' in vector_results and vector_results['ids'] else []

                            for i, doc_id in enumerate(ids):
                                metadata = metadatas[i] if i < len(metadatas) else {}
                                distance = distances[i] if i < len(distances) else 1.0
                                vector_score = 1.0 - distance
                                
                                collection_results[doc_id] = {
                                    "content": docs[i] if i < len(docs) else "",
                                    "source_collection": collection_name,
                                    "vector_score": vector_score,
                                    "bm25_score": 0.0,
                                    "metadata": metadata,
                                    "source": metadata.get('source', 'Unknown'),
                                    "page_number": metadata.get('page_number', 0)
                                }
                    except Exception as ve:
                        self.logger.warning(f"Vector search failed for {collection_name}: {ve}")

                # B. BM25 Search
                if BM25IndexManager:
                    try:
                        bm25_manager = BM25IndexManager(collection_name)
                        bm25_manager.load()
                        
                        if bm25_manager.bm25_index and bm25_manager.nodes:
                            query_tokens = _tokenize_text(query_text)
                            bm25_scores = bm25_manager.bm25_index.get_scores(query_tokens)
                            
                            # Get top BM25 results
                            top_bm25_indices = sorted(
                                range(len(bm25_scores)), 
                                key=lambda i: bm25_scores[i], 
                                reverse=True
                            )[:initial_k_per_collection * 2]
                            
                            for idx in top_bm25_indices:
                                score = float(bm25_scores[idx])
                                if score > 0:
                                    node = bm25_manager.nodes[idx]
                                    doc_id = node.id_
                                    
                                    if doc_id in collection_results:
                                        collection_results[doc_id]['bm25_score'] = score
                                    else:
                                        collection_results[doc_id] = {
                                            "content": node.text,
                                            "source_collection": collection_name,
                                            "vector_score": 0.0,
                                            "bm25_score": score,
                                            "metadata": node.metadata or {},
                                            "source": node.metadata.get('source', 'Unknown'),
                                            "page_number": node.metadata.get('page_number', 0)
                                        }
                    except Exception as be:
                        self.logger.warning(f"BM25 search failed for {collection_name}: {be}")

                # C. Combine Scores (Normalization & Weighted Sum)
                if collection_results:
                    # Get min/max for normalization
                    v_scores = [r['vector_score'] for r in collection_results.values()]
                    b_scores = [r['bm25_score'] for r in collection_results.values()]
                    
                    max_v = max(v_scores) if v_scores else 1.0
                    min_v = min(v_scores) if v_scores else 0.0
                    max_b = max(b_scores) if b_scores else 1.0
                    min_b = min(b_scores) if b_scores else 0.0
                    
                    for doc_id, res in collection_results.items():
                        # Normalize Vector
                        if max_v > min_v:
                            norm_v = (res['vector_score'] - min_v) / (max_v - min_v)
                        else:
                            norm_v = 1.0 if res['vector_score'] > 0 else 0.0
                            
                        # Normalize BM25
                        if max_b > min_b:
                            norm_b = (res['bm25_score'] - min_b) / (max_b - min_b)
                        else:
                            norm_b = 1.0 if res['bm25_score'] > 0 else 0.0
                            
                        # Hybrid Score (50/50)
                        hybrid_score = (0.5 * norm_v) + (0.5 * norm_b)
                        
                        ranked_node = RankedNode(
                            content=res['content'],
                            source_collection=collection_name,
                            relevance_score=hybrid_score,
                            metadata=res['metadata'],
                            distance=1.0 - hybrid_score,
                            source=res['source'],
                            page_number=res['page_number']
                        )
                        all_results.append(ranked_node)

            except Exception as col_error:
                self.logger.warning(f"Failed to query collection '{collection_name}': {col_error}")
                continue

        total_candidates = len(all_results)

        # Global sorting by relevance_score
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Apply advanced reranker if enabled (Phase 5)
        if use_reranker:
            self.logger.info("Applying cross-encoder reranker...")
            try:
                from src.core.retrievers.reranker import Reranker
                from llama_index.core.schema import NodeWithScore, TextNode
                
                # Convert RankedNodes to NodeWithScore for reranker
                nodes_with_score = []
                for ranked_node in all_results:
                    text_node = TextNode(
                        text=ranked_node.content,
                        metadata=ranked_node.metadata
                    )
                    node_with_score = NodeWithScore(
                        node=text_node,
                        score=ranked_node.relevance_score
                    )
                    nodes_with_score.append(node_with_score)
                
                # Apply reranker
                reranker = Reranker()
                reranked_nodes = reranker.rerank(
                    query=query_text,
                    nodes=nodes_with_score,
                    top_k=final_k
                )
                
                # Convert back to RankedNode
                all_results = []
                for node_with_score in reranked_nodes:
                    ranked_node = RankedNode(
                        content=node_with_score.node.text,
                        source_collection=node_with_score.node.metadata.get('source_collection', 'unknown'),
                        relevance_score=node_with_score.score,
                        metadata=node_with_score.node.metadata or {},
                        distance=1.0 - node_with_score.score,  # Approximate distance
                        source=node_with_score.node.metadata.get('source', ''),
                        page_number=node_with_score.node.metadata.get('page_number', 0)
                    )
                    all_results.append(ranked_node)
                
                self.logger.info(f"Reranker applied successfully, final results: {len(all_results)}")
            except Exception as rerank_error:
                self.logger.error(f"Reranker failed: {rerank_error}, falling back to score sort")
                # Fallback to original sorted results if reranker fails
                pass

        top_results = all_results[:final_k]
        self.logger.info(f"Retrieved and re-ranked {len(top_results)} final results from {total_candidates} candidates.")
        return top_results, total_candidates

    @tracer.start_as_current_span("answer_query")
    async def answer_query(
        self,
        query_text: str,
        collection_names: List[str],
        final_k: int = 10,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        llm_model: Optional[str] = None,
        initial_k_per_collection: int = 10,
        use_reranker: bool = False,
        user_id: str = "default_user"
    ) -> Dict[str, Any]:
        """
        Generates an LLM answer based on RAG, using advanced retrieval and flexible parameters.

        Args:
            query_text: The user's search query.
            collection_names: A list of collection names to search within.
            final_k: The total number of top-ranked results to use as context for the LLM.
            system_prompt: The system prompt to guide the LLM's behavior.
            temperature: The LLM's temperature setting.
            llm_model: The specific LLM model to use.
            initial_k_per_collection: Number of results to retrieve from each collection in the first pass.

        Returns:
            A dictionary containing the LLM's response, context, and metadata.
        """
        from src.core.llm_singleton import get_llm

        self.logger.info(f"Answering query: '{query_text}' with collections: {collection_names}")
        
        # Start timing for metrics
        start_time = time.time()
        
        # Get current span for attributes
        span = trace.get_current_span()
        span.set_attribute("query.text", query_text[:100])  # Truncate for privacy
        span.set_attribute("query.collections_count", len(collection_names))
        span.set_attribute("query.final_k", final_k)
        span.set_attribute("query.use_reranker", use_reranker)
        
        detected_domain = None
        original_collection_count = len(collection_names)
        
        # Track reranker usage
        reranker_enabled.labels(enabled=str(use_reranker)).inc()

        try:
            # Optional: Auto-filter collections based on query classification
            if self.query_classifier and len(collection_names) > 3:
                detected_domain = await self.query_classifier.classify_query(query_text)
                if detected_domain:
                    span.set_attribute("query.detected_domain", detected_domain)
                    domain_collections = self.query_classifier.get_collections_for_domain(detected_domain)
                    # Filter to collections that actually exist
                    filtered = [c for c in domain_collections if c in collection_names]
                    if filtered:
                        self.logger.info(f"Filtered collections from {len(collection_names)} to {len(filtered)} based on domain: {detected_domain}")
                        collection_names = filtered
            
            # Step 1: Query Understanding (Phase H.2)
            expanded_queries = [query_text]
            if self.config.get("ENABLE_QUERY_EXPANSION", "false").lower() == "true":
                try:
                    from src.core.rag.query_understanding import QueryUnderstanding
                    qu = QueryUnderstanding()
                    expanded_queries = await qu.expand_query(query_text)
                except Exception as e:
                    self.logger.warning(f"Query expansion failed: {e}")

            # Step 2: Retrieve and re-rank context (using expanded queries)
            # For now, we just use the original query for retrieval to keep it simple, 
            # or we could search for all and dedup. Let's stick to original for H.1 baseline,
            # but if expansion is enabled, we could loop.
            # IMPROVEMENT: Search for best variation or all.
            # For this iteration, we will use the original query but log the expansion.
            self.logger.info(f"Query variations: {expanded_queries}")
            
            with tracer.start_as_current_span("retrieve_and_rerank") as retr_span:
                retr_span.set_attribute("retrieval.initial_k", initial_k_per_collection)
                retr_start = time.time()
                
                context_chunks, total_candidates = await self.retrieve_and_rerank(
                    query_text=query_text, # Using original query for now
                    collection_names=collection_names,
                    initial_k_per_collection=initial_k_per_collection,
                    final_k=final_k,
                    use_reranker=use_reranker,
                    user_id=user_id
                )
                
                retr_duration = time.time() - retr_start
                query_latency.labels(stage='retrieval').observe(retr_duration)
                
                retr_span.set_attribute("retrieval.candidates_found", total_candidates)
                retr_span.set_attribute("retrieval.final_results", len(context_chunks))
                query_context_chunks.observe(len(context_chunks))

            if not context_chunks:
                # ... (existing no context handling) ...
                self.logger.warning("No relevant context found for the query.")
                query_total.labels(
                    status='success_no_context',
                    domain=detected_domain or 'unknown',
                    collections_count=str(len(collection_names))
                ).inc()
                return {
                    "response": "Es konnten keine relevanten Informationen in der Wissensdatenbank gefunden werden.",
                    "context": [],
                    "metadata": {
                        "collections_queried": collection_names,
                        "total_candidates": 0,
                        "final_k": final_k,
                        "model_used": llm_model or self.config.get("LLM_MODEL"),
                        "success": True,
                        "error": "No context found."
                    }
                }

            # Step 3: Format structured prompt
            effective_system_prompt = system_prompt if system_prompt else "You are a helpful assistant. Answer the user's query based on the provided context."
            prompt = self._format_structured_prompt(query_text, context_chunks, effective_system_prompt)

            # Step 4: Get LLM instance and generate response
            with tracer.start_as_current_span("llm_generation") as llm_span:
                # ... (existing LLM setup) ...
                provider = self.config.get("LLM_PROVIDER", "unknown")
                model = llm_model or self.config.get("LLM_MODEL", "unknown")
                
                llm_span.set_attribute("llm.provider", provider)
                llm_span.set_attribute("llm.model", model)
                llm_span.set_attribute("llm.temperature", temperature)
                
                llm_start = time.time()
                llm = get_llm(model=llm_model, temperature=temperature)
                llm_response = await llm.acomplete(prompt)
                llm_duration = time.time() - llm_start
                
                llm_span.set_attribute("llm.response_length", len(str(llm_response)))
                
                # Track LLM metrics
                llm_latency.labels(provider=provider, model=model).observe(llm_duration)
                
                # ... (existing token counting) ...
                prompt_tokens = len(prompt.split()) * 1.3
                completion_tokens = len(str(llm_response).split()) * 1.3
                
                llm_tokens_total.labels(provider=provider, model=model, type='prompt').inc(prompt_tokens)
                llm_tokens_total.labels(provider=provider, model=model, type='completion').inc(completion_tokens)

            # Step 5: Self-Correction (Phase H.3)
            verification_result = {"is_supported": True}
            if self.config.get("ENABLE_SELF_CORRECTION", "false").lower() == "true":
                try:
                    from src.core.rag.self_correction import SelfCorrection
                    sc = SelfCorrection()
                    verification_result = await sc.verify_answer(query_text, str(llm_response), context_chunks)
                    
                    if not verification_result["is_supported"]:
                        self.logger.warning(f"Answer verification failed: {verification_result.get('reasoning')}")
                        # Optional: Regenerate or flag answer
                        # For now, we just add a warning to metadata
                except Exception as e:
                    self.logger.warning(f"Self-correction failed: {e}")

            # Total latency
            total_duration = time.time() - start_time
            query_latency.labels(stage='total').observe(total_duration)
            
            # Track success
            span.set_attribute("query.success", True)
            query_total.labels(
                status='success',
                domain=detected_domain or 'unknown',
                collections_count=str(len(collection_names))
            ).inc()
            
            # Step 6: Return the standardized response
            final_response = {
                "response": str(llm_response),
                "context": [node.to_dict() for node in context_chunks],
                "metadata": {
                    "collections_queried": collection_names,
                    "total_candidates": total_candidates,
                    "final_k": len(context_chunks),
                    "model_used": getattr(llm, 'model', llm_model or self.config.get("LLM_MODEL")),
                    "detected_domain": detected_domain,
                    "original_collection_count": original_collection_count,
                    "success": True,
                    "error": None,
                    "expanded_queries": expanded_queries, # H.2 info
                    "verification": verification_result, # H.3 info
                    "source": "live"
                }
            }
            
            # Save to Cache (Phase I.1)
            try:
                from src.core.caching.redis_service import redis_cache
                redis_cache.set_query_result(query_text, cache_params, final_response)
            except Exception as e:
                self.logger.warning(f"Cache save failed: {e}")
            
            # Evaluate Query Quality (Phase 1a) - Fire & Forget
            try:
                from src.core.evaluation import get_evaluation_service
                eval_service = get_evaluation_service()
                if eval_service.enabled:
                    # Extract contexts as strings
                    context_strings = [node.content for node in context_chunks]
                    # Run evaluation asynchronously (don't wait for result)
                    asyncio.create_task(
                        eval_service.evaluate_query(
                            query=query_text,
                            answer=str(llm_response),
                            contexts=context_strings,
                            query_id=f"query_{int(time.time() * 1000)}"
                        )
                    )
            except Exception as e:
                self.logger.warning(f"Evaluation failed: {e}")
                
            return final_response

        except Exception as e:
            self.logger.error(f"Answer query failed: {e}", exc_info=True)
            
            # Track error
            span.set_attribute("query.success", False)
            span.set_attribute("query.error", str(e))
            span.record_exception(e)
            
            query_total.labels(
                status='error',
                domain=detected_domain or 'unknown',
                collections_count=str(len(collection_names))
            ).inc()
            
            error_message = self._translate_llm_error(str(e))
            return {
                "response": None,
                "context": [],
                "metadata": {
                    "collections_queried": collection_names,
                    "total_candidates": 0,
                    "final_k": final_k,
                    "model_used": llm_model or self.config.get("LLM_MODEL"),
                    "success": False,
                    "error": error_message
                }
            }

    def _format_structured_prompt(
        self,
        query_text: str,
        chunks: List[RankedNode],
        system_context: str = ""
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
        if system_context:
            prompt_parts.append("=== SYSTEM CONTEXT ===")
            prompt_parts.append(system_context)
            prompt_parts.append("")

        # 2. Retrieved Context
        if chunks:
            prompt_parts.append("=== RETRIEVED CONTEXT ===")
            prompt_parts.append(
                "The following information was retrieved from the knowledge base "
                "and may be relevant to answering the query:"
            )
            prompt_parts.append("")

            for i, chunk in enumerate(chunks, 1):
                prompt_parts.append(f"[Source {i}] ({chunk.source_collection})")
                prompt_parts.append(chunk.content)
                prompt_parts.append("")

        # 3. User Query
        prompt_parts.append("=== USER QUERY ===")
        prompt_parts.append(query_text)
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
