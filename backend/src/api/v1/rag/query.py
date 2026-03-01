"""
RAG Query endpoints.

Handles querying the vector database for relevant context.
"""

from fastapi import APIRouter, Depends, HTTPException
from src.core.exceptions import ChromaDBError, ValidationError
from typing import Dict, Any
import logging

from src.api.v1.dependencies import get_rag_adapter, get_rag_client  # Import both dependencies
from src.core.adapters.rag_adapter import RAGAdapter
from src.services.auth_service import get_current_user
from src.database.models import User
from .models import QueryRequest, IndexRequest

logger = logging.getLogger(__name__)
router = APIRouter()


def _convert_source_to_dual_format(source) -> Dict[str, Any]:
    """
    Convert SourceReference to dual format (legacy + new).
    
    Returns both old field names (relevance_score, source_collection, etc.)
    and new field names (score, collection_name, etc.) for compatibility.
    """
    return {
        # New format
        "content": source.content,
        "collection_name": source.collection_name,
        "score": source.score,
        "file": source.file,
        "page": source.page,
        "chunk_id": source.chunk_id,
        "metadata": source.metadata,
        
        # Legacy format (for backward compatibility)
        "source_collection": source.collection_name,
        "relevance_score": source.score,
        "source": source.file or source.metadata.get("source", ""),
        "page_number": source.page or source.metadata.get("page_number", 0),
        "distance": 1.0 - source.score if source.score else 0.0  # Calculate from score
    }


@router.post("/query")
async def query_rag(
    request: QueryRequest,
    adapter: RAGAdapter = Depends(get_rag_adapter),
    rag_client=Depends(get_rag_client),  # Added rag_client
    current_user: User = Depends(get_current_user)
):
    """
    Query RAG knowledge base and get a synthesized answer from the LLM.
    
    Returns both legacy format (llm_response, context_chunks) and new format
    (answer, sources) for backward compatibility.
    """
    logger.debug(f"Query request: collections={request.collections}, k={request.k}, query_len={len(request.query)}")

    try:
        # Handle both 'collection' (singular) and 'collections' (plural) from frontend
        collection_names = request.collections
        if not collection_names and request.collection:
            collection_names = [request.collection]
        
        # If still empty, fetch ALL collections from registry/ChromaDB
        if not collection_names:
            logger.info("No collections specified, querying ALL available collections.")
            response = await rag_client.list_collections()
            if response.is_success:
                collection_names = response.data
            
            if not collection_names:
                raise ValidationError("No collections available in the system. Please create one first.")
        
        # Build context for adapter
        context = {
            "collection_names": collection_names,
            "n_results": request.k,
            "temperature": request.temperature or 0.1,
            "use_reranker": request.use_reranker,
            "user_id": (
                current_user.id if hasattr(current_user, 'id') 
                else str(current_user.email) if hasattr(current_user, 'email') 
                else "unknown_user"
            )
        }
        
        # Call adapter
        rag_response = await adapter.query(
            question=request.query,
            context=context
        )

        logger.info(f"Query successful for {len(collection_names)} collections.")

        # Convert sources to dual format
        dual_sources = [
            _convert_source_to_dual_format(source) 
            for source in rag_response.sources
        ]

        # Return BOTH formats for backward compatibility
        return {
            # New format
            "answer": rag_response.answer,
            "sources": dual_sources,
            "mode": rag_response.mode,
            "confidence": rag_response.confidence,
            "latency_ms": rag_response.latency_ms,
            
            # Legacy format (for backward compatibility)
            "llm_response": rag_response.answer,
            "context_chunks": dual_sources,
            "query": request.query
        }

    except ValidationError as e:
        # Preserve ValidationError as a 400
        raise e
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        if hasattr(e, 'status_code'):
            raise e
        raise ChromaDBError(str(e))


@router.post("/search")
async def search_documents(
    request: QueryRequest,
    rag_client=Depends(get_rag_client),
    current_user: User = Depends(get_current_user)
):
    """Search for documents without LLM generation - instant results!"""
    logger.debug(f"Search request: collections={request.collections}, k={request.k}, query_len={len(request.query)}")

    try:
        import asyncio

        # Handle both 'collection' (singular) and 'collections' (plural) from frontend
        collection_names = []
        if request.collection:
            # Singular collection parameter takes priority
            collection_names = [request.collection]
        elif request.collections:
            # Use plural collections
            collection_names = request.collections
        else:
            raise ValidationError("Either 'collection' or 'collections' must be provided")
        
        logger.info(f"Searching in collections: {collection_names}")

        # Direct vector search without LLM/QueryService
        all_results = []
        total_candidates = 0

        # Get embeddings for the query
        embedding_instance = rag_client.embedding_manager.get_embeddings()
        if not embedding_instance:
            raise ChromaDBError("Embedding service not available")

        logger.info(f"Got embedding instance: {type(embedding_instance)}")

        # Generate query embedding using correct LlamaIndex methods
        try:
            if hasattr(embedding_instance, 'aget_query_embedding'):
                query_embedding = await embedding_instance.aget_query_embedding(request.query)
            elif hasattr(embedding_instance, 'get_query_embedding'):
                query_embedding = await asyncio.to_thread(embedding_instance.get_query_embedding, request.query)
            elif hasattr(embedding_instance, 'aget_text_embedding'):
                query_embedding = await embedding_instance.aget_text_embedding(request.query)
            elif hasattr(embedding_instance, 'get_text_embedding'):
                query_embedding = await asyncio.to_thread(embedding_instance.get_text_embedding, request.query)
            else:
                raise ChromaDBError(f"Embedding instance {type(embedding_instance)} has no compatible embed method")
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise ChromaDBError(f"Failed to generate query embedding: {e}")

        logger.info(f"Generated query embedding with {len(query_embedding) if query_embedding else 0} dimensions")

        # Query each collection
        for collection_name in collection_names:
            logger.info(f"Querying collection: {collection_name}")
            try:
                collection = await asyncio.to_thread(
                    rag_client.chroma_manager.get_collection,
                    collection_name
                )

                if collection:
                    logger.info(f"Collection {collection_name} found, querying...")
                    
                    # === HYBRID SEARCH: Vector + BM25 ===
                    
                    # 1. Vector Search
                    vector_results = await asyncio.to_thread(
                        collection.query,
                        query_embeddings=[query_embedding],
                        n_results=request.k * 2  # Get more candidates for hybrid ranking
                    )
                    
                    # 2. BM25 Keyword Search
                    bm25_results = []
                    try:
                        from src.core.bm25_index import BM25IndexManager, _tokenize_text
                        bm25_manager = BM25IndexManager(collection_name)
                        bm25_manager.load()
                        
                        if bm25_manager.bm25_index and bm25_manager.nodes:
                            query_tokens = _tokenize_text(request.query)
                            bm25_scores = bm25_manager.bm25_index.get_scores(query_tokens)
                            
                            # Get top BM25 results
                            top_bm25_indices = sorted(
                                range(len(bm25_scores)), 
                                key=lambda i: bm25_scores[i], 
                                reverse=True
                            )[:request.k * 2]
                            
                            for idx in top_bm25_indices:
                                if bm25_scores[idx] > 0:  # Only include relevant results
                                    node = bm25_manager.nodes[idx]
                                    bm25_results.append({
                                        'content': node.text,
                                        'bm25_score': float(bm25_scores[idx]),
                                        'metadata': node.metadata or {},
                                        'node_id': node.id_
                                    })
                            
                            logger.info(f"BM25 found {len(bm25_results)} results")
                    except Exception as e:
                        logger.warning(f"BM25 search failed for {collection_name}: {e}")
                    
                    # 3. Combine and rank results
                    combined_results = {}
                    
                    # Add vector results
                    if vector_results and 'documents' in vector_results and vector_results['documents']:
                        docs = vector_results['documents'][0] if vector_results['documents'] else []
                        metadatas = vector_results['metadatas'][0] if 'metadatas' in vector_results and vector_results['metadatas'] else []
                        distances = vector_results['distances'][0] if 'distances' in vector_results and vector_results['distances'] else []
                        ids = vector_results['ids'][0] if 'ids' in vector_results and vector_results['ids'] else []
                        
                        for i, doc in enumerate(docs):
                            doc_id = ids[i] if i < len(ids) else f"vec_{i}"
                            metadata = metadatas[i] if i < len(metadatas) else {}
                            distance = distances[i] if i < len(distances) else 1.0
                            vector_score = 1.0 - distance
                            
                            combined_results[doc_id] = {
                                "content": doc,
                                "source_collection": collection_name,
                                "vector_score": vector_score,
                                "bm25_score": 0.0,
                                "metadata": metadata,
                                "source": metadata.get('source', 'Unknown'),
                                "page_number": metadata.get('page_number')
                            }
                    
                    # Add/merge BM25 results
                    for bm25_result in bm25_results:
                        node_id = bm25_result['node_id']
                        if node_id in combined_results:
                            # Merge scores
                            combined_results[node_id]['bm25_score'] = bm25_result['bm25_score']
                        else:
                            # Add new result
                            combined_results[node_id] = {
                                "content": bm25_result['content'],
                                "source_collection": collection_name,
                                "vector_score": 0.0,
                                "bm25_score": bm25_result['bm25_score'],
                                "metadata": bm25_result['metadata'],
                                "source": bm25_result['metadata'].get('source', 'Unknown'),
                                "page_number": bm25_result['metadata'].get('page_number')
                            }
                    
                    # Calculate hybrid score (weighted combination)
                    # First, normalize both vector and BM25 scores to 0-1 range
                    if combined_results:
                        # Get min/max for normalization
                        vector_scores = [r['vector_score'] for r in combined_results.values()]
                        bm25_scores = [r['bm25_score'] for r in combined_results.values()]
                        
                        max_vector = max(vector_scores) if vector_scores else 1.0
                        min_vector = min(vector_scores) if vector_scores else 0.0
                        max_bm25 = max(bm25_scores) if bm25_scores else 1.0
                        min_bm25 = min(bm25_scores) if bm25_scores else 0.0
                        
                        for result in combined_results.values():
                            # Normalize vector score (0-1)
                            if max_vector > min_vector:
                                norm_vector = (result['vector_score'] - min_vector) / (max_vector - min_vector)
                            else:
                                norm_vector = 1.0 if result['vector_score'] > 0 else 0.0
                            
                            # Normalize BM25 score (0-1)
                            if max_bm25 > min_bm25:
                                norm_bm25 = (result['bm25_score'] - min_bm25) / (max_bm25 - min_bm25)
                            else:
                                norm_bm25 = 1.0 if result['bm25_score'] > 0 else 0.0
                            
                            # Hybrid score: 50% vector, 50% BM25 (equal weight for balanced results)
                            result['relevance_score'] = (0.5 * norm_vector) + (0.5 * norm_bm25)
                            result['distance'] = 1.0 - result['relevance_score']
                            
                            all_results.append(result)
                    
                    total_candidates += len(combined_results)

            except Exception as e:
                logger.warning(f"Failed to search collection {collection_name}: {e}")
                continue

        # Sort by hybrid relevance score
        ranked_nodes = sorted(all_results, key=lambda x: x['relevance_score'], reverse=True)[:request.k]

        logger.info(f"Search successful, returning {len(ranked_nodes)} documents from {total_candidates} candidates.")

        # ranked_nodes are already dicts, no conversion needed
        return {
            "documents": ranked_nodes,
            "total_found": len(ranked_nodes),
            "total_candidates": total_candidates,
            "query": request.query
        }

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise ChromaDBError(str(e))


@router.post("/index")
async def index_documents(
    request: IndexRequest,
    rag_client=Depends(get_rag_client),
    current_user: User = Depends(get_current_user)
):
    """Index documents into RAG knowledge base"""
    logger.debug(f"Index request: collection={request.collection}, docs_path={request.docs_path}")

    try:
        success = await rag_client.index_documents(
            docs_path=request.docs_path,
            collection_name=request.collection
        )

        logger.info(f"Indexing {'successful' if success else 'failed'}: {request.docs_path} → {request.collection}")

        return {
            "success": success,
            "collection": request.collection,
            "docs_path": request.docs_path
        }

    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        raise ChromaDBError(str(e))


@router.get("/stats")
async def get_rag_stats(
    collection: str = "default",
    rag_client=Depends(get_rag_client),
    current_user: User = Depends(get_current_user)
):
    """Get RAG collection statistics"""
    logger.debug(f"Stats request for collection: {collection}")

    try:
        response = await rag_client.collection_manager.get_collection_stats(collection)
        if not response.is_success:
            logger.error(f"Failed to get stats for '{collection}': {response.error}")
            raise ChromaDBError(response.error)

        logger.debug(f"Stats retrieved for {collection}: {response.data}")
        return response.data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}", exc_info=True)
        raise ChromaDBError(str(e))
