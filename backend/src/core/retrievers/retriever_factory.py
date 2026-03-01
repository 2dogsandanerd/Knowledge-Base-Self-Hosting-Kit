"""
Retriever Factory for creating appropriate retriever instances.

This module provides a factory function to create the right retriever
based on collection configuration and available services.
"""

from typing import Optional
from loguru import logger

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from src.core.retrievers.enhanced_hybrid_retriever import EnhancedHybridRetriever
from src.core.retrievers.bm25_retriever import BM25Retriever
from src.core.exceptions import (
    CollectionNotFoundError,
    RetrievalError,
    BM25SyncError
)


def get_retriever_for_collection(
    collection_name: str,
    chroma_client,
    embeddings,
    use_bm25: bool = True,
    weights: Optional[list] = None
) -> EnhancedHybridRetriever:
    """
    Factory function to get the appropriate retriever for a collection.

    Args:
        collection_name: Name of the collection to query
        chroma_client: ChromaDB client instance
        embeddings: Embedding model instance
        use_bm25: Whether to include BM25 in the hybrid retriever
        weights: Optional weights for retriever combination [vector_weight, bm25_weight]

    Returns:
        EnhancedHybridRetriever instance configured for the collection
    """
    log = logger.bind(component="RetrieverFactory")

    try:
        # Get the actual client through get_client method if it exists
        actual_client = chroma_client.get_client() if hasattr(chroma_client, 'get_client') else chroma_client
        
        # Get the collection from ChromaDB
        try:
            chroma_collection = actual_client.get_collection(collection_name)
        except Exception as e:
            log.error(f"Failed to get collection '{collection_name}': {e}")
            raise CollectionNotFoundError(collection_name)

        # Create vector store and index
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embeddings
        )

        # Create base retrievers
        vector_retriever = index.as_retriever()
        
        retrievers = [vector_retriever]
        
        if use_bm25:
            try:
                bm25_retriever = BM25Retriever(collection_name=collection_name)
                retrievers.append(bm25_retriever)
                
                # Use default weights if none provided
                if weights is None:
                    weights = [0.7, 0.3]  # Favor vector slightly over BM25
            except Exception as e:
                log.warning(f"Failed to initialize BM25 retriever for '{collection_name}': {e}")
                # Continue with vector-only retriever
                weights = [1.0]  # Only vector retriever
        else:
            weights = [1.0]  # Only vector retriever

        # Create enhanced hybrid retriever
        log.debug(f"Creating EnhancedHybridRetriever for collection '{collection_name}' with {len(retrievers)} retrievers")
        
        return EnhancedHybridRetriever(
            retrievers=retrievers,
            weights=weights
        )

    except CollectionNotFoundError:
        raise  # Re-raise collection not found errors
    except Exception as e:
        log.error(f"Failed to create retriever for collection '{collection_name}': {e}")
        raise RetrievalError(
            f"Failed to create retriever for collection '{collection_name}'",
            collection=collection_name,
            details={"error": str(e)}
        )