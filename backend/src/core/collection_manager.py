"""Collection Manager for RAG system.

This module manages ChromaDB collection operations with consistent response handling.
Extracted from RAGClient to improve separation of concerns and solve Issue #1.

Issue #1: Inconsistent RAG Status Handling
- Problem: Mixed return types (bool, Dict, None), unclear error states
- Solution: RAGResponse pattern with consistent structure (success, data, message, error)
"""

import asyncio
import json
from typing import Optional, Any, Dict, List
from loguru import logger
from datetime import datetime

# Import unified RAGResponse from circuit_breaker (Phase 5)
from src.core.circuit_breaker import RAGResponse


class CollectionManager:
    """Manages ChromaDB collection operations with RAGResponse pattern.

    This manager handles:
    - Collection CRUD operations (create, list, get, delete)
    - Collection metadata management
    - Document operations (list, delete, reset)
    - Consistent error handling via RAGResponse

    All methods return RAGResponse for uniform error handling (Issue #1).
    """

    def __init__(self, chroma_client=None, embedding_manager=None):
        """Initialize CollectionManager.

        Args:
            chroma_client: ChromaDB client (can be None, set later via set_client)
            embedding_manager: EmbeddingManager instance for embedding operations
        """
        self.chroma_client = chroma_client
        self.embedding_manager = embedding_manager
        self.logger = logger.bind(component="CollectionManager")

    def set_client(self, chroma_client):
        """Set or update the ChromaDB client.

        Args:
            chroma_client: ChromaDB client instance
        """
        self.chroma_client = chroma_client

    def set_embedding_manager(self, embedding_manager):
        """Set or update the EmbeddingManager.

        Args:
            embedding_manager: EmbeddingManager instance
        """
        self.embedding_manager = embedding_manager

    async def list_collections(self) -> RAGResponse:
        """List all collection names in ChromaDB.

        Returns:
            RAGResponse with data=List[str] of collection names
        """
        if not self.chroma_client:
            return RAGResponse.fail(
                error="ChromaDB client not initialized",
                message="Cannot list collections"
            )

        try:
            # Get the actual client through get_client method if it exists
            actual_client = self.chroma_client.get_client() if hasattr(self.chroma_client, 'get_client') else self.chroma_client
            collections = await asyncio.to_thread(actual_client.list_collections)
            collection_names = [col.name for col in collections]
            return RAGResponse.ok(
                data=collection_names,
                message=f"Found {len(collection_names)} collections"
            )
        except Exception as e:
            self.logger.error(f"Failed to list collections: {e}")
            return RAGResponse.fail(
                error=str(e),
                message="Failed to list collections"
            )

    async def create_collection(
        self,
        collection_name: str,
        embedding_config: Optional[Dict] = None
    ) -> RAGResponse:
        """Create a new collection in ChromaDB with metadata tracking.

        Args:
            collection_name: Name of the collection to create
            embedding_config: Optional dict with keys:
                - provider: "ollama" (default)
                - model: embedding model name (default: "nomic-embed-text:latest")
                - description: user-provided description (optional)

        Returns:
            RAGResponse with data=Dict containing collection metadata
        """
        if not self.chroma_client:
            return RAGResponse.fail(
                error="ChromaDB client not initialized",
                message=f"Cannot create collection '{collection_name}'"
            )

        try:
            # Get the actual ChromaDB client
            actual_client = self.chroma_client.get_client() if hasattr(self.chroma_client, 'get_client') else self.chroma_client

            # Check if collection already exists
            existing_collections = await asyncio.to_thread(actual_client.list_collections)
            if any(col.name == collection_name for col in existing_collections):
                self.logger.info(f"Collection {collection_name} already exists")
                return RAGResponse.ok(
                    data={"name": collection_name, "status": "already_exists"},
                    message=f"Collection '{collection_name}' already exists"
                )

            # Determine embedding configuration
            default_provider = "ollama"
            default_model = "nomic-embed-text:latest"
            
            if self.embedding_manager:
                manager_config = self.embedding_manager._load_config()
                default_provider = manager_config.get("EMBEDDING_PROVIDER", "ollama")
                default_model = manager_config.get("EMBEDDING_MODEL", "nomic-embed-text:latest")

            embedding_provider = embedding_config.get("provider", default_provider) if embedding_config else default_provider
            embedding_model = embedding_config.get("model", default_model) if embedding_config else default_model
            description = embedding_config.get("description", "") if embedding_config else ""

            # Get embeddings from EmbeddingManager
            if self.embedding_manager:
                embeddings = self.embedding_manager.get_embeddings(
                    provider=embedding_provider,
                    model=embedding_model,
                    use_fallback=True
                )
                embedding_dimensions = self.embedding_manager.get_dimensions(
                    provider=embedding_provider,
                    model=embedding_model
                )
            if not embeddings:
                return RAGResponse.fail(
                    error=f"Failed to initialize embeddings: {embedding_provider}:{embedding_model}",
                    message=f"Cannot create collection '{collection_name}'"
                )

            # Create collection via ChromaDB client directly
            # Create the collection without Langchain wrapper
            collection = await asyncio.to_thread(actual_client.create_collection, collection_name)

            # Save metadata
            metadata = {
                "embedding_provider": embedding_provider,
                "embedding_model": embedding_model,
                "embedding_dimensions": embedding_dimensions,
                "description": description,
                "created_at": datetime.now().isoformat(),
                "version": "1.0"
            }
            await self._save_collection_metadata(collection_name, metadata, embeddings)

            self.logger.success(
                f"Collection '{collection_name}' created with {embedding_model} ({embedding_dimensions} dims)"
            )
            return RAGResponse.ok(
                data={"name": collection_name, "metadata": metadata},
                message=f"Collection '{collection_name}' created successfully"
            )

        except Exception as e:
            self.logger.error(f"Failed to create collection '{collection_name}': {e}")
            return RAGResponse.fail(
                error=str(e),
                message=f"Failed to create collection '{collection_name}'"
            )

    async def delete_collection(self, collection_name: str) -> RAGResponse:
        """Delete a collection from ChromaDB.

        Args:
            collection_name: Name of the collection to delete

        Returns:
            RAGResponse with data=Dict containing deletion info
        """
        if not self.chroma_client:
            return RAGResponse.fail(
                error="ChromaDB client not initialized",
                message=f"Cannot delete collection '{collection_name}'"
            )

        try:
            # Get the actual client through get_client method if it exists
            actual_client = self.chroma_client.get_client() if hasattr(self.chroma_client, 'get_client') else self.chroma_client
            await asyncio.to_thread(actual_client.delete_collection, collection_name)
            self.logger.success(f"Collection '{collection_name}' deleted successfully")
            return RAGResponse.ok(
                data={"name": collection_name, "deleted": True},
                message=f"Collection '{collection_name}' deleted successfully"
            )
        except Exception as e:
            self.logger.error(f"Failed to delete collection '{collection_name}': {e}")
            return RAGResponse.fail(
                error=str(e),
                message=f"Failed to delete collection '{collection_name}'"
            )

    async def get_collection_stats(self, collection_name: str) -> RAGResponse:
        """Get statistics about a specific collection.

        Args:
            collection_name: Name of the collection

        Returns:
            RAGResponse with data=Dict containing:
                - name, total_documents, size_mb, embedding_model,
                - embedding_provider, embedding_dimensions, created_at, description, status
        """
        if not self.chroma_client:
            return RAGResponse.fail(
                error="ChromaDB client not initialized",
                message=f"Cannot get stats for '{collection_name}'"
            )

        try:
            # Get the actual client through get_client method if it exists
            actual_client = self.chroma_client.get_client() if hasattr(self.chroma_client, 'get_client') else self.chroma_client
            collection = await asyncio.to_thread(actual_client.get_collection, collection_name)
            count = await asyncio.to_thread(collection.count)

            # Get metadata
            metadata = await self._get_collection_metadata(collection_name)

            # Calculate size (rough estimate)
            embedding_dims = metadata.get("embedding_dimensions", 768)
            size_bytes = count * embedding_dims * 4  # 4 bytes per float32
            size_mb = round(size_bytes / (1024 * 1024), 2)

            stats = {
                "name": collection_name,
                "total_documents": count,
                "size_mb": size_mb,
                "embedding_model": metadata.get("embedding_model", "unknown"),
                "embedding_provider": metadata.get("embedding_provider", "unknown"),
                "embedding_dimensions": metadata.get("embedding_dimensions", 0),
                "created_at": metadata.get("created_at", ""),
                "description": metadata.get("description", ""),
                "status": "active" if count > 0 else "empty"
            }

            return RAGResponse.ok(
                data=stats,
                message=f"Stats retrieved for '{collection_name}'"
            )
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return RAGResponse.fail(
                error=str(e),
                message=f"Failed to get stats for '{collection_name}'"
            )

    async def reset_collection(self, collection_name: str) -> RAGResponse:
        """Remove all documents from collection EXCEPT metadata.

        Args:
            collection_name: Name of the collection to reset

        Returns:
            RAGResponse with data=Dict containing deletion count
        """
        if not self.chroma_client:
            return RAGResponse.fail(
                error="ChromaDB client not initialized",
                message=f"Cannot reset collection '{collection_name}'"
            )

        try:
            # Get the actual client through get_client method if it exists
            actual_client = self.chroma_client.get_client() if hasattr(self.chroma_client, 'get_client') else self.chroma_client
            collection = await asyncio.to_thread(actual_client.get_collection, collection_name)

            # Get all document IDs
            all_data = await asyncio.to_thread(collection.get, include=["metadatas"])
            all_ids = all_data["ids"]

            # Filter out metadata document
            ids_to_delete = [id for id in all_ids if id != "__collection_metadata__"]

            # Delete all except metadata
            if ids_to_delete:
                await asyncio.to_thread(collection.delete, ids=ids_to_delete)
                self.logger.success(
                    f"Collection '{collection_name}' reset. Deleted {len(ids_to_delete)} documents"
                )
                return RAGResponse.ok(
                    data={"name": collection_name, "deleted_count": len(ids_to_delete)},
                    message=f"Collection '{collection_name}' reset successfully"
                )
            else:
                self.logger.info(f"Collection '{collection_name}' already empty (only metadata)")
                return RAGResponse.ok(
                    data={"name": collection_name, "deleted_count": 0},
                    message=f"Collection '{collection_name}' is already empty"
                )

        except Exception as e:
            self.logger.error(f"Failed to reset collection '{collection_name}': {e}")
            return RAGResponse.fail(
                error=str(e),
                message=f"Failed to reset collection '{collection_name}'"
            )

    async def get_documents(
        self,
        collection_name: str,
        limit: int = 100,
        offset: int = 0
    ) -> RAGResponse:
        """Get documents from collection with pagination.

        Args:
            collection_name: Name of the collection
            limit: Maximum number of documents to return
            offset: Number of documents to skip

        Returns:
            RAGResponse with data=Dict containing:
                - documents: List[Dict], total: int, limit: int, offset: int
        """
        if not self.chroma_client:
            return RAGResponse.fail(
                error="ChromaDB client not initialized",
                message=f"Cannot get documents from '{collection_name}'"
            )

        try:
            # Get the actual client through get_client method if it exists
            actual_client = self.chroma_client.get_client() if hasattr(self.chroma_client, 'get_client') else self.chroma_client
            collection = await asyncio.to_thread(actual_client.get_collection, collection_name)

            # Get documents with pagination
            results = await asyncio.to_thread(collection.get,
                limit=limit,
                offset=offset,
                include=["documents", "metadatas"]
            )

            documents = []
            for i, doc_id in enumerate(results["ids"]):
                documents.append({
                    "id": doc_id,
                    "content": results["documents"][i][:500] if i < len(results["documents"]) else "",
                    "metadata": results["metadatas"][i] if i < len(results["metadatas"]) else {}
                })

            data = {
                "documents": documents,
                "total": await asyncio.to_thread(collection.count),
                "limit": limit,
                "offset": offset
            }

            return RAGResponse.ok(
                data=data,
                message=f"Retrieved {len(documents)} documents from '{collection_name}'"
            )
        except Exception as e:
            self.logger.error(f"Failed to get documents: {e}")
            return RAGResponse.fail(
                error=str(e),
                message=f"Failed to get documents from '{collection_name}'"
            )

    async def delete_document(self, doc_id: str, collection_name: str) -> RAGResponse:
        """Delete a document from collection.

        Args:
            doc_id: Document ID to delete
            collection_name: Name of the collection

        Returns:
            RAGResponse with data=Dict containing deletion info
        """
        if not self.chroma_client:
            return RAGResponse.fail(
                error="ChromaDB client not initialized",
                message=f"Cannot delete document '{doc_id}'"
            )

        try:
            # Get the actual client through get_client method if it exists
            actual_client = self.chroma_client.get_client() if hasattr(self.chroma_client, 'get_client') else self.chroma_client
            collection = await asyncio.to_thread(actual_client.get_collection, collection_name)
            await asyncio.to_thread(collection.delete, ids=[doc_id])
            self.logger.success(f"Document '{doc_id}' deleted from '{collection_name}'")
            return RAGResponse.ok(
                data={"doc_id": doc_id, "collection": collection_name, "deleted": True},
                message=f"Document '{doc_id}' deleted successfully"
            )
        except Exception as e:
            self.logger.error(f"Failed to delete document '{doc_id}': {e}")
            return RAGResponse.fail(
                error=str(e),
                message=f"Failed to delete document '{doc_id}'"
            )

    # ==================== PRIVATE HELPER METHODS ====================

    async def _save_collection_metadata(
        self,
        collection_name: str,
        metadata: Dict,
        embeddings
    ) -> bool:
        """Save collection metadata as a special document.

        Args:
            collection_name: Name of the collection
            metadata: Metadata dict to save
            embeddings: Embedding function to use

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get the actual client through get_client method if it exists
            actual_client = self.chroma_client.get_client() if hasattr(self.chroma_client, 'get_client') else self.chroma_client
            collection = await asyncio.to_thread(actual_client.get_collection, collection_name)

            if not embeddings:
                self.logger.error("No embedding function available for metadata storage")
                return False

            # Create metadata document
            metadata_doc = json.dumps(metadata)
            # Use appropriate method for embedding query
            if hasattr(embeddings, 'embed_query'):
                metadata_embedding = await asyncio.to_thread(embeddings.embed_query, "collection_metadata")
            elif hasattr(embeddings, 'get_query_embedding'):
                metadata_embedding = await asyncio.to_thread(embeddings.get_query_embedding, "collection_metadata")
            else:
                metadata_embedding = await asyncio.to_thread(embeddings.get_text_embedding, "collection_metadata")

            # Store metadata
            await asyncio.to_thread(collection.add,
                ids=["__collection_metadata__"],
                documents=[metadata_doc],
                embeddings=[metadata_embedding],
                metadatas=[{"type": "collection_metadata"}]
            )
            self.logger.debug(f"Metadata saved for collection '{collection_name}'")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save collection metadata: {e}")
            return False

    async def _get_collection_metadata(self, collection_name: str) -> Dict:
        """Get collection metadata from special document.

        Args:
            collection_name: Name of the collection

        Returns:
            Metadata dict or empty dict if not found
        """
        try:
            # Get the actual client through get_client method if it exists
            actual_client = self.chroma_client.get_client() if hasattr(self.chroma_client, 'get_client') else self.chroma_client
            collection = await asyncio.to_thread(actual_client.get_collection, collection_name)

            # Try to get metadata document
            result = await asyncio.to_thread(collection.get, ids=["__collection_metadata__"], include=["documents"])

            if result["ids"] and result["documents"]:
                return json.loads(result["documents"][0])
            else:
                self.logger.info(f"Using default configuration for collection '{collection_name}' (no custom metadata found)")
                return {}
        except Exception as e:
            self.logger.error(f"Failed to get collection metadata: {e}")
            return {}


# Singleton instance (can be configured later)
collection_manager = CollectionManager()
