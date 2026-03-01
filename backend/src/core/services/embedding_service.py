"""
Embedding Service - Manages embedding generation and validation.

This service handles all embedding-related operations:
- Embedding dimension queries
- Embedding generation
- Model compatibility checks
"""

import asyncio
from typing import Optional, Dict, Any, List
from loguru import logger


class EmbeddingService:
    """
    Service for embedding operations.

    Responsibilities:
    - Query embedding dimensions from models
    - Generate embeddings for text
    - Validate embedding compatibility
    """

    def __init__(self, embedding_manager, config: dict):
        """
        Initialize embedding service.

        Args:
            embedding_manager: EmbeddingManager instance
            config: Configuration dict with embedding settings
        """
        self.embedding_manager = embedding_manager
        self.config = config
        self.logger = logger.bind(component="EmbeddingService")

    async def get_embedding_dimensions(self, model_name: str) -> int:
        """
        Get embedding dimensions for a specific model.

        Args:
            model_name: Name of embedding model

        Returns:
            Number of dimensions (e.g., 768, 1536)

        Raises:
            Exception if model not found or query fails
        """
        self.logger.debug(f"Getting embedding dimensions for model: {model_name}")

        try:
            # Use embedding_manager's method
            dimensions = self.embedding_manager.get_dimensions(
                provider=self.config.get("EMBEDDING_PROVIDER", "ollama"),
                model=model_name
            )

            self.logger.debug(f"Model {model_name} has {dimensions} dimensions")
            return dimensions

        except Exception as e:
            self.logger.error(f"Failed to get embedding dimensions: {e}")
            raise

    async def generate_embedding(self, text: str) -> list:
        """
        Generate embedding for text.

        Args:
            text: Input text

        Returns:
            Embedding vector (list of floats)
        """
        return await asyncio.to_thread(
            self.embedding_manager.generate_embedding,
            text
        )

    async def generate_embeddings_batch(self, texts: List[str]) -> List[list]:
        """
        Generate embeddings for multiple texts in one call.

        10x faster than individual calls.
        """
        # Ollama supports batch embeddings
        if self.embedding_manager.provider == "ollama":
            return await asyncio.to_thread(
                self.embedding_manager.generate_embeddings_batch,
                texts
            )
        else:
            # Fallback: parallel individual calls
            tasks = [
                self.generate_embedding(text)
                for text in texts
            ]
            return await asyncio.gather(*tasks)

    async def validate_embedding_compatibility(
        self,
        collection_metadata: Dict[str, Any],
        current_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if current embedding config is compatible with collection.

        Args:
            collection_metadata: Collection's embedding metadata
            current_config: Current system embedding config

        Returns:
            Dict with compatibility info and details
        """
        collection_model = collection_metadata.get("embedding_model")
        collection_dims = collection_metadata.get("embedding_dimensions")

        current_model = current_config.get("EMBEDDING_MODEL")
        current_dims = await self.get_embedding_dimensions(current_model)

        compatible = (
            current_model == collection_model and
            current_dims == collection_dims
        )

        return {
            "compatible": compatible,
            "collection_model": collection_model,
            "collection_dimensions": collection_dims,
            "current_model": current_model,
            "current_dimensions": current_dims
        }
