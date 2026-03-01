"""Embedding Manager for RAG system.

This module manages embedding model initialization, caching, and fallback chains.
Extracted from RAGClient to improve separation of concerns and solve Issues #2 and #4.

Issue #2: Missing Embedding Re-initialization
- Auto-reinit embeddings after Ollama/ChromaDB restart
- Cached embeddings with validation

Issue #4: Lack of Fallback Mechanisms
- Fallback chain for embedding models (nomic → minilm → ada-002)
- Multi-provider support (ollama, openai, huggingface)
"""

import os
from typing import Dict, Any, Optional, List
from loguru import logger


class EmbeddingManager:
    """Manages embedding models with caching, fallback, and auto-reinit support.

    This manager handles:
    - Embedding model initialization (Ollama, OpenAI, etc.)
    - Caching of initialized embeddings
    - Fallback chains when primary model fails
    - Auto-reinitialization after service restarts
    - Dimension detection for compatibility checks
    """

    def __init__(self, config_override: Optional[Dict] = None):
        """Initialize EmbeddingManager.

        Args:
            config_override: Optional config dict to override defaults
        """
        self.config_override = config_override or {}

        # Embedding cache: key = "provider:model", value = embedding instance
        self._cache: Dict[str, Any] = {}

        # Fallback chains for different providers (Issue #4)
        self._fallback_chain = {
            "ollama": [
                "nomic-embed-text:latest",
                "all-minilm:latest",
                "all-minilm:l6-v2"
            ],
            "openai": [
                "text-embedding-3-small",
                "text-embedding-ada-002"
            ],
            "huggingface": [
                "sentence-transformers/all-MiniLM-L6-v2"
            ]
        }

        self._known_dimensions = {
            "nomic-embed-text:latest": 768,
            "nomic-embed-text": 768,
            "mxbai-embed-large": 1024,
            "all-minilm:l6-v2": 384,
            "all-minilm": 384,
            "all-minilm:latest": 384,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536
        }

        # Initialize provider from config immediately to prevent AttributeError
        initial_config = self._load_config()
        self.provider = initial_config.get("EMBEDDING_PROVIDER", "ollama")

    def _load_config(self) -> Dict[str, str]:
        """Load configuration from config_override or environment.

        Returns:
            Config dictionary
        """
        # Start with ENV variables as base
        base_config = {
            "OLLAMA_HOST": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest"),
            "EMBEDDING_PROVIDER": os.getenv("EMBEDDING_PROVIDER", "ollama"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
            "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL", ""),
        }

        # Try to load from config_service (if .env file exists)
        try:
            from src.services.config_service import config_service
            file_config = config_service.load_configuration()
            # Only override if key exists in file_config
            for key in base_config.keys():
                if key in file_config and file_config[key]:
                    base_config[key] = file_config[key]
        except (ImportError, Exception) as e:
            logger.debug(f"config_service not available, using ENV only: {e}")

        # Override with config_override (highest priority)
        if self.config_override:
            base_config.update(self.config_override)
        return base_config

    def get_embeddings(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        use_fallback: bool = True
    ) -> Optional[Any]:
        """Get cached or initialize embeddings (SYNC method for compatibility).

        Args:
            provider: Provider name (ollama, openai, huggingface). If None, uses config.
            model: Model name. If None, uses config.
            use_fallback: Whether to try fallback models on failure (default: True)

        Returns:
            Embedding instance or None if all attempts fail
        """
        config = self._load_config()

        # Use config defaults if not specified
        if provider is None:
            provider = config.get("EMBEDDING_PROVIDER", "ollama")
        if model is None:
            model = config.get("EMBEDDING_MODEL", "nomic-embed-text:latest")

        # Check cache first
        cache_key = f"{provider}:{model}"
        if cache_key in self._cache:
            cached_emb = self._cache[cache_key]
            if self._validate_embeddings(cached_emb):
                logger.debug(f"Using cached embeddings: {cache_key}")
                return cached_emb
            else:
                # Cache invalid, remove it
                logger.warning(f"Cached embeddings invalid, removing: {cache_key}")
                del self._cache[cache_key]

        # Try to initialize primary model
        embeddings = self._initialize_embeddings(provider, model, config)
        if embeddings:
            self._cache[cache_key] = embeddings
            return embeddings

        # If primary model failed and fallback enabled, try fallback chain (Issue #4)
        if use_fallback:
            logger.warning(f"Primary model failed: {provider}:{model}, trying fallback chain...")
            fallback_models = self._fallback_chain.get(provider, [])

            for fallback_model in fallback_models:
                if fallback_model == model:
                    # Skip the model we just tried
                    continue

                logger.info(f"Trying fallback model: {provider}:{fallback_model}")
                embeddings = self._initialize_embeddings(provider, fallback_model, config)
                if embeddings:
                    # Cache under ORIGINAL key so subsequent calls use fallback
                    self._cache[cache_key] = embeddings
                    logger.info(f"Fallback successful: {provider}:{fallback_model}")
                    return embeddings

        logger.error(f"All embedding initialization attempts failed for {provider}")
        return None

    def _initialize_embeddings(
        self,
        provider: str,
        model: str,
        config: Dict
    ) -> Optional[Any]:
        """Initialize embedding model for specific provider.

        Args:
            provider: Provider name (ollama, openai, huggingface)
            model: Model name
            config: Configuration dict

        Returns:
            Embedding instance or None on failure
        """
        try:
            if provider.lower() == "ollama":
                return self._initialize_ollama_embeddings(model, config)
            elif provider.lower() == "openai":
                return self._initialize_openai_embeddings(model, config)
            elif provider.lower() == "openai_compatible":
                return self._initialize_openai_compatible_embeddings(model, config)
            elif provider.lower() == "huggingface":
                return self._initialize_huggingface_embeddings(model, config)
            else:
                logger.error(f"Unknown embedding provider: {provider}")
                return None
        except Exception as e:
            logger.error(f"Failed to initialize {provider}:{model} embeddings: {e}")
            return None

    def _initialize_ollama_embeddings(self, model: str, config: Dict) -> Optional[Any]:
        """Initialize Ollama embeddings.

        Args:
            model: Ollama model name
            config: Configuration dict with OLLAMA_HOST

        Returns:
            OllamaEmbeddings instance or None
        """
        try:
            from llama_index.embeddings.ollama import OllamaEmbedding

            ollama_host = config.get("OLLAMA_HOST", "http://localhost:11434")
            embeddings = OllamaEmbedding(
                model_name=model,
                base_url=ollama_host
            )
            logger.info(f"Ollama embeddings initialized: {model} at {ollama_host}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to initialize Ollama embeddings: {e}")
            return None

    def _initialize_openai_embeddings(self, model: str, config: Dict) -> Optional[Any]:
        """Initialize OpenAI embeddings.

        Args:
            model: OpenAI model name
            config: Configuration dict with OPENAI_API_KEY

        Returns:
            OpenAIEmbeddings instance or None
        """
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding

            api_key = config.get("OPENAI_API_KEY")
            if not api_key:
                logger.error("OPENAI_API_KEY not found in config")
                return None

            embeddings = OpenAIEmbedding(
                model=model,
                api_key=api_key
            )
            logger.info(f"OpenAI embeddings initialized: {model}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings: {e}")
            return None

    def _initialize_openai_compatible_embeddings(self, model: str, config: Dict) -> Optional[Any]:
        """Initialize OpenAI-compatible embeddings (e.g. LM Studio, LocalAI).

        Args:
            model: Model name
            config: Configuration dict with OPENAI_BASE_URL

        Returns:
            OpenAIEmbedding instance or None
        """
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding

            base_url = config.get("OPENAI_BASE_URL")
            api_key = config.get("OPENAI_API_KEY") or "not-needed"
            
            if not base_url:
                logger.warning("OPENAI_BASE_URL not set for openai_compatible provider")
                
            embeddings = OpenAIEmbedding(
                model=model,
                api_key=api_key,
                base_url=base_url
            )
            logger.info(f"OpenAI-compatible embeddings initialized: {model} at {base_url}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI-compatible embeddings: {e}")
            return None

    def _initialize_huggingface_embeddings(self, model: str, config: Dict) -> Optional[Any]:
        """Initialize HuggingFace embeddings.

        Args:
            model: HuggingFace model name
            config: Configuration dict

        Returns:
            HuggingFaceEmbeddings instance or None
        """
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            embeddings = HuggingFaceEmbedding(model_name=model)
            logger.info(f"HuggingFace embeddings initialized: {model}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace embeddings: {e}")
            return None

    def _validate_embeddings(self, embeddings: Any) -> bool:
        """Validate that embeddings are still functional.

        Args:
            embeddings: Embedding instance to validate

        Returns:
            True if embeddings work, False otherwise
        """
        try:
            # Quick test with a simple query
            # Different methods depending on embedding type
            if hasattr(embeddings, 'embed_query'):
                test_result = embeddings.embed_query("test")
            elif hasattr(embeddings, 'get_query_embedding'):
                test_result = embeddings.get_query_embedding("test")
            else:
                # For some embeddings, we use a generic method
                test_result = embeddings.get_text_embedding("test")
            return len(test_result) > 0
        except Exception as e:
            logger.debug(f"Embeddings validation failed: {e}")
            return False

    def get_dimensions(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> int:
        """Get embedding dimensions for a model.

        This method tries to detect dimensions by:
        1. Using cached embeddings if available
        2. Creating temporary embeddings and testing
        3. Falling back to known dimensions lookup

        Args:
            provider: Provider name (ollama, openai, huggingface). If None, uses config.
            model: Model name. If None, uses config.

        Returns:
            Number of dimensions (int)
        """
        config = self._load_config()

        # Use config defaults if not specified
        if provider is None:
            provider = config.get("EMBEDDING_PROVIDER", "ollama")
        if model is None:
            model = config.get("EMBEDDING_MODEL", "nomic-embed-text:latest")

        try:
            # Try to use cached embeddings first
            cache_key = f"{provider}:{model}"
            if cache_key in self._cache:
                embedding_instance = self._cache[cache_key]
                # Use appropriate method for embedding query
                if hasattr(embedding_instance, 'embed_query'):
                    test_embedding = embedding_instance.embed_query("test")
                elif hasattr(embedding_instance, 'get_query_embedding'):
                    test_embedding = embedding_instance.get_query_embedding("test")
                else:
                    test_embedding = embedding_instance.get_text_embedding("test")
                dims = len(test_embedding)
                logger.info(f"Detected {dims} dimensions for cached {cache_key}")
                return dims

            # Otherwise create temporary embeddings and test
            temp_embeddings = self._initialize_embeddings(provider, model, config)
            if temp_embeddings:
                # Use appropriate method for embedding query
                if hasattr(temp_embeddings, 'embed_query'):
                    test_embedding = temp_embeddings.embed_query("test")
                elif hasattr(temp_embeddings, 'get_query_embedding'):
                    test_embedding = temp_embeddings.get_query_embedding("test")
                else:
                    test_embedding = temp_embeddings.get_text_embedding("test")
                dims = len(test_embedding)
                logger.info(f"Detected {dims} dimensions for {provider}:{model}")
                return dims
        except Exception as e:
            logger.warning(f"Failed to detect dimensions for {provider}:{model}: {e}")

        # Fallback to known dimensions
        dims = self._known_dimensions.get(model, 768)
        logger.info(f"Using known dimensions for {model}: {dims}")
        return dims

    def reinitialize(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> bool:
        """Re-initialize embeddings (useful after service restart, Issue #2).

        Args:
            provider: Provider name. If None, uses config.
            model: Model name. If None, uses config.

        Returns:
            True if reinitialization succeeded, False otherwise
        """
        config = self._load_config()

        # Use config defaults if not specified
        if provider is None:
            provider = config.get("EMBEDDING_PROVIDER", "ollama")
        if model is None:
            model = config.get("EMBEDDING_MODEL", "nomic-embed-text:latest")

        # Clear cache for this provider:model
        cache_key = f"{provider}:{model}"
        if cache_key in self._cache:
            logger.info(f"Clearing cached embeddings: {cache_key}")
            del self._cache[cache_key]

        # Try to reinitialize
        embeddings = self.get_embeddings(provider, model, use_fallback=True)
        return embeddings is not None

    def clear_cache(self, provider: Optional[str] = None) -> None:
        """Clear embedding cache.

        Args:
            provider: If specified, only clear cache for this provider.
                     If None, clear entire cache.
        """
        if provider is None:
            logger.info("Clearing entire embedding cache")
            self._cache.clear()
        else:
            # Clear only entries for specific provider
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"{provider}:")]
            for key in keys_to_remove:
                del self._cache[key]
            logger.info(f"Cleared {len(keys_to_remove)} cached embeddings for provider: {provider}")

    def get_fallback_models(self, provider: str) -> List[str]:
        """Get fallback model chain for a provider.

        Args:
            provider: Provider name

        Returns:
            List of fallback model names
        """
        return self._fallback_chain.get(provider, [])


# Singleton instance
embedding_manager = EmbeddingManager()
