"""Multi-provider LLM configuration for Gmail RAG Assistant.

IMPORTANT: This module now reads config from .env file via config_service
to enable hot-reload without backend restart.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env on module import (for backwards compatibility)
# But prefer using config_service for hot-reload!
load_dotenv()

@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: str
    model: str
    api_key: Optional[str]
    embedding_provider: str
    embedding_model: str
    chroma_path: str
    collection_name: str
    project_collection_name: str
    base_url: Optional[str] = None  # Für OpenAI-kompatible Server

@dataclass
class WebSocketConfig:
    """WebSocket configuration for real-time service status updates."""
    heartbeat_interval: int = 30  # Seconds between heartbeat pings
    heartbeat_timeout: int = 90   # Timeout for client response (3x interval)
    auth_required: bool = False   # Whether authentication is required
    localhost_only: bool = True   # Only allow localhost connections
    idle_timeout: int = 300       # Seconds of idle before closing (0 = disabled)
    max_queue_size: int = 16      # Maximum queue size for backpressure management

@dataclass
class HealthMonitorConfig:
    """Health monitoring configuration for background service checks."""
    check_interval: int = 5        # Seconds between health checks
    event_debounce: int = 2        # Seconds to debounce events (anti-flapping)
    max_subscribers: int = 10      # Maximum number of WebSocket subscribers

@dataclass
class RAGConfig:
    """RAG system configuration for advanced features."""
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_top_k: int = 5
    use_parent_retriever: bool = False
    use_advanced_pipeline: bool = False

# Multi-provider LLM mapping
LLM_PROVIDERS = {
    "openai": {
        "embeddings": "OpenAIEmbedding",  # LlamaIndex class
        "chat": "OpenAI",  # LlamaIndex class
        "module": "llama_index.llms.openai"  # LlamaIndex module
    },
    "gemini": {
        "embeddings": "GoogleAIEmbedding",  # LlamaIndex class
        "chat": "Gemini",  # LlamaIndex class
        "module": "llama_index.llms.gemini"  # LlamaIndex module
    },
    "ollama": {
        "embeddings": "OllamaEmbedding",  # LlamaIndex class
        "chat": "Ollama",  # LlamaIndex class
        "module": "llama_index.llms.ollama"  # LlamaIndex module
    },
    "anthropic": {
        "embeddings": None,  # Use OpenAI for embeddings
        "chat": "Anthropic",  # LlamaIndex class
        "module": "llama_index.llms.anthropic"  # LlamaIndex module
    }
}

def get_config(use_hot_reload: bool = True) -> LLMConfig:
    """Get configuration from environment variables.

    Args:
        use_hot_reload: If True, reads fresh config from .env file (hot-reload).
                       If False, uses cached os.getenv() values.
    """
    # Hot-reload: Read fresh from .env file
    if use_hot_reload:
        try:
            from src.services.config_service import config_service
            config_dict = config_service.load_configuration()

            # Try .env file first, fall back to ENV variables
            provider = config_dict.get("LLM_PROVIDER") or os.getenv("LLM_PROVIDER")
            if not provider:
                raise ValueError("LLM_PROVIDER not found in config or environment")

            model = config_dict.get("LLM_MODEL") or os.getenv("LLM_MODEL")
            if not model:
                raise ValueError("LLM_MODEL not found in config or environment")

            embedding_provider = config_dict.get("EMBEDDING_PROVIDER") or os.getenv("EMBEDDING_PROVIDER")
            if not embedding_provider:
                raise ValueError("EMBEDDING_PROVIDER not found in config or environment")

            embedding_model = config_dict.get("EMBEDDING_MODEL") or os.getenv("EMBEDDING_MODEL")
            if not embedding_model:
                raise ValueError("EMBEDDING_MODEL not found in config or environment")

            chroma_path = config_dict.get("CHROMA_PATH", "./chroma_data")
            collection_name = config_dict.get("CHROMA_COLLECTION", "mail_knowledge_base")
            project_collection_name = config_dict.get("PROJECT_COLLECTION", "project_context")

            # Hole base_url für OpenAI-kompatible Server
            base_url = config_dict.get("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")

            return LLMConfig(
                provider=provider,
                model=model,
                api_key=_get_api_key_from_dict(provider, config_dict),
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                chroma_path=chroma_path,
                collection_name=collection_name,
                project_collection_name=project_collection_name,
                base_url=base_url
            )
        except ImportError:
            # Fallback to os.getenv if config_service not available
            pass

    # Fallback: Use os.getenv() (cached values)
    provider = os.getenv("LLM_PROVIDER")
    if not provider:
        raise ValueError("LLM_PROVIDER environment variable is required")

    model = os.getenv("LLM_MODEL")
    if not model:
        raise ValueError("LLM_MODEL environment variable is required")

    embedding_provider = os.getenv("EMBEDDING_PROVIDER")
    if not embedding_provider:
        raise ValueError("EMBEDDING_PROVIDER environment variable is required")

    embedding_model = os.getenv("EMBEDDING_MODEL")
    if not embedding_model:
        raise ValueError("EMBEDDING_MODEL environment variable is required")

    chroma_path = os.getenv("CHROMA_PATH", "./chroma_data")
    collection_name = os.getenv("CHROMA_COLLECTION", "mail_knowledge_base")
    project_collection_name = os.getenv("PROJECT_COLLECTION", "project_context")

    # Hole base_url für OpenAI-kompatible Server
    base_url = os.getenv("OPENAI_BASE_URL")

    return LLMConfig(
        provider=provider,
        model=model,
        api_key=_get_api_key(provider),
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        chroma_path=chroma_path,
        collection_name=collection_name,
        project_collection_name=project_collection_name,
        base_url=base_url
    )

def _get_api_key(provider: str) -> Optional[str]:
    """Get API key for the specified provider from os.getenv()."""
    key_mapping = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "ollama": None  # Local, no API key needed
    }

    env_var = key_mapping.get(provider)
    return os.getenv(env_var) if env_var else None

def _get_api_key_from_dict(provider: str, config_dict: Dict[str, Any]) -> Optional[str]:
    """Get API key for the specified provider from config dictionary."""
    key_mapping = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "ollama": None  # Local, no API key needed
    }

    env_var = key_mapping.get(provider)
    return config_dict.get(env_var) if env_var else None

def get_websocket_config(use_hot_reload: bool = True) -> WebSocketConfig:
    """Get WebSocket configuration from environment variables.

    Args:
        use_hot_reload: If True, reads fresh config from .env file (hot-reload).
    """
    if use_hot_reload:
        try:
            from src.services.config_service import config_service
            config_dict = config_service.load_configuration()

            return WebSocketConfig(
                heartbeat_interval=int(config_dict.get("WEBSOCKET_HEARTBEAT_INTERVAL", 30)),
                heartbeat_timeout=int(config_dict.get("WEBSOCKET_HEARTBEAT_TIMEOUT", 90)),
                auth_required=config_dict.get("WEBSOCKET_AUTH_REQUIRED", "false").lower() == "true",
                localhost_only=config_dict.get("WEBSOCKET_LOCALHOST_ONLY", "true").lower() == "true",
                idle_timeout=int(config_dict.get("WEBSOCKET_IDLE_TIMEOUT", 300)),
                max_queue_size=int(config_dict.get("WEBSOCKET_MAX_QUEUE_SIZE", 16))
            )
        except ImportError:
            pass

    # Fallback to os.getenv()
    return WebSocketConfig(
        heartbeat_interval=int(os.getenv("WEBSOCKET_HEARTBEAT_INTERVAL", 30)),
        heartbeat_timeout=int(os.getenv("WEBSOCKET_HEARTBEAT_TIMEOUT", 90)),
        auth_required=os.getenv("WEBSOCKET_AUTH_REQUIRED", "false").lower() == "true",
        localhost_only=os.getenv("WEBSOCKET_LOCALHOST_ONLY", "true").lower() == "true",
        idle_timeout=int(os.getenv("WEBSOCKET_IDLE_TIMEOUT", 300)),
        max_queue_size=int(os.getenv("WEBSOCKET_MAX_QUEUE_SIZE", 16))
    )

def get_health_monitor_config(use_hot_reload: bool = True) -> HealthMonitorConfig:
    """Get health monitor configuration from environment variables.

    Args:
        use_hot_reload: If True, reads fresh config from .env file (hot-reload).
    """
    if use_hot_reload:
        try:
            from src.services.config_service import config_service
            config_dict = config_service.load_configuration()

            return HealthMonitorConfig(
                check_interval=int(config_dict.get("HEALTH_CHECK_INTERVAL", 5)),
                event_debounce=int(config_dict.get("HEALTH_EVENT_DEBOUNCE", 2)),
                max_subscribers=int(config_dict.get("HEALTH_MAX_SUBSCRIBERS", 10))
            )
        except ImportError:
            pass

    # Fallback to os.getenv()
    return HealthMonitorConfig(
        check_interval=int(os.getenv("HEALTH_CHECK_INTERVAL", 5)),
        event_debounce=int(os.getenv("HEALTH_EVENT_DEBOUNCE", 2)),
        max_subscribers=int(os.getenv("HEALTH_MAX_SUBSCRIBERS", 10))
    )

def get_rag_config(use_hot_reload: bool = True) -> RAGConfig:
    """Get RAG configuration from environment variables.
    
    Args:
        use_hot_reload: If True, reads fresh config from .env file (hot-reload).
    """
    if use_hot_reload:
        try:
            from src.services.config_service import config_service
            config_dict = config_service.load_configuration()
            
            return RAGConfig(
                reranker_enabled=config_dict.get("RERANKER_ENABLED", "false").lower() == "true",
                reranker_model=config_dict.get("RERANKER_MODEL", "BAAI/bge-reranker-base"),
                reranker_top_k=int(config_dict.get("RERANKER_TOP_K", "5")),
                use_parent_retriever=config_dict.get("USE_PARENT_RETRIEVER", "false").lower() == "true",
                use_advanced_pipeline=config_dict.get("USE_ADVANCED_PIPELINE", "false").lower() == "true"
            )
        except ImportError:
            pass
    
    # Fallback to os.getenv()
    return RAGConfig(
        reranker_enabled=os.getenv("RERANKER_ENABLED", "false").lower() == "true",
        reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base"),
        reranker_top_k=int(os.getenv("RERANKER_TOP_K", "5")),
        use_parent_retriever=os.getenv("USE_PARENT_RETRIEVER", "false").lower() == "true",
        use_advanced_pipeline=os.getenv("USE_ADVANCED_PIPELINE", "false").lower() == "true"
    )

def create_llm_instances(config: LLMConfig, use_hot_reload: bool = True) -> Dict[str, Any]:
    """Create LLM and embedding instances based on configuration.

    Args:
        config: LLM configuration
        use_hot_reload: If True, reads OLLAMA_HOST from fresh config
    """
    import importlib

    instances = {}

    # Get OLLAMA_HOST (with hot-reload if enabled)
    if use_hot_reload:
        try:
            from src.services.config_service import config_service
            config_dict = config_service.load_configuration()
            ollama_host = config_dict.get("OLLAMA_HOST", os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        except ImportError:
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    else:
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Create chat LLM
    provider_info = LLM_PROVIDERS[config.provider]
    
    # Import and create LLM instance based on provider
    if config.provider == "ollama":
        from llama_index.llms.ollama import Ollama
        instances["llm"] = Ollama(
            model=config.model,
            base_url=ollama_host,
            temperature=0.1,  # Niedrige Temperatur für fokussierte Antworten
            request_timeout=300.0, # 5 minutes timeout for slower models
            context_window=8192  # Limit context window to prevent OOM on 8GB GPUs
        )
    elif config.provider == "openai":
        from llama_index.llms.openai import OpenAI
        instances["llm"] = OpenAI(
            model=config.model,
            api_key=config.api_key,
            temperature=0.1  # Niedrige Temperatur für fokussierte Antworten
        )
    elif config.provider == "gemini":
        from llama_index.llms.gemini import Gemini
        instances["llm"] = Gemini(
            model=config.model,
            api_key=config.api_key,
            temperature=0.1  # Niedrige Temperatur für fokussierte Antworten
        )
    elif config.provider == "anthropic":
        from llama_index.llms.anthropic import Anthropic
        instances["llm"] = Anthropic(
            model=config.model,
            api_key=config.api_key,
            temperature=0.1  # Niedrige Temperatur für fokussierte Antworten
        )
    elif config.provider == "openai_compatible":
        from llama_index.llms.openai import OpenAI
        # Für OpenAI-kompatible Server wie LM Studio, Llama.cpp, etc.
        instances["llm"] = OpenAI(
            model=config.model,
            api_key=config.api_key or "not-required",  # Manche OpenAI-kompatible Server benötigen keinen API-Schlüssel
            base_url=config.base_url,  # Basis-URL des OpenAI-kompatiblen Servers
            temperature=0.1
        )

    # Create embeddings
    if config.embedding_provider == "ollama":
        from llama_index.embeddings.ollama import OllamaEmbedding
        instances["embeddings"] = OllamaEmbedding(
            model_name=config.embedding_model,
            base_url=ollama_host
        )
    elif config.embedding_provider == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding
        instances["embeddings"] = OpenAIEmbedding(
            model=config.embedding_model,
            api_key=_get_api_key(config.embedding_provider)
        )
    elif config.embedding_provider == "openai_compatible":
        from llama_index.embeddings.openai import OpenAIEmbedding
        instances["embeddings"] = OpenAIEmbedding(
            model=config.embedding_model,
            api_key=config.api_key or "not-required",
            base_url=config.base_url
        )

    return instances