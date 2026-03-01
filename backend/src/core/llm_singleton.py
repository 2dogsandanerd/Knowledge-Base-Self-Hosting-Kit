"""LLM Singleton implementation for memory optimization.

This module implements the Singleton pattern for LLM instances to prevent
duplicate creation and manage memory efficiently. The LLM and embedding models
are resource-intensive, so creating multiple instances would cause performance issues.
"""

import threading
import dataclasses
from typing import Dict, Any, Optional
from loguru import logger

from src.core.config import get_config, create_llm_instances

class LLMSingleton:
    """Singleton wrapper for LLM instances to prevent duplicate creation."""
    
    _instance = None
    _lock = threading.Lock()  # Thread-safety lock for singleton creation
    _llm_instances = None    # Cached LLM instances
    _config = None           # Cached config to avoid repeated loading
    
    def __new__(cls):
        """Thread-safe singleton creation with double-checked locking pattern."""
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking pattern to ensure thread safety
                if cls._instance is None:
                    cls._instance = super(LLMSingleton, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize LLM instances once.
        
        This method loads the configuration and creates LLM instances only once,
        preventing multiple resource-intensive model loads.
        """
        if self._llm_instances is None:
            logger.info("Initializing singleton LLM instances...")
            # Get configuration to determine which LLM provider and models to use
            self._config = get_config()
            # Create the actual LLM instances based on the configuration
            self._llm_instances = create_llm_instances(self._config)
            logger.info(f"Singleton LLM instances initialized for provider: {self._config.provider}")
    
    @property
    def llm_instances(self) -> Dict[str, Any]:
        """Get the singleton LLM instances.
        
        Returns:
            Dictionary containing all LLM instances (LLM, embeddings, etc.)
        """
        return self._llm_instances
    
    @property
    def llm(self):
        """Get the main LLM instance.
        
        Returns:
            The primary LLM instance for text generation
        """
        return self._llm_instances.get("llm")
    
    @property
    def embeddings(self):
        """Get the embeddings instance.
        
        Returns:
            The embedding model instance for vector generation
        """
        return self._llm_instances.get("embeddings")
    
    @property
    def config(self):
        """Get the config.
        
        Returns:
            The configuration object used to initialize the LLM instances
        """
        return self._config

# Global singleton instance to provide module-level access
_llm_singleton = None

def get_llm_singleton() -> LLMSingleton:
    """Get the LLM singleton instance.
    
    This function provides access to the singleton LLM manager instance,
    creating it if it doesn't exist yet.
    
    Returns:
        The global LLM singleton instance
    """
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = LLMSingleton()
    return _llm_singleton

def get_llm_instances() -> Dict[str, Any]:
    """Get the singleton LLM instances.
    
    Convenience function to access all LLM instances directly.
    
    Returns:
        Dictionary containing all LLM instances
    """
    return get_llm_singleton().llm_instances

def get_llm(model: Optional[str] = None, temperature: Optional[float] = None):
    """
    Get the main LLM instance.

    If model or temperature are provided, it creates a new instance with these
    parameters, bypassing the singleton for customized requests. Otherwise, it
    returns the default singleton instance.

    Args:
        model (Optional[str]): The specific model to use (e.g., "llama3:latest").
        temperature (Optional[float]): The temperature for the LLM.

    Returns:
        The main LLM instance for text generation.
    """
    singleton = get_llm_singleton()

    # If no custom parameters are given, return the default singleton instance
    if model is None and temperature is None:
        return singleton.llm

    # If only temperature is provided (no custom model), use singleton model
    if model is None:
        model = singleton.config.model

    # If custom parameters are provided, create a new instance
    logger.info(f"Creating a new LLM instance with custom parameters: model={model}, temp={temperature}")

    # Create a modified copy of the config using dataclasses.replace()
    config_updates = {"model": model}
    # Note: LLMConfig doesn't have a temperature field, so we ignore it for now
    # The temperature parameter is passed to the LLM during query execution

    config = dataclasses.replace(singleton.config, **config_updates)

    # create_llm_instances returns a dictionary, we need the 'llm' part
    custom_instances = create_llm_instances(config)
    return custom_instances.get("llm")

def get_config_singleton():
    """Get the singleton config.
    
    Convenience function to access the configuration used by the LLM instances.
    
    Returns:
        The configuration object used to initialize the LLM instances
    """
    return get_llm_singleton().config


class LLMServiceWrapper:
    """Wrapper to provide compatibility with systems expecting get_llm_instance() method."""
    
    def __init__(self, llm_instance=None):
        """
        Initialize the wrapper.
        
        Args:
            llm_instance: LLM instance to wrap. If None, gets from singleton.
        """
        if llm_instance is None:
            self._llm_instance = get_llm()
        else:
            self._llm_instance = llm_instance
    
    def get_llm_instance(self):
        """Get the LLM instance - compatibility method for existing code."""
        return self._llm_instance