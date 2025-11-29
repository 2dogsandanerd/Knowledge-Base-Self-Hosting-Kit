"""
Ingest v2 Configuration Module

Centralized configuration for RAG ingestion parameters, heuristic defaults per document type,
and feature flags for phased rollout of Agent integration.
"""

import os
from typing import Dict, List
from pydantic import BaseModel, Field


class ServiceConfig(BaseModel):
    """Configuration for external services"""
    ollama_host: str = Field(default="http://localhost:11434")
    chroma_host: str = Field(default="http://localhost:8000")
    tika_host: str = Field(default="http://localhost:9998")
    redis_url: str = Field(default="redis://localhost:6379/0")


class ChunkConfig(BaseModel):
    """Heuristic defaults for chunking per document type"""
    chunk_size: int  # Size in characters
    overlap: int  # Overlap in characters
    splitter_type: str  # "semantic", "fixed", "code", "row_based"
    embedding_model: str = "nomic-embed-text"


class IngestHeuristics(BaseModel):
    """Document type specific heuristics"""
    pdf: ChunkConfig = ChunkConfig(chunk_size=800, overlap=120, splitter_type="semantic")
    docx: ChunkConfig = ChunkConfig(chunk_size=600, overlap=100, splitter_type="semantic")
    html: ChunkConfig = ChunkConfig(chunk_size=500, overlap=80, splitter_type="semantic")
    markdown: ChunkConfig = ChunkConfig(chunk_size=400, overlap=60, splitter_type="semantic")
    csv: ChunkConfig = ChunkConfig(chunk_size=500, overlap=50, splitter_type="row_based")
    email: ChunkConfig = ChunkConfig(chunk_size=512, overlap=80, splitter_type="semantic")
    code: ChunkConfig = ChunkConfig(chunk_size=256, overlap=40, splitter_type="code")
    default: ChunkConfig = ChunkConfig(chunk_size=800, overlap=120, splitter_type="semantic")


class AgentTriggerConfig(BaseModel):
    """Conditions for triggering AI Agent"""
    confidence_threshold: float = 0.8  # confidence < 0.8 triggers agent
    text_length_min: int = 50  # extracted_text < 50 chars triggers agent
    quality_score_min: float = 0.6  # quality_score < 0.6 triggers agent
    problematic_mimes: List[str] = Field(
        default=["application/pdf", "image/tiff"]
    )


class FeatureFlags(BaseModel):
    """Feature flags for phased rollout"""
    agent_enabled: bool = False  # Phase 6: enable
    ml_classification_enabled: bool = False  # Phase 8: enable
    auto_tuning_enabled: bool = False  # Phase 9: enable


class IngestConfig(BaseModel):
    """Main configuration container for ingest_v2"""
    services: ServiceConfig = Field(default_factory=ServiceConfig)
    heuristics: IngestHeuristics = Field(default_factory=IngestHeuristics)
    agent_triggers: AgentTriggerConfig = Field(default_factory=AgentTriggerConfig)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    
    # Logging
    log_level: str = "INFO"
    
    # Database
    database_url: str = "sqlite:///./backend/data/app.db"
    
    # LLM Provider
    llm_provider: str = "ollama"  # "ollama" only for Phase 1-6
    llm_model: str = "llama3.1:8b-instruct-q4"
    
    # Embedding
    embedding_provider: str = "ollama"  # "ollama" only for Phase 1-3
    embedding_model_default: str = "nomic-embed-text"
    embedding_model_multilingual: str = "e5-large"
    embedding_model_code: str = "codebert"
    
    # Classification
    classification_engine: str = "spacy"  # Phase 1-6: spacy only
    classification_confidence_threshold: float = 0.8
    
    # Collection naming
    auto_create_collections: bool = True
    collection_naming_scheme: str = "doc_type"  # "doc_type" | "domain_type" | "auto"


def load_ingest_config_from_env() -> IngestConfig:
    """Load configuration from environment variables with fallback to defaults"""
    
    # Services
    services = ServiceConfig(
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        chroma_host=os.getenv("CHROMA_HOST", "http://localhost:8000"),
        tika_host=os.getenv("TIKA_HOST", "http://localhost:9998"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    )
    
    # Agent Triggers
    agent_triggers = AgentTriggerConfig(
        confidence_threshold=float(os.getenv("AGENT_TRIGGER_CONFIDENCE_THRESHOLD", "0.8")),
        text_length_min=int(os.getenv("AGENT_TRIGGER_TEXT_LENGTH_MIN", "50")),
        quality_score_min=float(os.getenv("AGENT_TRIGGER_QUALITY_SCORE_MIN", "0.6")),
        problematic_mimes=os.getenv(
            "AGENT_TRIGGER_PROBLEMATIC_MIMES",
            "application/pdf,image/tiff"
        ).split(",")
    )
    
    # Feature Flags
    features = FeatureFlags(
        agent_enabled=os.getenv("FEATURE_AGENT_ENABLED", "false").lower() == "true",
        ml_classification_enabled=os.getenv("FEATURE_ML_CLASSIFICATION", "false").lower() == "true",
        auto_tuning_enabled=os.getenv("FEATURE_AUTO_TUNING", "false").lower() == "true",
    )
    
    return IngestConfig(
        services=services,
        agent_triggers=agent_triggers,
        features=features,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        database_url=os.getenv("DATABASE_URL", "sqlite:///./backend/data/app.db"),
        llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
        llm_model=os.getenv("LLM_MODEL", "llama3.1:8b-instruct-q4"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "ollama"),
        embedding_model_default=os.getenv("EMBEDDING_MODEL_DEFAULT", "nomic-embed-text"),
        embedding_model_multilingual=os.getenv("EMBEDDING_MODEL_MULTILINGUAL", "e5-large"),
        embedding_model_code=os.getenv("EMBEDDING_MODEL_CODE", "codebert"),
        classification_engine=os.getenv("CLASSIFICATION_ENGINE", "spacy"),
        classification_confidence_threshold=float(
            os.getenv("CLASSIFICATION_CONFIDENCE_THRESHOLD", "0.8")
        ),
        auto_create_collections=os.getenv("AUTO_CREATE_COLLECTIONS", "true").lower() == "true",
        collection_naming_scheme=os.getenv("COLLECTION_NAMING_SCHEME", "doc_type"),
    )


# Global config instance
_ingest_config: IngestConfig | None = None


def get_ingest_config() -> IngestConfig:
    """Get or create global ingest config"""
    global _ingest_config
    if _ingest_config is None:
        _ingest_config = load_ingest_config_from_env()
    return _ingest_config


def reload_ingest_config() -> IngestConfig:
    """Reload config from environment"""
    global _ingest_config
    _ingest_config = load_ingest_config_from_env()
    return _ingest_config
