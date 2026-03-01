from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey, JSON, UniqueConstraint, Enum, BigInteger
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

# --- Enums for Multi-Tenancy ---

class SubscriptionTier(str, enum.Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class SubscriptionStatus(str, enum.Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"

class UserRole(str, enum.Enum):
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"

Base = declarative_base()

class Setting(Base):
    """Simple key‑value table for runtime configuration.
    Used by the SettingsService to store things like the selected LLM model.
    """
    __tablename__ = "settings"

    key = Column(String(64), primary_key=True, index=True)
    value = Column(Text, nullable=False)

class User(Base):
    """User accounts"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    
    # Multi-Tenancy Fields
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=True)  # Nullable for now to support migration
    role = Column(Enum(UserRole), default=UserRole.USER)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    tenant = relationship("Tenant", back_populates="users")
    email_accounts = relationship("EmailAccount", back_populates="user", cascade="all, delete-orphan")
    learning_pairs = relationship("LearningPair", back_populates="user")
    conversations = relationship("Conversation", back_populates="user")

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"

class EmailAccount(Base):
    """Stores configuration for a single email account, linked to a user."""
    __tablename__ = "email_accounts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    account_name = Column(String(100), nullable=False)
    email_address = Column(String(255), nullable=False)
    provider = Column(String(50), default="gmail") # gmail|imap
    config = Column(Text, nullable=False) # JSON string for credentials and settings
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="email_accounts")

    def __repr__(self):
        return f"<EmailAccount(id={self.id}, email_address='{self.email_address}')>"
class LearningPair(Base):
    """Draft-Sent email pairs for learning (FRESH START)"""
    __tablename__ = "learning_pairs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    thread_id = Column(String(255), nullable=False, index=True)
    draft_message_id = Column(String(255))
    draft_content = Column(Text)
    sent_message_id = Column(String(255), unique=True, nullable=True)
    sent_content = Column(Text, nullable=True)
    status = Column(String, default='DRAFT_CREATED') # DRAFT_CREATED, PAIR_COMPLETED, DELETED_NEGATIVE_EXAMPLE
    rating = Column(Float, default=0.0) # New column for rating
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="learning_pairs")

    def __repr__(self):
        return f"<LearningPair(id={self.id}, thread_id='{self.thread_id}', status='{self.status}')>"

# NOTE: QueryFeedback class is defined later (line 488) with enhanced features


# --- Multi-Tenancy Models ---

class Tenant(Base):
    """
    Tenant (Organization) model.
    Represents a customer or organization in the multi-tenant system.
    """
    __tablename__ = "tenants"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)  # Internal name (slug)
    display_name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    
    # Subscription
    subscription_tier = Column(Enum(SubscriptionTier), default=SubscriptionTier.FREE)
    subscription_status = Column(Enum(SubscriptionStatus), default=SubscriptionStatus.ACTIVE)
    
    # Quotas & Limits
    max_collections = Column(Integer, default=5)
    max_queries_per_day = Column(Integer, default=1000)
    max_api_keys = Column(Integer, default=5)
    
    # Features (JSON for flexibility)
    features = Column(JSON, default=dict)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    users = relationship("User", back_populates="tenant")
    api_keys = relationship("APIKey", back_populates="tenant")
    usage_records = relationship("TenantUsage", back_populates="tenant")
    collection_acls = relationship("CollectionACL", back_populates="tenant")


class CollectionACL(Base):
    """
    Access Control List for Collections.
    Defines which roles can access which collection within a tenant.
    """
    __tablename__ = "collection_acl"

    id = Column(Integer, primary_key=True, index=True)
    collection_name = Column(String, nullable=False)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False)
    
    # Access Control
    allowed_roles = Column(JSON, default=["admin", "user"])  # List of allowed roles
    is_public = Column(Boolean, default=False)  # If true, accessible by all tenant users
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    tenant = relationship("Tenant", back_populates="collection_acls")

    __table_args__ = (
        UniqueConstraint('collection_name', 'tenant_id', name='uix_collection_tenant'),
    )


class APIKey(Base):
    """
    API Key for programmatic access.
    Keys are hashed for security.
    """
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False)
    
    # Security
    key_hash = Column(String, unique=True, index=True, nullable=False)  # SHA-256 hash
    key_prefix = Column(String(20), nullable=False)  # First few chars for display (e.g. "sk-abc...")
    name = Column(String, nullable=False)
    
    # Permissions
    scopes = Column(JSON, default=["read"])
    
    # Limits
    rate_limit = Column(Integer, default=100)  # Requests per minute
    
    # Status
    is_active = Column(Boolean, default=True)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    tenant = relationship("Tenant", back_populates="api_keys")


class TenantUsage(Base):
    """
    Daily usage tracking for billing and quotas.
    """
    __tablename__ = "tenant_usage"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False)
    
    date = Column(DateTime, nullable=False)  # Date only
    
    # Metrics
    query_count = Column(Integer, default=0)
    document_count = Column(Integer, default=0)
    token_usage = Column(BigInteger, default=0)
    cost_usd = Column(Float, default=0.0)
    
    # Relationships
    tenant = relationship("Tenant", back_populates="usage_records")

    __table_args__ = (
        UniqueConstraint('tenant_id', 'date', name='uix_tenant_date'),
    )

class Conversation(Base):
    """Conversation history with RAG context (was in-memory in Streamlit)"""
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True)  # UUID
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Email data
    email_data = Column(Text, nullable=False)  # JSON string

    # Generated response
    generated_response = Column(Text, nullable=False)
    rag_context_used = Column(Text)
    model_used = Column(String(100))

    # Feedback
    feedback_score = Column(Integer, nullable=True)
    feedback_text = Column(Text, nullable=True)
    quality_score = Column(Float, nullable=True)

    # Learning status
    learned = Column(Boolean, default=False)
    draft_id = Column(String(255), nullable=True)
    draft_status = Column(String(50), nullable=True)
    sent_mail_content = Column(Text, nullable=True)
    user_action = Column(String(50), nullable=True)

    # Timestamps
    timestamp = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="conversations")

    def __repr__(self):
        return f"<Conversation(id='{self.id}', user_id={self.user_id})>"


class CollectionIndexConfig(Base):
    """
    Collection-Konfiguration für Index-Strategien.

    Speichert PRO Collection:
    - Welche Index-Strategie (Vector, Pandas, SQL, Hybrid)
    - Embedding-Konfiguration (für Vector/Hybrid)
    - Structured Data Settings (für Pandas/SQL)
    - Query-Routing Hints (Priority, Enabled)
    """

    __tablename__ = 'collection_index_configs'

    id = Column(Integer, primary_key=True)
    collection_name = Column(String(255), unique=True, nullable=False, index=True)

    # Index-Strategie
    index_strategy = Column(String(50), nullable=False)
    # 'vector', 'hybrid', 'pandas_agent', 'sql_agent'

    data_type = Column(String(50), nullable=False)
    # 'unstructured_text', 'structured_table', 'code', 'email', 'database'

    # Vector-Index Settings
    embedding_model = Column(String(255), nullable=True)
    embedding_provider = Column(String(50), nullable=True)
    embedding_dimensions = Column(Integer, nullable=True)
    chunk_size = Column(Integer, default=500)
    chunk_overlap = Column(Integer, default=50)

    # Structured Data Settings
    source_file_path = Column(String(512), nullable=True)
    connection_string = Column(Text, nullable=True)
    table_schema = Column(JSON, nullable=True)

    # Query-Routing
    priority = Column(String(20), default='medium')  # high, medium, low
    enabled_for_drafts = Column(Boolean, default=True, index=True)
    weight = Column(Float, default=1.0)

    # Metadata
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Statistics
    last_used = Column(DateTime(timezone=True), nullable=True)
    usage_count = Column(Integer, default=0)
    avg_relevance = Column(Float, default=0.0)

    def to_dict(self):
        """Konvertiert zu Dict für API-Response"""
        return {
            "id": self.id,
            "collection_name": self.collection_name,
            "index_strategy": self.index_strategy,
            "data_type": self.data_type,
            "embedding_model": self.embedding_model,
            "embedding_provider": self.embedding_provider,
            "embedding_dimensions": self.embedding_dimensions,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "source_file_path": self.source_file_path,
            "connection_string": self.connection_string,
            "table_schema": self.table_schema,
            "priority": self.priority,
            "enabled_for_drafts": self.enabled_for_drafts,
            "weight": self.weight,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "usage_count": self.usage_count,
            "avg_relevance": self.avg_relevance
        }

    def __repr__(self):
        return f"<CollectionIndexConfig(id={self.id}, collection_name='{self.collection_name}', strategy='{self.index_strategy}')>"


# --- New Tables for Extraction Pipeline ---

class ExtractionMetadata(Base):
    """
    Stores metadata for each processed file, identified by its unique hash.
    This table holds information that is independent of the extraction content itself.
    """
    __tablename__ = "extraction_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_hash = Column(String(64), nullable=False, index=True)  # Removed unique=True to allow versioning
    file_path = Column(String(1024), nullable=False)
    file_name = Column(String(255), nullable=True)
    file_size = Column(Integer, nullable=True)
    mime_type = Column(String(100), nullable=True, index=True)
    created_date = Column(DateTime, nullable=True)
    modified_date = Column(DateTime, nullable=True)
    language = Column(String(10), nullable=True, index=True)
    ocr_confidence = Column(Float, nullable=True)
    structure_score = Column(Float, nullable=True)
    extra = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Phase 3: Versioning and deduplication fields
    version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True, index=True)
    superseded_by_id = Column(Integer, ForeignKey("extraction_metadata.id"), nullable=True)
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Ensure only one active version per file_hash (will be implemented via partial index in Alembic)
    __table_args__ = (UniqueConstraint('file_hash', 'version', name='uq_file_hash_version'),)

    results = relationship("ExtractionResultDB", back_populates="extraction_metadata", cascade="all, delete-orphan")
    superseded_by = relationship("ExtractionMetadata", remote_side=[id])

    def __repr__(self):
        return f"<ExtractionMetadata(id={self.id}, file_hash='{self.file_hash[:8]}...', file_name='{self.file_name}', version={self.version}, is_active={self.is_active})>"


class DuplicationAuditLogAction(str, enum.Enum):
    """Enum for duplication audit log action types"""
    SKIPPED = "SKIPPED"
    REPLACED = "REPLACED"
    VERSIONED = "VERSIONED"
    NOTIFIED = "NOTIFIED"


class DuplicationAuditLog(Base):
    """
    Logs all duplication detection and handling events.
    """
    __tablename__ = "duplication_audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    original_metadata_id = Column(Integer, ForeignKey("extraction_metadata.id"), nullable=False, index=True)
    duplicate_file_path = Column(String(1024), nullable=False)
    duplicate_file_hash = Column(String(64), nullable=False, index=True)
    detection_timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    action_taken = Column(Enum(DuplicationAuditLogAction), nullable=False, index=True)
    user_notified = Column(Boolean, default=False)
    metadata_diff = Column(JSON, nullable=True)  # Stores differences between original and duplicate

    # Relationships
    original_metadata = relationship("ExtractionMetadata", foreign_keys=[original_metadata_id])

    def __repr__(self):
        return f"<DuplicationAuditLog(id={self.id}, original_id={self.original_metadata_id}, action='{self.action_taken}', hash='{self.duplicate_file_hash[:8]}...')>"


class BackupType(str, enum.Enum):
    """Enum for backup types"""
    FULL = "FULL"
    INCREMENTAL = "INCREMENTAL"


class BackupStatus(str, enum.Enum):
    """Enum for backup statuses"""
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ExtractionBackup(Base):
    """
    Stores backup information for extraction metadata and results.
    """
    __tablename__ = "extraction_backup"

    id = Column(Integer, primary_key=True, autoincrement=True)
    backup_timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    backup_type = Column(Enum(BackupType), nullable=False)
    records_count = Column(Integer, nullable=False)
    backup_file_path = Column(String(1024), nullable=False)
    checksum = Column(String(64), nullable=False)  # SHA256 checksum
    compression = Column(String(10), default="gzip")  # gzip, none
    status = Column(Enum(BackupStatus), default=BackupStatus.IN_PROGRESS, index=True)

    def __repr__(self):
        return f"<ExtractionBackup(id={self.id}, type='{self.backup_type}', status='{self.status}', timestamp='{self.backup_timestamp}')>"


class ExtractionResultDB(Base):
    """
    Stores the actual extracted content and quality metrics for a given file.
    There can be multiple results for a single metadata entry if re-extraction occurs.
    Name is ExtractionResultDB to avoid clash with Pydantic model.
    """
    __tablename__ = "extraction_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    metadata_id = Column(Integer, ForeignKey("extraction_metadata.id"), nullable=False, index=True)
    
    extracted_text = Column(Text, nullable=True)
    extraction_engine = Column(String(100), nullable=False)
    text_length = Column(Integer, nullable=False)
    quality_score = Column(Float, nullable=True)
    
    error_code = Column(String(50), nullable=True)
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    extraction_metadata = relationship("ExtractionMetadata", back_populates="results")

    def __repr__(self):
        return f"<ExtractionResultDB(id={self.id}, metadata_id={self.metadata_id}, engine='{self.extraction_engine}')>"

class QueryFeedback(Base):
    """User feedback on query responses for quality monitoring."""
    __tablename__ = "query_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Query and response
    query_text = Column(Text, nullable=False)
    response_text = Column(Text, nullable=False)
    
    # User feedback
    helpful = Column(Boolean, nullable=True)  # Thumbs up/down
    rating = Column(Integer, nullable=True)  # 1-5 stars (optional)
    comment = Column(Text, nullable=True)  # Free text feedback
    
    # Auto-evaluation scores (if available)
    faithfulness = Column(Float, nullable=True)
    answer_relevancy = Column(Float, nullable=True)
    overall_score = Column(Float, nullable=True)
    
    # Metadata
    collection_names = Column(JSON, nullable=True)  # Which collections were queried
    context_count = Column(Integer, nullable=True)  # Number of context chunks
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationship
    user = relationship("User")

    def __repr__(self):
        return f"<QueryFeedback(id={self.id}, query_id='{self.query_id}', helpful={self.helpful})>"
