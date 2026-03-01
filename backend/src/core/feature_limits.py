"""
Feature limits for different editions - Community Edition Reference

This is a simplified reference showing the edition-based architecture.
The full production version includes additional enterprise features and
dynamic license validation.

For production deployment with full feature set, contact sales.
"""

from typing import Dict, Any, Optional
from enum import Enum
import os


class Edition(Enum):
    """Application edition tiers."""
    COMMUNITY = "community"      # Open source, basic features
    DEVELOPER = "community"      # Alias for COMMUNITY (backwards compatibility)
    PROFESSIONAL = "professional"  # Extended features (contact sales)
    ENTERPRISE = "enterprise"     # Full features + support (contact sales)


class FeatureLimits:
    """
    Feature limits based on edition tier.

    Community Edition: Unlimited RAG functionality for self-hosting
    Professional/Enterprise: Contact sales for managed features & support

    NOTE: This is a reference implementation showing the architecture.
    Production versions include:
    - Dynamic license validation
    - Usage-based metering
    - Custom feature bundles
    - SSO and RBAC integration
    """

    # Community Edition - Self-Hosting Kit (No Limits!)
    COMMUNITY_LIMITS = {
        # Collection limits - UNLIMITED for self-hosting
        "max_collections": -1,  # -1 = unlimited
        "max_documents_per_collection": -1,  # -1 = unlimited
        "max_total_documents": -1,  # -1 = unlimited

        # File format support - All formats available (documents + code)
        "allowed_file_formats": [
            # Documents
            ".pdf", ".txt", ".md", ".docx", ".html", ".pptx", ".xlsx", ".csv", ".json", ".xml",
            # Code files
            ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h", ".hpp",
            ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".cs", ".sh", ".bash", ".yaml", ".yml",
            ".toml", ".ini", ".conf", ".sql", ".r", ".scala", ".pl", ".lua", ".vim"
        ],
        "max_file_size_mb": 100,  # Reasonable limit for stability

        # Feature flags - All features enabled
        "enable_advanced_rag": True,
        "enable_hybrid_search": True,
        "enable_reranking": True,
        "enable_multi_collection": True,
        "enable_batch_ingestion": True,
        "enable_custom_embeddings": True,

        # Performance limits - Generous defaults
        "max_concurrent_queries": 100,
        "query_timeout_seconds": 300,

        # API rate limits - No artificial limits for self-hosting
        "api_query_limit": -1,  # -1 = unlimited

        # Edition metadata
        "edition_name": "Community (Self-Hosted)",
        "support_level": "community",
        "upgrade_url": "https://github.com/yourusername/self-hosting-kit"
    }

    # Note: Professional and Enterprise limits are defined in the
    # production version with license key validation.
    # Contact sales for details: [your-contact-email]

    @classmethod
    def get_limits(cls, edition: Edition = None) -> Dict[str, Any]:
        """
        Get feature limits for specified edition.

        Community Edition: Returns public limits
        Professional/Enterprise: Requires license validation (production only)

        Args:
            edition: Edition tier (defaults to Community)

        Returns:
            Dict of feature limits
        """
        if edition is None or edition == Edition.COMMUNITY:
            return cls.COMMUNITY_LIMITS.copy()

        # Professional/Enterprise require license validation
        # This is a stub - production version validates licenses
        return {
            **cls.COMMUNITY_LIMITS,
            "license_required": True,
            "contact": "Contact sales for Professional/Enterprise features"
        }

    @classmethod
    def get_limit_value(cls, feature: str, edition: Edition = None) -> Any:
        """
        Get the limit value for a specific feature.

        Args:
            feature: Feature name (e.g., 'max_collections')
            edition: Edition tier

        Returns:
            The limit value for the feature
        """
        limits = cls.get_limits(edition)
        return limits.get(feature)

    @classmethod
    def check_limit(cls, feature: str, current_value: int, edition: Edition = None) -> bool:
        """
        Check if current value is within limits for edition.

        Args:
            feature: Feature name (e.g., 'max_collections')
            current_value: Current usage value
            edition: Edition tier

        Returns:
            True if within limits, False otherwise
        """
        limits = cls.get_limits(edition)
        max_value = limits.get(feature, 0)

        # -1 means unlimited (Enterprise feature)
        if max_value == -1:
            return True

        return current_value < max_value

    @classmethod
    def check_collection_limit(cls, current_count: int, edition: Edition = None) -> bool:
        """
        Check if current collection count is within limits.

        Args:
            current_count: Current number of collections
            edition: Edition tier

        Returns:
            True if within limits, False otherwise
        """
        return cls.check_limit('max_collections', current_count, edition)

    @classmethod
    def check_document_limit(cls, current_count: int, edition: Edition = None) -> bool:
        """
        Check if current document count in a collection is within limits.

        Args:
            current_count: Current number of documents
            edition: Edition tier

        Returns:
            True if within limits, False otherwise
        """
        return cls.check_limit('max_documents_per_collection', current_count, edition)

    @classmethod
    def is_feature_enabled(cls, feature: str, edition: Edition = None) -> bool:
        """
        Check if a feature is enabled for edition.

        Args:
            feature: Feature flag name (e.g., 'enable_advanced_rag')
            edition: Edition tier

        Returns:
            True if feature is enabled
        """
        limits = cls.get_limits(edition)
        return limits.get(feature, False)

    @classmethod
    def get_edition_from_env(cls) -> Edition:
        """
        Get edition from environment variable.

        Community Edition is default for self-hosted deployments.
        Production versions support license key validation.

        Returns:
            Edition enum value
        """
        edition_str = os.getenv("EDITION", "community").lower()

        try:
            return Edition(edition_str)
        except ValueError:
            # Default to community for invalid values
            return Edition.COMMUNITY


# Convenience function for common use case
def get_current_edition() -> Edition:
    """Get the currently configured edition."""
    return FeatureLimits.get_edition_from_env()
