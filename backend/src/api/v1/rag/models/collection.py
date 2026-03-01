"""Collection management models."""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
import re


class CollectionCreate(BaseModel):
    """Request model for creating a collection."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=63,
        description="Collection name (ChromaDB compatible: 3-63 chars, alphanumeric + - _ .)"
    )
    description: Optional[str] = Field(
        None,
        max_length=500,
        description="Optional collection description"
    )

    @field_validator("name")
    @classmethod
    def validate_collection_name(cls, v: str) -> str:
        """
        Validate collection name meets ChromaDB requirements.

        Rules:
        - 3-63 characters
        - Starts and ends with alphanumeric
        - Contains only: a-z, A-Z, 0-9, -, _, .
        - Cannot contain consecutive dots (..)
        """
        if len(v) < 3:
            raise ValueError("Collection name must be at least 3 characters")
        if len(v) > 63:
            raise ValueError("Collection name must not exceed 63 characters")

        # ChromaDB naming rules
        pattern = r'^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]$'
        if not re.match(pattern, v):
            raise ValueError(
                "Collection name must start/end with alphanumeric and "
                "contain only: a-z, A-Z, 0-9, -, _, ."
            )

        if ".." in v:
            raise ValueError("Collection name cannot contain consecutive dots")

        return v.lower()  # Normalize to lowercase


class CollectionResponse(BaseModel):
    """Response model for collection info."""
    name: str
    document_count: int = 0
    metadata: Optional[Dict[str, Any]] = None


class CollectionListResponse(BaseModel):
    """Response model for listing collections."""
    collections: List[CollectionResponse]
    total: int
