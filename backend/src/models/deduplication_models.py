from pydantic import BaseModel
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime


class DeduplicationPolicy(str, Enum):
    """Policy enum for handling duplicate files"""
    SKIP = "SKIP"
    VERSION = "VERSION"  # Create a new version
    REPLACE_IF_NEWER = "REPLACE_IF_NEWER"
    NOTIFY_ONLY = "NOTIFY_ONLY"


class DuplicationResult(BaseModel):
    """Result of duplication check and handling"""
    is_duplicate: bool
    action_taken: str
    original_metadata_id: Optional[int] = None
    new_metadata_id: Optional[int] = None
    version: Optional[int] = None
    message: str = ""


class DuplicateReport(BaseModel):
    """Report about duplicate detection"""
    original_file_path: str
    duplicate_file_path: str
    file_hash: str
    detection_timestamp: datetime
    action_taken: str
    metadata_diff: Optional[Dict[str, Any]] = None


class MetadataDiff(BaseModel):
    """Difference between two file metadata sets"""
    old_values: Dict[str, Any]
    new_values: Dict[str, Any]
    changed_fields: List[str]