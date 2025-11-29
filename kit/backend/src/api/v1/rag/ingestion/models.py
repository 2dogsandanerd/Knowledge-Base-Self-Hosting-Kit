"""
Models for RAG ingestion operations.
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from src.core.feature_limits import Edition


class ScanFolderRequest(BaseModel):
    folder_path: str
    recursive: bool = True
    max_depth: int = 10
    edition: Edition = Edition.DEVELOPER  # For feature limits


class ScanFolderResponse(BaseModel):
    files: List[Dict[str, Any]]
    total_files: int
    total_size: int
    summary: Dict[str, int]
    edition_limitations: Optional[Dict[str, Any]] = None


class AnalyzeFilesRequest(BaseModel):
    files: List[Dict[str, Any]]  # Contains 'path' and 'preview' fields
    edition: Edition = Edition.DEVELOPER  # For feature limits


class AnalyzeFilesResponse(BaseModel):
    analyses: List[Dict[str, Any]]
    edition_limitations: Optional[Dict[str, Any]] = None


class IngestBatchRequest(BaseModel):
    assignments: List[Dict[str, str]]  # {'filename': 'path', 'collection': 'collection_name'}
    chunk_size: int = 500
    chunk_overlap: int = 50
    async_mode: bool = True
    use_parent_child: bool = False
    edition: Edition = Edition.DEVELOPER  # For feature limits


class IngestBatchResponse(BaseModel):
    success: bool
    processed_files: int
    failed_files: int
    details: Dict[str, Any]
    edition_limitations: Optional[Dict[str, Any]] = None