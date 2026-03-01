# backend/src/models/extraction_result.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ExtractionResult(BaseModel):
    """
    Standardized data model for the result of any file extraction process.
    This serves as a unified in-memory representation for all loaders and the ExtractionService.
    """
    file_hash: str = Field(
        ...,
        description="SHA256 hash of the original file content for deduplication."
    )
    mime_type: str = Field(
        ...,
        description="Detected MIME type of the file (e.g., 'application/pdf')."
    )
    extraction_engine: str = Field(
        ...,
        description="Name of the engine used for extraction (e.g., 'docling', 'email_loader')."
    )
    text_length: int = Field(
        ...,
        description="Length of the extracted text in characters."
    )
    extracted_text: Optional[str] = Field(
        None,
        description="The main textual content extracted from the document."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="A dictionary of extracted metadata (e.g., file_size, language, author, tables_count)."
    )
    quality_score: Optional[float] = Field(
        None,
        description="A score from 0.0 to 1.0 representing the perceived quality of the extraction."
    )
    error: Optional[str] = Field(
        None,
        description="If an error occurred, this field contains the error message."
    )

    class Config:
        # Example for documentation purposes in FastAPI
        schema_extra = {
            "example": {
                "file_hash": "a1b2c3d4...",
                "mime_type": "application/pdf",
                "extraction_engine": "docling",
                "text_length": 12345,
                "extracted_text": "This is the full text of the document...",
                "metadata": {
                    "file_name": "example.pdf",
                    "file_size": 1024,
                    "created_date": "2025-11-13T10:00:00Z",
                    "language": "en",
                    "page_count": 5,
                    "tables_count": 2
                },
                "quality_score": 0.95,
                "error": None
            }
        }
