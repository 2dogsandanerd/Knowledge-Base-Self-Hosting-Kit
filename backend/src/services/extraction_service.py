# backend/src/services/extraction_service.py
import asyncio
import hashlib
import logging
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core import metadata_extractor, quality_scorer
from src.core.exceptions import ExtractionError
from src.database.models import ExtractionMetadata, ExtractionResultDB
from src.models.extraction_result import ExtractionResult as ExtractionResultModel
from src.models.deduplication_models import DeduplicationPolicy
from src.services.loaders import code_loader, email_loader, image_loader
from src.services.deduplication_service import DeduplicationService
# Assuming docling_loader and docling_service provide a unified interface
from src.core.docling_loader import DoclingLoader 

logger = logging.getLogger(__name__)


class ExtractionService:
    """
    Orchestrates the end-to-end process of file extraction.

    This service handles file reading, MIME type detection, metadata extraction,
    content extraction using various loaders (Docling, Email, Text, Image),
    quality scoring, and persistence of results to the database.

    Supported Formats:
    - PDFs, DOCX, HTML, Markdown, CSV (via Docling)
    - Plain text files (TXT)
    - Email files (EML, MSG)
    - Image files (metadata only)

    Return Structure:
    - ExtractionResult (Pydantic model)

    Error Codes:
    - ExtractionError (for file not found, unsupported types, extraction failures, database errors)
    """

    def __init__(self, db_session: AsyncSession, default_policy: DeduplicationPolicy = DeduplicationPolicy.VERSION):
        self.db = db_session
        self.docling_loader = DoclingLoader()
        self.deduplication_service = DeduplicationService(db_session)
        self.default_policy = default_policy

    async def extract_document(
        self, file_path: str, original_filename: Optional[str] = None, 
        policy: Optional[DeduplicationPolicy] = None
    ) -> ExtractionResultModel:
        """
        Main method to process a single file, extract its content and metadata,
        and persist the results to the database.

        Args:
            file_path: The absolute path to the file to be processed.
            original_filename: The original name of the file, if different from the path's basename.
            policy: Optional deduplication policy to use (defaults to service default).

        Returns:
            An ExtractionResult Pydantic model with the complete extraction data.

        Raises:
            ExtractionError: If the file is not found, unsupported, or an extraction/database error occurs.

        Example:
            >>> from src.database.database import AsyncSessionLocal
            >>> from src.services.extraction_service import ExtractionService
            >>> from src.models.deduplication_models import DeduplicationPolicy
            >>>
            >>> async def process_my_file(path: str):
            >>>     async with AsyncSessionLocal() as db:
            >>>         service = ExtractionService(db)
            >>>         try:
            >>>             result = await service.extract_document(path, policy=DeduplicationPolicy.VERSION)
            >>>             print(f"Extracted text length: {result.text_length}, Quality: {result.quality_score}")
            >>>         except ExtractionError as e:
            >>>             print(f"Extraction failed: {e.message}")
            >>>
            >>> # await process_my_file("/path/to/your/document.pdf")
        """
        start_time = time.time()
        logger.info(f"Starting extraction for file: {file_path}")
        if not os.path.exists(file_path):
            raise ExtractionError("File not found", file_path=file_path)

        # 1. Calculate SHA256 hash
        try:
            with open(file_path, "rb") as f:
                file_content = f.read()
                file_hash = hashlib.sha256(file_content).hexdigest()
        except IOError as e:
            raise ExtractionError(f"Could not read file: {e}", file_path=file_path)

        # 2. Use DeduplicationService to check and handle duplicates according to policy
        policy = policy or self.default_policy
        duplication_result = await self.deduplication_service.check_and_handle_duplicate(
            file_hash=file_hash,
            file_path=file_path,
            metadata={"file_path": file_path},  # We'll update this with full metadata later
            policy=policy
        )
        
        # If it's a duplicate and the policy action is to skip, raise an error
        if duplication_result.is_duplicate and duplication_result.action_taken in ["SKIPPED", "NOTIFIED"]:
            logger.info(f"Duplicate file handled with action '{duplication_result.action_taken}'. File: {file_path}")
            raise ExtractionError(f"Duplicate file handled with action '{duplication_result.action_taken}'", 
                                file_path=file_path, details={"file_hash": file_hash, "action": duplication_result.action_taken})
        
        # If it's a duplicate and was versioned or replaced, we might still need to process the content
        # In the VERSION and REPLACE_IF_NEWER policies, new content is created, so continue processing

        # 2. Basic metadata extraction
        mime_type = metadata_extractor.detect_mime_type(file_path)
        fs_meta = metadata_extractor.get_file_system_metadata(file_path)
        if original_filename:
            fs_meta["file_name"] = original_filename

        # 3. Loader Routing and Raw Extraction
        extracted_text = None
        extraction_engine = "unsupported"
        loader_artifacts = {}

        # Prioritize Docling for formats it supports well
        docling_formats = [
            "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword", "text/html", "text/markdown", "text/csv"
        ]

        if mime_type in docling_formats:
            extraction_engine = "docling"
            try:
                # Assuming docling_loader has an async method `extract`
                loop = asyncio.get_running_loop()
                docling_result = await loop.run_in_executor(
                    None, self.docling_loader.extract, file_path
                )
                extracted_text = docling_result.get("text")
                # TODO: Get structured artifacts from docling result
                loader_artifacts = {"tables_count": 0, "images_count": 0} 
            except Exception as e:
                raise ExtractionError(f"Docling extraction failed: {e}", file_path)

        elif mime_type and mime_type.startswith("text/"):
            extraction_engine = "text_reader"
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    extracted_text = f.read()
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decoding failed for {file_path}. Trying latin-1.")
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        extracted_text = f.read()
                except Exception as e:
                    raise ExtractionError(f"Could not read text file with fallback encoding: {e}", file_path)
            except Exception as e:
                raise ExtractionError(f"Failed to read text file: {e}", file_path)

        elif mime_type in ["message/rfc822", "application/vnd.ms-outlook"]:
            extraction_engine = "email_loader"
            try:
                loader = email_loader.EmailLoader(file_path)
                docs = loader.load()
                if docs:
                    doc = docs[0]
                    extracted_text = doc.page_content
                    # Merge email-specific metadata
                    fs_meta.update(doc.metadata)
                else:
                    extracted_text = ""
            except Exception as e:
                raise ExtractionError(f"Email parsing failed: {e}", file_path)

        elif mime_type and mime_type.startswith("image/"):
            extraction_engine = "image_loader"
            # Image loader only extracts metadata, no text
            image_meta = image_loader.extract_image_metadata(file_path)
            if image_meta:
                fs_meta.update(image_meta)
        else:
            raise ExtractionError(f"Unsupported MIME type: {mime_type}", file_path)

        # 4. Post-Extraction Metadata and Quality Scoring
        language = metadata_extractor.detect_language(extracted_text)
        structure_score = quality_scorer.estimate_structure_score(loader_artifacts)
        
        final_quality_score = quality_scorer.score_extraction(
            extracted_text=extracted_text,
            language=language,
            ocr_confidence=loader_artifacts.get("ocr_confidence"), # Assuming loader provides this
            structure_score=structure_score
        )

        # 5. Assemble the final metadata dictionary
        full_metadata = {
            **fs_meta,
            "language": language,
            "structure_score": structure_score,
            "mime_type": mime_type,
            **loader_artifacts
        }

        # --- Performance Metrics ---
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        file_size_bytes = full_metadata.get("file_size", 0)
        bytes_per_second = (file_size_bytes / (duration_ms / 1000)) if duration_ms > 0 else 0

        performance_metrics = {
            "duration_ms": round(duration_ms, 2),
            "bytes_per_second": round(bytes_per_second, 2)
        }
        full_metadata.update(performance_metrics)
        logger.info(f"Extraction for {file_path} completed in {duration_ms:.2f}ms. "
                    f"Engine: {extraction_engine}, Quality: {final_quality_score:.2f}")

        # 6. Persist to Database
        db_meta = ExtractionMetadata(
            file_hash=file_hash,
            file_path=file_path,
            file_name=full_metadata.get("file_name"),
            file_size=file_size_bytes,
            mime_type=mime_type,
            created_date=datetime.fromisoformat(full_metadata.get("created_date")),
            modified_date=datetime.fromisoformat(full_metadata.get("modified_date")),
            language=language,
            structure_score=structure_score,
            extra={**performance_metrics, **{k: v for k, v in full_metadata.items() if k not in [
                "file_name", "file_size", "created_date", "modified_date", "language", "structure_score", "mime_type"
            ]}}
        )
        
        db_result = ExtractionResultDB(
            extracted_text=extracted_text,
            extraction_engine=extraction_engine,
            text_length=len(extracted_text) if extracted_text else 0,
            quality_score=final_quality_score,
            metadata=db_meta # Associate with the metadata object
        )

        try:
            self.db.add(db_meta)
            await self.db.commit()
            await self.db.refresh(db_meta)
            await self.db.refresh(db_result)
            logger.info(f"Successfully persisted extraction for {file_path} (Metadata ID: {db_meta.id})")
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Database error while saving extraction for {file_path}: {e}")
            raise ExtractionError(f"Database commit failed: {e}", file_path)

        # 7. Build final Pydantic model for return
        return ExtractionResultModel(
            file_hash=file_hash,
            mime_type=mime_type,
            extraction_engine=extraction_engine,
            text_length=len(extracted_text) if extracted_text else 0,
            extracted_text=extracted_text,
            metadata=full_metadata,
            quality_score=final_quality_score,
            error=None
        )

    async def _get_metadata_by_hash(self, file_hash: str) -> Optional[ExtractionMetadata]:
        """Checks the database for an existing record by file hash."""
        stmt = select(ExtractionMetadata).where(ExtractionMetadata.file_hash == file_hash)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
