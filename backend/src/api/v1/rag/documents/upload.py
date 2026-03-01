"""
Document Upload Endpoints.

Handles file uploads and ingestion into ChromaDB collections using the Central Docling Service.
Features:
- Multi-file upload with validation
- Centralized Docling processing (Repair -> Convert -> Refine)
- Direct Markdown chunking
"""

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    File,
    UploadFile,
    Form,
    BackgroundTasks,
    status
)
from fastapi.responses import JSONResponse
from typing import List, Optional
import logging
import os
import shutil
from pathlib import Path
import asyncio
import tempfile
import uuid
from asyncio import Semaphore

from src.api.v1.dependencies import get_rag_client
from src.services.auth_service import get_current_user
from src.database.models import User
from src.services.ingestion_task_manager import ingestion_task_manager
from src.services.docling_service import docling_service
from src.services.progress_service import progress_manager
from src.core.exceptions import (
    CollectionNotFoundError,
    ValidationError,
    InvalidFileTypeError,
    ChromaDBError,
    IngestionError,
    ServiceUnavailableError
)

# FIX BUG #7: File type validation based on magic bytes
ALLOWED_MIME_TYPES = {
    '.pdf': [b'%PDF'],  # PDF magic bytes
    '.docx': [b'PK\x03\x04'],  # ZIP archive (DOCX is a ZIP)
    '.xlsx': [b'PK\x03\x04'],  # ZIP archive (XLSX is a ZIP)
    '.pptx': [b'PK\x03\x04'],  # ZIP archive (PPTX is a ZIP)
    '.md': [b'#', b'##', b'', b'\n'],  # Markdown can start with anything (text)
    '.html': [b'<!DOCTYPE', b'<html', b'<?xml'],  # HTML/XML
    '.csv': [b'', b'\n', b',']  # CSV can start with anything (text)
}

logger = logging.getLogger(__name__)
router = APIRouter()

# Configuration
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

# FIX ISSUE #1: Limit concurrent uploads to prevent resource exhaustion
# This prevents DoS attacks where a user starts 100+ parallel uploads
MAX_CONCURRENT_UPLOADS = int(os.getenv("MAX_CONCURRENT_UPLOADS", "5"))
UPLOAD_SEMAPHORE = Semaphore(MAX_CONCURRENT_UPLOADS)


def validate_file_type(file_path: str, claimed_extension: str) -> bool:
    """Validate file type matches extension using magic bytes.

    FIX BUG #7: Prevent arbitrary file uploads by checking magic bytes.
    This prevents attacks like uploading malware.exe renamed to document.pdf.

    Args:
        file_path: Path to uploaded file
        claimed_extension: File extension from filename (e.g., '.pdf')

    Returns:
        True if file type matches extension, False otherwise
    """
    try:
        # Read first 512 bytes for magic byte detection
        with open(file_path, 'rb') as f:
            header = f.read(512)

        if not header:
            logger.warning(f"Empty file: {file_path}")
            return False

        # Get expected magic bytes for this extension
        expected_magics = ALLOWED_MIME_TYPES.get(claimed_extension.lower(), [])

        # Text files (md, csv, html) are harder to validate - allow if printable
        text_extensions = ['.md', '.csv', '.html']
        if claimed_extension.lower() in text_extensions:
            # Check if file is mostly ASCII/UTF-8 printable
            try:
                header.decode('utf-8')
                return True
            except UnicodeDecodeError:
                logger.warning(f"File {file_path} claims to be {claimed_extension} but contains binary data")
                return False

        # Binary files: check magic bytes
        for magic in expected_magics:
            if header.startswith(magic):
                return True

        logger.warning(f"File type mismatch: {file_path} does not match extension {claimed_extension}")
        return False

    except Exception as e:
        logger.error(f"File type validation failed for {file_path}: {e}")
        return False


def flatten_metadata(metadata: dict) -> dict:
    """Flatten nested metadata for ChromaDB compatibility.

    ChromaDB only accepts str, int, float, bool, or None as metadata values.
    This function converts nested dicts and lists to JSON strings.
    """
    flattened = {}
    for key, value in metadata.items():
        if value is None:
            flattened[key] = None
        elif isinstance(value, (str, int, float, bool)):
            flattened[key] = value
        elif isinstance(value, (dict, list)):
            # Convert nested structures to JSON string
            import json
            flattened[key] = json.dumps(value)
        else:
            # Convert other types to string
            flattened[key] = str(value)
    return flattened


@router.post("/validate-upload")
async def validate_upload(
    collection_name: str = Form(...),
    rag_client=Depends(get_rag_client),
    current_user: User = Depends(get_current_user)
):
    """Validate if upload is compatible with collection embedding configuration."""
    logger.debug(f"Validating upload for collection: {collection_name}")

    try:
        metadata = await rag_client.get_collection_metadata(collection_name)

        if not metadata:
            logger.warning(f"No metadata for '{collection_name}', allowing upload with warning")
            return {
                "valid": True,
                "warning": "No metadata found for this collection. Upload may fail if embedding models don't match.",
                "collection_model": "unknown",
                "current_model": "unknown"
            }

        # Get system-wide defaults from embedding_manager
        manager_config = rag_client.embedding_manager._load_config()
        current_model = manager_config.get("EMBEDDING_MODEL", "nomic-embed-text:latest")
        collection_model = metadata.get("embedding_model")
        collection_dims = metadata.get("embedding_dimensions")

        try:
            current_dims = await rag_client.get_embedding_dimensions(current_model)
        except Exception:
            current_dims = 768

        if current_model != collection_model or current_dims != collection_dims:
            logger.warning(f"Embedding mismatch: {current_model} vs {collection_model}")
            raise ValidationError(
                "Embedding mismatch - collection and current model incompatible",
                details={
                    "collection_model": collection_model,
                    "current_model": current_model,
                    "collection_dims": collection_dims,
                    "current_dims": current_dims,
                    "suggestion": "Please change EMBEDDING_MODEL in Settings to match the collection"
                }
            )

        logger.debug("Upload validation passed")
        return {
            "valid": True,
            "message": "Upload is compatible",
            "collection_model": collection_model,
            "current_model": current_model
        }

    except ValidationError:
        raise  # Re-raise validation errors
    except Exception as e:
        logger.error(f"Upload validation failed: {e}", exc_info=True)
        raise ServiceUnavailableError("validation", str(e))


@router.post("/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    collection_name: str = Form("default"),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
    rag_client=Depends(get_rag_client),
    current_user: User = Depends(get_current_user)
):
    """Upload and index documents to ChromaDB with embedding validation."""
    # FIX ISSUE #1: Limit concurrent uploads to prevent resource exhaustion
    # The semaphore protects the ENTIRE upload process, not just logging
    async with UPLOAD_SEMAPHORE:
        logger.debug(f"Uploading {len(files)} files to collection '{collection_name}' (concurrent slots: {MAX_CONCURRENT_UPLOADS})")

        try:
            # STEP 1: Validate embedding compatibility
            metadata = await rag_client.get_collection_metadata(collection_name)

            if metadata:
                # Get system-wide defaults from embedding_manager
                manager_config = rag_client.embedding_manager._load_config()
                current_model = manager_config.get("EMBEDDING_MODEL", "nomic-embed-text:latest")
                collection_model = metadata.get("embedding_model")
                collection_dims = metadata.get("embedding_dimensions")

                try:
                    current_dims = await rag_client.get_embedding_dimensions(current_model)
                except Exception:
                    current_dims = 768

                # FIX BUG #8: Normalize model names for comparison (Ollama auto-adds :latest)
                # nomic-embed-text should match nomic-embed-text:latest
                def normalize_model_name(model_name: str) -> str:
                    """Add :latest tag if no tag is present"""
                    if model_name and ':' not in model_name:
                        return f"{model_name}:latest"
                    return model_name

                normalized_current = normalize_model_name(current_model)
                normalized_collection = normalize_model_name(collection_model)

                if normalized_current != normalized_collection or current_dims != collection_dims:
                    logger.error(f"Embedding mismatch: {normalized_current} vs {normalized_collection}")
                    raise ValidationError(
                        f"Embedding mismatch! Collection requires '{collection_model}' ({collection_dims} dims), but current settings use '{current_model}' ({current_dims} dims).",
                        details={
                            "collection_model": collection_model,
                            "current_model": current_model,
                            "collection_dims": collection_dims,
                            "current_dims": current_dims
                        }
                    )

            results = []
            total_chunks = 0

            # FIX BUG #2: Create unique temp directory per request to avoid race conditions
            # This prevents concurrent uploads from interfering with each other
            temp_dir = tempfile.mkdtemp(prefix="rag_upload_")
            logger.debug(f"Created temporary directory: {temp_dir}")

            try:
                for file in files:
                    try:
                        # FIX EDGE CASE #3: Generate unique filename to prevent overwrites
                        # Even if multiple files in same request have identical names
                        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
                        file_path = os.path.join(temp_dir, unique_filename)

                        # FIX BUG #1: Ensure UploadFile is properly closed
                        try:
                            with open(file_path, "wb") as buffer:
                                content = await file.read()
                                buffer.write(content)
                        finally:
                            # Explicitly close the UploadFile to prevent file descriptor leak
                            await file.close()

                        # FIX BUG #7: Validate file type matches extension (security)
                        extension = Path(file.filename).suffix.lower()
                        if not validate_file_type(file_path, extension):
                            allowed_types = ['.pdf', '.txt', '.md', '.docx', '.html', '.pptx', '.xlsx']
                            raise InvalidFileTypeError(extension, allowed_types)

                        # Use Central Docling Service
                        # This handles Repair, Analysis, and Conversion (OCR/Tables)
                        logger.info(f"Docling converting: {file.filename}")
                        progress_manager.set_status(f"Converting {file.filename}...")
                        process_result = await docling_service.process_file(file_path)
                        
                        if not process_result["success"]:
                            raise IngestionError(f"Docling processing failed: {process_result.get('error')}")

                        markdown_content = process_result["content"]
                        doc_metadata = process_result["metadata"]
                        
                        # Add ingestion metadata
                        doc_metadata.update({
                            'source': file.filename,
                            'file_type': Path(file.filename).suffix.lower(),
                            'collection': collection_name,
                            'processed_by': 'docling_service_v1'
                        })

                        # Use IndexingService via RAGClient
                        # This ensures both Vector Store and BM25 Index are updated
                        from src.core.indexing_service import Document, ChunkConfig

                        # Create Document object
                        doc = Document(
                            content=markdown_content,
                            metadata=doc_metadata
                        )

                        # Create ChunkConfig
                        chunk_config = ChunkConfig(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )

                        # Index using RAGClient (which uses IndexingService)
                        logger.info(f"Indexing in ChromaDB: {file.filename}")
                        progress_manager.set_status(f"Indexing {file.filename} in ChromaDB...")
                        response = await rag_client.index_documents(
                            documents=[doc],
                            collection_name=collection_name,
                            chunk_config=chunk_config
                        )

                        if not response.is_success:
                             raise IngestionError(f"Indexing failed: {response.error}")
                        
                        # Extract stats from response
                        chunk_count = response.data.get("indexed_nodes", 0)
                        total_chunks += chunk_count
                        
                        results.append({
                            "filename": file.filename,
                            "success": True,
                            "chunks": chunk_count
                        })

                        logger.info(f"Successfully uploaded and indexed '{file.filename}': {chunk_count} chunks")



                    except Exception as file_error:
                        logger.error(f"Failed to upload '{file.filename}': {file_error}")
                        results.append({
                            "filename": file.filename,
                            "success": False,
                            "chunks": 0,
                            "error": str(file_error)
                        })

            finally:
                # Cleanup: Remove unique temp directory (safe - no concurrent conflicts)
                if os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                        logger.debug(f"Cleaned up temporary directory: {temp_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")

            successful_files = sum(1 for r in results if r["success"])
            logger.info(f"Upload complete: {successful_files}/{len(files)} files, {total_chunks} chunks")

            return {
                "success": successful_files > 0,
                "files_processed": successful_files,
                "total_files": len(files),
                "total_chunks": total_chunks,
                "results": results,
                "collection": collection_name,
                "chunk_config": {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                }
            }

        except (ValidationError, ChromaDBError, ServiceUnavailableError):
            raise  # Re-raise custom exceptions
        except Exception as e:
            logger.error(f"Upload failed: {e}", exc_info=True)
            raise IngestionError(str(e))
