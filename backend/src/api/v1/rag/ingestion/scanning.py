"""
Folder Scanning Endpoint.

Scans directories for Docling-compatible files using the Central Docling Service.
Features:
- Recursive directory traversal
- File type detection via DoclingService
- Metadata extraction (size, type)
"""

from fastapi import APIRouter, Depends, HTTPException, Form
from src.core.exceptions import ValidationError, RAGFileNotFoundError
from typing import Optional
import logging
import os

from src.api.v1.dependencies import get_rag_client
from src.services.auth_service import get_current_user
from src.database.models import User
from src.services.docling_service import docling_service
from src.core.feature_limits import FeatureLimits, Edition

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/scan-folder")
async def scan_folder_endpoint(
    folder_path: str = Form(...),
    recursive: bool = Form(True),
    max_depth: int = Form(10),
    allowed_extensions_str: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    rag_client=Depends(get_rag_client)
):
    """
    Scan a folder for Docling-compatible files.

    Args:
        folder_path: Path to directory to scan
        recursive: Whether to scan subdirectories
        max_depth: Maximum recursion depth
        allowed_extensions_str: Comma-separated string of extensions (e.g., .py,.js)

    Returns:
        List of files with metadata
    """
    logger.debug(f"Scanning folder: {folder_path} (recursive: {recursive}, max_depth: {max_depth})")

    # Check file format limitations based on edition
    edition = rag_client.edition
    allowed_extensions = None

    if allowed_extensions_str:
        allowed_extensions = [ext.strip().lower() for ext in allowed_extensions_str.split(',')]
    else:
        # Use edition's default allowed extensions
        allowed_extensions = FeatureLimits.get_limit_value('allowed_file_formats', edition)
        logger.info(f"Using edition default extensions: {allowed_extensions}")

    # For Developer Edition, only allow specified file formats
    if edition == Edition.DEVELOPER:
        developer_formats = FeatureLimits.get_limit_value('allowed_file_formats', Edition.DEVELOPER)
        allowed_extensions = [ext for ext in allowed_extensions if ext in developer_formats]

    logger.info(f"Filtering by extensions: {allowed_extensions}")

    try:
        # Validate folder path
        if not os.path.exists(folder_path):
            logger.error(f"Folder does not exist: {folder_path}")
            raise HTTPException(status_code=404, detail=f"Folder not found: {folder_path}")

        if not os.path.isdir(folder_path):
            logger.error(f"Path is not a directory: {folder_path}")
            raise HTTPException(status_code=400, detail=f"Path is not a directory: {folder_path}")

        # Perform scan using DoclingService
        # Note: DoclingService.process_directory actually processes the files (heavy).
        # For scanning, we just want to LIST them.
        # So we implement a lightweight listing logic here that respects Docling's supported types.
        
        files = []
        total_size = 0
        extension_counts = {}

        # Use configured allowed_extensions (includes code files + documents)
        # Don't limit to Docling - we want to scan ALL configured formats

        for root, _, filenames in os.walk(folder_path):
            # Depth check
            depth = root[len(folder_path):].count(os.sep)
            if depth > max_depth:
                continue

            for filename in filenames:
                file_path = os.path.join(root, filename)
                ext = os.path.splitext(filename)[1].lower()

                # Filter logic - only check against allowed_extensions
                if allowed_extensions and ext not in allowed_extensions:
                    continue
                    
                try:
                    size = os.path.getsize(file_path)
                    
                    # Edition check for Developer
                    error = None
                    if edition == Edition.DEVELOPER:
                         if ext not in FeatureLimits.get_limit_value('allowed_file_formats', Edition.DEVELOPER):
                            error = f"File format {ext} not supported in Developer Edition."

                    files.append({
                        "path": file_path,
                        "original_path": file_path,
                        "filename": filename,
                        "extension": ext,
                        "size_bytes": size,
                        "size_human": f"{size / 1024:.1f} KB",
                        "is_txt_converted": False, # Legacy field
                        "error": error
                    })
                    
                    total_size += size
                    extension_counts[ext] = extension_counts.get(ext, 0) + 1
                    
                except OSError:
                    continue
            
            if not recursive:
                break

        response = {
            "files": files,
            "total_files": len(files),
            "total_size": total_size,
            "summary": extension_counts
        }

        # Add edition limitations info if applicable
        if edition == Edition.DEVELOPER:
            response["edition_limitations"] = {
                "message": "Only PDF files are supported in Developer Edition. Upgrade to Team Edition for more document formats.",
                "supported_formats": FeatureLimits.get_limit_value('allowed_file_formats', Edition.DEVELOPER)
            }

        logger.info(f"Folder scan complete: {len(files)} files found in {folder_path}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Folder scan failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Folder scan failed: {str(e)}")
