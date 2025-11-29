# backend/src/core/metadata_extractor.py
import os
import mimetypes
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Attempt to import python-magic, but fall back gracefully if not installed
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    logging.warning("python-magic not found. Falling back to mimetypes for MIME type detection. "
                    "For more accurate results, run 'pip install python-magic'.")

# Attempt to import langdetect
try:
    from langdetect import detect, LangDetectException
    from langdetect.detector_factory import DetectorFactory
    # Enforce consistent results from langdetect
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect not found. Language detection will be disabled. "
                    "To enable it, run 'pip install langdetect'.")

logger = logging.getLogger(__name__)

def detect_mime_type(file_path: str) -> Optional[str]:
    """
    Detects the MIME type of a file.
    Uses python-magic for accuracy if available, otherwise falls back to the standard mimetypes library.

    Args:
        file_path: The path to the file.

    Returns:
        The detected MIME type as a string (e.g., 'application/pdf'), or None if detection fails.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found for MIME detection: {file_path}")
        return None

    try:
        if MAGIC_AVAILABLE:
            mime_type = magic.from_file(file_path, mime=True)
            if mime_type:
                return mime_type
        
        # Fallback to mimetypes if magic fails or is not available
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            return mime_type
            
        logger.warning(f"Could not determine MIME type for file: {file_path}")
        return "application/octet-stream" # Generic fallback
    except Exception as e:
        logger.error(f"An error occurred during MIME type detection for {file_path}: {e}")
        return None

def get_file_system_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extracts basic file system metadata.

    Args:
        file_path: The path to the file.

    Returns:
        A dictionary containing file_name, file_path, file_size,
        created_date, and modified_date.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found for file system metadata extraction: {file_path}")
        return {}
        
    try:
        stat = os.stat(file_path)
        return {
            "file_name": os.path.basename(file_path),
            "file_path": file_path,
            "file_size": stat.st_size,
            "created_date": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to get file system metadata for {file_path}: {e}")
        return {}

def detect_language(text: str, min_length: int = 50) -> Optional[str]:
    """
    Detects the language of a given text block.

    Args:
        text: The text to analyze.
        min_length: The minimum number of characters required to attempt detection.

    Returns:
        The two-letter language code (e.g., 'en', 'de'), or None if detection is not possible.
    """
    if not LANGDETECT_AVAILABLE:
        return None
        
    if not text or len(text) < min_length:
        return None
        
    try:
        return detect(text)
    except LangDetectException:
        logger.warning("Could not detect a specific language for the provided text.")
        return "unknown"
    except Exception as e:
        logger.error(f"An unexpected error occurred during language detection: {e}")
        return None
