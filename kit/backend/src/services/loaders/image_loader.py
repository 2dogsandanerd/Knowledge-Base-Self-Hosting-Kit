# backend/src/services/loaders/image_loader.py
import logging
from typing import Dict, Any, Optional

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    logging.warning("Pillow not found. Image metadata extraction will be limited. "
                    "To enable it, run 'pip install Pillow'.")

try:
    import exifread
    EXIFREAD_AVAILABLE = True
except ImportError:
    EXIFREAD_AVAILABLE = False
    logging.warning("exifread not found. Detailed EXIF metadata extraction will be disabled. "
                    "To enable it, run 'pip install exifread'.")

logger = logging.getLogger(__name__)

def extract_image_metadata(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Extracts metadata from an image file, including basic info from Pillow
    and detailed EXIF data using exifread.

    This loader does NOT perform OCR. It only extracts metadata.

    Args:
        file_path: The path to the image file.

    Returns:
        A dictionary containing extracted metadata, or None if the file
        cannot be processed.
    """
    if not PILLOW_AVAILABLE:
        logger.error("Cannot extract image metadata because Pillow is not installed.")
        return None

    metadata = {}

    # --- Basic metadata with Pillow ---
    try:
        with Image.open(file_path) as img:
            metadata['format'] = img.format
            metadata['mode'] = img.mode
            metadata['width'], metadata['height'] = img.size
    except Exception as e:
        logger.error(f"Could not process image file with Pillow: {file_path}. Error: {e}")
        return None

    # --- Detailed EXIF data with exifread ---
    if EXIFREAD_AVAILABLE:
        try:
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
                exif_data = {}
                for tag, value in tags.items():
                    if tag not in ('JPEGThumbnail', 'TIFFThumbnail'): # Exclude thumbnails
                        # Convert value to a printable format
                        if hasattr(value, 'printable'):
                            exif_data[str(tag)] = value.printable
                        else:
                            try:
                                exif_data[str(tag)] = str(value)
                            except Exception:
                                # Skip tags that can't be easily converted
                                pass
                if exif_data:
                    metadata['exif'] = exif_data
        except Exception as e:
            logger.warning(f"Could not extract EXIF data from {file_path}. Error: {e}")

    return metadata
