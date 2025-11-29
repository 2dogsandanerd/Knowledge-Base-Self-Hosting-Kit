"""
Duplicate Detection Service for RAG Ingestion.

Detects duplicate files based on SHA256 hashing to prevent re-ingestion
of identical content across different filenames.
"""

import hashlib
from typing import Dict, Optional
from loguru import logger


class DuplicateDetector:
    """
    Detects duplicate files during ingestion based on content hash.

    Usage:
        detector = DuplicateDetector()

        # Check if file is duplicate
        duplicate = detector.check_duplicate(file_content)
        if duplicate:
            print(f"Duplicate of: {duplicate}")
        else:
            detector.register_file_hash(file_content, filename)
    """

    def __init__(self):
        self._file_hashes: Dict[str, str] = {}  # hash -> filename
        self.logger = logger.bind(component="DuplicateDetector")

    def check_duplicate(self, file_content: bytes) -> Optional[str]:
        """
        Check if file content was already processed.

        Args:
            file_content: File bytes

        Returns:
            Filename of duplicate if exists, None otherwise
        """
        file_hash = hashlib.sha256(file_content).hexdigest()

        if existing_filename := self._file_hashes.get(file_hash):
            self.logger.debug(
                f"Duplicate detected: hash {file_hash[:8]}... "
                f"(original: {existing_filename})"
            )
            return existing_filename

        return None

    def register_file_hash(self, file_content: bytes, filename: str):
        """
        Register file hash for duplicate detection.

        Args:
            file_content: File bytes
            filename: Original filename
        """
        file_hash = hashlib.sha256(file_content).hexdigest()
        self._file_hashes[file_hash] = filename
        self.logger.debug(f"Registered hash {file_hash[:8]}... for {filename}")

    def clear_cache(self):
        """Clear all registered file hashes (useful for testing)."""
        count = len(self._file_hashes)
        self._file_hashes.clear()
        self.logger.info(f"Cleared {count} file hashes from cache")

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about registered files."""
        return {
            "total_files_registered": len(self._file_hashes)
        }


# Global singleton instance
duplicate_detector = DuplicateDetector()
