"""
Folder Scanner Service for Smart Docling Uploader (Phase 4)

Provides recursive directory scanning with:
- Docling-compatible file detection
- Symlink loop prevention
- Hidden file exclusion
- TXT→MD transparent conversion
- File validation (size, type)
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# Docling-supported file extensions (from phase_4_smart_uploader.md)
DOCLING_SUPPORTED_EXTENSIONS = {
    '.pdf', '.docx', '.pptx', '.xlsx',
    '.html', '.md', '.csv'
}

# Hidden file/folder patterns to exclude
HIDDEN_PATTERNS = {
    '.git', '.env', '.venv', 'venv', '__pycache__',
    'node_modules', '.pytest_cache', '.mypy_cache'
}

# Max file size (100 MB as per deployment guide)
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024


@dataclass
class FileInfo:
    """Information about a scanned file."""
    path: str
    original_path: str  # For TXT files, this is the .txt path
    filename: str
    extension: str
    size_bytes: int
    size_human: str
    is_txt_converted: bool = False
    error: Optional[str] = None


@dataclass
class ValidationResult:
    """File validation result."""
    valid: bool
    error: Optional[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class FolderScanner:
    """
    Scans folders recursively for Docling-compatible documents.
    """

    def __init__(self, max_file_size: int = MAX_FILE_SIZE_BYTES):
        self.max_file_size = max_file_size
        self.temp_files: List[str] = []  # Track temp .md files from TXT conversion
        self._visited_dirs: Set[str] = set()  # Prevent symlink loops

    def scan_directory(
        self,
        path: str,
        recursive: bool = True,
        max_depth: int = 10,
        current_depth: int = 0,
        allowed_extensions: Optional[List[str]] = None
    ) -> List[FileInfo]:
        """
        Scans a directory for Docling-compatible files.

        Args:
            path: Directory path to scan
            recursive: Whether to scan subdirectories
            max_depth: Maximum recursion depth
            current_depth: Current recursion level (internal)
            allowed_extensions: Optional list of file extensions to include (e.g., ['.py', '.js'])

        Returns:
            List of FileInfo objects for valid files
        """
        files: List[FileInfo] = []

        if not os.path.exists(path):
            logger.error(f"Path does not exist: {path}")
            return files

        if not os.path.isdir(path):
            logger.error(f"Path is not a directory: {path}")
            return files

        # Resolve symlinks and check for loops
        real_path = os.path.realpath(path)
        if real_path in self._visited_dirs:
            logger.warning(f"Symlink loop detected, skipping: {path}")
            return files
        self._visited_dirs.add(real_path)

        # Check depth limit
        if current_depth >= max_depth:
            logger.warning(f"Max depth {max_depth} reached, skipping: {path}")
            return files

        try:
            for entry in os.scandir(path):
                # Skip hidden files/folders
                if entry.name.startswith('.') or entry.name in HIDDEN_PATTERNS:
                    continue

                if entry.is_file(follow_symlinks=False):
                    file_info = self._process_file(entry.path, allowed_extensions)
                    if file_info:
                        files.append(file_info)

                elif entry.is_dir(follow_symlinks=False) and recursive:
                    # Recursive scan
                    subdir_files = self.scan_directory(
                        entry.path,
                        recursive=True,
                        max_depth=max_depth,
                        current_depth=current_depth + 1,
                        allowed_extensions=allowed_extensions
                    )
                    files.extend(subdir_files)

        except PermissionError as e:
            logger.error(f"Permission denied: {path} - {e}")
        except Exception as e:
            logger.error(f"Error scanning directory {path}: {e}")

        return files

    def _process_file(self, file_path: str, allowed_extensions: Optional[List[str]] = None) -> Optional[FileInfo]:
        """
        Processes a single file and returns FileInfo if valid.

        Handles TXT→MD conversion transparently.
        """
        try:
            file_stat = os.stat(file_path)
            file_size = file_stat.st_size
            file_ext = Path(file_path).suffix.lower()

            # Check if file is too large
            if file_size > self.max_file_size:
                logger.warning(
                    f"File too large ({self._human_size(file_size)}): {file_path}"
                )
                return FileInfo(
                    path=file_path,
                    original_path=file_path,
                    filename=os.path.basename(file_path),
                    extension=file_ext,
                    size_bytes=file_size,
                    size_human=self._human_size(file_size),
                    error=f"File exceeds max size of {self._human_size(self.max_file_size)}"
                )

            # If a specific list of extensions is given, use it
            if allowed_extensions is not None:
                if file_ext not in allowed_extensions:
                    return None # Skip files not in the allowed list
            # Otherwise, use the default Docling-supported extensions
            else:
                if file_ext == '.txt':
                    md_path = self.convert_txt_to_md(file_path)
                    return FileInfo(
                        path=md_path,
                        original_path=file_path,
                        filename=os.path.basename(file_path),
                        extension='.md',  # Report as .md after conversion
                        size_bytes=file_size,
                        size_human=self._human_size(file_size),
                        is_txt_converted=True
                    )

                if file_ext not in DOCLING_SUPPORTED_EXTENSIONS:
                    return None  # Silently skip unsupported files

            return FileInfo(
                path=file_path,
                original_path=file_path,
                filename=os.path.basename(file_path),
                extension=file_ext,
                size_bytes=file_size,
                size_human=self._human_size(file_size),
                is_txt_converted=False
            )

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None

    def convert_txt_to_md(self, file_path: str) -> str:
        """
        Converts a .txt file to .md by copying to a temp file.

        Args:
            file_path: Path to .txt file

        Returns:
            Path to temporary .md file
        """
        # Create temp .md file
        md_path = file_path.replace('.txt', '.txt.md')

        try:
            shutil.copy(file_path, md_path)
            self.temp_files.append(md_path)  # Track for cleanup
            logger.debug(f"Converted TXT→MD: {file_path} → {md_path}")
            return md_path
        except Exception as e:
            logger.error(f"Failed to convert TXT→MD: {file_path} - {e}")
            return file_path  # Fallback to original

    def validate_files(
        self,
        files: List[str]
    ) -> Dict[str, ValidationResult]:
        """
        Validates a list of file paths.

        Args:
            files: List of file paths to validate

        Returns:
            Dict mapping file path to ValidationResult
        """
        results = {}

        for file_path in files:
            if not os.path.exists(file_path):
                results[file_path] = ValidationResult(
                    valid=False,
                    error="File does not exist"
                )
                continue

            if not os.path.isfile(file_path):
                results[file_path] = ValidationResult(
                    valid=False,
                    error="Path is not a file"
                )
                continue

            try:
                file_size = os.path.getsize(file_path)
                file_ext = Path(file_path).suffix.lower()

                warnings = []

                # Size check
                if file_size > self.max_file_size:
                    results[file_path] = ValidationResult(
                        valid=False,
                        error=f"File size ({self._human_size(file_size)}) exceeds max {self._human_size(self.max_file_size)}"
                    )
                    continue

                # Empty file warning
                if file_size == 0:
                    warnings.append("File is empty")

                # Extension check
                if file_ext not in DOCLING_SUPPORTED_EXTENSIONS and file_ext != '.txt':
                    results[file_path] = ValidationResult(
                        valid=False,
                        error=f"Unsupported file type: {file_ext}"
                    )
                    continue

                # Readable check
                try:
                    with open(file_path, 'rb') as f:
                        f.read(1)  # Try reading first byte
                except Exception as e:
                    results[file_path] = ValidationResult(
                        valid=False,
                        error=f"File is not readable: {e}"
                    )
                    continue

                results[file_path] = ValidationResult(
                    valid=True,
                    warnings=warnings
                )

            except Exception as e:
                results[file_path] = ValidationResult(
                    valid=False,
                    error=f"Validation error: {e}"
                )

        return results

    def cleanup_temp_files(self):
        """
        Removes temporary .md files created from TXT conversion.
        Call this after upload is complete.
        """
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.debug(f"Cleaned up temp file: {temp_file}")
            except Exception as e:
                logger.error(f"Failed to clean up {temp_file}: {e}")

        self.temp_files.clear()

    @staticmethod
    def _human_size(size_bytes: int) -> str:
        """Converts bytes to human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"


# Convenience function for quick scanning
def scan_folder(
    path: str,
    recursive: bool = True,
    max_depth: int = 10,
    allowed_extensions: Optional[List[str]] = None
) -> List[FileInfo]:
    """
    Quick folder scan function.

    Args:
        path: Directory path to scan
        recursive: Whether to scan subdirectories
        max_depth: Maximum recursion depth
        allowed_extensions: Optional list of file extensions to include

    Returns:
        List of FileInfo objects
    """
    scanner = FolderScanner()
    return scanner.scan_directory(path, recursive, max_depth, allowed_extensions=allowed_extensions)
