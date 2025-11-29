"""
Data Type Detector - Automatic detection of file types for optimal ingestion.

This service analyzes files and determines the best ingestion strategy:
- UNSTRUCTURED_TEXT: PDFs, documents → Vector index
- STRUCTURED_TABLE: Excel, CSV with tabular data → Pandas agent
- CODE: Source code files → Code-aware splitting
- EMAIL: Email files → Metadata extraction
- UNKNOWN: Unsupported formats

Part of Phase 7: Smart Ingestion System
"""

from enum import Enum
from pathlib import Path
from typing import Optional
import mimetypes
from loguru import logger


class DataType(Enum):
    """
    Enumeration of supported data types for ingestion.

    Each type corresponds to a specific ingestion strategy:
    - UNSTRUCTURED_TEXT → Docling loader + Vector index
    - STRUCTURED_TABLE → Pandas agent (queryable with natural language)
    - CODE → Code-aware splitting (preserve functions/classes)
    - EMAIL → Email loader with metadata extraction
    - UNKNOWN → Not supported or cannot be determined
    """

    UNSTRUCTURED_TEXT = "unstructured_text"
    STRUCTURED_TABLE = "structured_table"
    CODE = "code"
    EMAIL = "email"
    UNKNOWN = "unknown"


class DataTypeDetector:
    """
    Detects file type and recommends optimal ingestion strategy.

    Uses a combination of:
    - File extension analysis
    - MIME type detection
    - Content validation (for Excel/CSV)

    Example:
        detector = DataTypeDetector()
        data_type = detector.detect("customer_data.xlsx")
        # → DataType.STRUCTURED_TABLE (if valid table)
        # → DataType.UNSTRUCTURED_TEXT (if just text in Excel)
    """

    # File extensions mapped to data types
    EMAIL_EXTENSIONS = {'.eml', '.mbox', '.msg'}
    CODE_EXTENSIONS = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c',
                       '.h', '.hpp', '.go', '.rs', '.rb', '.php', '.cs', '.swift'}
    TABLE_EXTENSIONS = {'.xlsx', '.xls', '.csv', '.tsv'}
    DOCLING_EXTENSIONS = {'.pdf', '.docx', '.pptx', '.html', '.md'}
    TEXT_EXTENSIONS = {'.txt'}

    def __init__(self):
        """Initialize the DataTypeDetector."""
        self.logger = logger.bind(component="DataTypeDetector")

    def detect(self, file_path: str) -> DataType:
        """
        Detect the data type of a file.

        Args:
            file_path: Absolute or relative path to the file

        Returns:
            DataType enum indicating the detected type

        Example:
            >>> detector = DataTypeDetector()
            >>> detector.detect("code.py")
            DataType.CODE
            >>> detector.detect("customers.xlsx")
            DataType.STRUCTURED_TABLE
        """
        try:
            path = Path(file_path)

            if not path.exists():
                self.logger.warning(f"File not found: {file_path}")
                return DataType.UNKNOWN

            extension = path.suffix.lower()

            # 1. Email formats
            if extension in self.EMAIL_EXTENSIONS:
                self.logger.debug(f"Detected EMAIL: {path.name}")
                return DataType.EMAIL

            # 2. Code formats
            if extension in self.CODE_EXTENSIONS:
                self.logger.debug(f"Detected CODE: {path.name}")
                return DataType.CODE

            # 3. Tables (requires validation)
            if extension in self.TABLE_EXTENSIONS:
                return self._validate_table(file_path)

            # 4. Docling-supported formats (unstructured)
            if extension in self.DOCLING_EXTENSIONS:
                self.logger.debug(f"Detected UNSTRUCTURED_TEXT (Docling): {path.name}")
                return DataType.UNSTRUCTURED_TEXT

            # 5. Plain text
            if extension in self.TEXT_EXTENSIONS:
                self.logger.debug(f"Detected UNSTRUCTURED_TEXT (Text): {path.name}")
                return DataType.UNSTRUCTURED_TEXT

            # 6. Unknown
            self.logger.warning(f"Unknown file type: {extension} ({path.name})")
            return DataType.UNKNOWN

        except Exception as e:
            self.logger.error(f"Error detecting file type for {file_path}: {e}")
            return DataType.UNKNOWN

    def _validate_table(self, file_path: str) -> DataType:
        """
        Validates if Excel/CSV file contains a real table.

        A "real table" is defined as:
        - At least 2 columns
        - At least 2 rows (excluding header)

        If validation fails, treats as unstructured text.

        Args:
            file_path: Path to Excel or CSV file

        Returns:
            DataType.STRUCTURED_TABLE if valid table
            DataType.UNSTRUCTURED_TEXT if just text/invalid
            DataType.UNKNOWN if cannot be read
        """
        try:
            import pandas as pd

            path = Path(file_path)
            extension = path.suffix.lower()

            # Try to read first few rows
            if extension in {'.xlsx', '.xls'}:
                df = pd.read_excel(file_path, nrows=10)
            elif extension in {'.csv', '.tsv'}:
                # Auto-detect separator
                separator = '\t' if extension == '.tsv' else None
                df = pd.read_csv(file_path, nrows=10, sep=separator)
            else:
                return DataType.UNKNOWN

            # Validation criteria
            num_columns = len(df.columns)
            num_rows = len(df)

            # At least 2 columns and 2 rows → Real table
            if num_columns >= 2 and num_rows >= 2:
                self.logger.info(
                    f"Detected STRUCTURED_TABLE: {path.name} "
                    f"({num_columns} cols, {num_rows} rows)"
                )
                return DataType.STRUCTURED_TABLE
            else:
                self.logger.info(
                    f"Detected UNSTRUCTURED_TEXT: {path.name} "
                    f"(insufficient structure: {num_columns} cols, {num_rows} rows)"
                )
                return DataType.UNSTRUCTURED_TEXT

        except Exception as e:
            self.logger.warning(f"Failed to validate table {file_path}: {e}")
            # If we can't read it as table, treat as unstructured
            return DataType.UNSTRUCTURED_TEXT

    def get_recommended_strategy(self, data_type: DataType) -> str:
        """
        Get recommended index strategy for a data type.

        Maps DataType to CollectionIndexConfig.index_strategy values.

        Args:
            data_type: The detected data type

        Returns:
            Recommended strategy string ('vector', 'pandas_agent', etc.)
        """
        strategy_map = {
            DataType.UNSTRUCTURED_TEXT: "vector",
            DataType.STRUCTURED_TABLE: "pandas_agent",
            DataType.CODE: "hybrid",  # Could use code-aware chunking
            DataType.EMAIL: "vector",  # With metadata filtering
            DataType.UNKNOWN: "vector"  # Default fallback
        }

        return strategy_map.get(data_type, "vector")

    def analyze(self, file_path: str) -> dict:
        """
        Complete analysis of a file.

        Returns detailed information about the file for ingestion planning.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with analysis results:
            {
                "file_path": str,
                "file_name": str,
                "extension": str,
                "data_type": DataType,
                "index_strategy": str,
                "mime_type": str (optional)
            }
        """
        path = Path(file_path)
        data_type = self.detect(file_path)

        # Get MIME type
        mime_type, _ = mimetypes.guess_type(file_path)

        analysis = {
            "file_path": str(path.absolute()),
            "file_name": path.name,
            "extension": path.suffix.lower(),
            "data_type": data_type.value,
            "index_strategy": self.get_recommended_strategy(data_type),
            "mime_type": mime_type
        }

        self.logger.debug(f"File analysis: {analysis}")

        return analysis


# Convenience function for quick detection
def detect_file_type(file_path: str) -> DataType:
    """
    Quick convenience function for file type detection.

    Args:
        file_path: Path to the file

    Returns:
        DataType enum
    """
    detector = DataTypeDetector()
    return detector.detect(file_path)
