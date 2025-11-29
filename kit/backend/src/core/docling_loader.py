"""
Docling Document Loader - LangChain-compatible loader for enhanced document parsing.

This module provides a standalone, drop-in replacement for PyPDFLoader and TextLoader
with superior document structure extraction and format support.

Supported formats: PDF, DOCX, PPTX, XLSX, HTML, MD, TXT, CSV
Feature-flag controlled via USE_DOCLING_INGESTION environment variable.
"""

import os
from pathlib import Path
from typing import List
from llama_index.core.schema import Document
from loguru import logger


class DoclingLoader:
    """LangChain-compatible document loader using Docling framework.

    This loader provides enhanced document parsing with structure preservation
    for PDF, Office documents, HTML, and text files. It's designed as a drop-in
    replacement for standard LangChain loaders like PyPDFLoader and TextLoader.

    Usage:
        loader = DoclingLoader("/path/to/document.pdf")
        documents = loader.load()  # Returns List[Document]
    """

    def __init__(self, file_path: str):
        """Initialize the Docling loader.

        Args:
            file_path: Absolute path to the document file
        """
        self.file_path = Path(file_path)
        self.logger = logger.bind(component="DoclingLoader")

        if not self.file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

    def load(self) -> List[Document]:
        """Load and parse the document using Docling.

        Returns:
            List of LlamaIndex Document objects with parsed content and metadata

        Raises:
            Exception: If document parsing fails
        """
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions

            self.logger.info(f"Loading document with Docling: {self.file_path.name}")

            # Configure PDF pipeline to extract images (graphs, figures) and tables
            pipeline_options = PdfPipelineOptions()
            pipeline_options.generate_picture_images = True
            pipeline_options.generate_table_images = True

            # Initialize converter with custom options
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )

            # Auto-detection for all supported formats
            # Note: TXT files should be pre-converted to MD by customer workflow
            result = converter.convert(str(self.file_path))

            # Extract content as Markdown (preserves structure)
            markdown_content = result.document.export_to_markdown()

            # Get file extension for metadata
            file_extension = self.file_path.suffix.lower()

            # Create LlamaIndex Document with metadata
            doc = Document(
                text=markdown_content,
                metadata={
                    'source': str(self.file_path),
                    'source_file_path': str(self.file_path),  # For migration compatibility
                    'file_type': file_extension,
                    'file_name': self.file_path.name,
                    'loader': 'docling',
                    'format': 'markdown'
                }
            )

            self.logger.success(
                f"Successfully loaded {self.file_path.name} "
                f"({len(markdown_content)} chars) via Docling"
            )

            return [doc]

        except ImportError as e:
            self.logger.error(f"Docling not installed: {e}")
            raise RuntimeError(
                "Docling is not installed. Install with: pip install docling"
            ) from e

        except Exception as e:
            self.logger.error(f"Failed to load {self.file_path.name} with Docling: {e}")
            raise


class DoclingLoaderFactory:
    """Factory for creating appropriate document loaders based on feature flags.

    This factory handles the logic of choosing between Docling and traditional
    loaders based on the USE_DOCLING_INGESTION environment variable.

    Usage:
        loader = DoclingLoaderFactory.create_loader("/path/to/file.pdf")
        documents = loader.load()
    """

    DOCLING_SUPPORTED_EXTENSIONS = {
        '.pdf', '.docx', '.pptx', '.xlsx',
        '.html', '.md', '.csv'
        # Note: .txt NOT supported by Docling - use TextLoader fallback
        # Customer workflow: Rename important .txt files to .md for Docling processing
    }

    @staticmethod
    def is_docling_enabled() -> bool:
        """Check if Docling ingestion is enabled via environment variable."""
        return os.getenv("USE_DOCLING_INGESTION", "False").lower() == "true"

    @staticmethod
    def create_loader(file_path: str):
        """Create appropriate loader based on file type and feature flags.

        Args:
            file_path: Path to the document

        Returns:
            Loader instance (DoclingLoader or traditional LlamaIndex loader)
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        # Check if Docling is enabled and file is supported
        if (DoclingLoaderFactory.is_docling_enabled() and
            extension in DoclingLoaderFactory.DOCLING_SUPPORTED_EXTENSIONS):

            logger.debug(f"Using DoclingLoader for {path.name}")
            return DoclingLoader(str(file_path))

        else:
            # Fallback to LlamaIndex loaders
            logger.debug(f"Using LlamaIndex loader for {path.name}")

            if extension == '.pdf':
                from llama_index.readers.file import PDFReader
                return PDFReader()
            elif extension in {'.txt', '.md'}:
                from llama_index.core import SimpleDirectoryReader
                # SimpleDirectoryReader needs to be configured to load a specific file
                # and its load method is what the processor will call.
                # We can wrap it to present a consistent .load() interface.
                class SingleFileWrapper:
                    def __init__(self, file_path):
                        self.file_path = file_path
                    def load(self):
                        return SimpleDirectoryReader(input_files=[self.file_path]).load_data()
                return SingleFileWrapper(str(file_path))
            elif extension == '.json':
                from llama_index.core import SimpleDirectoryReader
                # Wrapper für JSON-Datei
                class JSONFileWrapper:
                    def __init__(self, file_path):
                        self.file_path = file_path
                    def load(self):
                        return SimpleDirectoryReader(input_files=[self.file_path]).load_data()
                return JSONFileWrapper(str(file_path))
            elif extension == '.xml':
                from llama_index.readers.file import XMLReader
                return XMLReader()
            else:
                from llama_index.readers.file import FlatReader
                return FlatReader()


def test_docling_loader():
    """Quick test function for standalone validation."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python docling_loader.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        loader = DoclingLoader(file_path)
        documents = loader.load()

        print(f"\n✅ Successfully loaded: {file_path}")
        print(f"📄 Documents: {len(documents)}")
        print(f"📊 Content length: {len(documents[0].text)} chars")
        print(f"🔖 Metadata: {documents[0].metadata}")
        print(f"\n--- Content Preview (first 500 chars) ---")
        print(documents[0].text[:500])

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    test_docling_loader()
