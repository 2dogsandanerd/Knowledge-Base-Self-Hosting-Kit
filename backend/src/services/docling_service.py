"""
Docling Service - Centralized Document Processing Engine.

This service implements the "Industry Standard" pipeline:
1. Repair (pikepdf)
2. Analyze (pypdf) -> Smart Routing (Fast vs. Heavy Mode)
3. Convert (Docling)
4. Refine (LLM - optional)
"""

import os
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from loguru import logger

# Docling Imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat

# Internal Services
from src.services.pdf_repair_service import PDFRepairService
from src.services.pdf_analyzer import PDFAnalyzer
from src.core.rag_client import RAGClient  # RAG Client for LLM access


class DoclingService:
    """
    Central service for batch document processing using Docling.
    Integrates repair, analysis, smart routing, and optional LLM refinement.
    """

    SUPPORTED_EXTENSIONS = {
        '.pdf', '.docx', '.pptx', '.xlsx',
        '.html', '.md', '.csv'
    }

    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
    CACHE_DIR = Path("/app/cache/docling")  # Persistent cache mounted via Docker

    def __init__(self):
        """Initialize the Docling service with dual-pipeline configuration."""
        self.logger = logger.bind(component="DoclingService")
        
        # Initialize helpers
        self.repair_service = PDFRepairService()
        self.analyzer = PDFAnalyzer()
        
        # Ensure cache dir exists
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # --- Pipeline A: Fast Mode (Digital PDFs) ---
        # No OCR, Fast Table Mode
        self.pipeline_fast = PdfPipelineOptions()
        self.pipeline_fast.do_ocr = False
        self.pipeline_fast.do_table_structure = True
        self.pipeline_fast.table_structure_options.mode = TableFormerMode.FAST

        self.converter_fast = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_fast)
            }
        )

        # --- Pipeline B: Heavy Mode (Scans/Complex) ---
        # OCR Enabled, Accurate Table Mode
        self.pipeline_heavy = PdfPipelineOptions()
        self.pipeline_heavy.do_ocr = True
        self.pipeline_heavy.do_table_structure = True
        self.pipeline_heavy.table_structure_options.mode = TableFormerMode.ACCURATE

        self.converter_heavy = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_heavy)
            }
        )
        
        self.logger.info("DoclingService initialized with Smart Routing (Fast/Heavy).")

    def is_supported_file(self, file_path: str) -> bool:
        """Check if file type is supported."""
        path = Path(file_path)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def validate_file(self, file_path: str) -> tuple[bool, str]:
        """Validate file before processing."""
        path = Path(file_path)

        if not path.exists():
            return False, f"File not found: {file_path}"

        if not self.is_supported_file(file_path):
            return False, f"Unsupported file type: {path.suffix}"

        file_size = path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            size_mb = file_size / (1024 * 1024)
            return False, f"File too large: {size_mb:.1f}MB"

        return True, ""

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file for caching."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _get_from_cache(self, file_hash: str) -> Optional[Dict]:
        """Retrieve result from cache."""
        cache_file = self.CACHE_DIR / f"{file_hash}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _save_to_cache(self, file_hash: str, data: Dict):
        """Save result to cache."""
        cache_file = self.CACHE_DIR / f"{file_hash}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            self.logger.warning(f"Failed to write cache: {e}")

    async def refine_with_llm(self, markdown_content: str, rag_client: Any) -> str:
        """
        Use LLM to refine and clean up the Markdown content.
        This is an optional step for 'hard cases'.
        """
        if not markdown_content:
            return ""

        # Truncate for safety if too huge (LLM context limits)
        # In a real scenario, we would chunk this.
        content_sample = markdown_content[:15000] 
        
        prompt = f"""
        You are an expert Document Cleaner. Your task is to fix formatting issues in the following Markdown text, which was extracted from a PDF.
        
        Rules:
        1. Fix broken line breaks (hyphenation at end of lines).
        2. Ensure headers (#, ##) are logically nested.
        3. Fix obvious OCR errors (e.g., '1l' instead of 'll', 'rn' instead of 'm').
        4. Do NOT summarize. Do NOT change the meaning. Keep the content exact.
        5. Output ONLY the cleaned Markdown.
        
        TEXT TO CLEAN:
        {content_sample}
        """
        
        try:
            # We assume rag_client has a method to get the LLM and complete
            # This depends on how RAGClient exposes the LLM. 
            # If rag_client.llm is the LlamaIndex LLM:
            response = await rag_client.llm.acomplete(prompt)
            return str(response)
        except Exception as e:
            self.logger.error(f"LLM refinement failed: {e}")
            return markdown_content  # Fallback to original

    async def process_file(self, file_path: str, enable_llm_refinement: bool = False, rag_client: Any = None) -> Dict[str, Any]:
        """
        Process a single file through the full pipeline.
        
        Args:
            file_path: Path to file
            enable_llm_refinement: Whether to use LLM to clean up result (slow, costs tokens)
            rag_client: RAGClient instance (required if enable_llm_refinement is True)
        """
        try:
            # 1. Validate
            is_valid, error_msg = self.validate_file(file_path)
            if not is_valid:
                return {"success": False, "error": error_msg, "file_path": file_path}

            # 2. Check Cache
            file_hash = self._calculate_file_hash(file_path)
            cached = self._get_from_cache(file_hash)
            if cached:
                self.logger.info(f"Cache hit for {Path(file_path).name}")
                return cached

            work_path = file_path
            metadata = {}
            converter = self.converter_fast # Default to fast

            # PDF Specific Steps (Repair & Smart Routing)
            if file_path.lower().endswith(".pdf"):
                # Repair
                work_path = str(self.repair_service.repair(file_path))
                
                # Analyze
                analysis = self.analyzer.analyze(work_path)
                metadata["pdf_analysis"] = analysis
                
                if analysis.get("is_encrypted"):
                    return {"success": False, "error": "PDF is encrypted", "file_path": file_path}

                # Smart Routing Logic
                # If no text found -> It's a scan -> Use Heavy Mode
                if not analysis.get("has_text", False):
                    self.logger.info(f"Smart Routing: Detected SCAN/IMAGE PDF. Switching to HEAVY mode (OCR).")
                    converter = self.converter_heavy
                else:
                    self.logger.info(f"Smart Routing: Detected DIGITAL PDF. Using FAST mode.")
                    converter = self.converter_fast

            # 3. Convert with Docling
            self.logger.info(f"Converting file: {work_path}")
            # Run conversion in thread pool to avoid blocking async loop
            import asyncio
            result = await asyncio.to_thread(converter.convert, work_path)
            
            # Export to Markdown
            markdown_content = result.document.export_to_markdown()
            
            # 4. Optional LLM Refinement
            if enable_llm_refinement and rag_client:
                self.logger.info("Refining content with LLM...")
                markdown_content = await self.refine_with_llm(markdown_content, rag_client)
                metadata["llm_refined"] = True

            # Cleanup temp file
            if work_path != file_path and os.path.exists(work_path):
                try:
                    os.remove(work_path)
                except OSError:
                    pass

            response = {
                "success": True,
                "content": markdown_content,
                "metadata": {**metadata, "docling_meta": result.document.export_to_dict().get("metadata", {})},
                "file_path": file_path,
                "content_length": len(markdown_content)
            }
            
            # Save to Cache
            self._save_to_cache(file_hash, response)
            
            return response

        except Exception as e:
            self.logger.error(f"Failed to process {file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }

    def process_directory(self, directory_path: str, recursive: bool = True) -> List[Dict]:
        """Process all supported files in a directory (Sync wrapper for compatibility)."""
        # Note: This method is synchronous but process_file is async. 
        # For full directory processing, we should use an async runner.
        # This is a simplified version that might block.
        # Ideally, callers should use process_file individually.
        self.logger.warning("process_directory is deprecated for heavy workloads. Use process_file.")
        return []

    def get_supported_extensions(self) -> List[str]:
        return sorted(self.SUPPORTED_EXTENSIONS)


# Singleton instance
docling_service = DoclingService()
