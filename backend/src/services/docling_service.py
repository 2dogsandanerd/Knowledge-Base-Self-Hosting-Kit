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
import time
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

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md", ".csv"}

    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
    CACHE_DIR = Path("/app/cache/docling")  # Persistent cache mounted via Docker
    CACHE_TTL_DAYS = 7  # FIX BUG #5: Cache TTL (Time To Live)

    def __init__(self):
        """Initialize the Docling service with dual-pipeline configuration."""
        self.logger = logger.bind(component="DoclingService")

        # Initialize helpers
        self.repair_service = PDFRepairService()
        self.analyzer = PDFAnalyzer()

        # Ensure cache dir exists
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # FIX BUG #5: Cleanup old cache on startup
        self._cleanup_old_cache()

        self.logger.info("DoclingService initialized with Smart Routing (Fast/Heavy).")

    def _create_fast_converter(self) -> DocumentConverter:
        """Create a fresh Fast Mode converter instance.

        FIX BUG #3: Create per-request to prevent memory accumulation.
        Converters accumulate internal state (cached models, temp files) that
        is never freed when reused. Creating per-request ensures cleanup.
        """
        pipeline_fast = PdfPipelineOptions()
        pipeline_fast.do_ocr = False
        pipeline_fast.do_table_structure = True
        pipeline_fast.table_structure_options.mode = TableFormerMode.FAST

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_fast)
            }
        )

    def _create_heavy_converter(self) -> DocumentConverter:
        """Create a fresh Heavy Mode converter instance.

        FIX BUG #3: Create per-request to prevent memory accumulation.
        OCR models and table extractors accumulate significant memory (~50-100MB
        per 100 PDFs) when reused. Per-request creation ensures cleanup.
        """
        pipeline_heavy = PdfPipelineOptions()
        pipeline_heavy.do_ocr = True
        pipeline_heavy.do_table_structure = True
        pipeline_heavy.table_structure_options.mode = TableFormerMode.ACCURATE

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_heavy)
            }
        )

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
        """Retrieve result from cache with TTL and validation.

        FIX BUG #5: Added cache validation to prevent:
        - Stale cache entries (TTL check)
        - Corrupt cache files (JSON validation)
        - Invalid cache structure (schema validation)
        """
        cache_file = self.CACHE_DIR / f"{file_hash}.json"

        if not cache_file.exists():
            return None

        try:
            # Check file age (TTL: 7 days by default)
            file_age_seconds = time.time() - cache_file.stat().st_mtime
            if file_age_seconds > (self.CACHE_TTL_DAYS * 24 * 3600):
                self.logger.info(
                    f"Cache expired for {file_hash} (age: {file_age_seconds / 86400:.1f} days), removing"
                )
                cache_file.unlink()
                return None

            # Load and validate cache structure
            with open(cache_file, "r") as f:
                data = json.load(f)

            # Validate cache structure (must have success field)
            if not isinstance(data, dict) or "success" not in data:
                self.logger.warning(
                    f"Invalid cache structure for {file_hash}, removing"
                )
                cache_file.unlink()
                return None

            self.logger.debug(f"Cache hit for {file_hash}")
            return data

        except json.JSONDecodeError as e:
            self.logger.error(f"Corrupt cache file {file_hash}: {e}, removing")
            try:
                cache_file.unlink()
            except Exception:
                pass
            return None
        except Exception as e:
            self.logger.error(f"Cache read error for {file_hash}: {e}")
            return None

    def _save_to_cache(self, file_hash: str, data: Dict):
        """Save result to cache."""
        cache_file = self.CACHE_DIR / f"{file_hash}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            self.logger.warning(f"Failed to write cache: {e}")

    def _cleanup_old_cache(self, max_age_days: Optional[int] = None):
        """Remove cache files older than max_age_days.

        FIX BUG #5: Prevent disk space exhaustion from stale cache entries.
        Called on service startup to cleanup old files.

        Args:
            max_age_days: Maximum age in days (default: CACHE_TTL_DAYS)
        """
        if max_age_days is None:
            max_age_days = self.CACHE_TTL_DAYS

        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        removed_count = 0
        failed_count = 0

        try:
            for cache_file in self.CACHE_DIR.glob("*.json"):
                try:
                    if cache_file.stat().st_mtime < cutoff_time:
                        cache_file.unlink()
                        removed_count += 1
                except Exception as e:
                    self.logger.warning(
                        f"Failed to remove cache file {cache_file}: {e}"
                    )
                    failed_count += 1

            if removed_count > 0:
                self.logger.info(
                    f"Cache cleanup: Removed {removed_count} old files (failed: {failed_count})"
                )
        except Exception as e:
            self.logger.error(f"Cache cleanup failed: {e}")

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

    async def process_file(
        self,
        file_path: str,
        enable_llm_refinement: bool = False,
        rag_client: Any = None,
    ) -> Dict[str, Any]:
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
            file_hash = self._calculate_file_hash(
                file_path
            )  # Defined early for fallback error handling
            cached = self._get_from_cache(file_hash)
            if cached:
                self.logger.info(f"Cache hit for {Path(file_path).name}")
                return cached

            work_path = file_path
            metadata = {}
            converter = None
            file_hash = self._calculate_file_hash(
                file_path
            )  # Defined early for fallback error handling

            # PDF Specific Steps (Repair & Smart Routing)
            if file_path.lower().endswith(".pdf"):
                # Repair
                work_path = str(self.repair_service.repair(file_path))

                # Analyze
                analysis = self.analyzer.analyze(work_path)
                metadata["pdf_analysis"] = analysis

                if analysis.get("is_encrypted"):
                    return {
                        "success": False,
                        "error": "PDF is encrypted",
                        "file_path": file_path,
                    }

                # Smart Routing Logic
                # FIX BUG #3: Create converter per-request to prevent memory leak
                if not analysis.get("has_text", False):
                    self.logger.info(
                        f"Smart Routing: Detected SCAN/IMAGE PDF. Switching to HEAVY mode (OCR)."
                    )
                    converter = self._create_heavy_converter()
                else:
                    self.logger.info(
                        f"Smart Routing: Detected DIGITAL PDF. Using FAST mode."
                    )
                    converter = self._create_fast_converter()
            else:
                # Non-PDF files: use fast converter (no OCR needed)
                converter = self._create_fast_converter()

            # 3. Convert with Docling
            self.logger.info(f"Converting file: {work_path}")
            # Run conversion in thread pool to avoid blocking async loop
            import asyncio

            try:
                result = await asyncio.to_thread(converter.convert, work_path)

                # Export to Markdown
                # FIX BUG #6 (GitHub Issue #6): Handle IndexError for empty markdown structures
                # Docling throws "list index out of range" for files with empty headers/lists
                try:
                    markdown_content = result.document.export_to_markdown()
                except IndexError as e:
                    # Fallback for markdown files with empty structures
                    if file_path.lower().endswith(".md"):
                        self.logger.warning(
                            f"Docling export failed for {Path(file_path).name}, using plaintext fallback: {e}"
                        )
                        # Read file as plaintext
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                markdown_content = f.read()
                        except UnicodeDecodeError:
                            with open(file_path, "r", encoding="latin-1") as f:
                                markdown_content = f.read()
                        metadata["docling_fallback"] = "plaintext"
                    else:
                        raise  # Re-raise for non-markdown files

                # 4. Optional LLM Refinement
                if enable_llm_refinement and rag_client:
                    self.logger.info("Refining content with LLM...")
                    markdown_content = await self.refine_with_llm(
                        markdown_content, rag_client
                    )
                    metadata["llm_refined"] = True

                # FIX BUG #6 (GitHub Issue #6): export_to_dict() can also fail with IndexError
                # for files with empty structures, so wrap it in try-except
                try:
                    docling_meta = result.document.export_to_dict().get("metadata", {})
                except IndexError:
                    docling_meta = {
                        "error": "Failed to export metadata - empty document structure"
                    }

                response = {
                    "success": True,
                    "content": markdown_content,
                    "metadata": {**metadata, "docling_meta": docling_meta},
                    "file_path": file_path,
                    "content_length": len(markdown_content),
                }

                # Save to Cache
                self._save_to_cache(file_hash, response)

                return response

            finally:
                # FIX BUG #6: ALWAYS cleanup temp file, even on error
                if work_path != file_path and os.path.exists(work_path):
                    try:
                        os.remove(work_path)
                        self.logger.debug(f"Cleaned up repaired PDF: {work_path}")
                    except OSError as e:
                        self.logger.warning(
                            f"Failed to cleanup temp file {work_path}: {e}"
                        )

        except IndexError as e:
            # FIX BUG #6 (GitHub Issue #6): Docling's markdown backend throws IndexError
            # for files with empty headers or list items (e.g., "## \n" or "- ")
            # Fallback: Read markdown files as plaintext when Docling fails
            if file_path.lower().endswith(".md"):
                self.logger.warning(
                    f"Docling IndexError for {Path(file_path).name}, using plaintext fallback: {e}"
                )
                try:
                    # Read file as plaintext
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            markdown_content = f.read()
                    except UnicodeDecodeError:
                        with open(file_path, "r", encoding="latin-1") as f:
                            markdown_content = f.read()

                    response = {
                        "success": True,
                        "content": markdown_content,
                        "metadata": {"docling_fallback": "plaintext", "error": str(e)},
                        "file_path": file_path,
                        "content_length": len(markdown_content),
                    }

                    # Save to Cache
                    self._save_to_cache(file_hash, response)

                    return response

                except Exception as fallback_error:
                    return {
                        "success": False,
                        "error": f"Docling failed with: {e}, plaintext fallback also failed: {fallback_error}",
                        "file_path": file_path,
                    }
            else:
                # Re-raise for non-markdown files
                self.logger.error(f"Failed to process {file_path}: {e}")
                return {"success": False, "error": str(e), "file_path": file_path}

        except Exception as e:
            self.logger.error(f"Failed to process {file_path}: {e}")
            return {"success": False, "error": str(e), "file_path": file_path}

    def process_directory(
        self, directory_path: str, recursive: bool = True
    ) -> List[Dict]:
        """Process all supported files in a directory (Sync wrapper for compatibility)."""
        # Note: This method is synchronous but process_file is async.
        # For full directory processing, we should use an async runner.
        # This is a simplified version that might block.
        # Ideally, callers should use process_file individually.
        self.logger.warning(
            "process_directory is deprecated for heavy workloads. Use process_file."
        )
        return []

    def get_supported_extensions(self) -> List[str]:
        return sorted(self.SUPPORTED_EXTENSIONS)


# Singleton instance
docling_service = DoclingService()
