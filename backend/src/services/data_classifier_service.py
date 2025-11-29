"""
DataClassifierService for intelligent file content analysis using LLMs.

This service classifies files into predefined categories and suggests optimal
RAG ingestion parameters (e.g., chunk size, embedding model) based on content.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from src.core.config import get_config, LLMConfig
from src.core.llm_singleton import LLMSingleton
from src.services.folder_scanner import scan_folder, FileInfo
from src.utils.llm_response_parser import parse_json_response_with_llm

logger = logging.getLogger(__name__)

# Predefined categories and default suggestions
CATEGORIES = {
    "documents": {"description": "General text documents, reports, letters (PDF, DOCX).", "suggested_chunk_size": 512},
    "spreadsheets": {"description": "Tabular data, calculations, financial reports (XLSX, CSV).", "suggested_chunk_size": 1024},
    "correspondence": {"description": "Emails, faxes, memos (.eml, scanned PDFs).", "suggested_chunk_size": 512},
    "source_code": {"description": "Programming files (.py, .js, .ts, .jsx, .html, .css).", "suggested_chunk_size": 256},
    "presentation": {"description": "Presentations and slideshows (PPTX).", "suggested_chunk_size": 512},
    "generic": {"description": "Content that doesn't fit specific categories or cannot be determined.", "suggested_chunk_size": 512},
}

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text" # Or from config


class DataClassifierService:
    def __init__(self, llm_singleton: LLMSingleton, llm_config: LLMConfig):
        self.llm_singleton = llm_singleton
        self.llm_config = llm_config
        self.llm_client = llm_singleton.get_client()

    async def analyze_folder_contents(
        self,
        folder_path: str,
        recursive: bool = True,
        max_depth: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Scans a folder, reads file previews, and uses LLM to classify content
        and suggest RAG ingestion parameters.
        """
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            raise ValueError(f"Invalid folder path: {folder_path}")

        # Use a broad set of extensions for the initial scan
        # We need to explicitly list all possible extensions that might contain text
        # that an LLM can classify. Image/CAD files would need separate processing.
        broad_extensions = {
            ".pdf", ".docx", ".xlsx", ".eml", ".md", ".txt", ".csv", ".html",
            ".py", ".js", ".ts", ".jsx", ".css", ".json", ".xml", ".yaml", ".yml",
            ".java", ".cpp", ".c", ".h", ".go", ".rs", ".php", ".rb", ".swift", ".kt"
        }
        
        # Filter out .txt because folder_scanner handles it by converting to .md
        broad_extensions.discard(".txt")

        # Perform initial scan to get file list
        files_info: List[FileInfo] = scan_folder(
            folder_path,
            recursive=recursive,
            max_depth=max_depth,
            allowed_extensions=list(broad_extensions)
        )

        analysis_results: List[Dict[str, Any]] = []

        for file_info in files_info:
            try:
                # Read a small preview of the file content
                file_preview = self._get_file_preview(file_info.path)
                
                # Use LLM to classify and suggest parameters
                llm_classification = await self._classify_with_llm(file_info.path, file_preview)
                
                # Combine scan info with LLM classification
                analysis_results.append({
                    "file_path": file_info.path,
                    "filename": file_info.filename,
                    "extension": file_info.extension,
                    "size_bytes": file_info.size_bytes,
                    "size_human": file_info.size_human,
                    **llm_classification
                })
            except Exception as e:
                logger.error(f"Error analyzing file {file_info.path}: {e}")
                analysis_results.append({
                    "file_path": file_info.path,
                    "filename": file_info.filename,
                    "extension": file_info.extension,
                    "size_bytes": file_info.size_bytes,
                    "size_human": file_info.size_human,
                    "recommended_collection": "generic",
                    "confidence": 0.1,
                    "reasoning": f"Analysis failed: {e}",
                    "suggested_chunk_size": CATEGORIES["generic"]["suggested_chunk_size"],
                    "suggested_embedding_model": DEFAULT_EMBEDDING_MODEL
                })

        return analysis_results

    def _get_file_preview(self, file_path: str, max_chars: int = 4096) -> str:
        """
        Reads a small preview of the file content.
        Needs to handle various file types or rely on external tools for complex ones.
        For now, just reads text/markdown files.
        """
        file_ext = Path(file_path).suffix.lower()
        
        # Simple text-based preview for now.
        # For DOCX, PDF, XLSX, etc., we'd need libraries like python-docx, PyPDF2, openpyxl
        # or Docling integration to extract text.
        if file_ext in [".md", ".txt", ".py", ".js", ".ts", ".jsx", ".css", ".html", ".json", ".xml", ".yaml", ".yml",
                        ".java", ".cpp", ".c", ".h", ".go", ".rs", ".php", ".rb", ".swift", ".kt", ".csv"]:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read(max_chars)
            except Exception as e:
                logger.warning(f"Could not read text preview from {file_path}: {e}")
                return ""
        elif file_ext in [".pdf", ".docx", ".xlsx", ".pptx", ".eml"]:
            # Placeholder for complex document types.
            # Real implementation would call Docling's text extraction capabilities.
            return f"Binary file preview for {file_ext} at {file_path}. Content extraction pending."
        else:
            return f"No text preview available for file type {file_ext}."


    async def _classify_with_llm(self, file_path: str, file_preview: str) -> Dict[str, Any]:
        """
        Uses LLM to classify the file content and suggest RAG parameters.
        """
        prompt_template = """
        You are an expert data architect for a Retrieval-Augmented Generation system.
        Analyze the following file content and return a JSON object with the following structure:

        ```json
        {{
          "recommended_collection": "...",
          "confidence": <float_between_0_and_1>,
          "reasoning": "...",
          "suggested_chunk_size": <int>,
          "suggested_embedding_model": "..."
        }}
        ```

        **Categories:**
        {categories_description}

        **Chunk Size Suggestions (based on typical content structure):**
        - For `documents`: 512
        - For `spreadsheets`: 1024 (to keep rows/contexts together)
        - For `correspondence`: 512
        - For `source_code`: 256 (to keep functions/blocks atomic)
        - For `presentation`: 512
        - For `generic`: 512

        **Embedding Model Suggestion:**
        Always suggest "{default_embedding_model}" unless the content strongly implies a need for a specialized model for which you have knowledge (e.g., highly domain-specific, but generally stick to the default).

        Give a `confidence` score between 0.0 and 1.0 based on how certain you are about the classification.

        **File Content Preview (from {file_path}):**
        ```
        {file_preview}
        ```
        """

        categories_description = ""
        for cat, details in CATEGORIES.items():
            categories_description += f"- `{cat}`: {details['description']}\n"

        # Dynamically set embedding model based on configuration
        embedding_model = self.llm_config.embedding_model or DEFAULT_EMBEDDING_MODEL

        prompt = prompt_template.format(
            categories_description=categories_description.strip(),
            default_embedding_model=embedding_model,
            file_path=file_path,
            file_preview=file_preview[:2048] # Truncate preview for prompt length
        )

        try:
            # Use the LLM to get a JSON response
            raw_response = await self.llm_client.predict(prompt)
            
            # Attempt to parse the JSON response. If it's malformed, try to fix with LLM.
            parsed_response = parse_json_response_with_llm(raw_response, self.llm_client)

            # Validate and enrich response
            recommended_collection = parsed_response.get("recommended_collection", "generic")
            suggested_chunk_size = parsed_response.get("suggested_chunk_size")
            suggested_embedding_model = parsed_response.get("suggested_embedding_model", embedding_model)

            # Ensure chunk size is within range and is an integer
            if not isinstance(suggested_chunk_size, int) or not (128 <= suggested_chunk_size <= 2048):
                  suggested_chunk_size = CATEGORIES.get(recommended_collection, CATEGORIES["generic"])["suggested_chunk_size"]

            # If confidence is missing or not a float, set a default low value
            confidence = parsed_response.get("confidence")
            if not isinstance(confidence, (float, int)) or not (0.0 <= confidence <= 1.0):
                confidence = 0.2


            return {
                "recommended_collection": recommended_collection,
                "confidence": confidence,
                "reasoning": parsed_response.get("reasoning", "No specific reasoning from LLM."),
                "suggested_chunk_size": suggested_chunk_size,
                "suggested_embedding_model": suggested_embedding_model,
                "llm_raw_response": raw_response # For debugging
            }
        except Exception as e:
            logger.error(f"LLM classification failed for {file_path}: {e}")
            return {
                "recommended_collection": "generic",
                "confidence": 0.0,
                "reasoning": f"LLM classification failed: {e}",
                "suggested_chunk_size": CATEGORIES["generic"]["suggested_chunk_size"],
                "suggested_embedding_model": embedding_model
            }

# Dependency for FastAPI
async def get_data_classifier_service() -> DataClassifierService:
    llm_singleton = LLMSingleton()
    llm_config = get_config()
    return DataClassifierService(llm_singleton, llm_config)
