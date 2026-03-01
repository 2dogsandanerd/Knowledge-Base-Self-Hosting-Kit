"""
DataClassifierService for intelligent file content analysis using LLMs and Heuristics.

This service classifies files into predefined categories and suggests optimal
RAG ingestion parameters (e.g., chunk size, embedding model) based on content.
Integrates logic from DataTypeDetector for fast, rule-based initial classification.
"""

import os
import logging
import mimetypes
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import json

from src.core.config import get_config, LLMConfig
from src.core.llm_singleton import LLMSingleton
from src.services.folder_scanner import scan_folder, FileInfo
from src.utils.llm_response_parser import parse_json_response_with_llm

logger = logging.getLogger(__name__)

# Predefined categories and default suggestions
# Extended with specific document types from the implementation plan
CATEGORIES = {
    # Original categories
    "documents": {"description": "General text documents, reports, letters (PDF, DOCX).", "suggested_chunk_size": 512},
    "spreadsheets": {"description": "Tabular data, calculations, financial reports (XLSX, CSV).", "suggested_chunk_size": 1024},
    "correspondence": {"description": "Emails, faxes, memos (.eml, scanned PDFs).", "suggested_chunk_size": 512},
    "source_code": {"description": "Programming files (.py, .js, .ts, .jsx, .html, .css).", "suggested_chunk_size": 256},
    "presentation": {"description": "Presentations and slideshows (PPTX).", "suggested_chunk_size": 512},
    "generic": {"description": "Content that doesn't fit specific categories or cannot be determined.", "suggested_chunk_size": 512},
    
    # New specific categories
    "legal_documents": {"description": "Legal documents, contracts, agreements, terms of service.", "suggested_chunk_size": 1024},
    "financial_reports": {"description": "Financial statements, invoices, balance sheets, tax documents.", "suggested_chunk_size": 512},
    "technical_manuals": {"description": "Technical documentation, manuals, specifications, API docs.", "suggested_chunk_size": 512},
    "emails": {"description": "Email communications (specific alias for correspondence).", "suggested_chunk_size": 512}
}

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest" # Or from config

class DataClassifierService:
    # Heuristic mapping similar to DataTypeDetector
    EMAIL_EXTENSIONS = {'.eml', '.mbox', '.msg'}
    CODE_EXTENSIONS = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c',
                       '.h', '.hpp', '.go', '.rs', '.rb', '.php', '.cs', '.swift', 
                       '.html', '.css', '.json', '.xml', '.yaml', '.yml', '.sql'}
    TABLE_EXTENSIONS = {'.xlsx', '.xls', '.csv', '.tsv'}
    DOCLING_EXTENSIONS = {'.pdf', '.docx', '.pptx', '.md', '.txt', '.rst'}
    
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
        Scans a folder, reads file previews, and uses heuristics + LLM to classify content
        and suggest RAG ingestion parameters.
        """
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            raise ValueError(f"Invalid folder path: {folder_path}")

        # Use a broad set of extensions for the initial scan
        broad_extensions = self.EMAIL_EXTENSIONS | self.CODE_EXTENSIONS | self.TABLE_EXTENSIONS | self.DOCLING_EXTENSIONS

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
                # 1. Heuristic Classification
                heuristic_result = self._heuristic_classify(file_info.path)
                
                # 2. LLM Validation/Refinement (only if heuristic is not definitive or needs specific subtype)
                # For now, we always perform LLM analysis for higher accuracy on content subtype,
                # but we use heuristic as a baseline/fallback.
                
                # Read a small preview of the file content
                file_preview = self._get_file_preview(file_info.path)
                
                # Use LLM to classify and suggest parameters
                llm_classification = await self._classify_with_llm(file_info.path, file_preview, heuristic_result)
                
                # Combine scan info with Classification
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

    def _heuristic_classify(self, file_path: str) -> Dict[str, Any]:
        """
        Performs fast, rule-based classification based on extension and simple checks.
        Returns a simplified classification result dict.
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        result = {
            "category": "generic",
            "confidence": 0.3, # Low confidence by default logic
            "heuristic_hint": ""
        }

        if extension in self.CODE_EXTENSIONS:
            result["category"] = "source_code"
            result["confidence"] = 0.95
            result["heuristic_hint"] = "File extension indicates source code."
            
        elif extension in self.EMAIL_EXTENSIONS:
            result["category"] = "emails" # Using specific alias
            result["confidence"] = 0.95
            result["heuristic_hint"] = "File extension indicates email."
            
        elif extension in self.TABLE_EXTENSIONS:
            result["category"] = "spreadsheets"
            result["confidence"] = 0.9
            result["heuristic_hint"] = "File extension indicates tabular data."
            
        elif extension in self.DOCLING_EXTENSIONS:
            # Further check for PPTX
            if extension == '.pptx':
                result["category"] = "presentation"
                result["confidence"] = 0.9
            else:
                # PDF, DOCX, etc. could be anything (contract, manual, etc.)
                result["category"] = "documents"
                result["confidence"] = 0.6
                result["heuristic_hint"] = "File extension indicates general document."

        return result

    def _get_file_preview(self, file_path: str, max_chars: int = 4096) -> str:
        """
        Reads a small preview of the file content.
        Safe for large files as it only reads the first block.
        """
        file_ext = Path(file_path).suffix.lower()
        
        # Simple text-based preview for now.
        # For DOCX, PDF, XLSX, etc., we'd need libraries like python-docx, PyPDF2, openpyxl
        # or Docling integration to extract text. 
        # CAUTION: Docling/PyPDF might read whole file. 
        # Here we just try to read as text for text-like files, 
        # and placeholder for binaries until we implement safe binary preview.
        
        if file_ext in self.CODE_EXTENSIONS | {'.md', '.txt', '.csv', '.rst'}:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read(max_chars)
            except Exception as e:
                logger.warning(f"Could not read text preview from {file_path}: {e}")
                return ""
        else:
            # TODO: Integrate safe preview readers for PDF/DOCX if needed for better LLM context.
            # For now, relying on filename and extension for binary formats in LLM prompt often suffices,
            # or we need proper ingestion tools here.
            return f"Binary file preview for {file_ext} at {file_path}. Content extraction pending."

    async def _classify_with_llm(self, file_path: str, file_preview: str, heuristic_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uses LLM to classify the file content, using heuristic result as a hint.
        """
        prompt_template = """
        You are an expert data architect for a Retrieval-Augmented Generation system.
        Analyze the following file to determine its optimal processing category.

        **Heuristic Analysis Hint:**
        The system automatically guessed: category="{heuristic_category}" with confidence {heuristic_confidence}.
        Hint: {heuristic_hint}

        **Task:**
        Confirm or refine this classification based on the file content preview and filename.
        Return a JSON object:

        ```json
        {{
          "recommended_collection": "...",
          "confidence": <float_between_0_and_1>,
          "reasoning": "...",
          "suggested_chunk_size": <int>,
          "suggested_embedding_model": "..."
        }}
        ```

        **Available Categories:**
        {categories_description}

        **Guidance:**
        - If the file is a 'contract', 'agreement', 'terms', classify as `legal_documents`.
        - If the file is an 'invoice', 'balance sheet', classify as `financial_reports`.
        - If the file is a manual, API doc, classify as `technical_manuals`.
        - `documents` is a fallback for general documents that don't fit specific types.
        - `emails` is for correspondence.
        - `source_code` is for programming files.

        **Embedding Model Suggestion:**
        Always suggest "{default_embedding_model}" unless you have a very specific reason not to.

        **File Info:**
        Path: {file_path}
        
        **Content Preview:**
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
            file_preview=file_preview[:2048], # Truncate preview
            heuristic_category=heuristic_result.get("category"),
            heuristic_confidence=heuristic_result.get("confidence"),
            heuristic_hint=heuristic_result.get("heuristic_hint")
        )

        try:
            # Use the LLM to get a JSON response
            raw_response = await self.llm_client.predict(prompt)
            
            # Attempt to parse
            parsed_response = parse_json_response_with_llm(raw_response)

            # Validate and enrich response
            recommended_collection = parsed_response.get("recommended_collection", "generic")
            
            # Basic normalization of collection name if LLM hallucinated
            if recommended_collection not in CATEGORIES:
                # Fallback logic: check if it matches a key part
                found = False
                for cat in CATEGORIES:
                    if cat in recommended_collection.lower():
                        recommended_collection = cat
                        found = True
                        break
                if not found:
                    recommended_collection = "generic"

            suggested_chunk_size = parsed_response.get("suggested_chunk_size")
            suggested_embedding_model = parsed_response.get("suggested_embedding_model", embedding_model)

            # Ensure chunk size is valid
            if not isinstance(suggested_chunk_size, int) or not (128 <= suggested_chunk_size <= 4096):
                  suggested_chunk_size = CATEGORIES.get(recommended_collection, CATEGORIES["generic"])["suggested_chunk_size"]

            confidence = parsed_response.get("confidence")
            if not isinstance(confidence, (float, int)) or not (0.0 <= confidence <= 1.0):
                confidence = 0.5

            return {
                "recommended_collection": recommended_collection,
                "confidence": confidence,
                "reasoning": parsed_response.get("reasoning", "No specific reasoning from LLM."),
                "suggested_chunk_size": suggested_chunk_size,
                "suggested_embedding_model": suggested_embedding_model,
                "llm_raw_response": raw_response 
            }
        except Exception as e:
            logger.error(f"LLM classification failed for {file_path}: {e}")
            # Fallback to heuristic result if LLM fails
            return {
                "recommended_collection": heuristic_result.get("category", "generic"),
                "confidence": heuristic_result.get("confidence", 0.0),
                "reasoning": f"LLM classification failed: {e}. Used heuristic fallback.",
                "suggested_chunk_size": CATEGORIES.get(heuristic_result.get("category", "generic"), CATEGORIES["generic"])["suggested_chunk_size"],
                "suggested_embedding_model": embedding_model
            }

# Dependency for FastAPI
async def get_data_classifier_service() -> DataClassifierService:
    llm_singleton = LLMSingleton()
    llm_config = get_config()
    return DataClassifierService(llm_singleton, llm_config)
