"""
Document Analysis Endpoints.

Provides AI-powered analysis of files before ingestion:
- Analyze individual files for collection recommendations
- Analyze entire folders for data classification
"""

from fastapi import APIRouter, Depends, HTTPException, Form
from src.core.exceptions import ValidationError, RAGFileNotFoundError, IngestionError
from typing import List
import logging

from src.api.v1.dependencies import get_rag_client
from src.services.auth_service import get_current_user
from src.database.models import User
from src.services.data_classifier_service import (
    get_data_classifier_service,
    DataClassifierService
)
from src.api.v1.rag.models import AnalyzeFilesRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze-files")
async def analyze_files_batch(
    request: AnalyzeFilesRequest,
    rag_client=Depends(get_rag_client),
    current_user: User = Depends(get_current_user)
):
    """
    Analyze multiple files in parallel and recommend collections.

    Uses AI to determine the best collection for each file based on content preview.
    """
    logger.debug(f"Analyzing {len(request.files)} files for collection recommendation")

    try:
        analyses = []

        # Check if LLM is available
        if not rag_client.llm:
            logger.warning("LLM not available, using default recommendations")
            for file in request.files:
                analyses.append({
                    "file": file.path,
                    "recommended_collection": "generic",
                    "confidence": 0.5,
                    "reasoning": "LLM not available, using default collection"
                })

            return {
                "analyses": analyses,
                "warning": "LLM not active, using default recommendations"
            }

        # Process each file
        for file_preview in request.files:
            try:
                logger.debug(f"Analyzing file: {file_preview.path}")

                result = await rag_client.analyze_document_sample(file_preview.preview[:2000])

                recommended_collection = result.get("recommended_collection", "generic")
                reasoning = result.get("chunk_strategy", "No specific reasoning provided")
                confidence = 0.8 if "recommend" in str(reasoning).lower() else 0.6

                analyses.append({
                    "file": file_preview.path,
                    "recommended_collection": recommended_collection,
                    "confidence": confidence,
                    "reasoning": reasoning
                })

                logger.info(f"File '{file_preview.path}' → collection '{recommended_collection}' (confidence: {confidence})")

            except Exception as file_error:
                logger.error(f"Analysis failed for '{file_preview.path}': {file_error}")
                analyses.append({
                    "file": file_preview.path,
                    "recommended_collection": "generic",
                    "confidence": 0.3,
                    "reasoning": f"Analysis failed: {str(file_error)}",
                    "error": str(file_error)
                })

        logger.info(f"Batch analysis complete: {len(analyses)} files analyzed")
        return {"analyses": analyses}

    except Exception as e:
        logger.error(f"Batch analysis failed: {e}", exc_info=True)
        raise IngestionError(str(e))


@router.post("/analyze-folder-contents")
async def analyze_folder_contents_endpoint(
    folder_path: str = Form(...),
    recursive: bool = Form(True),
    max_depth: int = Form(10),
    classifier_service: DataClassifierService = Depends(get_data_classifier_service),
    current_user: User = Depends(get_current_user)
):
    """
    Intelligently analyzes the contents of a specified folder using an LLM.

    Classifies files into categories and suggests optimal RAG ingestion parameters.
    """
    logger.debug(f"Analyzing folder contents: {folder_path}")

    try:
        analysis_results = await classifier_service.analyze_folder_contents(
            folder_path=folder_path,
            recursive=recursive,
            max_depth=max_depth
        )
        return {"analysis": analysis_results}
    except ValueError as e:
        raise ValidationError(str(e))
    except Exception as e:
        logger.error(f"Error analyzing folder contents: {e}", exc_info=True)
        raise IngestionError(f"Failed to analyze folder contents: {e}")
