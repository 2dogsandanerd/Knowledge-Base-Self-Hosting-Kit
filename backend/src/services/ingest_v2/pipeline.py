import asyncio
from typing import Dict, Any
from loguru import logger
import os

# Import the singleton instance of the classification service
from src.services.classification import classification_service

async def process_document_pipeline(file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    The main processing pipeline for a single document.

    This pipeline was missing and has been created to unblock the ingestion process.
    The first step is document classification.

    Args:
        file_path: The absolute path to the temporary file to be processed.
        metadata: A dictionary of metadata (e.g., filename, content_type).

    Returns:
        A dictionary containing the results of the processing steps.
    """
    logger.info(f"Starting new V2 ingestion pipeline for: {metadata.get('filename')}")
    pipeline_results = {}

    try:
        # --- 1. Read file content ---
        logger.debug(f"Reading content from file: {file_path}")
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text_content = f.read()
        
        if not text_content:
            logger.warning("Document is empty or could not be read.")
            # Early exit if no content
            os.remove(file_path)
            logger.debug(f"Removed temporary file: {file_path}")
            return {"status": "failed", "error": "Document is empty."}

        # --- 2. Classify Document ---
        logger.info("Step 1: Classifying document...")
        classification_result = await classification_service.classify_document(
            text=text_content,
            metadata=metadata
        )
        pipeline_results['classification'] = classification_result.dict()
        logger.success(f"Classification complete. Label: {classification_result.label}, Confidence: {classification_result.confidence}")

        # --- Future steps would go here ---
        # e.g., chunking, embedding, indexing
        logger.info("Pipeline finished (for now).")


    except Exception as e:
        logger.error(f"Error during document processing pipeline for {file_path}: {e}")
        raise  # Re-raise the exception to be caught by the Celery task

    finally:
        # --- 5. Cleanup ---
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Removed temporary file: {file_path}")

    return pipeline_results
