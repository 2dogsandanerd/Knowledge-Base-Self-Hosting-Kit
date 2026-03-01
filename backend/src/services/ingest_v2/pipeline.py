import asyncio
from typing import Dict, Any, List
from loguru import logger
import os
import re

# Import services
from src.services.data_classifier_service import get_data_classifier_service
from src.services.document_router_service import DocumentRouterService

async def process_document_pipeline(file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    The main processing pipeline for a single document with intelligent routing.

    Args:
        file_path: The absolute path to the temporary file to be processed.
        metadata: A dictionary of metadata (e.g., filename, content_type).

    Returns:
        A dictionary containing the results of the processing steps.
    """
    logger.info(f"Starting intelligent routing pipeline for: {metadata.get('filename')}")
    pipeline_results = {}

    try:
        # --- 1. Get Classifier and Router ---
        data_classifier = await get_data_classifier_service()
        router_service = DocumentRouterService(data_classifier)
        
        # --- 2. Route Document ---
        logger.info("Step 1: Routing document...")
        routing_decision = await router_service.route_document(file_path, metadata)
        pipeline_results['routing_decision'] = routing_decision
        
        target_collection = routing_decision.get('target_collection')
        confidence = routing_decision.get('confidence')
        logger.success(f"Routed to '{target_collection}' (Conf: {confidence})")
        
        # --- 3. Get Processing Params ---
        params = routing_decision.get('processing_params', {})
        chunk_size = params.get('chunk_size', 512)
        chunk_overlap = params.get('chunk_overlap', 128)
        preprocessing_steps = params.get('preprocessing_steps', [])
        postprocessing_steps = params.get('postprocessing_steps', [])
        
        # --- 4. Read Content (Safe) ---
        # For text extraction, we still need to read the file. 
        # CAUTION: 'open().read()' is dangerous for huge files. 
        # But for text processing we eventually need the text.
        # Ideally we stream it. For now, we limit the read size as a safety guard 
        # or assume the file is reasonable size since it wasn't rejected by upload.
        # We will implement a size check before reading.
        
        MAX_TEXT_SIZE = 10 * 1024 * 1024 # 10 MB limit for text reading
        file_size = os.path.getsize(file_path)
        
        if file_size > MAX_TEXT_SIZE:
             logger.warning(f"File size {file_size} exceeds safety limit {MAX_TEXT_SIZE}. Truncating read.")
             # Read only up to limit
             with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                 text_content = f.read(MAX_TEXT_SIZE)
        else:
             with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                 text_content = f.read()

        if not text_content:
            logger.warning("Document is empty or could not be read.")
            return {"status": "failed", "error": "Document is empty."}
            
        # --- 5. Pre-processing ---
        processed_content = text_content
        for step in preprocessing_steps:
            if step == "clean_text":
                processed_content = _clean_text(processed_content)
            elif step == "extract_clauses":
                pipeline_results['clauses'] = _extract_clauses(processed_content)
            elif step == "extract_amounts":
                pipeline_results['amounts'] = _extract_amounts(processed_content)
            elif step == "preserve_syntax":
                # Do nothing, just mark logic
                pass
        
        # --- 6. Chunking ---
        logger.info(f"Step 2: Chunking (Size: {chunk_size}, Overlap: {chunk_overlap})...")
        chunks = _chunk_text(processed_content, chunk_size, chunk_overlap)
        pipeline_results['chunks_created'] = len(chunks)
        pipeline_results['chunks_sample'] = chunks[:1] if chunks else []
        
        # --- 7. Post-processing / Validation ---
        validation_passed = True
        if routing_decision.get('requires_validation', False):
            logger.info("Step 3: Validating...")
            for step in postprocessing_steps:
                if step == "validate_clauses":
                    res = _validate_clauses(pipeline_results.get('clauses', []))
                    pipeline_results['clause_validation'] = res
                    if not res['validation_passed']: validation_passed = False
                elif step == "validate_amounts":
                    res = _validate_amounts(pipeline_results.get('amounts', []))
                    pipeline_results['amount_validation'] = res
                    if not res['validation_passed']: validation_passed = False
                    
        pipeline_results['validation_passed'] = validation_passed
        logger.success("Pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Error during document processing pipeline for {file_path}: {e}")
        raise

    finally:
        # --- Cleanup ---
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Removed temporary file: {file_path}")

    return pipeline_results

def _clean_text(text: str) -> str:
    """Basic text cleaning."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def _extract_clauses(text: str) -> List[str]:
    """Simple clause extraction."""
    clause_patterns = [
        r'(Abschnitt|Section)\s+\d+[.:]?\s*[A-Z][^.]*?(?=(?:Abschnitt|Section)\s+\d+|$)',
        r'(Klausel|Clause)\s+\d+[.:]?\s*[A-Z][^.]*?(?=(?:Klausel|Clause)\s+\d+|$)'
    ]
    clauses = []
    for pattern in clause_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        clauses.extend(matches)
    return clauses

def _extract_amounts(text: str) -> List[str]:
    """Simple amount extraction."""
    amount_pattern = r'(\d{1,3}(?:[,.]\d{3})*(?:\.\d{2})?)\s*(€|\$|USD|EUR|GBP)?'
    amounts = re.findall(amount_pattern, text)
    return [" ".join(m) for m in amounts]

def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Simple chunking."""
    chunks = []
    start = 0
    text_length = len(text)
    if text_length == 0: return []
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start >= text_length - overlap and end >= text_length: break # prevent infinite loop near end
    return chunks

def _validate_clauses(clauses: List[str]) -> Dict[str, Any]:
    return {"total_clauses": len(clauses), "validation_passed": len(clauses) > 0}

def _validate_amounts(amounts: List[str]) -> Dict[str, Any]:
    return {"total_amounts": len(amounts), "validation_passed": len(amounts) > 0}
