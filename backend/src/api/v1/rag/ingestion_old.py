"""
RAG Docling Ingestion endpoints (Phase 4).

Advanced document processing with Docling integration:
- Multi-file analysis
- Smart collection routing
- Async batch processing
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Form

from typing import List, Dict, Any, Optional

import logging

import os

from pathlib import Path

import asyncio



from src.api.v1.dependencies import get_rag_client

from src.services.auth_service import get_current_user

from src.database.models import User

from src.services.data_classifier_service import get_data_classifier_service, DataClassifierService

from .models import (

    FilePreview,

    AnalyzeFilesRequest,

    FileAssignment,

    IngestBatchRequest,

    IngestionResponse

)



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

        raise HTTPException(status_code=500, detail=str(e))





@router.post("/ingest-batch", response_model=IngestionResponse)

async def ingest_batch_multi_collection(

    background_tasks: BackgroundTasks,

    request: IngestBatchRequest,

    rag_client=Depends(get_rag_client),

    current_user: User = Depends(get_current_user)

):

    """

    Start batch ingestion with files assigned to multiple collections.



    This supports:

    - Files going to different collections

    - Auto-creation of missing collections

    - Background processing with task tracking



    Returns task_id for status polling.

    """

    logger.debug(f"Starting batch ingestion: {len(request.assignments)} files")



    try:

        from src.services.ingestion_task_manager import ingestion_task_manager



        # STEP 1: Validate all files exist and are readable

        validated_assignments = []

        collections_affected = set()



        for assignment in request.assignments:

            file_path = assignment.file  # Assuming file contains path



            if not os.path.exists(file_path):

                logger.error(f"File not found: {file_path}")

                raise HTTPException(

                    status_code=400,

                    detail=f"File not found: {file_path}"

                )



            if not os.path.isfile(file_path):

                logger.error(f"Path is not a file: {file_path}")

                raise HTTPException(

                    status_code=400,

                    detail=f"Path is not a file: {file_path}"

                )



            # Read file content for validation

            try:

                with open(file_path, 'rb') as f:

                    content = f.read()

            except Exception as e:

                logger.error(f"Cannot read file {file_path}: {e}")

                raise HTTPException(

                    status_code=400,

                    detail=f"Cannot read file {file_path}: {str(e)}"

                )



            validated_assignments.append({

                "file_path": file_path,

                "filename": Path(file_path).name,

                "collection": assignment.collection,

                "content": content,

                "size": len(content)

            })



            collections_affected.add(assignment.collection)



        logger.info(f"Validated {len(validated_assignments)} files for {len(collections_affected)} collections")



        # STEP 2: Ensure all collections exist

        for collection_name in collections_affected:

            try:

                collection = await asyncio.to_thread(rag_client.chroma_manager.get_collection, collection_name)

                if not collection:

                    logger.info(f"Creating missing collection: {collection_name}")

                    response = await rag_client.create_collection(name=collection_name)

                    if not response.is_success:

                        logger.error(f"Failed to create collection {collection_name}: {response.error}")

                        raise HTTPException(status_code=500, detail=f"Failed to create collection: {response.error}")

            except Exception as e:

                logger.info(f"Creating collection: {collection_name}")

                response = await rag_client.create_collection(name=collection_name)

                if not response.is_success:

                    logger.error(f"Failed to create collection {collection_name}: {response.error}")

                    raise HTTPException(status_code=500, detail=f"Failed to create collection: {response.error}")



        # STEP 3: Create background task

        # For batch ingest with multiple collections, use first collection or "batch" as task name

        primary_collection = list(collections_affected)[0] if collections_affected else "batch"



        task_id = ingestion_task_manager.create_task(

            collection_name=primary_collection,

            user_id=str(current_user.id),

            assignments=validated_assignments,

            chunk_size=request.chunk_size if hasattr(request, 'chunk_size') else 500,

            chunk_overlap=request.chunk_overlap if hasattr(request, 'chunk_overlap') else 50

        )



        logger.info(f"Created ingestion task: {task_id}")



        # STEP 4: Start async processing if requested

        if request.async_mode:

            # Phase 8f: Use new async process_task_async() method via processor

            processor = ingestion_task_manager.get_processor()

            background_tasks.add_task(

                processor.process_task_async,

                task_id=task_id,

                rag_client=rag_client

            )

            logger.info(f"Started async processing for task {task_id}")



            return IngestionResponse(

                success=True,

                processed_files=0,

                failed_files=0,

                details={

                    "task_id": task_id,

                    "status": "processing",

                    "total_files": len(validated_assignments),

                    "collections": list(collections_affected)

                },

                task_id=task_id

            )

        else:

            # Synchronous processing (Phase 8f: using async method via processor)

            logger.info("Processing batch synchronously")

            processor = ingestion_task_manager.get_processor()

            result = await processor.process_task_async(task_id, rag_client)



            return IngestionResponse(

                success=result.get("success", False),

                processed_files=result.get("processed_files", 0),

                failed_files=result.get("failed_files", 0),

                details=result,

                task_id=task_id

            )



    except HTTPException:

        raise

    except Exception as e:

        logger.error(f"Batch ingestion failed: {e}", exc_info=True)

        raise HTTPException(status_code=500, detail=str(e))





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

        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:

        logger.error(f"Error analyzing folder contents: {e}", exc_info=True)

        raise HTTPException(status_code=500, detail=f"Failed to analyze folder contents: {e}")





@router.post("/scan-folder")

async def scan_folder_endpoint(

    folder_path: str = Form(...),

    recursive: bool = Form(True),

    max_depth: int = Form(10),

    allowed_extensions_str: Optional[str] = Form(None),

    current_user: User = Depends(get_current_user)

):

    """

    Scan a folder for Docling-compatible files.

    

    Args:

        folder_path: Path to directory to scan

        recursive: Whether to scan subdirectories

        max_depth: Maximum recursion depth

        allowed_extensions_str: Comma-separated string of extensions (e.g., .py,.js)

    

    Returns:

        List of files with metadata

    """

    from src.services.folder_scanner import scan_folder, FileInfo

    

    logger.debug(f"Scanning folder: {folder_path} (recursive: {recursive}, max_depth: {max_depth})")

    

    allowed_extensions = None

    if allowed_extensions_str:

        allowed_extensions = [ext.strip() for ext in allowed_extensions_str.split(',')]

        logger.info(f"Filtering by extensions: {allowed_extensions}")



    try:

        # Validate folder path

        if not os.path.exists(folder_path):

            logger.error(f"Folder does not exist: {folder_path}")

            raise HTTPException(status_code=400, detail=f"Folder does not exist: {folder_path}")

        

        if not os.path.isdir(folder_path):

            logger.error(f"Path is not a directory: {folder_path}")

            raise HTTPException(status_code=400, detail=f"Path is not a directory: {folder_path}")

        

        # Perform scan

        files_info = scan_folder(

            folder_path, 

            recursive=recursive, 

            max_depth=max_depth, 

            allowed_extensions=allowed_extensions

        )

        

        # Convert to response format

        files = []

        total_size = 0

        extension_counts = {}

        

        for file_info in files_info:

            if file_info.error is None:  # Only include valid files

                files.append({

                    "path": file_info.path,

                    "original_path": file_info.original_path,

                    "filename": file_info.filename,

                    "extension": file_info.extension,

                    "size_bytes": file_info.size_bytes,

                    "size_human": file_info.size_human,

                    "is_txt_converted": file_info.is_txt_converted,

                    "error": file_info.error

                })

                total_size += file_info.size_bytes

                

                # Count by extension

                ext = file_info.extension.lower()

                extension_counts[ext] = extension_counts.get(ext, 0) + 1

        

        response = {

            "files": files,

            "total_files": len(files),

            "total_size": total_size,

            "summary": extension_counts

        }

        

        logger.info(f"Folder scan complete: {len(files)} files found in {folder_path}")

        return response

        

    except HTTPException:

        raise

    except Exception as e:

        logger.error(f"Folder scan failed: {e}", exc_info=True)

        raise HTTPException(status_code=500, detail=str(e))





@router.get("/ingest-status/{task_id}")

async def get_ingestion_status(

    task_id: str,

    rag_client=Depends(get_rag_client),

    current_user: User = Depends(get_current_user)

):

    """

    Get status of a background ingestion task.

    """

    logger.debug(f"Checking status for task: {task_id}")



    try:

        from src.services.ingestion_task_manager import ingestion_task_manager



        task = ingestion_task_manager.get_task(task_id)



        if not task:

            logger.warning(f"Task not found: {task_id}")

            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")



        logger.debug(f"Task {task_id} status: {task.status.value}")

        return task.to_dict()



    except HTTPException:

        raise

    except Exception as e:

        logger.error(f"Failed to get task status: {e}", exc_info=True)

        raise HTTPException(status_code=500, detail=str(e))
