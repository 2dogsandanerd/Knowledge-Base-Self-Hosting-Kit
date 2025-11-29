"""
Batch Ingestion Endpoint.

Handles multi-file, multi-collection batch ingestion with:
- File validation
- Auto-collection creation
- Async task processing
- Progress tracking
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Dict, Any
import logging
import os
from pathlib import Path
import asyncio

from src.api.v1.dependencies import get_rag_client
from src.services.auth_service import get_current_user
from src.database.models import User
from src.api.v1.rag.models import IngestBatchRequest, IngestionResponse
from src.core.exceptions import (
    RAGFileNotFoundError,
    InvalidFileTypeError,
    IngestionError,
    CollectionNotFoundError,
    ChromaDBError,
    ValidationError
)
from src.core.feature_limits import FeatureLimits, Edition

logger = logging.getLogger(__name__)
router = APIRouter()


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

    # Check feature availability based on edition
    edition = rag_client.edition
    if not FeatureLimits.is_feature_enabled("batch_processing", edition):
        raise HTTPException(
            status_code=403,
            detail="Batch processing is not available in Developer Edition. Upgrade to Team Edition to use this feature."
        )

    # For Developer Edition, limit to single collection
    collections_affected = set(assignment.collection for assignment in request.assignments)
    if len(collections_affected) > 1 and not FeatureLimits.is_feature_enabled("multi_collection_search", edition):
        raise HTTPException(
            status_code=403,
            detail="Multi-collection batch ingestion is not available in Developer Edition. "
                   "Upgrade to Team Edition to ingest files into multiple collections simultaneously."
        )

    # For Developer Edition, check document limits before ingestion
    if edition == Edition.DEVELOPER:
        # Check if we're approaching document limit
        total_files = len(request.assignments)
        if total_files > FeatureLimits.get_limit_value('max_documents_per_collection', edition):
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Batch size exceeds document limit for Developer Edition. "
                    f"Maximum {FeatureLimits.get_limit_value('max_documents_per_collection', edition)} documents per collection allowed. "
                    "Upgrade to Team Edition for more documents."
                )
            )

    try:
        from src.services.ingestion_task_manager import ingestion_task_manager

        # STEP 1: Validate all files exist and are readable
        validated_assignments = []

        for assignment in request.assignments:
            file_path = assignment.file  # FIXED: .file statt .filename

            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                raise RAGFileNotFoundError(file_path)

            if not os.path.isfile(file_path):
                logger.error(f"Path is not a file: {file_path}")
                raise ValidationError(
                    f"Path is not a file: {file_path}",
                    details={"file_path": file_path, "type": "directory"}
                )

            # For Developer Edition, check file format
            if edition == Edition.DEVELOPER:
                file_ext = Path(file_path).suffix.lower()
                allowed_formats = FeatureLimits.get_limit_value('allowed_file_formats', Edition.DEVELOPER)
                if file_ext not in allowed_formats:
                    raise ValidationError(
                        f"File format {file_ext} not supported in Developer Edition. "
                        f"Only {', '.join(allowed_formats)} formats are supported. "
                        "Upgrade to Team Edition for more formats.",
                        details={"file_path": file_path, "format": file_ext}
                    )

            # Read file content for validation
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Cannot read file {file_path}: {e}")
                raise ValidationError(
                    f"Cannot read file: {Path(file_path).name}",
                    details={"file_path": file_path, "error": str(e)}
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

        # STEP 2: Ensure all collections exist and check limits
        for collection_name in collections_affected:
            # For Developer Edition, ensure we don't exceed collection limit
            if edition == Edition.DEVELOPER:
                all_collections_response = await rag_client.list_collections()
                if all_collections_response.is_success and all_collections_response.data:
                    current_count = len(all_collections_response.data)
                    if current_count >= FeatureLimits.get_limit_value('max_collections', edition):
                        raise HTTPException(
                            status_code=403,
                            detail=(
                                f"Collection limit reached for Developer Edition. "
                                f"Maximum {FeatureLimits.get_limit_value('max_collections', edition)} collection(s) allowed. "
                                "Upgrade to Team Edition for more collections."
                            )
                        )

            try:
                collection = await asyncio.to_thread(rag_client.chroma_manager.get_collection, collection_name)
                if not collection:
                    logger.info(f"Creating missing collection: {collection_name}")
                    response = await rag_client.create_collection(name=collection_name)
                    if not response.is_success:
                        logger.error(f"Failed to create collection {collection_name}: {response.error}")
                        raise ChromaDBError(f"Failed to create collection: {response.error}")
            except ChromaDBError:
                raise
            except Exception as e:
                logger.info(f"Creating collection: {collection_name}")
                response = await rag_client.create_collection(name=collection_name)
                if not response.is_success:
                    logger.error(f"Failed to create collection {collection_name}: {response.error}")
                    raise ChromaDBError(f"Failed to create collection: {response.error}")

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

    except (RAGFileNotFoundError, ValidationError, ChromaDBError, IngestionError):
        raise  # Re-raise custom exceptions
    except Exception as e:
        logger.error(f"Batch ingestion failed: {e}", exc_info=True)
        raise IngestionError(str(e), details={"files_count": len(request.assignments)})
