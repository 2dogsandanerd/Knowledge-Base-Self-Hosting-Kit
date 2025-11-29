"""
Ingestion Processor - File processing logic for RAG ingestion.

This module handles the actual file processing during ingestion tasks,
separated from task management for better testability and maintainability.
"""

import asyncio
import time
from typing import Dict, Any, List, Callable, Optional
from loguru import logger

from src.services.ingestion_task_manager import (
    FileResult,
    IngestionTask,
    TaskStatus
)

class IngestionProcessor:
    """
    Processes ingestion tasks by loading, chunking, and indexing files.
    
    Decoupled from IngestionTaskManager to allow usage in Celery workers.
    """

    def __init__(self):
        """Initialize processor."""
        self.logger = logger.bind(component="IngestionProcessor")

    async def process(
        self,
        assignments: List[Dict],
        collection_name: str,
        rag_client,
        progress_callback: Optional[Callable[[Dict], None]] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        use_retry_policy: bool = True
    ) -> Dict[str, Any]:
        """
        Process ingestion task asynchronously with parallel file processing.

        Args:
            assignments: List of file assignment objects
            collection_name: Target ChromaDB collection
            rag_client: RAGClient instance
            progress_callback: Optional callback for status updates
            chunk_size: Chunk size
            chunk_overlap: Chunk overlap
            use_retry_policy: Enable retry policy

        Returns:
            Dict with success status, counts, and any errors
        """
        from src.core.retry_policy import INGESTION_RETRY

        # Initialize counters
        stats = {
            'processed_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'skipped_files': 0,
            'total_chunks': 0,
            'file_results': []
        }
        
        total_files = len(assignments)

        # Initialize retry policy if enabled
        retry_policy = INGESTION_RETRY if use_retry_policy else None

        # Concurrency control: max 5 files in parallel
        semaphore = asyncio.Semaphore(5)

        try:
            if not assignments:
                return {'success': False, 'error': 'No file assignments found'}

            # Process all files in parallel
            file_results = await asyncio.gather(
                *[self._process_single_file(
                    assignment, 
                    collection_name, 
                    rag_client, 
                    semaphore, 
                    retry_policy
                  ) for assignment in assignments],
                return_exceptions=False
            )

            # Aggregate results
            for result in file_results:
                stats['processed_files'] += 1
                stats['file_results'].append(result)
                
                if result.skipped:
                    stats['skipped_files'] += 1
                elif result.success:
                    stats['successful_files'] += 1
                    stats['total_chunks'] += result.chunks
                else:
                    stats['failed_files'] += 1
                
                # Report progress if callback provided
                if progress_callback:
                    progress = int((stats['processed_files'] / total_files) * 100)
                    progress_callback({
                        'progress': progress,
                        'processed': stats['processed_files'],
                        'total': total_files,
                        'current_file': result.filename
                    })

            # Phase 8e: Register collections in registry (graceful degradation)
            if stats['successful_files'] > 0:
                await self._register_collections_in_registry(
                    file_results, 
                    assignments,
                    None  # Detector not available in this context yet
                )

            return {
                'success': True,
                **stats
            }

        except Exception as e:
            self.logger.error(f"Ingestion process failed: {e}")
            return {'success': False, 'error': str(e)}

    async def _process_single_file(
        self,
        assignment: dict,
        collection_name: str,
        rag_client,
        semaphore: asyncio.Semaphore,
        retry_policy=None
    ) -> FileResult:
        """
        Processes a single file by calling the ExtractionService and then the IndexingService.
        """
        from src.database.database import AsyncSessionLocal
        from src.services.extraction_service import ExtractionService
        from src.core.indexing_service import Document as IndexingDocument
        from src.core.exceptions import ExtractionError

        async with semaphore:
            start_time = time.time()
            file_path = assignment['file_path']
            filename = assignment['filename']
            # collection_name passed as arg, but assignment might override (legacy support)
            target_collection = assignment.get('collection', collection_name)

            try:
                # 1. Get DB Session and instantiate services
                async with AsyncSessionLocal() as db:
                    extraction_svc = ExtractionService(db)
                    indexing_svc = rag_client.indexing_service

                    # 2. Call Extraction Service
                    try:
                        extraction_result = await extraction_svc.extract_document(
                            file_path=file_path,
                            original_filename=filename
                        )
                    except ExtractionError as e:
                        if "Duplicate file" in str(e):
                            logger.info(f"Skipping duplicate file: {filename}")
                            return FileResult(
                                file_path=file_path,
                                filename=filename,
                                success=False,
                                error="Duplicate file",
                                skipped=True,
                                processing_time=time.time() - start_time
                            )
                        raise e

                    # 3. Indexing
                    if extraction_result and extraction_result.extracted_text:
                        doc_to_index = IndexingDocument(
                            content=extraction_result.extracted_text,
                            metadata=extraction_result.metadata,
                            doc_id=extraction_result.file_hash
                        )

                        indexing_response = await indexing_svc.index_documents(
                            documents=[doc_to_index],
                            collection_name=target_collection
                        )

                        if indexing_response.status != "SUCCESS":
                            raise Exception(f"Indexing failed: {indexing_response.error}")

                        total_chunks = indexing_response.data.get("indexed_nodes", 0)
                        
                        return FileResult(
                            file_path=file_path,
                            filename=filename,
                            success=True,
                            chunks=total_chunks,
                            processing_time=time.time() - start_time
                        )
                    else:
                        return FileResult(
                            file_path=file_path,
                            filename=filename,
                            success=False,
                            error="Extraction resulted in no text",
                            skipped=True,
                            processing_time=time.time() - start_time
                        )

            except Exception as e:
                logger.error(f"Failed to process {filename}: {e}")
                return FileResult(
                    file_path=file_path,
                    filename=filename,
                    success=False,
                    error=str(e),
                    processing_time=time.time() - start_time
                )

    async def _register_collections_in_registry(
        self,
        file_results: List[FileResult],
        assignments: List[Dict],
        detector
    ):
        """Register processed collections in the Collection Registry."""
        try:
            from src.database.database import AsyncSessionLocal
            from src.core.collection_registry import CollectionRegistry

            # Collect unique collections from successful files
            collections_used = set()
            for result in file_results:
                if result.success and not result.skipped:
                    for assignment in assignments:
                        if assignment['filename'] == result.filename:
                            collections_used.add(assignment.get('collection', 'default'))
                            break

            if not collections_used:
                return

            async with AsyncSessionLocal() as db:
                registry = CollectionRegistry(db)
                for collection_name in collections_used:
                    # Default registration for now
                    await registry.register_collection(
                        collection_name=collection_name,
                        index_strategy="vector",
                        data_type="unstructured_text"
                    )

        except Exception as e:
            self.logger.warning(f"Failed to register collections in registry: {e}")

# Factory function
def create_ingestion_processor(task_manager=None):
    """Create IngestionProcessor instance."""
    return IngestionProcessor()
