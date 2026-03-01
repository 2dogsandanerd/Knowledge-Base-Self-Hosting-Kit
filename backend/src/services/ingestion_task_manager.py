"""
Task Manager for asynchronous document ingestion.

This module provides in-memory task tracking for document processing.
Tasks are processed using FastAPI BackgroundTasks and status is stored
in memory for polling.

For production with multiple workers, consider using Redis or a database
for task state persistence.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass, field
from loguru import logger
from pathlib import Path

BM25_INDEX_DIR = Path("/app/data/bm25_indices")


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"  # Some files succeeded, some failed


@dataclass
class FileResult:
    """Result of processing a single file."""
    file_path: str
    filename: str
    success: bool
    chunks: int = 0
    error: Optional[str] = None
    file_hash: Optional[str] = None
    skipped: bool = False  # True if duplicate
    processing_time: float = 0.0


@dataclass
class IngestionTask:
    """Represents a document ingestion task."""
    task_id: str
    status: TaskStatus
    total_files: int
    processed_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    total_chunks: int = 0
    file_results: List[FileResult] = field(default_factory=list)
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    collection_name: str = "default"
    assignments: List[Dict] = field(default_factory=list)  # List of file assignment objects
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Phase 8 additions
    user_id: Optional[str] = None  # User who created the task
    retry_count: int = 0  # Number of retries performed
    max_retries: int = 3  # Maximum retries per file
    use_phase_7: bool = True  # Use DataTypeDetector for intelligent routing
    use_retry_policy: bool = True  # Enable retry policy for transient errors

    def to_dict(self) -> Dict:
        """Convert task to dictionary for API response."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "progress": {
                "total_files": self.total_files,
                "processed_files": self.processed_files,
                "successful_files": self.successful_files,
                "failed_files": self.failed_files,
                "skipped_files": self.skipped_files,
                "total_chunks": self.total_chunks
            },
            "collection_name": self.collection_name,
            "user_id": self.user_id,
            "retry_count": self.retry_count,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "file_results": [
                {
                    "filename": r.filename,
                    "success": r.success,
                    "chunks": r.chunks,
                    "error": r.error,
                    "skipped": r.skipped,
                    "processing_time": round(r.processing_time, 3)
                }
                for r in self.file_results
            ] if self.status in [TaskStatus.SUCCESS, TaskStatus.FAILED, TaskStatus.PARTIAL] else []
        }


def _tokenize_text(text: str) -> List[str]:
    """Simple tokenizer for BM25."""
    return text.lower().split()


class IngestionTaskManager:
    """Manages asynchronous document ingestion tasks.

    This is a singleton in-memory task tracker. For production with
    multiple workers, use Redis or database-backed storage.
    """

    def __init__(self):
        self._tasks: Dict[str, IngestionTask] = {}
        self.logger = logger.bind(component="IngestionTaskManager")

    def create_task(
        self,
        file_count: Optional[int] = None,
        collection_name: str = "default",
        assignments: List[Dict] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        user_id: Optional[str] = None,
        use_phase_7: bool = True,
        use_retry_policy: bool = True,
        max_retries: int = 3
    ) -> str:
        """Create a new ingestion task."""
        # Auto-calculate file_count from assignments if not provided
        if file_count is None:
            if assignments:
                file_count = len(assignments)
            else:
                raise ValueError("Either file_count or assignments must be provided")

        task_id = str(uuid.uuid4())

        task = IngestionTask(
            task_id=task_id,
            status=TaskStatus.PENDING,
            total_files=file_count,
            collection_name=collection_name,
            assignments=assignments or [],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            user_id=user_id,
            use_phase_7=use_phase_7,
            use_retry_policy=use_retry_policy,
            max_retries=max_retries
        )

        self._tasks[task_id] = task
        self.logger.info(f"Created task {task_id} for {file_count} files → {collection_name} (user: {user_id or 'anonymous'})")

        return task_id

    def get_task(self, task_id: str) -> Optional[IngestionTask]:
        """Get task by ID."""
        return self._tasks.get(task_id)

    def get_processor(self):
        """
        Get IngestionProcessor instance.
        
        ADAPTER PATTERN:
        Wraps the new decoupled IngestionProcessor to work with the old TaskManager interface.
        """
        from src.services.ingestion_processor import create_ingestion_processor
        processor = create_ingestion_processor()
        
        # Create a wrapper that mimics the old process_task_async signature
        class ProcessorAdapter:
            def __init__(self, manager, real_processor):
                self.manager = manager
                self.processor = real_processor
                
            async def process_task_async(self, task_id, rag_client):
                task = self.manager.get_task(task_id)
                if not task:
                    return {'success': False, 'error': 'Task not found'}
                
                self.manager.start_task(task_id)
                
                def progress_callback(status):
                    # This is a simplified callback, real implementation would update task state
                    pass
                
                # Call the real processor
                result = await self.processor.process(
                    assignments=task.assignments,
                    collection_name=task.collection_name,
                    rag_client=rag_client,
                    progress_callback=progress_callback
                )
                
                # Update task status based on result
                if result['success']:
                    task.successful_files = result['successful_files']
                    task.failed_files = result['failed_files']
                    task.skipped_files = result['skipped_files']
                    task.total_chunks = result['total_chunks']
                    task.file_results = result['file_results']
                    self.manager.complete_task(task_id)
                else:
                    self.manager.complete_task(task_id, error=result.get('error'))
                    
                return result

        return ProcessorAdapter(self, processor)
    
    # Legacy methods kept for compatibility but mostly unused by new processor
    def start_task(self, task_id: str):
        if task := self._tasks.get(task_id):
            task.status = TaskStatus.PROCESSING
            task.started_at = datetime.now()

    def complete_task(self, task_id: str, error: Optional[str] = None):
        if task := self._tasks.get(task_id):
            task.completed_at = datetime.now()
            if error:
                task.status = TaskStatus.FAILED
                task.error_message = error
            else:
                task.status = TaskStatus.SUCCESS


# Global singleton instance
ingestion_task_manager = IngestionTaskManager()
