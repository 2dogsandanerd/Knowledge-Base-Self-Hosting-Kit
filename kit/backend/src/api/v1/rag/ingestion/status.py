"""
Ingestion Status Endpoint.

Provides real-time status polling for background ingestion tasks.
"""

from fastapi import APIRouter, Depends, HTTPException
from src.core.exceptions import TaskNotFoundError
import logging

from src.api.v1.dependencies import get_rag_client
from src.services.auth_service import get_current_user
from src.database.models import User

logger = logging.getLogger(__name__)
router = APIRouter()


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
            raise TaskNotFoundError(task_id)

        logger.debug(f"Task {task_id} status: {task.status.value}")
        return task.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status: {e}", exc_info=True)
        raise TaskNotFoundError(task_id)
