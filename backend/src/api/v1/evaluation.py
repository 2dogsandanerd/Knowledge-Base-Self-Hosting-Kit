"""
Evaluation API endpoints.

Provides access to RAG quality metrics and evaluation statistics.
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any
from loguru import logger

from src.core.evaluation import get_evaluation_service
from src.services.auth_service import get_current_user
from src.database.models import User

router = APIRouter()


@router.get("/stats")
async def get_evaluation_stats(
    limit: int = 100,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get aggregated evaluation statistics.
    
    Args:
        limit: Number of recent evaluations to analyze
        
    Returns:
        Dict with evaluation metrics averages
    """
    eval_service = get_evaluation_service()
    
    if not eval_service.enabled:
        return {
            "enabled": False,
            "message": "RAGAS evaluation is not enabled. Install 'ragas' and 'datasets' packages."
        }
    
    stats = await eval_service.get_stats(limit=limit)
    
    return {
        "enabled": True,
        **stats
    }


@router.get("/health")
async def evaluation_health(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Check if evaluation service is healthy.
    
    Returns:
        Dict with health status
    """
    eval_service = get_evaluation_service()
    
    return {
        "enabled": eval_service.enabled,
        "log_file": str(eval_service.log_file) if eval_service.enabled else None,
        "status": "healthy" if eval_service.enabled else "disabled"
    }
