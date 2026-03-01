from src.core.exceptions import ServiceUnavailableError
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from datetime import datetime

from src.api.v1.dependencies import get_statistics_service, get_current_user
from src.services.statistics_service import StatisticsService
from src.database.models import User

router = APIRouter()


@router.get("/email")
async def get_email_statistics(
    stats_service: StatisticsService = Depends(get_statistics_service),
    current_user: User = Depends(get_current_user)
):
    """Async get email processing statistics."""
    try:
        return await stats_service.get_email_statistics()
    except Exception as e:
        raise ServiceUnavailableError("statistics", str(e))


@router.get("/daily-counts")
async def get_daily_counts(
    days: int = 30,
    stats_service: StatisticsService = Depends(get_statistics_service),
    current_user: User = Depends(get_current_user)
):
    """Async get email counts per day."""
    try:
        counts = await stats_service.get_daily_email_counts(days)
        return {"email_counts": counts}
    except Exception as e:
        raise ServiceUnavailableError("statistics", str(e))


@router.get("/performance")
async def get_performance_metrics(
    stats_service: StatisticsService = Depends(get_statistics_service),
    current_user: User = Depends(get_current_user)
):
    """Async get system performance metrics."""
    try:
        return await stats_service.get_performance_metrics()
    except Exception as e:
        raise ServiceUnavailableError("statistics", str(e))


@router.get("/llm")
async def get_llm_statistics(
    stats_service: StatisticsService = Depends(get_statistics_service),
    current_user: User = Depends(get_current_user)
):
    """Async get LLM usage statistics."""
    try:
        return await stats_service.get_llm_statistics()
    except Exception as e:
        raise ServiceUnavailableError("statistics", str(e))


@router.get("/rag")
async def get_rag_statistics(
    stats_service: StatisticsService = Depends(get_statistics_service),
    current_user: User = Depends(get_current_user)
):
    """Async get RAG system statistics."""
    try:
        return await stats_service.get_rag_statistics()
    except Exception as e:
        raise ServiceUnavailableError("statistics", str(e))


@router.get("/health")
async def get_system_health(
    stats_service: StatisticsService = Depends(get_statistics_service),
    current_user: User = Depends(get_current_user)
):
    """Async get system health status."""
    try:
        return await stats_service.get_system_health()
    except Exception as e:
        raise ServiceUnavailableError("statistics", str(e))


@router.get("/activities")
async def get_recent_activities(
    limit: int = 10,
    stats_service: StatisticsService = Depends(get_statistics_service),
    current_user: User = Depends(get_current_user)
):
    """Async get recent system activities."""
    try:
        activities = await stats_service.get_recent_activities(limit)
        return {"activities": activities}
    except Exception as e:
        raise ServiceUnavailableError("statistics", str(e))


@router.get("/dashboard")
async def get_dashboard_stats(
    stats_service: StatisticsService = Depends(get_statistics_service),
    current_user: User = Depends(get_current_user)
):
    """Async get quick dashboard statistics."""
    try:
        return await stats_service.get_dashboard_stats()
    except Exception as e:
        raise ServiceUnavailableError("statistics", str(e))


@router.get("/export/csv")
async def export_statistics_csv(
    stats_service: StatisticsService = Depends(get_statistics_service),
    current_user: User = Depends(get_current_user)
):
    """Async export statistics as CSV."""
    try:
        csv_data = await stats_service.export_statistics_csv()
        return {
            "success": True,
            "data": csv_data,
            "filename": f"statistics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        }
    except Exception as e:
        raise ServiceUnavailableError("statistics", str(e))


@router.get("/export/json")
async def export_statistics_json(
    stats_service: StatisticsService = Depends(get_statistics_service),
    current_user: User = Depends(get_current_user)
):
    """Async export statistics as JSON."""
    try:
        json_data = await stats_service.export_statistics_json()
        return {
            "success": True,
            "data": json_data,
            "filename": f"statistics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        }
    except Exception as e:
        raise ServiceUnavailableError("statistics", str(e))