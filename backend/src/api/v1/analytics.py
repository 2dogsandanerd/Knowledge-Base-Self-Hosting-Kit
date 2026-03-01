from src.core.exceptions import ServiceUnavailableError
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from typing import Dict, Any

from src.database.database import get_db
from src.services.analytics_service import AnalyticsService
from src.services.auth_service import get_current_user
from src.database.models import User

router = APIRouter()

@router.get("/engagement-score")
async def get_engagement_score(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Calculate customer engagement score."""
    try:
        analytics = AnalyticsService(db)
        
        score = analytics.calculate_engagement_score()
        
        return {
            "score": score,  # 0-100
            "effort_level": analytics.get_effort_level(score),
            "metrics": {
                "total_conversations": analytics.get_total_conversations(),
                "avg_response_time": analytics.get_avg_response_time(),
                "reply_rate": analytics.get_reply_rate(),
                "avg_conversation_length": analytics.get_avg_conversation_length()
            }
        }
    except Exception as e:
        raise ServiceUnavailableError("analytics", str(e))

@router.get("/learning-progress")
async def get_learning_progress(
    days: int = 30,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get learning progress over time."""
    try:
        analytics = AnalyticsService(db)
        
        # Daten für letzte N Tage
        data = analytics.get_daily_learning_counts(days)
        
        return {
            "labels": [d["date"] for d in data],  # ["2024-01-01", ...]
            "values": [d["count"] for d in data],  # [5, 8, 12, ...]
            "trend": analytics.calculate_trend(data)  # Linear regression
        }
    except Exception as e:
        raise ServiceUnavailableError("analytics", str(e))

@router.get("/conversation-stats")
async def get_conversation_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get conversation statistics by status."""
    try:
        analytics = AnalyticsService(db)
        
        stats = analytics.get_conversation_by_status()
        
        return {
            "labels": ["Completed", "In Progress", "Failed"],
            "values": [
                stats.get("PAIR_COMPLETED", 0),
                stats.get("DRAFT_CREATED", 0),
                stats.get("FAILED", 0)
            ]
        }
    except Exception as e:
        raise ServiceUnavailableError("analytics", str(e))

@router.get("/recommendations")
async def get_recommendations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get AI-generated learning recommendations."""
    try:
        analytics = AnalyticsService(db)
        
        return {
            "recommendations": analytics.generate_recommendations()
            # [{ type: "focus|success|warning", message: "..." }, ...]
        }
    except Exception as e:
        raise ServiceUnavailableError("analytics", str(e))

@router.get("/export")
async def export_learning_data(
    format: str = "csv",  # csv|json
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Export learning data."""
    try:
        analytics = AnalyticsService(db)
        
        if format == "csv":
            csv_data = analytics.export_to_csv()
            return Response(
                content=csv_data,
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=learning_data.csv"}
            )
        elif format == "json":
            json_data = analytics.export_to_json()
            return Response(
                content=json_data,
                media_type="application/json",
                headers={"Content-Disposition": "attachment; filename=learning_data.json"}
            )
        else:
            raise ValidationError("Invalid format. Use csv or json")
    except Exception as e:
        raise ServiceUnavailableError("analytics", str(e))

@router.delete("/reset")
async def reset_learning_data(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Reset all learning data (DANGEROUS)."""
    try:
        analytics = AnalyticsService(db)
        
        deleted_count = analytics.reset_all_data()
        
        return {
            "success": True,
            "deleted_count": deleted_count,
            "message": f"Successfully deleted {deleted_count} learning pairs"
        }
    except Exception as e:
        raise ServiceUnavailableError("analytics", str(e))

