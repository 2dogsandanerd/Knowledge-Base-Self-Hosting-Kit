"""
User Feedback API for RAG Query Responses.

Allows users to provide feedback on query responses for quality monitoring.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timedelta
from sqlalchemy import func

from src.database.models import QueryFeedback, User
from src.database.database import get_db
from src.api.v1.dependencies import get_current_user

router = APIRouter(prefix="/feedback", tags=["feedback"])


class FeedbackRequest(BaseModel):
    """Request model for submitting feedback."""
    query_id: str = Field(..., description="Unique query ID")
    helpful: Optional[bool] = Field(None, description="Was the response helpful?")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating 1-5 stars")
    comment: Optional[str] = Field(None, max_length=1000, description="Optional comment")


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    message: str
    feedback_id: int


class FeedbackStats(BaseModel):
    """Statistics about user feedback."""
    total_feedback: int
    helpful_count: int
    unhelpful_count: int
    helpful_rate: float
    average_rating: float
    period_days: int


@router.post("/", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackRequest,
    user: User = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Submit user feedback on a query response.
    
    Users can provide:
    - Thumbs up/down (helpful)
    - Star rating (1-5)
    - Free text comment
    """
    # Check if feedback already exists for this query
    existing = db.query(QueryFeedback).filter(
        QueryFeedback.query_id == feedback.query_id
    ).first()
    
    if existing:
        # Update existing feedback
        if feedback.helpful is not None:
            existing.helpful = feedback.helpful
        if feedback.rating is not None:
            existing.rating = feedback.rating
        if feedback.comment is not None:
            existing.comment = feedback.comment
        
        db.commit()
        
        return FeedbackResponse(
            message="Feedback updated",
            feedback_id=existing.id
        )
    
    # Create new feedback
    # Note: query_text and response_text should be provided by the client
    # or fetched from a query log. For now, we'll use placeholders.
    db_feedback = QueryFeedback(
        query_id=feedback.query_id,
        user_id=user.id,
        query_text="",  # Should be populated from query log
        response_text="",  # Should be populated from query log
        helpful=feedback.helpful,
        rating=feedback.rating,
        comment=feedback.comment
    )
    
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    
    return FeedbackResponse(
        message="Feedback recorded",
        feedback_id=db_feedback.id
    )


@router.get("/stats", response_model=FeedbackStats)
async def get_feedback_stats(
    days: int = 7,
    user: User = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Get feedback statistics for the last N days.
    
    Returns:
    - Total feedback count
    - Helpful/unhelpful counts
    - Helpful rate percentage
    - Average star rating
    """
    since = datetime.utcnow() - timedelta(days=days)
    
    # Total feedback
    total = db.query(QueryFeedback).filter(
        QueryFeedback.created_at >= since
    ).count()
    
    # Helpful count
    helpful = db.query(QueryFeedback).filter(
        QueryFeedback.created_at >= since,
        QueryFeedback.helpful == True
    ).count()
    
    # Unhelpful count
    unhelpful = db.query(QueryFeedback).filter(
        QueryFeedback.created_at >= since,
        QueryFeedback.helpful == False
    ).count()
    
    # Helpful rate
    helpful_rate = helpful / total if total > 0 else 0.0
    
    # Average rating
    avg_rating = db.query(func.avg(QueryFeedback.rating)).filter(
        QueryFeedback.created_at >= since,
        QueryFeedback.rating.isnot(None)
    ).scalar() or 0.0
    
    return FeedbackStats(
        total_feedback=total,
        helpful_count=helpful,
        unhelpful_count=unhelpful,
        helpful_rate=helpful_rate,
        average_rating=float(avg_rating),
        period_days=days
    )


@router.get("/recent")
async def get_recent_feedback(
    limit: int = 10,
    user: User = Depends(get_current_user),
    db = Depends(get_db)
):
    """Get recent feedback entries."""
    feedback = db.query(QueryFeedback).filter(
        QueryFeedback.user_id == user.id
    ).order_by(
        QueryFeedback.created_at.desc()
    ).limit(limit).all()
    
    return [
        {
            "id": f.id,
            "query_id": f.query_id,
            "helpful": f.helpful,
            "rating": f.rating,
            "comment": f.comment,
            "created_at": f.created_at.isoformat()
        }
        for f in feedback
    ]
