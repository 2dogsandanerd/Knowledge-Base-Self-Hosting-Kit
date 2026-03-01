from src.core.exceptions import ValidationError, DocumentNotFoundError
from fastapi import APIRouter, Depends, HTTPException

from src.api.v1.dependencies import get_learning_manager, get_email_client, get_current_user
from src.services.learning_manager import LearningManager
from src.core.email_clients.base_client import AbstractEmailClient
from src.database.models import User

router = APIRouter()


@router.post("/match-sent-emails")
async def match_sent_emails(
    learning_manager: LearningManager = Depends(get_learning_manager),
    email_client: AbstractEmailClient = Depends(get_email_client),
    current_user: User = Depends(get_current_user)
):
    """Async match sent emails with pending drafts"""
    try:
        report = await learning_manager.match_sent_emails(email_client, current_user.id)
        return report
    except Exception as e:
        raise ValidationError(str(e))


@router.get("/stats")
async def get_learning_stats(
    learning_manager: LearningManager = Depends(get_learning_manager),
    current_user: User = Depends(get_current_user)
):
    """Async get learning statistics for current user"""
    try:
        stats = await learning_manager.get_stats(current_user.id)
        return stats
    except Exception as e:
        raise ValidationError(str(e))


@router.get("/pairs")
async def get_all_pairs(
    learning_manager: LearningManager = Depends(get_learning_manager),
    current_user: User = Depends(get_current_user)
):
    """Async get all draft-sent pairs for current user"""
    try:
        pairs = await learning_manager.get_all_pairs(current_user.id)

        pairs_data = [
            {
                "id": p.id,
                "thread_id": p.thread_id,
                "draft_message_id": p.draft_message_id,
                "sent_message_id": p.sent_message_id,
                "status": p.status,
                "created_at": p.created_at.isoformat() if p.created_at else None
            }
            for p in pairs
        ]

        return {
            "pairs": pairs_data,
            "count": len(pairs_data)
        }

    except Exception as e:
        raise ValidationError(str(e))


@router.delete("/draft/{draft_id}")
async def delete_draft(
    draft_id: int,
    learning_manager: LearningManager = Depends(get_learning_manager),
    current_user: User = Depends(get_current_user)
):
    """Async delete a draft from the learning database"""
    try:
        success = await learning_manager.delete_pair(draft_id)
        if not success:
            raise DocumentNotFoundError(f"draft_{draft_id}", collection="drafts")
        return {"success": True, "message": f"Draft {draft_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise ValidationError(str(e))
