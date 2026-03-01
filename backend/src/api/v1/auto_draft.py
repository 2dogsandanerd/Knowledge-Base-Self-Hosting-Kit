from fastapi import APIRouter, Depends

from src.api.v1.dependencies import get_auto_draft_service, get_current_user
from src.services.auto_draft_service import AutoDraftService
from src.database.models import User

router = APIRouter()


@router.post("/start")
async def start_auto_draft(
    service: AutoDraftService = Depends(get_auto_draft_service)
):
    """Start auto-draft monitoring service"""
    # In a real async setup, this would likely trigger a persistent task.
    # For now, we use the service's state.
    success = await service.start_monitoring()

    if not success:
        return {
            "status": "already_running",
            "message": "Auto-draft service is already running."
        }

    status = await service.get_status()
    return {
        "status": "started",
        "interval": status.get("interval"),
        "worker_id": status.get("worker_id")
    }


@router.post("/stop")
async def stop_auto_draft(
    service: AutoDraftService = Depends(get_auto_draft_service)
):
    """Stop auto-draft monitoring service"""
    await service.stop_monitoring()
    return {"status": "stopped"}


@router.get("/status")
async def get_status(service: AutoDraftService = Depends(get_auto_draft_service)):
    """Get auto-draft service status"""
    return await service.get_status()