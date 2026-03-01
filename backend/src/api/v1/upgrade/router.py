from fastapi import APIRouter, HTTPException
from loguru import logger

router = APIRouter()

@router.get("/", tags=["Upgrade"])
async def health_check():
    """Placeholder health endpoint for upgrade service.
    Future implementation will handle version migrations and data upgrades.
    """
    logger.info("Upgrade health check called")
    return {"status": "ready", "message": "Upgrade service is operational"}
