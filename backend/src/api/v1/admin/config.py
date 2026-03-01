"""
Admin Config API (Phase J.1).

Endpoints for reading and updating system configuration.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Union
from src.services.config_service import config_service
from src.services.auth_service import get_current_user
from src.database.models import User

router = APIRouter()

class ConfigUpdateRequest(BaseModel):
    """Request model for updating configuration."""
    key: str
    value: Union[str, int, float, bool]

@router.get("/config")
async def get_config(current_user: User = Depends(get_current_user)):
    """Get all tunable configurations."""
    # In a real app, check for admin role here
    return config_service.load_configuration()

@router.patch("/config")
async def update_config(
    request: ConfigUpdateRequest,
    current_user: User = Depends(get_current_user)
):
    """Update a specific configuration parameter."""
    # In a real app, check for admin role here
    try:
        # Load, update, and save pattern
        current_config = config_service.load_configuration()
        current_config[request.key] = request.value
        config_service.save_configuration(current_config)
        
        return {"status": "success", "key": request.key, "value": request.value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
