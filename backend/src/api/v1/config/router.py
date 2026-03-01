from fastapi import APIRouter, HTTPException
from loguru import logger
from .models import ModelConfig
from src.core.services.settings_service import SettingsService

router = APIRouter()
settings = SettingsService()

@router.get("/model", response_model=ModelConfig)
async def get_model() -> ModelConfig:
    model = await settings.get("model_name")
    if not model:
        raise HTTPException(status_code=404, detail="Model not configured")
    return ModelConfig(model_name=model)

@router.post("/model", response_model=ModelConfig)
async def set_model(cfg: ModelConfig) -> ModelConfig:
    await settings.set("model_name", cfg.model_name)
    logger.success(f"LLM model switched to {cfg.model_name}")
    return cfg
