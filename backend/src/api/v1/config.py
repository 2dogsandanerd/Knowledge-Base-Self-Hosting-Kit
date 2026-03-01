from src.core.exceptions import ValidationError
from fastapi import APIRouter, HTTPException, Body, Depends
from typing import Dict, Any, List
import httpx
import os
from loguru import logger

from src.services.config_service import config_service as default_config_service, ConfigService
from src.services.connection_service import connection_service as default_connection_service, ConnectionService
from src.core.services.settings_service import SettingsService

router = APIRouter()

# Dependency for SettingsService
def get_settings_service() -> SettingsService:
    return SettingsService()

@router.get("/config", response_model=Dict[str, Any])
async def get_configuration(
    config_service: ConfigService = Depends(lambda: default_config_service)
):
    """Endpoint to retrieve the current application configuration."""
    try:
        config = config_service.load_configuration()
        return config
    except Exception as e:
        raise ValidationError(str(e))

@router.post("/config")
async def save_configuration(
    config_data: Dict[str, Any] = Body(...),
    config_service: ConfigService = Depends(lambda: default_config_service)
):
    """Endpoint to save the application configuration."""
    try:
        success = config_service.save_configuration(config_data)
        if success:
            return {"message": "Configuration saved successfully."}
        else:
            raise ValidationError("Failed to save configuration.")
    except Exception as e:
        raise ValidationError(str(e))

@router.post("/config/test", response_model=List[Dict[str, Any]])
async def test_connections(
    config_data: Dict[str, Any] = Body(...),
    connection_service: ConnectionService = Depends(lambda: default_connection_service)
):
    """Endpoint to test connections to external services using the provided config."""
    try:
        results = await connection_service.test_all_connections(config_data)
        return results
    except Exception as e:
        raise ValidationError(str(e))

@router.get("/models")
async def get_available_models():
    """Fetches available models from Ollama and other providers."""
    models = []
    
    # 1. Ollama Models
    ollama_host = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{ollama_host}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                for model in data.get("models", []):
                    # Use the model name as ID. 
                    # Note: The frontend/backend logic needs to handle 'ollama/' prefix if we use it.
                    # For simplicity, we just use the name, assuming the backend knows it's ollama if not gpt/claude.
                    models.append({
                        "id": model['name'], 
                        "name": f"Ollama: {model['name']}",
                        "provider": "ollama"
                    })
    except Exception as e:
        logger.warning(f"Could not fetch Ollama models: {e}")

    # 2. Add OpenAI/Anthropic placeholders if keys are present (or always show them as options)
    models.append({"id": "gpt-4", "name": "OpenAI GPT-4", "provider": "openai"})
    models.append({"id": "gpt-3.5-turbo", "name": "OpenAI GPT-3.5 Turbo", "provider": "openai"})
    models.append({"id": "claude-3-opus", "name": "Anthropic Claude 3 Opus", "provider": "anthropic"})
    models.append({"id": "gemini-pro", "name": "Google Gemini Pro", "provider": "google"})

    return {"models": models}

@router.get("/settings/security/allowed-roots")
async def get_allowed_roots(
    settings_service: SettingsService = Depends(get_settings_service)
):
    """
    Get the list of allowed root directories for path validation.

    Returns:
        List of absolute paths that are allowed for file operations.
    """
    try:
        paths = await settings_service.get_allowed_roots()
        return {"allowed_roots": paths}
    except Exception as e:
        logger.error(f"Failed to get allowed_roots: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/settings/security/allowed-roots")
async def update_allowed_roots(
    data: Dict[str, Any] = Body(...),
    settings_service: SettingsService = Depends(get_settings_service)
):
    """
    Update the list of allowed root directories.

    Body:
        {
            "allowed_roots": ["/path1", "/path2", ...]
        }

    Returns:
        Success message

    Raises:
        ValidationError: If any path does not exist or is not a directory
    """
    try:
        paths = data.get("allowed_roots", [])
        if not paths:
            raise ValidationError("At least one allowed root directory must be specified")

        if not isinstance(paths, list):
            raise ValidationError("allowed_roots must be a list")

        await settings_service.update_allowed_roots(paths)
        return {"message": "Security settings updated successfully", "allowed_roots": paths}
    except ValueError as e:
        raise ValidationError(str(e))
    except Exception as e:
        logger.error(f"Failed to update allowed_roots: {e}")
        raise HTTPException(status_code=500, detail=str(e))