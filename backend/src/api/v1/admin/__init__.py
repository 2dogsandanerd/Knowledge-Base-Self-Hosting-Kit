from fastapi import APIRouter
from src.api.v1.admin import config

router = APIRouter()

router.include_router(config.router, tags=["Admin Config"])
