"""API routers module."""
from fastapi import APIRouter

from .auth import router as auth_router
from .generation import router as generation_router
from .gallery import router as gallery_router
from .history import router as history_router
from .settings import router as settings_router
from .models import router as models_router
from .llm import router as llm_router
from .workflow import router as workflow_router

# Create main API router
api_router = APIRouter()

# Include all sub-routers
api_router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
api_router.include_router(generation_router, prefix="/generation", tags=["Generation"])
api_router.include_router(gallery_router, prefix="/gallery", tags=["Gallery"])
api_router.include_router(history_router, prefix="/history", tags=["History"])
api_router.include_router(settings_router, prefix="/settings", tags=["Settings"])
api_router.include_router(models_router, prefix="/models", tags=["Models"])
api_router.include_router(llm_router, prefix="/llm", tags=["LLM"])
api_router.include_router(workflow_router, prefix="/workflow", tags=["Workflow"])

__all__ = ["api_router"]
