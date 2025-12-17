"""User settings API endpoints."""
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import get_db, User
from backend.api.auth import get_current_user

router = APIRouter()


class UserSettings(BaseModel):
    """User settings model."""
    theme: str = "dark"
    language: str = "ko"
    realtime_preview: bool = True
    auto_save_workflow: bool = True
    generation_defaults: Optional[dict] = None
    llm_preferences: Optional[dict] = None


class SettingsUpdate(BaseModel):
    """Settings update request."""
    settings: UserSettings


@router.get("/", response_model=UserSettings)
async def get_settings(
    current_user: User = Depends(get_current_user),
):
    """Get current user settings."""
    if current_user.settings:
        return UserSettings(**current_user.settings)
    return UserSettings()


@router.put("/", response_model=UserSettings)
async def update_settings(
    settings_update: SettingsUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update user settings."""
    current_user.settings = settings_update.settings.model_dump()
    await db.commit()
    
    return settings_update.settings


@router.patch("/")
async def patch_settings(
    partial_settings: dict,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Partially update user settings."""
    current_settings = current_user.settings or {}
    current_settings.update(partial_settings)
    current_user.settings = current_settings
    await db.commit()
    
    return current_user.settings
