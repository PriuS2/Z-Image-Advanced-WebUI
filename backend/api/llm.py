"""LLM API endpoints for translation and prompt enhancement."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from backend.db import User
from backend.api.auth import get_current_user
from backend.config import get_config

router = APIRouter()


class TranslateRequest(BaseModel):
    """Translation request."""
    text: str
    provider: Optional[str] = None


class TranslateResponse(BaseModel):
    """Translation response."""
    original: str
    translated: str
    provider: str


class EnhanceRequest(BaseModel):
    """Prompt enhancement request."""
    prompt: str
    provider: Optional[str] = None


class EnhanceResponse(BaseModel):
    """Enhancement response."""
    original: str
    enhanced: str
    provider: str


@router.post("/translate", response_model=TranslateResponse)
async def translate(
    request: TranslateRequest,
    current_user: User = Depends(get_current_user),
):
    """Translate Korean text to English for image generation."""
    config = get_config()
    provider = request.provider or config.llm.default_provider
    
    # TODO: Implement actual LLM translation
    # For now, return placeholder
    return TranslateResponse(
        original=request.text,
        translated=f"[Translated] {request.text}",  # Placeholder
        provider=provider
    )


@router.post("/enhance", response_model=EnhanceResponse)
async def enhance(
    request: EnhanceRequest,
    current_user: User = Depends(get_current_user),
):
    """Enhance a prompt for better image generation results."""
    config = get_config()
    provider = request.provider or config.llm.default_provider
    
    # TODO: Implement actual LLM enhancement
    # For now, return placeholder
    return EnhanceResponse(
        original=request.prompt,
        enhanced=f"{request.prompt}, highly detailed, 8k, masterpiece",  # Placeholder
        provider=provider
    )


@router.get("/providers")
async def get_providers(
    current_user: User = Depends(get_current_user),
):
    """Get available LLM providers."""
    config = get_config()
    
    providers = []
    for name, provider_config in config.llm.providers.items():
        providers.append({
            "name": name,
            "model": provider_config.model,
            "configured": bool(provider_config.api_key or provider_config.base_url)
        })
    
    return {
        "default_provider": config.llm.default_provider,
        "providers": providers
    }
