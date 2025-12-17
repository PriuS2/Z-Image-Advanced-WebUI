"""LLM API endpoints for translation and prompt enhancement."""
import logging
import httpx
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from backend.db import User
from backend.api.auth import get_current_user
from backend.config import get_config

logger = logging.getLogger(__name__)
router = APIRouter()


async def call_openai(api_key: str, model: str, system_prompt: str, user_message: str) -> str:
    """Call OpenAI API."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


async def call_claude(api_key: str, model: str, system_prompt: str, user_message: str) -> str:
    """Call Claude/Anthropic API."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 1024,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_message}],
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"].strip()


async def call_gemini(api_key: str, model: str, system_prompt: str, user_message: str) -> str:
    """Call Google Gemini API."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
            json={
                "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_message}"}]}],
                "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1024},
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()


async def call_ollama(base_url: str, model: str, system_prompt: str, user_message: str) -> str:
    """Call local Ollama API."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": f"{system_prompt}\n\nUser: {user_message}",
                "stream": False,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        return data["response"].strip()


async def call_llm(provider: str, system_prompt: str, user_message: str) -> str:
    """Call LLM based on provider."""
    config = get_config()
    
    try:
        if provider == "openai":
            provider_config = config.llm.providers.openai
            if not provider_config.api_key:
                raise ValueError("OpenAI API key not configured")
            return await call_openai(provider_config.api_key, provider_config.model, system_prompt, user_message)
        
        elif provider == "claude":
            provider_config = config.llm.providers.claude
            if not provider_config.api_key:
                raise ValueError("Claude API key not configured")
            return await call_claude(provider_config.api_key, provider_config.model, system_prompt, user_message)
        
        elif provider == "gemini":
            provider_config = config.llm.providers.gemini
            if not provider_config.api_key:
                raise ValueError("Gemini API key not configured")
            return await call_gemini(provider_config.api_key, provider_config.model, system_prompt, user_message)
        
        elif provider == "ollama":
            provider_config = config.llm.providers.ollama
            return await call_ollama(provider_config.base_url, provider_config.model, system_prompt, user_message)
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
            
    except httpx.HTTPStatusError as e:
        logger.error(f"LLM API error: {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"LLM call error: {e}")
        raise


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
    
    # Get system prompt
    system_prompt = config.llm.system_prompts.translate.custom or config.llm.system_prompts.translate.default
    
    try:
        translated = await call_llm(provider, system_prompt, request.text)
        
        return TranslateResponse(
            original=request.text,
            translated=translated,
            provider=provider
        )
    except ValueError as e:
        # API key not configured - return simple placeholder
        logger.warning(f"LLM not configured: {e}")
        return TranslateResponse(
            original=request.text,
            translated=request.text,  # Return original if no LLM configured
            provider=provider
        )
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}"
        )


@router.post("/enhance", response_model=EnhanceResponse)
async def enhance(
    request: EnhanceRequest,
    current_user: User = Depends(get_current_user),
):
    """Enhance a prompt for better image generation results."""
    config = get_config()
    provider = request.provider or config.llm.default_provider
    
    # Get system prompt
    system_prompt = config.llm.system_prompts.enhance.custom or config.llm.system_prompts.enhance.default
    
    try:
        enhanced = await call_llm(provider, system_prompt, request.prompt)
        
        return EnhanceResponse(
            original=request.prompt,
            enhanced=enhanced,
            provider=provider
        )
    except ValueError as e:
        # API key not configured - return simple enhancement
        logger.warning(f"LLM not configured: {e}")
        return EnhanceResponse(
            original=request.prompt,
            enhanced=f"{request.prompt}, highly detailed, 8k, masterpiece, best quality",
            provider=provider
        )
    except Exception as e:
        logger.error(f"Enhancement error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Enhancement failed: {str(e)}"
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
