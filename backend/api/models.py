"""Model management API endpoints."""
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from backend.db import User
from backend.api.auth import get_current_user
from backend.config import get_config

router = APIRouter()


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    path: str
    size: Optional[int] = None
    loaded: bool = False
    type: str  # base, controlnet, lora


class ModelStatus(BaseModel):
    """Current model status."""
    base_model_loaded: bool = False
    controlnet_loaded: bool = False
    current_model: Optional[str] = None
    vram_usage: Optional[float] = None
    gpu_memory_mode: str = "model_cpu_offload_and_qfloat8"


class ModelLoadRequest(BaseModel):
    """Model load request."""
    model_path: str
    gpu_memory_mode: Optional[str] = None
    weight_dtype: Optional[str] = None


class ModelDownloadRequest(BaseModel):
    """Model download request."""
    repo_id: str
    filename: Optional[str] = None
    destination: str


@router.get("/list", response_model=List[ModelInfo])
async def list_models(
    model_type: Optional[str] = None,
    current_user: User = Depends(get_current_user),
):
    """List available models."""
    # TODO: Scan model directories and return available models
    config = get_config()
    
    models = []
    
    # Placeholder - actual implementation will scan directories
    if model_type is None or model_type == "base":
        models.append(ModelInfo(
            name="Z-Image-Turbo",
            path=config.models.base_model_path,
            loaded=False,
            type="base"
        ))
    
    return models


@router.get("/status", response_model=ModelStatus)
async def get_model_status(
    current_user: User = Depends(get_current_user),
):
    """Get current model loading status."""
    # TODO: Get actual model status from model manager
    config = get_config()
    
    return ModelStatus(
        base_model_loaded=False,
        controlnet_loaded=False,
        gpu_memory_mode=config.optimization.gpu_memory_mode
    )


@router.post("/load")
async def load_model(
    request: ModelLoadRequest,
    current_user: User = Depends(get_current_user),
):
    """Load a model."""
    # TODO: Implement model loading
    return {
        "message": "Model loading not yet implemented",
        "model_path": request.model_path
    }


@router.post("/unload")
async def unload_model(
    current_user: User = Depends(get_current_user),
):
    """Unload current model to free VRAM."""
    # TODO: Implement model unloading
    return {"message": "Model unloading not yet implemented"}


@router.post("/download")
async def download_model(
    request: ModelDownloadRequest,
    current_user: User = Depends(get_current_user),
):
    """Download a model from HuggingFace."""
    # TODO: Implement model downloading with progress
    return {
        "message": "Model downloading not yet implemented",
        "repo_id": request.repo_id
    }


@router.get("/loras", response_model=List[ModelInfo])
async def list_loras(
    current_user: User = Depends(get_current_user),
):
    """List available LoRA models."""
    # TODO: Scan LoRA directory
    return []


@router.post("/loras/apply")
async def apply_lora(
    lora_path: str,
    weight: float = 0.8,
    current_user: User = Depends(get_current_user),
):
    """Apply a LoRA to the current model."""
    # TODO: Implement LoRA application
    return {
        "message": "LoRA application not yet implemented",
        "lora_path": lora_path,
        "weight": weight
    }
