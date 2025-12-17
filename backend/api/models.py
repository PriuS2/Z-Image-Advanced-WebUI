"""Model management API endpoints."""
import asyncio
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel

from backend.db import User
from backend.api.auth import get_current_user
from backend.config import get_config
from backend.services.model_manager import get_model_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# Track download progress globally
download_progress = {
    "status": "idle",  # idle, downloading, completed, error
    "progress": 0,
    "message": "",
    "repo_id": ""
}


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
    import os
    from pathlib import Path
    
    config = get_config()
    model_manager = get_model_manager()
    status_info = model_manager.get_model_status()
    loaded_models = status_info.get("loaded_models", [])
    
    models = []
    
    # Define model paths
    model_paths = {
        "base": config.models.base_model_path,
        "controlnet": config.models.controlnet_path,
        "lora": config.models.lora_path,
        "annotator": config.models.annotator_path,
    }
    
    for mtype, base_path in model_paths.items():
        # Skip if filtering by type
        if model_type and mtype != model_type:
            continue
        
        path = Path(base_path)
        
        if not path.exists():
            continue
        
        if path.is_file():
            # Single model file
            size = path.stat().st_size
            models.append(ModelInfo(
                name=path.stem,
                path=str(path),
                size=size,
                loaded=mtype in loaded_models,
                type=mtype
            ))
        else:
            # Directory - scan for models
            # Check if it's a diffusers model directory (has model_index.json or config.json)
            if (path / "model_index.json").exists() or (path / "config.json").exists():
                # It's a diffusers model directory
                size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                models.append(ModelInfo(
                    name=path.name,
                    path=str(path),
                    size=size,
                    loaded=mtype in loaded_models,
                    type=mtype
                ))
            else:
                # Scan for model files/subdirectories
                for item in path.iterdir():
                    if item.is_dir():
                        # Check if subdirectory is a model
                        if (item / "model_index.json").exists() or (item / "config.json").exists():
                            size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                            models.append(ModelInfo(
                                name=item.name,
                                path=str(item),
                                size=size,
                                loaded=item.name in loaded_models,
                                type=mtype
                            ))
                    elif item.suffix in ['.safetensors', '.bin', '.ckpt', '.pt', '.pth']:
                        # Model file
                        models.append(ModelInfo(
                            name=item.stem,
                            path=str(item),
                            size=item.stat().st_size,
                            loaded=item.stem in loaded_models,
                            type=mtype
                        ))
    
    return models


@router.get("/status", response_model=ModelStatus)
async def get_model_status(
    current_user: User = Depends(get_current_user),
):
    """Get current model loading status."""
    config = get_config()
    model_manager = get_model_manager()
    status_info = model_manager.get_model_status()
    
    return ModelStatus(
        base_model_loaded="base" in status_info.get("loaded_models", []),
        controlnet_loaded="controlnet" in status_info.get("loaded_models", []),
        current_model=config.models.base_model_path,
        vram_usage=status_info.get("vram_usage_gb"),
        gpu_memory_mode=config.optimization.gpu_memory_mode
    )


@router.post("/load")
async def load_model(
    request: ModelLoadRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    """Load a model."""
    config = get_config()
    model_manager = get_model_manager()
    
    # Determine model path
    model_path = request.model_path
    if model_path == "default":
        model_path = config.models.base_model_path
    
    # Get GPU memory mode and weight dtype
    gpu_memory_mode = request.gpu_memory_mode or config.optimization.gpu_memory_mode
    weight_dtype = request.weight_dtype or config.optimization.weight_dtype
    
    try:
        logger.info(f"Loading model: {model_path} with mode={gpu_memory_mode}, dtype={weight_dtype}")
        
        # Load model asynchronously
        success = await model_manager.load_model(
            model_path=model_path,
            model_type="base",
            gpu_memory_mode=gpu_memory_mode,
            weight_dtype=weight_dtype,
        )
        
        if success:
            return {
                "message": f"모델 로드 완료: {model_path}",
                "model_path": model_path,
                "gpu_memory_mode": gpu_memory_mode,
                "weight_dtype": weight_dtype,
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to load model"
            )
    except Exception as e:
        logger.error(f"Model load error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/unload")
async def unload_model(
    current_user: User = Depends(get_current_user),
):
    """Unload current model to free VRAM."""
    model_manager = get_model_manager()
    
    try:
        success = await model_manager.unload_model("base")
        
        if success:
            return {"message": "모델 언로드 완료. VRAM이 해제되었습니다."}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to unload model"
            )
    except Exception as e:
        logger.error(f"Model unload error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


async def _download_model_task(repo_id: str, destination: str, filename: Optional[str] = None):
    """Background task to download model."""
    global download_progress
    
    try:
        download_progress["status"] = "downloading"
        download_progress["repo_id"] = repo_id
        download_progress["message"] = f"Downloading {repo_id}..."
        download_progress["progress"] = 0
        
        logger.info(f"Starting download: {repo_id} to {destination}")
        
        model_manager = get_model_manager()
        
        # Download the model
        local_path = await model_manager.download_model(
            repo_id=repo_id,
            filename=filename,
            destination=destination,
        )
        
        download_progress["status"] = "completed"
        download_progress["progress"] = 100
        download_progress["message"] = f"Download completed: {local_path}"
        logger.info(f"Download completed: {local_path}")
        
    except Exception as e:
        download_progress["status"] = "error"
        download_progress["message"] = str(e)
        logger.error(f"Download error: {e}")


@router.post("/download")
async def download_model(
    request: ModelDownloadRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    """Download a model from HuggingFace."""
    global download_progress
    
    # Check if already downloading
    if download_progress["status"] == "downloading":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A download is already in progress"
        )
    
    # Start download in background
    background_tasks.add_task(
        _download_model_task,
        request.repo_id,
        request.destination,
        request.filename
    )
    
    download_progress["status"] = "downloading"
    download_progress["repo_id"] = request.repo_id
    download_progress["message"] = "Download started..."
    download_progress["progress"] = 0
    
    return {
        "message": f"Download started for {request.repo_id}",
        "repo_id": request.repo_id,
        "status": "downloading"
    }


@router.get("/download/progress")
async def get_download_progress(
    current_user: User = Depends(get_current_user),
):
    """Get current download progress."""
    return download_progress


@router.get("/loras", response_model=List[ModelInfo])
async def list_loras(
    current_user: User = Depends(get_current_user),
):
    """List available LoRA models."""
    from pathlib import Path
    
    config = get_config()
    lora_path = Path(config.models.lora_path)
    
    loras = []
    
    if not lora_path.exists():
        return loras
    
    # Scan for LoRA files
    for item in lora_path.iterdir():
        if item.suffix in ['.safetensors', '.bin', '.ckpt', '.pt', '.pth']:
            loras.append(ModelInfo(
                name=item.stem,
                path=str(item),
                size=item.stat().st_size,
                loaded=False,
                type="lora"
            ))
        elif item.is_dir():
            # Check for LoRA files inside directory
            for lora_file in item.glob("*.safetensors"):
                loras.append(ModelInfo(
                    name=f"{item.name}/{lora_file.stem}",
                    path=str(lora_file),
                    size=lora_file.stat().st_size,
                    loaded=False,
                    type="lora"
                ))
    
    return loras


class LoRAApplyRequest(BaseModel):
    """LoRA apply request."""
    lora_path: str
    weight: float = 0.8


@router.post("/loras/apply")
async def apply_lora(
    request: LoRAApplyRequest,
    current_user: User = Depends(get_current_user),
):
    """Apply a LoRA to the current model."""
    from pathlib import Path
    
    config = get_config()
    model_manager = get_model_manager()
    
    # Check if model is loaded
    status_info = model_manager.get_model_status()
    if "base" not in status_info.get("loaded_models", []):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="기본 모델이 로드되지 않았습니다. 먼저 모델을 로드해주세요."
        )
    
    # Validate LoRA path
    lora_file = Path(request.lora_path)
    if not lora_file.exists():
        # Try relative to lora directory
        lora_file = Path(config.models.lora_path) / request.lora_path
        if not lora_file.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"LoRA 파일을 찾을 수 없습니다: {request.lora_path}"
            )
    
    try:
        logger.info(f"Applying LoRA: {lora_file} with weight={request.weight}")
        
        success = await model_manager.apply_lora(
            lora_path=str(lora_file),
            weight=request.weight,
        )
        
        if success:
            return {
                "message": f"LoRA 적용 완료: {lora_file.stem}",
                "lora_path": str(lora_file),
                "weight": request.weight
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="LoRA 적용에 실패했습니다."
            )
    except Exception as e:
        logger.error(f"LoRA apply error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/loras/remove")
async def remove_lora(
    current_user: User = Depends(get_current_user),
):
    """Remove applied LoRA from the current model."""
    model_manager = get_model_manager()
    
    # Check if model is loaded
    status_info = model_manager.get_model_status()
    if "base" not in status_info.get("loaded_models", []):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="기본 모델이 로드되지 않았습니다."
        )
    
    try:
        success = await model_manager.remove_lora()
        
        if success:
            return {"message": "LoRA가 제거되었습니다."}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="LoRA 제거에 실패했습니다."
            )
    except Exception as e:
        logger.error(f"LoRA remove error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
