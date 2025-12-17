"""Model management service for downloading, loading, and managing AI models."""
import os
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Callable, Any
from dataclasses import dataclass

from huggingface_hub import hf_hub_download, scan_cache_dir

from backend.config import get_config


@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    path: str
    size: Optional[int] = None
    loaded: bool = False
    model_type: str = "base"  # base, controlnet, lora, annotator


class ModelManager:
    """Manager for AI model operations."""
    
    def __init__(self):
        self.config = get_config()
        self._loaded_models: Dict[str, Any] = {}
    
    def list_models(self, model_type: Optional[str] = None) -> List[ModelInfo]:
        """List available models.
        
        Args:
            model_type: Filter by model type (base, controlnet, lora, annotator).
        
        Returns:
            List of ModelInfo objects.
        """
        models = []
        
        paths = {
            "base": self.config.models.base_model_path,
            "controlnet": self.config.models.controlnet_path,
            "lora": self.config.models.lora_path,
            "annotator": self.config.models.annotator_path,
        }
        
        for mtype, path in paths.items():
            if model_type and mtype != model_type:
                continue
            
            model_path = Path(path)
            if model_path.exists():
                # Check for model files
                if model_path.is_file():
                    models.append(ModelInfo(
                        name=model_path.stem,
                        path=str(model_path),
                        size=model_path.stat().st_size,
                        model_type=mtype,
                        loaded=mtype in self._loaded_models
                    ))
                else:
                    # Scan directory for models
                    for item in model_path.iterdir():
                        if item.is_dir() or item.suffix in ['.safetensors', '.bin', '.ckpt']:
                            size = None
                            if item.is_file():
                                size = item.stat().st_size
                            models.append(ModelInfo(
                                name=item.stem,
                                path=str(item),
                                size=size,
                                model_type=mtype,
                                loaded=item.name in self._loaded_models
                            ))
        
        return models
    
    async def download_model(
        self,
        repo_id: str,
        filename: Optional[str] = None,
        destination: str = "models/",
        progress_callback: Optional[Callable[[int, int], Any]] = None
    ) -> str:
        """Download a model from HuggingFace.
        
        Args:
            repo_id: HuggingFace repository ID (e.g., "username/model-name").
            filename: Specific file to download. Downloads entire repo if None.
            destination: Local destination directory.
            progress_callback: Callback for progress updates (downloaded, total).
        
        Returns:
            Path to downloaded model.
        """
        os.makedirs(destination, exist_ok=True)
        
        try:
            # Download from HuggingFace
            local_path = await asyncio.to_thread(
                hf_hub_download,
                repo_id=repo_id,
                filename=filename,
                local_dir=destination,
            )
            return local_path
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")
    
    async def load_model(
        self,
        model_path: str,
        model_type: str = "base",
        gpu_memory_mode: Optional[str] = None,
        weight_dtype: Optional[str] = None,
    ) -> bool:
        """Load a model into memory.
        
        Args:
            model_path: Path to the model.
            model_type: Type of model (base, controlnet, lora).
            gpu_memory_mode: GPU memory optimization mode.
            weight_dtype: Weight data type (float16, bfloat16, etc.).
        
        Returns:
            True if successful.
        """
        # TODO: Implement actual model loading
        self._loaded_models[model_type] = model_path
        return True
    
    async def unload_model(self, model_type: str = "base") -> bool:
        """Unload a model from memory.
        
        Args:
            model_type: Type of model to unload.
        
        Returns:
            True if successful.
        """
        import torch
        
        if model_type in self._loaded_models:
            del self._loaded_models[model_type]
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model loading status.
        
        Returns:
            Dictionary with model status information.
        """
        import torch
        
        vram_usage = None
        if torch.cuda.is_available():
            vram_usage = torch.cuda.memory_allocated() / (1024 ** 3)
        
        return {
            "loaded_models": list(self._loaded_models.keys()),
            "vram_usage_gb": vram_usage,
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    
    def list_loras(self) -> List[ModelInfo]:
        """List available LoRA models.
        
        Returns:
            List of LoRA ModelInfo objects.
        """
        return self.list_models("lora")
    
    async def apply_lora(
        self,
        lora_path: str,
        weight: float = 0.8,
    ) -> bool:
        """Apply a LoRA to the current model.
        
        Args:
            lora_path: Path to the LoRA file.
            weight: LoRA weight (0.0 to 1.0).
        
        Returns:
            True if successful.
        """
        # TODO: Implement LoRA application
        return True


# Global instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
