"""Model management service for downloading, loading, and managing AI models."""
import os
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Callable, Any
from dataclasses import dataclass

from huggingface_hub import hf_hub_download, snapshot_download, scan_cache_dir

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
            if filename:
                # Download specific file
                local_path = await asyncio.to_thread(
                    hf_hub_download,
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=destination,
                )
            else:
                # Download entire repository
                local_path = await asyncio.to_thread(
                    snapshot_download,
                    repo_id=repo_id,
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
        import logging
        logger = logging.getLogger(__name__)
        
        if model_type == "base":
            # Use ImageGeneratorService for base model
            from backend.services.image_generator import get_generator
            
            generator = get_generator()
            
            # Update config if parameters provided
            if gpu_memory_mode:
                self.config.optimization.gpu_memory_mode = gpu_memory_mode
            if weight_dtype:
                self.config.optimization.weight_dtype = weight_dtype
            
            logger.info(f"Loading base model: {model_path}")
            success = await generator.load_model(model_path)
            
            if success:
                self._loaded_models[model_type] = model_path
                logger.info("Base model loaded successfully")
            
            return success
        else:
            # For other model types, just track the path for now
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
        import logging
        logger = logging.getLogger(__name__)
        
        if model_type == "base":
            # Use ImageGeneratorService for base model
            from backend.services.image_generator import get_generator
            
            generator = get_generator()
            success = await generator.unload_model()
            
            if success:
                if model_type in self._loaded_models:
                    del self._loaded_models[model_type]
                logger.info("Base model unloaded")
            
            return success
        else:
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
        vram_total = None
        if torch.cuda.is_available():
            vram_usage = torch.cuda.memory_allocated() / (1024 ** 3)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        
        # Check actual generator status
        from backend.services.image_generator import get_generator
        generator = get_generator()
        
        loaded_models = list(self._loaded_models.keys())
        if generator.is_loaded and generator.pipe is not None:
            if "base" not in loaded_models:
                loaded_models.append("base")
        
        return {
            "loaded_models": loaded_models,
            "vram_usage_gb": round(vram_usage, 2) if vram_usage else None,
            "vram_total_gb": round(vram_total, 2) if vram_total else None,
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "pipeline_loaded": generator.pipe is not None,
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
        import logging
        import torch
        logger = logging.getLogger(__name__)
        
        from backend.services.image_generator import get_generator
        generator = get_generator()
        
        if generator.pipe is None:
            logger.error("Pipeline not loaded, cannot apply LoRA")
            return False
        
        try:
            # Try to use VideoX-Fun's merge_lora utility
            try:
                from videox_fun.utils.lora_utils import merge_lora
                
                logger.info(f"Applying LoRA: {lora_path} with weight={weight}")
                
                generator.pipe = merge_lora(
                    generator.pipe,
                    lora_path,
                    weight,
                    device=generator.device,
                    dtype=generator.dtype,
                )
                
                # Store current LoRA info for later removal
                self._loaded_models["current_lora"] = {
                    "path": lora_path,
                    "weight": weight,
                }
                
                logger.info("LoRA applied successfully")
                return True
                
            except ImportError:
                logger.warning("VideoX-Fun lora_utils not available, trying diffusers method")
                
                # Fallback to diffusers PEFT method if available
                if hasattr(generator.pipe, 'load_lora_weights'):
                    await asyncio.to_thread(
                        generator.pipe.load_lora_weights,
                        lora_path,
                    )
                    generator.pipe.fuse_lora(lora_scale=weight)
                    
                    self._loaded_models["current_lora"] = {
                        "path": lora_path,
                        "weight": weight,
                    }
                    
                    logger.info("LoRA applied using diffusers method")
                    return True
                else:
                    logger.error("No LoRA loading method available")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to apply LoRA: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def remove_lora(self) -> bool:
        """Remove currently applied LoRA from the model.
        
        Returns:
            True if successful.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        from backend.services.image_generator import get_generator
        generator = get_generator()
        
        if generator.pipe is None:
            logger.error("Pipeline not loaded")
            return False
        
        if "current_lora" not in self._loaded_models:
            logger.warning("No LoRA currently applied")
            return True
        
        try:
            lora_info = self._loaded_models["current_lora"]
            
            # Try VideoX-Fun's unmerge_lora utility
            try:
                from videox_fun.utils.lora_utils import unmerge_lora
                
                generator.pipe = unmerge_lora(
                    generator.pipe,
                    lora_info["path"],
                    lora_info["weight"],
                    device=generator.device,
                    dtype=generator.dtype,
                )
                
                del self._loaded_models["current_lora"]
                logger.info("LoRA removed successfully")
                return True
                
            except ImportError:
                logger.warning("VideoX-Fun lora_utils not available, trying diffusers method")
                
                # Fallback to diffusers method
                if hasattr(generator.pipe, 'unfuse_lora'):
                    generator.pipe.unfuse_lora()
                    generator.pipe.unload_lora_weights()
                    
                    del self._loaded_models["current_lora"]
                    logger.info("LoRA removed using diffusers method")
                    return True
                else:
                    logger.error("No LoRA removal method available")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to remove LoRA: {e}")
            import traceback
            traceback.print_exc()
            return False


# Global instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
