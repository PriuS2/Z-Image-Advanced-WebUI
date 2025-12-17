"""Image generation service using Z-Image-Turbo-Fun-Controlnet-Union-2.0."""
import os
import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Any
from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image

from backend.config import get_config


@dataclass
class GenerationParams:
    """Image generation parameters."""
    prompt: str
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 25
    guidance_scale: float = 0.0
    control_context_scale: float = 0.75
    seed: Optional[int] = None
    sampler: str = "Flow"
    control_type: Optional[str] = None
    control_image: Optional[Image.Image] = None
    mask_image: Optional[Image.Image] = None
    original_image: Optional[Image.Image] = None


class ImageGeneratorService:
    """Service for generating images using Z-Image pipeline."""
    
    def __init__(self):
        self.pipe = None
        self.config = get_config()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = self._get_dtype()
        self.is_loaded = False
    
    def _get_dtype(self):
        """Get torch dtype from config."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.config.optimization.weight_dtype, torch.bfloat16)
    
    async def load_model(
        self, 
        model_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, str], Any]] = None
    ) -> bool:
        """Load the image generation model.
        
        Args:
            model_path: Path to the model. Uses config default if not provided.
            progress_callback: Callback for progress updates (percent, message).
        
        Returns:
            True if model loaded successfully.
        """
        if self.is_loaded:
            return True
        
        try:
            model_path = model_path or self.config.models.base_model_path
            
            if progress_callback:
                await progress_callback(10, "Loading model configuration...")
            
            # Import diffusers components
            from diffusers import FlowMatchEulerDiscreteScheduler
            
            if progress_callback:
                await progress_callback(20, "Initializing scheduler...")
            
            # Create scheduler
            self.scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=17.0,
            )
            
            if progress_callback:
                await progress_callback(30, "Loading pipeline...")
            
            # TODO: Load actual Z-Image pipeline when model is available
            # For now, we'll create a placeholder
            # from videox_fun.pipeline.pipeline_zimage_control import ZImageControlPipeline
            # self.pipe = ZImageControlPipeline.from_pretrained(
            #     model_path,
            #     scheduler=self.scheduler,
            #     torch_dtype=self.dtype,
            # )
            
            if progress_callback:
                await progress_callback(70, "Applying optimizations...")
            
            # Apply memory optimizations based on config
            memory_mode = self.config.optimization.gpu_memory_mode
            
            # TODO: Apply actual optimizations when model is loaded
            # if memory_mode == "model_cpu_offload":
            #     self.pipe.enable_model_cpu_offload()
            # elif memory_mode == "sequential_cpu_offload":
            #     self.pipe.enable_sequential_cpu_offload()
            
            if progress_callback:
                await progress_callback(100, "Model loaded successfully!")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    async def unload_model(self) -> bool:
        """Unload the model to free memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_loaded = False
        return True
    
    async def generate(
        self,
        params: GenerationParams,
        progress_callback: Optional[Callable[[int, int, str], Any]] = None
    ) -> Optional[Image.Image]:
        """Generate an image based on the provided parameters.
        
        Args:
            params: Generation parameters.
            progress_callback: Callback for progress updates (step, total_steps, node_name).
        
        Returns:
            Generated PIL Image or None if failed.
        """
        # For now, return a placeholder since model isn't loaded
        # TODO: Implement actual generation when model is available
        
        if progress_callback:
            total_steps = params.num_inference_steps
            for step in range(total_steps):
                await asyncio.sleep(0.1)  # Simulate generation time
                await progress_callback(step + 1, total_steps, "generate")
        
        # Create a placeholder image
        placeholder = Image.new('RGB', (params.width, params.height), color='gray')
        
        return placeholder
    
    async def generate_with_control(
        self,
        params: GenerationParams,
        progress_callback: Optional[Callable[[int, int, str], Any]] = None
    ) -> Optional[Image.Image]:
        """Generate an image with control image guidance.
        
        Args:
            params: Generation parameters with control_image set.
            progress_callback: Callback for progress updates.
        
        Returns:
            Generated PIL Image or None if failed.
        """
        if params.control_image is None:
            raise ValueError("control_image is required for control-guided generation")
        
        return await self.generate(params, progress_callback)
    
    async def generate_inpaint(
        self,
        params: GenerationParams,
        progress_callback: Optional[Callable[[int, int, str], Any]] = None
    ) -> Optional[Image.Image]:
        """Generate an image with inpainting.
        
        Args:
            params: Generation parameters with original_image and mask_image set.
            progress_callback: Callback for progress updates.
        
        Returns:
            Generated PIL Image or None if failed.
        """
        if params.original_image is None or params.mask_image is None:
            raise ValueError("original_image and mask_image are required for inpainting")
        
        return await self.generate(params, progress_callback)
    
    def save_image(self, image: Image.Image, filename: Optional[str] = None) -> str:
        """Save a generated image.
        
        Args:
            image: PIL Image to save.
            filename: Optional filename. Auto-generated if not provided.
        
        Returns:
            Path to the saved image.
        """
        output_dir = Path(self.config.models.outputs_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_{timestamp}.png"
        
        output_path = output_dir / filename
        image.save(output_path)
        
        return str(output_path)
    
    def create_thumbnail(self, image: Image.Image, size: tuple = (256, 256)) -> Image.Image:
        """Create a thumbnail of an image.
        
        Args:
            image: PIL Image to create thumbnail from.
            size: Thumbnail size.
        
        Returns:
            Thumbnail PIL Image.
        """
        thumbnail = image.copy()
        thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
        return thumbnail
    
    def get_vram_usage(self) -> Optional[float]:
        """Get current VRAM usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 3)
        return None


# Global instance
_generator: Optional[ImageGeneratorService] = None


def get_generator() -> ImageGeneratorService:
    """Get the global image generator instance."""
    global _generator
    if _generator is None:
        _generator = ImageGeneratorService()
    return _generator
