"""Image generation service using Z-Image-Turbo-Fun-Controlnet-Union-2.0."""
import os
import sys
import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Any, Literal
from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image

from backend.config import get_config

logger = logging.getLogger(__name__)


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
    control_image_path: Optional[str] = None  # Path to control image file


# Type alias for pipeline types
PipelineType = Literal['base', 'control']


class ImageGeneratorService:
    """Service for generating images using Z-Image pipeline."""
    
    def __init__(self):
        self.pipe = None
        self.pipe_type: Optional[PipelineType] = None  # 'base' or 'control'
        self.transformer = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None
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
    
    def _find_controlnet_path(self) -> Optional[str]:
        """Find ControlNet safetensors file.
        
        Searches for the controlnet file in multiple possible locations:
        1. Direct file: models/Personalized_Model/Z-Image-Turbo-Fun-Controlnet-Union-2.0.safetensors
        2. In folder: models/Personalized_Model/Z-Image-Turbo-Fun-Controlnet-Union-2.0/*.safetensors
        
        Returns:
            Path to the safetensors file or None if not found.
        """
        base_path = self.config.models.controlnet_path
        controlnet_name = "Z-Image-Turbo-Fun-Controlnet-Union-2.0"
        
        # 방법 1: 직접 파일 경로
        direct_file = os.path.join(base_path, f"{controlnet_name}.safetensors")
        if os.path.isfile(direct_file):
            logger.info(f"Found ControlNet file directly: {direct_file}")
            return direct_file
        
        # 방법 2: 폴더 안에서 safetensors 파일 찾기
        folder_path = os.path.join(base_path, controlnet_name)
        if os.path.isdir(folder_path):
            # 폴더 안에서 .safetensors 파일 찾기
            for filename in os.listdir(folder_path):
                if filename.endswith(".safetensors"):
                    found_path = os.path.join(folder_path, filename)
                    logger.info(f"Found ControlNet file in folder: {found_path}")
                    return found_path
            
            # safetensors 파일이 없으면 폴더 자체를 반환 (diffusers 형식일 수 있음)
            logger.info(f"ControlNet folder found (diffusers format): {folder_path}")
            return folder_path
        
        # 방법 3: 폴더 내 모든 하위 폴더에서 검색
        if os.path.isdir(base_path):
            for item in os.listdir(base_path):
                item_path = os.path.join(base_path, item)
                if os.path.isdir(item_path) and "controlnet" in item.lower():
                    # 폴더 안에서 safetensors 찾기
                    for filename in os.listdir(item_path):
                        if filename.endswith(".safetensors"):
                            found_path = os.path.join(item_path, filename)
                            logger.info(f"Found ControlNet file: {found_path}")
                            return found_path
        
        logger.warning(f"ControlNet not found in {base_path}")
        return None
    
    async def load_model(
        self, 
        model_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, str], Any]] = None,
        use_control: bool = True
    ) -> bool:
        """Load the image generation model.
        
        Args:
            model_path: Path to the model. Uses config default if not provided.
            progress_callback: Callback for progress updates (percent, message).
            use_control: If True, load ControlPipeline; if False, load base Pipeline.
        
        Returns:
            True if model loaded successfully.
        """
        target_type: PipelineType = 'control' if use_control else 'base'
        
        # Check if already loaded with the correct type
        if self.is_loaded and self.pipe is not None and self.pipe_type == target_type:
            logger.info(f"Model already loaded as {target_type} pipeline")
            return True
        
        # If loaded with different type, unload first
        if self.is_loaded and self.pipe is not None and self.pipe_type != target_type:
            logger.info(f"Switching pipeline from {self.pipe_type} to {target_type}")
            await self.unload_model()
        
        try:
            model_path = model_path or self.config.models.base_model_path
            
            logger.info(f"Loading model from: {model_path}")
            logger.info(f"Pipeline type: {target_type}")
            
            if progress_callback:
                await progress_callback(5, "Loading configuration...")
            
            # Load config
            from omegaconf import OmegaConf
            
            if use_control:
                # ControlNet 경로 찾기 - 폴더 또는 파일 모두 지원
                controlnet_path = self._find_controlnet_path()
                config_path = "config/z_image/z_image_control_2.0.yaml"
                logger.info(f"ControlNet path: {controlnet_path}")
                
                if os.path.exists(config_path):
                    model_config = OmegaConf.load(config_path)
                else:
                    # Default config for Z-Image 2.0 Control
                    model_config = OmegaConf.create({
                        "transformer_additional_kwargs": {
                            "control_layers_places": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
                            "control_refiner_layers_places": [0, 1],
                            "add_control_noise_refiner": True,
                            "control_in_dim": 33,
                        }
                    })
            else:
                controlnet_path = None
                model_config = None
            
            if progress_callback:
                await progress_callback(10, "Importing VideoX-Fun modules...")
            
            # Try to import VideoX-Fun modules
            try:
                from videox_fun.models import (
                    AutoencoderKL, 
                    AutoTokenizer,
                    Qwen3ForCausalLM,
                )
                from diffusers import FlowMatchEulerDiscreteScheduler
                
                if use_control:
                    from videox_fun.models import ZImageControlTransformer2DModel
                    from videox_fun.pipeline import ZImageControlPipeline
                else:
                    from videox_fun.models import ZImageTransformer2DModel
                    from videox_fun.pipeline import ZImagePipeline
            except ImportError as e:
                logger.error(f"VideoX-Fun not installed: {e}")
                logger.info("Falling back to placeholder mode")
                self.is_loaded = True
                return True
            
            if progress_callback:
                await progress_callback(20, "Loading Transformer...")
            
            # Load Transformer based on pipeline type
            if use_control:
                self.transformer = await asyncio.to_thread(
                    ZImageControlTransformer2DModel.from_pretrained,
                    model_path,
                    subfolder="transformer",
                    low_cpu_mem_usage=True,
                    torch_dtype=self.dtype,
                    transformer_additional_kwargs=OmegaConf.to_container(model_config['transformer_additional_kwargs']),
                )
            else:
                self.transformer = await asyncio.to_thread(
                    ZImageTransformer2DModel.from_pretrained,
                    model_path,
                    subfolder="transformer",
                    low_cpu_mem_usage=True,
                    torch_dtype=self.dtype,
                )
            self.transformer = self.transformer.to(self.dtype)
            
            if progress_callback:
                await progress_callback(35, "Loading ControlNet weights..." if use_control else "Transformer loaded...")
            
            # Load ControlNet Union weights (only for control pipeline)
            if use_control and controlnet_path and os.path.exists(controlnet_path):
                if os.path.isfile(controlnet_path) and controlnet_path.endswith(".safetensors"):
                    # 단일 safetensors 파일
                    from safetensors.torch import load_file
                    state_dict = await asyncio.to_thread(load_file, controlnet_path)
                    m, u = self.transformer.load_state_dict(state_dict, strict=False)
                    logger.info(f"ControlNet loaded from file - missing keys: {len(m)}, unexpected keys: {len(u)}")
                elif os.path.isdir(controlnet_path):
                    # 폴더인 경우 - 폴더 안의 safetensors 파일들을 찾아서 로드
                    loaded = False
                    for filename in os.listdir(controlnet_path):
                        if filename.endswith(".safetensors"):
                            file_path = os.path.join(controlnet_path, filename)
                            from safetensors.torch import load_file
                            state_dict = await asyncio.to_thread(load_file, file_path)
                            m, u = self.transformer.load_state_dict(state_dict, strict=False)
                            logger.info(f"ControlNet loaded from {filename} - missing keys: {len(m)}, unexpected keys: {len(u)}")
                            loaded = True
                            break
                    if not loaded:
                        logger.warning(f"No safetensors file found in ControlNet folder: {controlnet_path}")
            elif use_control:
                logger.warning(f"ControlNet not found: {controlnet_path}")
            
            if progress_callback:
                await progress_callback(50, "Loading VAE...")
            
            # Load VAE
            self.vae = await asyncio.to_thread(
                AutoencoderKL.from_pretrained,
                model_path,
                subfolder="vae",
            )
            self.vae = self.vae.to(self.dtype)
            
            if progress_callback:
                await progress_callback(65, "Loading Text Encoder...")
            
            # Load Tokenizer
            self.tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained,
                model_path,
                subfolder="tokenizer",
            )
            
            # Load Text Encoder
            self.text_encoder = await asyncio.to_thread(
                Qwen3ForCausalLM.from_pretrained,
                model_path,
                subfolder="text_encoder",
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
            
            if progress_callback:
                await progress_callback(80, "Creating pipeline...")
            
            # Create scheduler
            self.scheduler = await asyncio.to_thread(
                FlowMatchEulerDiscreteScheduler.from_pretrained,
                model_path,
                subfolder="scheduler",
            )
            
            # Create pipeline based on type
            if use_control:
                self.pipe = ZImageControlPipeline(
                    vae=self.vae,
                    tokenizer=self.tokenizer,
                    text_encoder=self.text_encoder,
                    transformer=self.transformer,
                    scheduler=self.scheduler,
                )
                self.pipe_type = 'control'
            else:
                self.pipe = ZImagePipeline(
                    vae=self.vae,
                    tokenizer=self.tokenizer,
                    text_encoder=self.text_encoder,
                    transformer=self.transformer,
                    scheduler=self.scheduler,
                )
                self.pipe_type = 'base'
            
            if progress_callback:
                await progress_callback(90, "Applying memory optimizations...")
            
            # Apply memory optimizations based on config
            memory_mode = self.config.optimization.gpu_memory_mode
            
            if "qfloat8" in memory_mode:
                try:
                    from videox_fun.utils.fp8_optimization import (
                        convert_model_weight_to_float8,
                        convert_weight_dtype_wrapper
                    )
                    # Exclude modules that need to maintain original dtype
                    # pad tokens cause dtype mismatch error if converted to FP8
                    convert_model_weight_to_float8(
                        self.transformer,
                        exclude_module_name=["img_in", "txt_in", "timestep", "x_pad_token", "cap_pad_token"],
                        device=self.device
                    )
                    convert_weight_dtype_wrapper(self.transformer, self.dtype)
                    logger.info("FP8 quantization applied")
                except ImportError:
                    logger.warning("FP8 optimization not available")
            
            if "sequential_cpu_offload" in memory_mode:
                self.pipe.enable_sequential_cpu_offload(device=self.device)
            elif "cpu_offload" in memory_mode:
                self.pipe.enable_model_cpu_offload(device=self.device)
            else:
                self.pipe.to(device=self.device)
            
            if progress_callback:
                await progress_callback(100, "Model loaded successfully!")
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully as {self.pipe_type} pipeline")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to placeholder mode
            self.is_loaded = True
            return True
    
    async def unload_model(self) -> bool:
        """Unload the model to free memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        if self.transformer is not None:
            del self.transformer
            self.transformer = None
        if self.vae is not None:
            del self.vae
            self.vae = None
        if self.text_encoder is not None:
            del self.text_encoder
            self.text_encoder = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.scheduler is not None:
            del self.scheduler
            self.scheduler = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_loaded = False
        self.pipe_type = None
        logger.info("Model unloaded, VRAM freed")
        return True
    
    def _needs_control_pipeline(self, params: GenerationParams) -> bool:
        """Check if control pipeline is needed based on params."""
        return (
            params.control_image is not None or 
            params.control_image_path is not None or
            params.mask_image is not None or 
            params.original_image is not None
        )
    
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
        # Determine if we need control pipeline
        needs_control = self._needs_control_pipeline(params)
        
        # Check if model is loaded with the correct pipeline type
        if self.pipe is None:
            logger.warning("Model not loaded, attempting auto-load...")
            try:
                success = await self.load_model(use_control=needs_control)
                if not success or self.pipe is None:
                    logger.error("Auto-load failed, model not available")
                    raise RuntimeError("모델이 로드되지 않았습니다. 설정 탭에서 모델을 먼저 로드해주세요.")
            except Exception as e:
                logger.error(f"Auto-load error: {e}")
                raise RuntimeError(f"모델 로드 실패: {e}")
        
        # Switch pipeline if needed
        if needs_control and self.pipe_type != 'control':
            logger.info("Switching to control pipeline for this generation...")
            await self.load_model(use_control=True)
        elif not needs_control and self.pipe_type == 'control':
            logger.info("Switching to base pipeline for this generation...")
            await self.load_model(use_control=False)
        
        try:
            # Set seed
            seed = params.seed if params.seed is not None else int(time.time()) % (2**32)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Load control image from path if provided
            control_image_pil = params.control_image
            if control_image_pil is None and params.control_image_path:
                try:
                    control_image_pil = Image.open(params.control_image_path).convert('RGB')
                    logger.info(f"Loaded control image from: {params.control_image_path}")
                except Exception as e:
                    logger.warning(f"Failed to load control image from path: {e}")
            
            # Prepare control image if available (for control pipeline)
            control_image = None
            if control_image_pil is not None and self.pipe_type == 'control':
                try:
                    from videox_fun.utils.utils import get_image_latent
                    control_image = get_image_latent(
                        control_image_pil,
                        sample_size=[params.height, params.width]
                    )[:, :, 0]
                except Exception as e:
                    logger.warning(f"Failed to process control image: {e}")
            
            # Prepare inpaint images if available
            inpaint_image = None
            mask_image = None
            if params.original_image is not None and params.mask_image is not None:
                try:
                    from videox_fun.utils.utils import get_image_latent
                    inpaint_image = get_image_latent(
                        params.original_image,
                        sample_size=[params.height, params.width]
                    )[:, :, 0]
                    mask_image = get_image_latent(
                        params.mask_image,
                        sample_size=[params.height, params.width]
                    )[:, :1, 0]
                except Exception as e:
                    logger.warning(f"Failed to process inpaint images: {e}")
            
            logger.info(f"Generating image with {self.pipe_type} pipeline: {params.prompt[:50]}...")
            
            # Get the current event loop for thread-safe callback
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            
            # Progress callback wrapper for diffusers
            step_count = [0]
            def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
                step_count[0] = step_index + 1
                if progress_callback and loop is not None:
                    # Use run_coroutine_threadsafe for calling async from thread
                    asyncio.run_coroutine_threadsafe(
                        progress_callback(step_index + 1, params.num_inference_steps, "generate"),
                        loop
                    )
                return callback_kwargs
            
            # Generate based on pipeline type
            with torch.no_grad():
                if self.pipe_type == 'control':
                    # Control pipeline with control image support
                    result = await asyncio.to_thread(
                        lambda: self.pipe(
                            prompt=params.prompt,
                            negative_prompt=" ",
                            height=params.height,
                            width=params.width,
                            generator=generator,
                            guidance_scale=params.guidance_scale,
                            control_image=control_image,
                            image=inpaint_image,
                            mask_image=mask_image,
                            num_inference_steps=params.num_inference_steps,
                            control_context_scale=params.control_context_scale,
                            callback_on_step_end=callback_on_step_end,
                        ).images
                    )
                else:
                    # Base pipeline without control
                    result = await asyncio.to_thread(
                        lambda: self.pipe(
                            prompt=params.prompt,
                            negative_prompt=" ",
                            height=params.height,
                            width=params.width,
                            generator=generator,
                            guidance_scale=params.guidance_scale,
                            num_inference_steps=params.num_inference_steps,
                            callback_on_step_end=callback_on_step_end,
                        ).images
                    )
            
            if result and len(result) > 0:
                return result[0]
            return None
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            import traceback
            traceback.print_exc()
            # Return placeholder on error
            return Image.new('RGB', (params.width, params.height), color='gray')
    
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
