"""Depth extraction service using transformers depth-estimation pipeline."""
import logging
import numpy as np
from PIL import Image
from typing import Optional

logger = logging.getLogger(__name__)

# Global instance for singleton pattern
_depth_extractor: Optional["DepthExtractor"] = None


class DepthExtractor:
    """Depth extraction using transformers depth-estimation model."""
    
    def __init__(self):
        self._pipe = None
        self._model_loaded = False
    
    def _load_model(self):
        """Lazy load the depth estimation model."""
        if self._model_loaded:
            return
        
        try:
            import torch
            from transformers import pipeline
            
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading depth estimation model on {device}...")
            
            # Use Intel's DPT model for depth estimation
            # Alternatives: "Intel/dpt-large", "Intel/dpt-hybrid-midas", "LiheYoung/depth-anything-small-hf"
            self._pipe = pipeline(
                "depth-estimation",
                model="Intel/dpt-hybrid-midas",
                device=device
            )
            
            self._model_loaded = True
            logger.info("Depth estimation model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load depth estimation model: {e}")
            raise RuntimeError(f"Failed to load depth model: {e}")
    
    def extract_depth(self, image: Image.Image) -> Image.Image:
        """Extract depth map from image.
        
        Args:
            image: Source PIL Image (RGB).
        
        Returns:
            Depth map as PIL Image (grayscale converted to RGB).
        """
        # Load model if not already loaded
        self._load_model()
        
        # Ensure image is in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Run depth estimation
        result = self._pipe(image)
        
        # Get depth map (PIL Image)
        depth_map = result["depth"]
        
        # Normalize depth values for better visualization
        depth_array = np.array(depth_map)
        
        # Normalize to 0-255 range using percentile for better contrast
        vmin = np.percentile(depth_array, 2)
        vmax = np.percentile(depth_array, 98)
        
        if vmax - vmin > 0:
            depth_normalized = (depth_array - vmin) / (vmax - vmin)
        else:
            depth_normalized = depth_array
        
        # Invert depth (closer objects are brighter, like ZoeDepth output)
        depth_normalized = 1.0 - depth_normalized
        
        # Convert to uint8
        depth_image = (depth_normalized * 255.0).clip(0, 255).astype(np.uint8)
        
        # Convert to 3-channel image
        depth_rgb = np.stack([depth_image, depth_image, depth_image], axis=-1)
        
        return Image.fromarray(depth_rgb)
    
    def unload_model(self):
        """Unload the model to free memory."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            self._model_loaded = False
            
            # Force garbage collection
            import gc
            gc.collect()
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            
            logger.info("Depth estimation model unloaded")


def get_depth_extractor() -> DepthExtractor:
    """Get or create the global depth extractor instance."""
    global _depth_extractor
    
    if _depth_extractor is None:
        _depth_extractor = DepthExtractor()
    
    return _depth_extractor

