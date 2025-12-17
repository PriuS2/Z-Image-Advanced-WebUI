"""Control image extraction service (Canny, Depth, Pose, etc.)."""
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple

from backend.config import get_config


class ControlExtractorService:
    """Service for extracting control images from source images."""
    
    def __init__(self):
        self.config = get_config()
        self._depth_model = None
        self._pose_model = None
        self._hed_model = None
    
    def extract_canny(
        self,
        image: Image.Image,
        low_threshold: int = 100,
        high_threshold: int = 200
    ) -> Image.Image:
        """Extract Canny edge detection from image.
        
        Args:
            image: Source PIL Image.
            low_threshold: Lower threshold for edge detection.
            high_threshold: Upper threshold for edge detection.
        
        Returns:
            Canny edge image.
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # Convert back to RGB (3 channel)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(edges_rgb)
    
    def extract_hed(self, image: Image.Image) -> Image.Image:
        """Extract HED (Holistically-Nested Edge Detection) from image.
        
        Args:
            image: Source PIL Image.
        
        Returns:
            HED edge image.
        """
        # TODO: Implement HED using ONNX model
        # For now, return Canny as fallback
        return self.extract_canny(image, 50, 150)
    
    def extract_mlsd(self, image: Image.Image) -> Image.Image:
        """Extract MLSD (Mobile Line Segment Detection) from image.
        
        Args:
            image: Source PIL Image.
        
        Returns:
            MLSD line image.
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Use Hough Line Transform as a simple line detection
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
        
        # Create output image
        output = np.zeros_like(img_array)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(output, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        return Image.fromarray(output)
    
    def extract_depth(self, image: Image.Image) -> Image.Image:
        """Extract depth map from image using ZoeDepth.
        
        Args:
            image: Source PIL Image.
        
        Returns:
            Depth map image.
        """
        # TODO: Implement depth extraction using ZoeDepth
        # For now, return a gradient placeholder
        width, height = image.size
        depth_array = np.zeros((height, width), dtype=np.uint8)
        
        # Create a simple gradient as placeholder
        for y in range(height):
            depth_array[y, :] = int(255 * y / height)
        
        depth_rgb = cv2.cvtColor(depth_array, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(depth_rgb)
    
    def extract_pose(self, image: Image.Image) -> Image.Image:
        """Extract pose from image using DWPose.
        
        Args:
            image: Source PIL Image.
        
        Returns:
            Pose skeleton image.
        """
        # TODO: Implement pose extraction using DWPose/ONNX
        # For now, return a black image as placeholder
        width, height = image.size
        pose_array = np.zeros((height, width, 3), dtype=np.uint8)
        return Image.fromarray(pose_array)
    
    def extract(
        self,
        image: Image.Image,
        control_type: str,
        **kwargs
    ) -> Image.Image:
        """Extract control image based on type.
        
        Args:
            image: Source PIL Image.
            control_type: Type of control extraction (canny, hed, mlsd, depth, pose).
            **kwargs: Additional arguments for specific extractors.
        
        Returns:
            Extracted control image.
        """
        extractors = {
            "canny": self.extract_canny,
            "hed": self.extract_hed,
            "mlsd": self.extract_mlsd,
            "depth": self.extract_depth,
            "pose": self.extract_pose,
        }
        
        extractor = extractors.get(control_type.lower())
        if extractor is None:
            raise ValueError(f"Unknown control type: {control_type}")
        
        return extractor(image, **kwargs)
    
    def resize_control_image(
        self,
        image: Image.Image,
        target_width: int,
        target_height: int
    ) -> Image.Image:
        """Resize control image to match generation size.
        
        Args:
            image: Control image.
            target_width: Target width.
            target_height: Target height.
        
        Returns:
            Resized image.
        """
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


# Global instance
_extractor: Optional[ControlExtractorService] = None


def get_extractor() -> ControlExtractorService:
    """Get the global control extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = ControlExtractorService()
    return _extractor
