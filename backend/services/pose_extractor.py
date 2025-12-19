"""Pose extraction service using DWPose ONNX models."""
import cv2
import numpy as np
import math
import requests
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, List, NamedTuple
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Model download URLs from HuggingFace
MODEL_URLS = {
    "yolox_l.onnx": "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx",
    "dw-ll_ucoco_384.onnx": "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx",
}


def download_file(url: str, dest_path: Path, desc: str = "Downloading") -> bool:
    """Download a file with progress bar.
    
    Args:
        url: URL to download from.
        dest_path: Destination file path.
        desc: Description for progress bar.
    
    Returns:
        True if download successful, False otherwise.
    """
    try:
        # Create parent directory if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress bar
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"Downloaded {dest_path.name} successfully")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()  # Remove partial download
        return False
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False

# ===== Data structures =====
class Keypoint(NamedTuple):
    x: float
    y: float
    score: float = 1.0
    id: int = -1


# ===== ONNX Detection functions =====
def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def demo_postprocess(outputs, img_size, p6=False):
    grids = []
    expanded_strides = []
    strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


def preprocess_det(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def inference_detector(session, oriImg, detect_classes=[0]):
    input_shape = (640, 640)
    img, ratio = preprocess_det(oriImg, input_shape)

    input_data = img[None, :, :, :]
    outNames = session.getUnconnectedOutLayersNames()
    session.setInput(input_data)
    output = session.forward(outNames)

    predictions = demo_postprocess(output[0], input_shape)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is None:
        return None
    final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
    isscore = final_scores > 0.3
    iscat = np.isin(final_cls_inds, detect_classes)
    isbbox = [i and j for (i, j) in zip(isscore, iscat)]
    final_boxes = final_boxes[isbbox]
    return final_boxes


# ===== ONNX Pose functions =====
def bbox_xyxy2cs(bbox: np.ndarray, padding: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale


def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
    w, h = np.hsplit(bbox_scale, [1])
    bbox_scale = np.where(w > h * aspect_ratio,
                          np.hstack([w, w / aspect_ratio]),
                          np.hstack([h * aspect_ratio, h]))
    return bbox_scale


def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c


def get_warp_matrix(center: np.ndarray, scale: np.ndarray, rot: float,
                    output_size: Tuple[int, int], shift: Tuple[float, float] = (0., 0.),
                    inv: bool = False) -> np.ndarray:
    shift = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0., src_w * -0.5]), rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return warp_mat


def top_down_affine(input_size: Tuple[int, int], bbox_scale: np.ndarray,
                    bbox_center: np.ndarray, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    w, h = input_size
    warp_size = (int(w), int(h))

    bbox_scale = _fix_aspect_ratio(bbox_scale, aspect_ratio=w / h)

    center = bbox_center
    scale = bbox_scale
    rot = 0
    warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

    img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)

    return img, bbox_scale


def preprocess_pose(img: np.ndarray, out_bbox, input_size: Tuple[int, int] = (192, 256)):
    img_shape = img.shape[:2]
    out_img, out_center, out_scale = [], [], []
    if len(out_bbox) == 0:
        out_bbox = [[0, 0, img_shape[1], img_shape[0]]]
    for i in range(len(out_bbox)):
        x0 = out_bbox[i][0]
        y0 = out_bbox[i][1]
        x1 = out_bbox[i][2]
        y1 = out_bbox[i][3]
        bbox = np.array([x0, y0, x1, y1])

        center, scale = bbox_xyxy2cs(bbox, padding=1.25)
        resized_img, scale = top_down_affine(input_size, scale, center, img)

        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        resized_img = (resized_img - mean) / std

        out_img.append(resized_img)
        out_center.append(center)
        out_scale.append(scale)

    return out_img, out_center, out_scale


def get_simcc_maximum(simcc_x: np.ndarray, simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)

    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.] = -1

    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)

    return locs, vals


def decode_simcc(simcc_x: np.ndarray, simcc_y: np.ndarray,
                 simcc_split_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints /= simcc_split_ratio
    return keypoints, scores


def postprocess_pose(outputs: List[np.ndarray], model_input_size: Tuple[int, int],
                     center: List[np.ndarray], scale: List[np.ndarray],
                     simcc_split_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    all_key = []
    all_score = []
    for i in range(len(outputs)):
        simcc_x, simcc_y = outputs[i]
        keypoints, scores = decode_simcc(simcc_x, simcc_y, simcc_split_ratio)
        keypoints = keypoints / model_input_size * scale[i] + center[i] - scale[i] / 2
        all_key.append(keypoints[0])
        all_score.append(scores[0])

    return np.array(all_key), np.array(all_score)


def inference_pose(session, out_bbox, oriImg, model_input_size: Tuple[int, int] = (288, 384)):
    resized_img, center, scale = preprocess_pose(oriImg, out_bbox, model_input_size)
    
    all_out = []
    for i in range(len(resized_img)):
        input_data = resized_img[i].transpose(2, 0, 1)
        input_data = input_data[None, :, :, :].astype(np.float32)

        outNames = session.getUnconnectedOutLayersNames()
        session.setInput(input_data)
        outputs = session.forward(outNames)
        all_out.append(outputs)

    keypoints, scores = postprocess_pose(all_out, model_input_size, center, scale)
    return keypoints, scores


# ===== Drawing functions =====
eps = 0.01

def is_normalized(keypoints) -> bool:
    point_normalized = [
        0 <= abs(k.x) <= 1 and 0 <= abs(k.y) <= 1
        for k in keypoints
        if k is not None
    ]
    if not point_normalized:
        return False
    return all(point_normalized)


def draw_bodypose(canvas: np.ndarray, keypoints) -> np.ndarray:
    if not is_normalized(keypoints):
        H, W = 1.0, 1.0
    else:
        H, W, _ = canvas.shape

    stickwidth = 4

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5],
        [6, 7], [7, 8], [2, 9], [9, 10],
        [10, 11], [2, 12], [12, 13], [13, 14],
        [2, 1], [1, 15], [15, 17], [1, 16],
        [16, 18],
    ]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for (k1_index, k2_index), color in zip(limbSeq, colors):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if keypoint1 is None or keypoint2 is None:
            continue

        Y = np.array([keypoint1.x, keypoint2.x]) * float(W)
        X = np.array([keypoint1.y, keypoint2.y]) * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

    for keypoint, color in zip(keypoints, colors):
        if keypoint is None:
            continue

        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)

    return canvas


# ===== Main Pose Extractor Class =====
class PoseExtractorService:
    """Service for extracting pose from images using DWPose."""

    def __init__(self):
        self._det_session = None
        self._pose_session = None
        self._models_dir = Path("models/Annotators")
        self._det_model_path = self._models_dir / "yolox_l.onnx"
        self._pose_model_path = self._models_dir / "dw-ll_ucoco_384.onnx"

    def _download_models_if_needed(self) -> bool:
        """Download models if they don't exist.
        
        Returns:
            True if all models are available, False otherwise.
        """
        # Ensure models directory exists
        self._models_dir.mkdir(parents=True, exist_ok=True)
        
        # Check and download detection model
        if not self._det_model_path.exists():
            logger.info(f"Detection model not found. Downloading {self._det_model_path.name}...")
            url = MODEL_URLS.get("yolox_l.onnx")
            if url:
                if not download_file(url, self._det_model_path, f"Downloading {self._det_model_path.name}"):
                    return False
            else:
                logger.error("Detection model URL not found")
                return False
        
        # Check and download pose model
        if not self._pose_model_path.exists():
            logger.info(f"Pose model not found. Downloading {self._pose_model_path.name}...")
            url = MODEL_URLS.get("dw-ll_ucoco_384.onnx")
            if url:
                if not download_file(url, self._pose_model_path, f"Downloading {self._pose_model_path.name}"):
                    return False
            else:
                logger.error("Pose model URL not found")
                return False
        
        return True

    def _load_models(self):
        """Load ONNX models if not already loaded. Downloads if needed."""
        if self._det_session is not None and self._pose_session is not None:
            return True

        # Try to download models if they don't exist
        if not self._download_models_if_needed():
            logger.warning("Failed to download DWPose models")
            return False

        try:
            # Use OpenCV DNN for ONNX inference
            logger.info("Loading DWPose detection model...")
            self._det_session = cv2.dnn.readNetFromONNX(str(self._det_model_path))
            self._det_session.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._det_session.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            logger.info("Loading DWPose pose model...")
            self._pose_session = cv2.dnn.readNetFromONNX(str(self._pose_model_path))
            self._pose_session.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._pose_session.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            logger.info("DWPose models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load DWPose models: {e}")
            return False

    def extract_pose(self, image: Image.Image) -> Image.Image:
        """Extract pose from image using DWPose.

        Args:
            image: Source PIL Image.

        Returns:
            Pose skeleton image on black background.
        """
        if not self._load_models():
            # Return placeholder if models not available
            return self._create_placeholder(image)

        # Convert PIL to numpy array (RGB -> BGR for OpenCV)
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Detect persons
        det_result = inference_detector(self._det_session, img_array)
        if det_result is None or len(det_result) == 0:
            # No person detected, return black image
            return Image.fromarray(np.zeros_like(img_array))

        # Extract pose keypoints
        keypoints, scores = inference_pose(self._pose_session, det_result, img_array)

        # Process keypoints
        keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
        
        # Compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3,
            keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
        
        # Reorder keypoints from MMPose to OpenPose format
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        # Create output canvas (black background)
        H, W = img_array.shape[:2]
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        # Draw pose for each person
        for instance in keypoints_info:
            # Format keypoints for drawing
            body_keypoints = []
            for i, (x, y, score) in enumerate(instance[:18]):
                if score >= 0.3:
                    body_keypoints.append(Keypoint(x, y, score, i))
                else:
                    body_keypoints.append(None)

            # Draw body pose
            canvas = draw_bodypose(canvas, body_keypoints)

        # Convert back to RGB
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return Image.fromarray(canvas_rgb)

    def _create_placeholder(self, image: Image.Image, message: str = "DWPose model download failed") -> Image.Image:
        """Create a placeholder image when models are not available."""
        width, height = image.size
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add text message
        text = message
        text2 = "Check your internet connection"
        text3 = "and try again"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1
        
        # Get text sizes
        (w1, h1), _ = cv2.getTextSize(text, font, font_scale, thickness)
        (w2, h2), _ = cv2.getTextSize(text2, font, font_scale * 0.8, thickness)
        (w3, h3), _ = cv2.getTextSize(text3, font, font_scale * 0.8, thickness)
        
        # Center text
        x1 = max(10, (width - w1) // 2)
        y1 = height // 2 - 30
        x2 = max(10, (width - w2) // 2)
        y2 = height // 2
        x3 = max(10, (width - w3) // 2)
        y3 = height // 2 + 25
        
        cv2.putText(canvas, text, (x1, y1), font, font_scale, color, thickness)
        cv2.putText(canvas, text2, (x2, y2), font, font_scale * 0.8, (200, 200, 200), thickness)
        cv2.putText(canvas, text3, (x3, y3), font, font_scale * 0.8, (200, 200, 200), thickness)
        
        return Image.fromarray(canvas)

    def is_available(self) -> bool:
        """Check if pose extraction is available."""
        return self._det_model_path.exists() and self._pose_model_path.exists()


# Global instance
_pose_extractor: Optional[PoseExtractorService] = None


def get_pose_extractor() -> PoseExtractorService:
    """Get the global pose extractor instance."""
    global _pose_extractor
    if _pose_extractor is None:
        _pose_extractor = PoseExtractorService()
    return _pose_extractor

