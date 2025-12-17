# Z-Image-Turbo-Fun-Controlnet-Union-2.0 분석

## 1. 개요

**Z-Image-Turbo-Fun-Controlnet-Union-2.0**은 VideoX-Fun 프로젝트의 이미지 생성 모델로, **다중 컨트롤 조건**(Canny, HED, Depth, Pose, MLSD 등)을 지원하는 통합 ControlNet 모델입니다.

---

## 2. 지원하는 Control 타입

Z-Image-Turbo-Fun-Controlnet-Union-2.0은 **6가지 Control 타입**을 지원합니다:

| Control 타입 | 설명 | 용도 |
|--------------|------|------|
| **Canny** | 이미지의 윤곽선/엣지를 추출 | 구조적 외곽선 기반 생성 |
| **HED** | Holistically-Nested Edge Detection, 세밀한 엣지 감지 | 더 부드러운 윤곽선 기반 생성 |
| **Depth** | 깊이 맵 (거리 정보) | 3D 공간감 있는 이미지 생성 |
| **Pose** | 인체 관절 위치 감지 (DWPose) | 캐릭터 포즈 제어 |
| **MLSD** | Multi-Line Segment Detection, 직선 감지 | 건축물/구조물 기반 생성 |
| **Inpaint** | 마스크 기반 영역 수정 (2.0 전용) | 이미지 부분 수정/편집 |

### Control 이미지 예시

```
┌─────────────────────────────────────────────────────────────────────────┐
│  원본 이미지     →    Control 추출    →    새 이미지 생성              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [사람 사진]    →    [Pose 스켈레톤]  →    [다른 스타일의 캐릭터]      │
│  [건물 사진]    →    [Canny 엣지]     →    [같은 구조의 다른 건물]     │
│  [풍경 사진]    →    [Depth 맵]       →    [같은 구도의 다른 풍경]     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Control 추출 및 사용 예시 코드

#### 1. Canny Edge Detection

```python
import cv2
import numpy as np
from PIL import Image

def extract_canny(image_path, low_threshold=100, high_threshold=200):
    """Canny 엣지 추출"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return edges_colored

# 사용 예시
canny_image = extract_canny("input.jpg", low=100, high=200)
Image.fromarray(canny_image).save("canny_control.png")
```

#### 2. Depth Map (ZoeDepth 사용)

```python
import torch
import cv2
import numpy as np
from einops import rearrange

# ZoeDepth 모델 로드 (VideoX-Fun 내장)
from comfyui.annotator.zoe.zoedepth.models.zoedepth.zoedepth_v1 import ZoeDepth
from comfyui.annotator.zoe.zoedepth.utils.config import get_config

def extract_depth(image_path, model_path="ZoeD_M12_N.pt"):
    """Depth 맵 추출 (ZoeDepth)"""
    # 모델 로드
    model = ZoeDepth.build_from_config(get_config("zoedepth", "infer"))
    model.load_state_dict(torch.load(model_path, map_location="cpu")['model'], strict=False)
    model = model.to("cuda").eval()
    
    # 이미지 처리
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image).to("cuda", torch.float32) / 255.0
    image_tensor = rearrange(image_tensor, 'h w c -> 1 c h w')
    
    # Depth 추론
    with torch.no_grad():
        depth = model.infer(image_tensor)
        depth = depth[0, 0].cpu().numpy()
        
        # 정규화
        vmin, vmax = np.percentile(depth, 2), np.percentile(depth, 85)
        depth = (depth - vmin) / (vmax - vmin)
        depth = 1.0 - depth  # 반전
        depth_image = (depth * 255).clip(0, 255).astype(np.uint8)
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2RGB)
    
    return depth_image

# 사용 예시
depth_image = extract_depth("input.jpg")
Image.fromarray(depth_image).save("depth_control.png")
```

#### 3. Pose Detection (DWPose 사용)

```python
import cv2
import numpy as np
from PIL import Image

# DWPose 모델 로드 (VideoX-Fun 내장)
from comfyui.annotator.dwpose_utils import DWposeDetector

def extract_pose(image_path, det_model="yolox_l.onnx", pose_model="dw-ll_ucoco_384.onnx"):
    """Pose 스켈레톤 추출 (DWPose)"""
    # 모델 로드
    detector = DWposeDetector(det_model, pose_model)
    
    # 이미지 처리
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Pose 추론
    pose_image = detector(image)
    
    return pose_image

# 사용 예시
pose_image = extract_pose("input.jpg")
Image.fromarray(pose_image).save("pose_control.png")
```

#### 4. HED (Holistically-Nested Edge Detection)

```python
import cv2
import numpy as np
from PIL import Image

def extract_hed(image_path, prototxt="deploy.prototxt", caffemodel="hed_pretrained_bsds.caffemodel"):
    """HED 엣지 추출 (OpenCV DNN)"""
    # HED 모델 로드
    net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    
    # 이미지 로드
    image = cv2.imread(image_path)
    (H, W) = image.shape[:2]
    
    # Blob 생성
    blob = cv2.dnn.blobFromImage(
        image, 
        scalefactor=1.0, 
        size=(W, H),
        mean=(104.00698793, 116.66876762, 122.67891434),
        swapRB=False, 
        crop=False
    )
    
    # 추론
    net.setInput(blob)
    hed = net.forward()
    hed = cv2.resize(hed[0, 0], (W, H))
    hed = (255 * hed).astype('uint8')
    hed = cv2.cvtColor(hed, cv2.COLOR_GRAY2RGB)
    
    return hed

# 사용 예시 (HED 모델 파일 필요)
# hed_image = extract_hed("input.jpg")
# Image.fromarray(hed_image).save("hed_control.png")
```

#### 5. MLSD (Multi-Line Segment Detection)

```python
import cv2
import numpy as np
from PIL import Image

def extract_mlsd(image_path, score_thr=0.1, dist_thr=20.0):
    """MLSD 직선 감지 (OpenCV LSD)"""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # LSD 라인 감지
    lsd = cv2.createLineSegmentDetector(0)
    lines, width, prec, nfa = lsd.detect(gray)
    
    # 결과 이미지 생성
    result = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x0, y0, x1, y1 = map(int, line[0])
            cv2.line(result, (x0, y0), (x1, y1), (255, 255, 255), 1)
    
    return result

# 사용 예시
mlsd_image = extract_mlsd("input.jpg")
Image.fromarray(mlsd_image).save("mlsd_control.png")
```

#### 6. Inpainting (마스크 기반 편집) - 2.0 전용

```python
import torch
from PIL import Image
import numpy as np

# 마스크 이미지 생성 예시
def create_mask(image_size, mask_region):
    """마스크 이미지 생성 (흰색=수정할 영역)"""
    mask = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)
    x, y, w, h = mask_region  # (x, y, width, height)
    mask[y:y+h, x:x+w] = 255
    return mask

# Inpainting 전체 파이프라인 (2.0 전용)
from videox_fun.utils.utils import get_image_latent

sample_size = [1728, 992]

# 원본 이미지, 마스크, 컨트롤 이미지 로드
inpaint_image = get_image_latent("original.png", sample_size=sample_size)[:, :, 0]
mask_image = get_image_latent("mask.png", sample_size=sample_size)[:, :1, 0]
control_image = get_image_latent("pose.png", sample_size=sample_size)[:, :, 0]

# 파이프라인 실행
result = pipeline(
    prompt="새로운 내용 설명",
    height=sample_size[0],
    width=sample_size[1],
    image=inpaint_image,           # 원본 이미지
    mask_image=mask_image,         # 마스크 (흰색=수정 영역)
    control_image=control_image,   # 컨트롤 이미지
    num_inference_steps=25,
    control_context_scale=0.75,
).images
```

### 필요한 Annotator 모델 파일

| 모델 | 다운로드 경로 |
|------|---------------|
| **ZoeDepth** | `ZoeD_M12_N.pt` - [HuggingFace](https://huggingface.co/lllyasviel/Annotators/resolve/main/ZoeD_M12_N.pt) |
| **DWPose (Det)** | `yolox_l.onnx` - [HuggingFace](https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx) |
| **DWPose (Pose)** | `dw-ll_ucoco_384.onnx` - [HuggingFace](https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx) |
| **HED** | `hed_pretrained_bsds.caffemodel` - [GitHub](https://github.com/s9xie/hed) |

### 주요 특징

- **Union 모델**: 하나의 모델로 여러 Control 타입을 모두 처리 (별도 모델 불필요)
- **학습 데이터**: 100만 장의 고품질 이미지로 10,000 스텝 학습
- **학습 해상도**: 1328 (BFloat16 정밀도)
- **권장 control_context_scale**: 0.65 ~ 0.80

---

## 3. 핵심 구성 요소

| 구성 요소 | 설명 |
|-----------|------|
| **ZImageControlTransformer2DModel** | Control 기능이 추가된 Transformer 모델 |
| **ZImageControlPipeline** | 이미지 생성 파이프라인 |
| **Qwen3ForCausalLM** | 텍스트 인코더 |
| **AutoencoderKL** | VAE 모델 |
| **FlowMatchEulerDiscreteScheduler** | 스케줄러 |

---

## 4. 2.0 버전 설정

### 설정 파일 (`z_image_control_2.0.yaml`)

```yaml
transformer_additional_kwargs:
    control_layers_places: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    control_refiner_layers_places: [0, 1]
    add_control_noise_refiner: true   # Noise Refiner에서도 Control 적용
    control_in_dim: 33                # 33채널 입력 (16 latent + 1 mask + 16 control)
```

### 2.0 핵심 특징

| 특징 | 설명 |
|------|------|
| **Noise Refiner** | Refiner 단계에서도 Control 적용 |
| **Inpainting 지원** | 마스크 기반 이미지 편집 가능 |
| **15개 Control Layers** | 더 정교한 Control 적용 |
| **권장 스텝 수** | 25 스텝 |

---

## 5. Python 사용 코드 (단순화 버전)

```python
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image

# VideoX-Fun 모듈 import
from videox_fun.dist import set_multi_gpus_devices
from videox_fun.models import (AutoencoderKL, AutoTokenizer,
                               Qwen3ForCausalLM, ZImageControlTransformer2DModel)
from videox_fun.pipeline import ZImageControlPipeline
from videox_fun.utils.utils import get_image_latent

# ================== 설정 ==================
config_path = "config/z_image/z_image_control_2.0.yaml"
model_name = "models/Diffusion_Transformer/Z-Image-Turbo/"
transformer_path = "models/Personalized_Model/Z-Image-Turbo-Fun-Controlnet-Union-2.0.safetensors"

weight_dtype = torch.bfloat16  # RTX 30/40 시리즈 사용시
# weight_dtype = torch.float16  # RTX 20 시리즈 이하

# ================== 모델 로드 ==================
device = set_multi_gpus_devices(1, 1)
config = OmegaConf.load(config_path)

# 1. Transformer 로드
transformer = ZImageControlTransformer2DModel.from_pretrained(
    model_name, 
    subfolder="transformer",
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
).to(weight_dtype)

# 2. ControlNet Union 가중치 로드
from safetensors.torch import load_file
state_dict = load_file(transformer_path)
m, u = transformer.load_state_dict(state_dict, strict=False)
print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# 3. VAE 로드
vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(weight_dtype)

# 4. 텍스트 인코더 로드
tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
text_encoder = Qwen3ForCausalLM.from_pretrained(
    model_name, subfolder="text_encoder", 
    torch_dtype=weight_dtype,
    low_cpu_mem_usage=True,
)

# 5. 스케줄러
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")

# 6. 파이프라인 구성
pipeline = ZImageControlPipeline(
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    transformer=transformer,
    scheduler=scheduler,
)
pipeline.enable_model_cpu_offload(device=device)

# ================== 이미지 생성 ==================
sample_size = [1728, 992]  # [height, width]
control_image = get_image_latent("asset/pose.jpg", sample_size=sample_size)[:, :, 0]

prompt = "A beautiful woman with purple hair on the beach"
seed = 43
generator = torch.Generator(device=device).manual_seed(seed)

with torch.no_grad():
    result = pipeline(
        prompt=prompt, 
        negative_prompt=" ",
        height=sample_size[0],
        width=sample_size[1],
        generator=generator,
        guidance_scale=0.0,          # 0.0 = 가이던스 비활성화 (Turbo 모델 특성)
        control_image=control_image,
        num_inference_steps=25,       # 2.0은 25 스텝 권장
        control_context_scale=0.75,   # 컨트롤 강도 (0.65~0.80 권장)
    ).images

# 저장
result[0].save("output.png")
```

---

## 6. GPU 메모리 모드 및 양자화 옵션

### 6.1 GPU 메모리 모드

| 모드 | 속도 | VRAM 사용량 | 설명 |
|------|------|-------------|------|
| `model_full_load` | ⚡ 가장 빠름 | 가장 많음 | 전체 모델 GPU 로드 |
| `model_full_load_and_qfloat8` | 빠름 | 중간 | GPU 로드 + FP8 양자화 |
| `model_cpu_offload` | 보통 | 적음 | 사용 후 CPU로 오프로드 |
| `model_cpu_offload_and_qfloat8` | 보통 | 더 적음 | CPU 오프로드 + FP8 양자화 |
| `sequential_cpu_offload` | 🐢 가장 느림 | 최소 | 레이어별 CPU 오프로드 |

### 6.2 FP8 양자화 사용법

FP8 (`torch.float8_e4m3fn`) 양자화를 사용하면 **VRAM을 약 50% 절약**할 수 있습니다.

```python
from videox_fun.utils.fp8_optimization import (
    convert_model_weight_to_float8,
    convert_weight_dtype_wrapper
)

# ================== FP8 양자화 적용 예시 ==================

GPU_memory_mode = "model_cpu_offload_and_qfloat8"  # 권장

# 파이프라인 생성 후 메모리 모드 적용
if GPU_memory_mode == "sequential_cpu_offload":
    pipeline.enable_sequential_cpu_offload(device=device)

elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    # FP8 양자화 적용 (제외할 모듈 지정)
    convert_model_weight_to_float8(
        transformer, 
        exclude_module_name=["img_in", "txt_in", "timestep"],  # 입출력 레이어 제외
        device=device
    )
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)

elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)

elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(
        transformer, 
        exclude_module_name=["img_in", "txt_in", "timestep"],
        device=device
    )
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.to(device=device)

else:  # model_full_load
    pipeline.to(device=device)
```

### 6.3 FP8 양자화 주의사항

| 항목 | 설명 |
|------|------|
| **제외 모듈** | `img_in`, `txt_in`, `timestep` 등 입출력 레이어는 양자화에서 제외 |
| **품질 영향** | 약간의 품질 저하 가능 (대부분 무시할 수준) |
| **호환성** | RTX 40 시리즈 이상에서 최적 성능 (FP8 하드웨어 지원) |
| **dtype 래퍼** | `convert_weight_dtype_wrapper`로 forward 시 자동 dtype 변환 |

---

## 7. 샘플러 (Scheduler) 옵션

### 지원 샘플러

| 샘플러 | 클래스 | 특징 |
|--------|--------|------|
| `Flow` | `FlowMatchEulerDiscreteScheduler` | 기본 샘플러, 안정적 |
| `Flow_Unipc` | `FlowUniPCMultistepScheduler` | 더 빠른 수렴, 적은 스텝 |
| `Flow_DPM++` | `FlowDPMSolverMultistepScheduler` | 고품질 결과 |

### 샘플러 선택 코드

```python
from diffusers import FlowMatchEulerDiscreteScheduler
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

# 샘플러 선택
sampler_name = "Flow"  # "Flow", "Flow_Unipc", "Flow_DPM++" 중 선택

scheduler_dict = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}

Chosen_Scheduler = scheduler_dict[sampler_name]
scheduler = Chosen_Scheduler.from_pretrained(model_name, subfolder="scheduler")
```

### 권장 설정

| 샘플러 | 권장 스텝 수 | 용도 |
|--------|-------------|------|
| `Flow` | 25 | 기본, 안정적인 결과 |
| `Flow_Unipc` | 15~20 | 빠른 생성 |
| `Flow_DPM++` | 20~25 | 고품질 출력 |

---

## 8. LoRA 사용법

### LoRA 적용/해제

```python
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora

# LoRA 경로 및 가중치
lora_path = "models/Lora/your_lora.safetensors"
lora_weight = 0.8  # 0.0 ~ 1.0

# LoRA 적용 (모델에 병합)
pipeline = merge_lora(
    pipeline, 
    lora_path, 
    lora_weight, 
    device=device, 
    dtype=weight_dtype
)

# 이미지 생성...
result = pipeline(prompt=prompt, ...).images

# LoRA 해제 (원본 복원)
pipeline = unmerge_lora(
    pipeline, 
    lora_path, 
    lora_weight, 
    device=device, 
    dtype=weight_dtype
)
```

### LoRA 특징

- **동적 로딩**: 런타임에 LoRA를 적용/해제 가능
- **가중치 조절**: `lora_weight`로 LoRA 영향도 조절 (0.0=없음, 1.0=100%)
- **다중 LoRA**: 여러 LoRA를 순차적으로 적용 가능

---

## 9. Inpainting 사용법 (2.0 전용)

```python
# Inpainting에 필요한 추가 입력
inpaint_image = get_image_latent("asset/8.png", sample_size=sample_size)[:, :, 0]
mask_image = get_image_latent("asset/mask.png", sample_size=sample_size)[:, :1, 0]

result = pipeline(
    prompt=prompt,
    height=sample_size[0],
    width=sample_size[1],
    image=inpaint_image,           # 원본 이미지
    mask_image=mask_image,         # 마스크 (흰색=수정할 영역)
    control_image=control_image,   # 컨트롤 이미지 (Pose, Canny 등)
    num_inference_steps=25,
    control_context_scale=0.75,
).images
```

---

## 10. 핵심 파라미터 정리

### 기본 파라미터

| 파라미터 | 범위/타입 | 기본값 | 설명 |
|----------|-----------|--------|------|
| `prompt` | str / List[str] | - | 생성할 이미지 설명 |
| `negative_prompt` | str / List[str] | None | 피할 내용 (Turbo는 " " 권장) |
| `height` | int | 1024 | 출력 높이 (16 배수) |
| `width` | int | 1024 | 출력 너비 (16 배수) |
| `num_inference_steps` | int | 50 | 추론 스텝 수 |
| `guidance_scale` | float | 5.0 | CFG 스케일 (**Turbo는 0.0**) |
| `seed` | int | - | 랜덤 시드 |

### Control 관련 파라미터

| 파라미터 | 범위/타입 | 기본값 | 설명 |
|----------|-----------|--------|------|
| `control_image` | torch.FloatTensor | None | 컨트롤 이미지 (Pose, Canny 등) |
| `control_context_scale` | 0.0~1.0 | 1.0 | 컨트롤 강도 (**0.65~0.80 권장**) |
| `image` | torch.FloatTensor | None | Inpaint용 원본 이미지 |
| `mask_image` | torch.FloatTensor | None | Inpaint용 마스크 |

### 고급 파라미터

| 파라미터 | 범위/타입 | 기본값 | 설명 |
|----------|-----------|--------|------|
| `sigmas` | List[float] | None | 커스텀 시그마 스케줄 |
| `cfg_normalization` | bool | False | CFG 정규화 |
| `cfg_truncation` | float | 1.0 | CFG 잘라내기 비율 |
| `max_sequence_length` | int | 512 | 최대 토큰 길이 |
| `num_images_per_prompt` | int | 1 | 프롬프트당 이미지 수 |
| `output_type` | str | "pil" | 출력 타입 ("pil", "latent") |

### 권장 설정

| 설정 | 권장값 |
|------|--------|
| `num_inference_steps` | **25** |
| `control_context_scale` | **0.65~0.80** |
| `guidance_scale` | **0.0** (Turbo 모델) |
| `sample_size` | **[1728, 992]** (학습 해상도) |

---

## 11. 필요 라이브러리

### CUDA 지원 GPU 사용 시 (권장)

```bash
# PyTorch CUDA 버전 설치 (CUDA 12.6 기준)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 나머지 라이브러리 설치
pip install diffusers>=0.30.1 transformers>=4.46.2 safetensors omegaconf Pillow
```

> ⚠️ **주의**: 일반 `pip install torch`는 CPU 버전이 설치됩니다. GPU 가속을 위해서는 반드시 위 명령어로 CUDA 버전을 설치해야 합니다.


### 전체 requirements.txt

```txt
# 핵심 라이브러리
torch>=2.1.2
torchvision
diffusers>=0.30.1
transformers>=4.46.2
safetensors
omegaconf
Pillow
accelerate>=0.25.0

# 추가 필수 라이브러리
einops              # 텐서 연산 (Depth 추출 등)
opencv-python       # 이미지 처리 (Canny, MLSD 등)
onnxruntime         # Pose 추출 (DWPose)
numpy               # 수치 연산
scikit-image        # 이미지 처리

# 선택 라이브러리
gradio>=3.41.2      # WebUI 사용 시
decord              # 비디오 처리 시
imageio[ffmpeg]     # 비디오 저장 시
```

---

## 12. 아키텍처 흐름도

```
┌─────────────────────────────────────────────────────────────┐
│                    ZImageControlPipeline                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Text Encoding (Qwen3ForCausalLM)                       │
│     prompt → text_embeds                                    │
│                                                             │
│  2. Control Image Processing                                │
│     control_image → VAE.encode → control_latents           │
│                                                             │
│  3. Noise Refiner (2.0 신규)                               │
│     control_noise_refiner → refiner_hints                  │
│                                                             │
│  4. Main Transformer Blocks (30 layers)                    │
│     latents + hints → denoised_latents                     │
│                                                             │
│  5. VAE Decode                                             │
│     latents → output_image                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 13. 모델 다운로드 경로

> ⚠️ **중요**: Z-Image-Turbo-Fun-Controlnet-Union-2.0을 사용하려면 **두 모델이 모두 필요합니다!**

### 필요한 모델

| 모델 | 제공하는 것 | 크기 |
|------|-------------|------|
| **Base 모델 (Z-Image-Turbo)** | Transformer 기본 구조, VAE, Tokenizer, Text Encoder, Scheduler | ~**26GB** (전체 폴더) |
| **ControlNet Union 2.0** | Control 관련 레이어 가중치 | ~**3.1GB** (단일 `.safetensors`) |

### Annotator 모델 (Control 추출용, 선택)

| 모델 | 용도 | 크기 |
|------|------|------|
| **ZoeD_M12_N.pt** | Depth 맵 추출 | ~**1.44GB** |
| **yolox_l.onnx** | Pose - 사람 감지 | ~**217MB** |
| **dw-ll_ucoco_384.onnx** | Pose - 관절 추출 | ~**134MB** |
| **Canny** | 윤곽선 추출 | 별도 모델 불필요 (OpenCV 내장) |
| **MLSD** | 직선 감지 | 별도 모델 불필요 (OpenCV 내장) |

### 총 필요 용량

| 구성 | 용량 |
|------|------|
| **최소 (Base + ControlNet)** | ~**29GB** |
| **전체 (+ Annotator 모델)** | ~**31GB** |

### 다운로드 링크

- **Base 모델**: [Z-Image-Turbo (HuggingFace)](https://huggingface.co/alibaba-pai/Z-Image-Turbo)
- **ControlNet Union 2.0**: [Z-Image-Turbo-Fun-Controlnet-Union (HuggingFace)](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union)

### 로드 순서

```python
# 1단계: Base 모델에서 기본 구조 로드
transformer = ZImageControlTransformer2DModel.from_pretrained(
    "models/Diffusion_Transformer/Z-Image-Turbo/",
    subfolder="transformer",
    ...
)

# 2단계: ControlNet Union 2.0 가중치를 추가로 로드
state_dict = load_file("models/Personalized_Model/Z-Image-Turbo-Fun-Controlnet-Union-2.0.safetensors")
transformer.load_state_dict(state_dict, strict=False)
```

### 디렉토리 구조

```
models/
├── Diffusion_Transformer/
│   └── Z-Image-Turbo/
│       ├── transformer/
│       ├── vae/
│       ├── tokenizer/
│       ├── text_encoder/
│       └── scheduler/
└── Personalized_Model/
    └── Z-Image-Turbo-Fun-Controlnet-Union-2.0.safetensors
```

---

## 14. 관련 파일 경로 (VideoX-Fun)

### 예제 스크립트

| 파일 | 설명 |
|------|------|
| `examples/z_image_fun/predict_t2i_control_2.0.py` | **Control T2I** (Text-to-Image + Control) |
| `examples/z_image_fun/predict_i2i_inpaint_2.0.py` | **Inpainting** (이미지 편집 + Control) |

### 모델/파이프라인 구현

| 파일 | 설명 |
|------|------|
| `videox_fun/pipeline/pipeline_z_image_control.py` | Control 파이프라인 (ZImageControlPipeline) |
| `videox_fun/models/z_image_transformer2d_control.py` | Control Transformer (ZImageControlTransformer2DModel) |

### 설정 파일

| 파일 | 설명 |
|------|------|
| `config/z_image/z_image_control_2.0.yaml` | 2.0 버전 설정 |

### 모델 다운로드

| 모델 | 파일명 |
|------|--------|
| **Base 모델** | `Z-Image-Turbo/` 폴더 전체 (~26GB) |
| **ControlNet 2.0** | `Z-Image-Turbo-Fun-Controlnet-Union-2.0.safetensors` (~3.1GB) |

---

## 15. 고급 옵션 (Z-Image 전용)

> **참고**: `predict_t2i_control_2.0.py` 예제에서 지원하는 옵션들입니다.

### 15.1 사용 가능한 파라미터

#### 샘플링 파라미터

| 파라미터 | 범위/옵션 | 기본값 | 설명 |
|----------|-----------|--------|------|
| `width` | 64~2048 (step 16) | 992 | 출력 너비 |
| `height` | 64~2048 (step 16) | 1728 | 출력 높이 |
| `seed` | 0~2^64 | 43 | 랜덤 시드 |
| `num_inference_steps` | 1~200 | 25 | 추론 스텝 수 |
| `guidance_scale` | 0.0~20.0 | **0.0** | 가이던스 스케일 (Turbo=0) |
| `control_context_scale` | 0.0~1.0 | 0.75 | 컨트롤 강도 (0.65~0.80 권장) |

#### LoRA 옵션

| 파라미터 | 범위/옵션 | 기본값 | 설명 |
|----------|-----------|--------|------|
| `lora_path` | 파일 경로 | None | LoRA 파일 |
| `lora_weight` | 0.0~1.0 | 0.55 | LoRA 가중치 |

#### Control 추출 옵션

| 타입 | 파라미터 | 범위 | 설명 |
|------|----------|------|------|
| **Canny** | `low_threshold` | 0~255 (기본 100) | 하한 임계값 |
|           | `high_threshold` | 0~255 (기본 200) | 상한 임계값 |
| **Depth** | - | - | ZoeDepth 모델 사용 |
| **Pose** | - | - | DWPose 모델 사용 |

### 15.2 Multi-GPU / 분산 처리

```python
from videox_fun.dist import set_multi_gpus_devices, shard_model

# Multi-GPU 설정
ulysses_degree = 1      # Ulysses 분산 차수
ring_degree = 1         # Ring 분산 차수
# 참고: ulysses_degree × ring_degree = GPU 개수

device = set_multi_gpus_devices(ulysses_degree, ring_degree)

# FSDP (Fully Sharded Data Parallel) - 대규모 GPU에서 메모리 절약
fsdp_dit = False        # Transformer FSDP 활성화
fsdp_text_encoder = False  # Text Encoder FSDP 활성화

if ulysses_degree > 1 or ring_degree > 1:
    transformer.enable_multi_gpus_inference()
    if fsdp_dit:
        from functools import partial
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype, 
                          module_to_wrapper=list(transformer.transformer_blocks))
        pipeline.transformer = shard_fn(pipeline.transformer)
```

### 15.3 torch.compile 최적화

고정 해상도에서 **속도 향상** (첫 실행 시 컴파일 시간 필요):

```python
compile_dit = True

if compile_dit:
    for i in range(len(pipeline.transformer.transformer_blocks)):
        pipeline.transformer.transformer_blocks[i] = torch.compile(
            pipeline.transformer.transformer_blocks[i]
        )
    print("Add Compile")
```

> ⚠️ **주의**: `sequential_cpu_offload`와 호환되지 않습니다.

### 15.4 Attention 타입 선택

```python
import os

# Attention 백엔드 선택 (pipeline 생성 전에 설정)
os.environ['VIDEOX_ATTENTION_TYPE'] = "FLASH_ATTENTION"

# 옵션:
# - "FLASH_ATTENTION": Flash Attention 2 (기본, 가장 빠름)
# - "SAGE_ATTENTION": Sage Attention
# - "TORCH_SCALED_DOT": PyTorch 기본 Scaled Dot Product
```

### 15.5 Inpainting 예제 (2.0 전용)

> 2.0 전용 기능: Inpainting + Control 동시 사용

```python
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf

from videox_fun.dist import set_multi_gpus_devices
from videox_fun.models import (AutoencoderKL, AutoTokenizer,
                               Qwen3ForCausalLM, ZImageControlTransformer2DModel)
from videox_fun.pipeline import ZImageControlPipeline
from videox_fun.utils.utils import get_image_latent

# ================== 2.0 버전 설정 ==================
config_path = "config/z_image/z_image_control_2.0.yaml"  # 2.0 설정
model_name = "models/Diffusion_Transformer/Z-Image-Turbo/"
transformer_path = "models/Personalized_Model/Z-Image-Turbo-Fun-Controlnet-Union-2.0.safetensors"  # 2.0 모델

weight_dtype = torch.bfloat16
sample_size = [1728, 992]

# Inpainting 입력
control_image_path = "asset/pose.jpg"
inpaint_image_path = "asset/8.png"      # 원본 이미지
mask_image_path = "asset/mask.png"       # 마스크 (흰색=수정 영역)

prompt = "A beautiful woman with purple hair on the beach"
seed = 43
num_inference_steps = 25   # 2.0은 25 스텝 권장
guidance_scale = 0.0
control_context_scale = 0.75

# ================== 모델 로드 ==================
device = set_multi_gpus_devices(1, 1)
config = OmegaConf.load(config_path)

transformer = ZImageControlTransformer2DModel.from_pretrained(
    model_name, subfolder="transformer",
    low_cpu_mem_usage=True, torch_dtype=weight_dtype,
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
).to(weight_dtype)

from safetensors.torch import load_file
state_dict = load_file(transformer_path)
transformer.load_state_dict(state_dict, strict=False)

vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(weight_dtype)
tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
text_encoder = Qwen3ForCausalLM.from_pretrained(
    model_name, subfolder="text_encoder", torch_dtype=weight_dtype, low_cpu_mem_usage=True
)
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")

pipeline = ZImageControlPipeline(
    vae=vae, tokenizer=tokenizer, text_encoder=text_encoder,
    transformer=transformer, scheduler=scheduler,
)
pipeline.enable_model_cpu_offload(device=device)

# ================== 이미지 로드 ==================
control_image = get_image_latent(control_image_path, sample_size=sample_size)[:, :, 0]
inpaint_image = get_image_latent(inpaint_image_path, sample_size=sample_size)[:, :, 0]
mask_image = get_image_latent(mask_image_path, sample_size=sample_size)[:, :1, 0]

# ================== Inpainting 실행 ==================
generator = torch.Generator(device=device).manual_seed(seed)

with torch.no_grad():
    result = pipeline(
        prompt=prompt,
        negative_prompt=" ",
        height=sample_size[0],
        width=sample_size[1],
        generator=generator,
        guidance_scale=guidance_scale,
        image=inpaint_image,              # 원본 이미지
        mask_image=mask_image,            # 마스크
        control_image=control_image,      # 컨트롤 이미지
        num_inference_steps=num_inference_steps,
        control_context_scale=control_context_scale,
    ).images

result[0].save("output_inpaint_2.0.png")
```

### 15.6 종합 Python 예제 (모든 옵션 포함)

```python
import os
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf

from videox_fun.dist import set_multi_gpus_devices
from videox_fun.models import (AutoencoderKL, AutoTokenizer,
                               Qwen3ForCausalLM, ZImageControlTransformer2DModel)
from videox_fun.pipeline import ZImageControlPipeline
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import get_image_latent

# ================== 환경 설정 ==================
os.environ['VIDEOX_ATTENTION_TYPE'] = "FLASH_ATTENTION"

# ================== 파라미터 설정 ==================
config_path = "config/z_image/z_image_control_2.0.yaml"
model_name = "models/Diffusion_Transformer/Z-Image-Turbo/"
transformer_path = "models/Personalized_Model/Z-Image-Turbo-Fun-Controlnet-Union-2.0.safetensors"

# 메모리 & 성능 옵션
GPU_memory_mode = "model_cpu_offload_and_qfloat8"
weight_dtype = torch.bfloat16
compile_dit = False

# 샘플링 파라미터
sample_size = [1728, 992]  # [height, width]
prompt = "A beautiful woman with purple hair on the beach"
negative_prompt = " "
seed = 43
num_inference_steps = 25
guidance_scale = 0.0        # Turbo 모델은 0 사용
control_context_scale = 0.75

# LoRA 옵션
lora_path = None
lora_weight = 0.55

# ================== 모델 로드 ==================
device = set_multi_gpus_devices(1, 1)
config = OmegaConf.load(config_path)

# Transformer
transformer = ZImageControlTransformer2DModel.from_pretrained(
    model_name, subfolder="transformer",
    low_cpu_mem_usage=True, torch_dtype=weight_dtype,
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
).to(weight_dtype)

# ControlNet 가중치 로드
from safetensors.torch import load_file
state_dict = load_file(transformer_path)
transformer.load_state_dict(state_dict, strict=False)

# VAE, Tokenizer, Text Encoder
vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(weight_dtype)
tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
text_encoder = Qwen3ForCausalLM.from_pretrained(
    model_name, subfolder="text_encoder", torch_dtype=weight_dtype, low_cpu_mem_usage=True
)
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler")

# 파이프라인 구성
pipeline = ZImageControlPipeline(
    vae=vae, tokenizer=tokenizer, text_encoder=text_encoder,
    transformer=transformer, scheduler=scheduler,
)

# Compile (선택)
if compile_dit:
    for i in range(len(pipeline.transformer.transformer_blocks)):
        pipeline.transformer.transformer_blocks[i] = torch.compile(
            pipeline.transformer.transformer_blocks[i]
        )

# 메모리 모드 적용
if GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["img_in", "txt_in", "timestep"], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
else:
    pipeline.to(device=device)

# LoRA 적용
if lora_path:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

# ================== 이미지 생성 ==================
generator = torch.Generator(device=device).manual_seed(seed)
control_image = get_image_latent("asset/pose.jpg", sample_size=sample_size)[:, :, 0]

with torch.no_grad():
    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=sample_size[0],
        width=sample_size[1],
        generator=generator,
        guidance_scale=guidance_scale,
        control_image=control_image,
        num_inference_steps=num_inference_steps,
        control_context_scale=control_context_scale,
    ).images

# 저장
result[0].save("output.png")

# 정리
if lora_path:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)
```
