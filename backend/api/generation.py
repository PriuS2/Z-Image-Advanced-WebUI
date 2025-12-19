"""Image generation API endpoints."""
import logging
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import get_db, User, TaskQueue
from backend.api.auth import get_current_user
from backend.services.queue_manager import get_queue_manager, QueuedTask

logger = logging.getLogger(__name__)
router = APIRouter()


class GenerationRequest(BaseModel):
    """Image generation request model."""
    prompt: str
    prompt_ko: Optional[str] = None
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 25
    guidance_scale: float = 0.0
    control_context_scale: float = 0.75
    seed: Optional[int] = None
    sampler: str = "Flow"
    control_type: Optional[str] = None
    control_image_path: Optional[str] = None
    mask_image_path: Optional[str] = None
    original_image_path: Optional[str] = None  # For inpainting: the original image to inpaint
    lora_name: Optional[str] = None
    lora_weight: float = 0.8


class TaskResponse(BaseModel):
    """Task response model."""
    id: int
    status: str
    progress: int
    created_at: datetime
    result_path: Optional[str] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True


class ControlExtractionRequest(BaseModel):
    """Control image extraction request."""
    control_type: str  # canny, depth, pose, hed, mlsd
    low_threshold: int = 100  # for canny
    high_threshold: int = 200  # for canny


@router.post("/generate", response_model=TaskResponse)
async def generate_image(
    request: GenerationRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Queue an image generation task."""
    # Create task
    task = TaskQueue(
        user_id=current_user.id,
        status="pending",
        params=request.model_dump(),
    )
    
    db.add(task)
    await db.commit()
    await db.refresh(task)
    
    # Add task to generation queue for processing
    queue_manager = get_queue_manager()
    queued_task = QueuedTask(
        task_id=task.id,
        user_id=current_user.id,
        params=request.model_dump()
    )
    await queue_manager.add_task(queued_task)
    
    logger.info(f"Task {task.id} added to generation queue")
    
    return task


@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """Upload an image for use in generation."""
    import os
    import uuid
    from pathlib import Path
    
    # Create uploads directory
    upload_dir = Path("uploads") / str(current_user.id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    ext = Path(file.filename).suffix if file.filename else ".png"
    filename = f"{uuid.uuid4()}{ext}"
    filepath = upload_dir / filename
    
    # Save file
    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)
    
    return {
        "path": str(filepath),
        "filename": filename
    }


@router.post("/extract-control")
async def extract_control(
    control_type: str = Form(...),
    low_threshold: int = Form(100),
    high_threshold: int = Form(200),
    image: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """Extract control image from uploaded image."""
    import os
    import uuid
    import cv2
    import numpy as np
    from pathlib import Path
    from PIL import Image
    import io
    
    # Read image content
    content = await image.read()
    
    # Convert to numpy array
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file"
        )
    
    # Create output directory for control images
    control_dir = Path("uploads") / str(current_user.id) / "control"
    control_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = f"{uuid.uuid4()}_control_{control_type}.png"
    output_path = control_dir / output_filename
    
    # Process based on control type
    if control_type == "canny":
        # Canny edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
    elif control_type == "hed":
        # HED-like edge detection (simplified using bilateral filter + Canny)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(blurred, low_threshold // 2, high_threshold // 2)
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
    elif control_type == "depth":
        # Simple depth estimation using grayscale (placeholder)
        # Real depth would use ZoeDepth or MiDaS
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply some contrast enhancement to simulate depth
        result = cv2.equalizeHist(gray)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
    elif control_type == "mlsd":
        # Line segment detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
        result = np.zeros_like(img)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
    elif control_type == "pose":
        # Pose detection requires DWPose model - return placeholder
        # This would need actual pose detection model
        result = img.copy()
        cv2.putText(result, "Pose detection requires DWPose model", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown control type: {control_type}"
        )
    
    # Save result
    cv2.imwrite(str(output_path), result)
    
    return {
        "control_image_path": str(output_path),
        "control_type": control_type
    }


@router.get("/status/{task_id}", response_model=TaskResponse)
async def get_task_status(
    task_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get the status of a generation task."""
    result = await db.execute(
        select(TaskQueue).where(
            TaskQueue.id == task_id,
            TaskQueue.user_id == current_user.id
        )
    )
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    return task


@router.delete("/cancel/{task_id}")
async def cancel_task(
    task_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Cancel a pending generation task."""
    result = await db.execute(
        select(TaskQueue).where(
            TaskQueue.id == task_id,
            TaskQueue.user_id == current_user.id
        )
    )
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    if task.status not in ["pending", "running"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Task cannot be cancelled"
        )
    
    task.status = "cancelled"
    await db.commit()
    
    return {"message": "Task cancelled successfully"}


@router.get("/queue", response_model=List[TaskResponse])
async def get_queue(
    status_filter: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's task queue."""
    query = select(TaskQueue).where(TaskQueue.user_id == current_user.id)
    
    if status_filter:
        query = query.where(TaskQueue.status == status_filter)
    
    query = query.order_by(TaskQueue.created_at.desc())
    
    result = await db.execute(query)
    tasks = result.scalars().all()
    
    return tasks
