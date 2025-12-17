"""Image generation API endpoints."""
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import get_db, User, TaskQueue
from backend.api.auth import get_current_user

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
    
    # TODO: Add task to generation queue for processing
    
    return task


@router.post("/extract-control")
async def extract_control(
    request: ControlExtractionRequest,
    image: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """Extract control image from uploaded image."""
    # TODO: Implement control extraction
    return {
        "message": "Control extraction not yet implemented",
        "control_type": request.control_type
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
