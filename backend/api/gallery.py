"""Gallery API endpoints."""
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import get_db, User, Image
from backend.api.auth import get_current_user

router = APIRouter()


class ImageResponse(BaseModel):
    """Image response model."""
    id: int
    path: str
    thumbnail_path: Optional[str] = None
    prompt: Optional[str] = None
    params: Optional[dict] = None
    generation_info: Optional[dict] = None
    created_at: datetime
    is_favorite: bool

    class Config:
        from_attributes = True


class ImageListResponse(BaseModel):
    """Paginated image list response."""
    images: List[ImageResponse]
    total: int
    page: int
    page_size: int


@router.get("/images", response_model=ImageListResponse)
async def get_images(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    favorite_only: bool = False,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's images with pagination."""
    # Build query
    query = select(Image).where(Image.user_id == current_user.id)
    
    if favorite_only:
        query = query.where(Image.is_favorite == True)
    
    # Get total count
    count_result = await db.execute(
        select(Image.id).where(Image.user_id == current_user.id)
    )
    total = len(count_result.all())
    
    # Apply pagination
    offset = (page - 1) * page_size
    query = query.order_by(desc(Image.created_at)).offset(offset).limit(page_size)
    
    result = await db.execute(query)
    images = result.scalars().all()
    
    return ImageListResponse(
        images=images,
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/images/{image_id}", response_model=ImageResponse)
async def get_image(
    image_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific image."""
    result = await db.execute(
        select(Image).where(
            Image.id == image_id,
            Image.user_id == current_user.id
        )
    )
    image = result.scalar_one_or_none()
    
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
    
    return image


@router.delete("/images/{image_id}")
async def delete_image(
    image_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete an image."""
    result = await db.execute(
        select(Image).where(
            Image.id == image_id,
            Image.user_id == current_user.id
        )
    )
    image = result.scalar_one_or_none()
    
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
    
    await db.delete(image)
    await db.commit()
    
    # TODO: Delete actual image files from disk
    
    return {"message": "Image deleted successfully"}


@router.post("/images/{image_id}/favorite")
async def toggle_favorite(
    image_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Toggle favorite status of an image."""
    result = await db.execute(
        select(Image).where(
            Image.id == image_id,
            Image.user_id == current_user.id
        )
    )
    image = result.scalar_one_or_none()
    
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
    
    image.is_favorite = not image.is_favorite
    await db.commit()
    
    return {"is_favorite": image.is_favorite}


@router.get("/images/{image_id}/download")
async def download_image(
    image_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Download an image file."""
    from fastapi.responses import FileResponse
    import os
    
    result = await db.execute(
        select(Image).where(
            Image.id == image_id,
            Image.user_id == current_user.id
        )
    )
    image = result.scalar_one_or_none()
    
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
    
    if not os.path.exists(image.path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image file not found"
        )
    
    return FileResponse(
        path=image.path,
        filename=f"image_{image_id}.png",
        media_type="image/png"
    )


@router.post("/images/{image_id}/regenerate")
async def regenerate_image(
    image_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Regenerate an image with the same parameters but new seed."""
    from backend.db import TaskQueue
    
    result = await db.execute(
        select(Image).where(
            Image.id == image_id,
            Image.user_id == current_user.id
        )
    )
    image = result.scalar_one_or_none()
    
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
    
    # Get original parameters
    params = image.params or {}
    params['seed'] = None  # New random seed
    
    # Create new task
    task = TaskQueue(
        user_id=current_user.id,
        status="pending",
        params=params,
    )
    
    db.add(task)
    await db.commit()
    await db.refresh(task)
    
    return {"task_id": task.id}


@router.get("/images/{image_id}/settings")
async def get_image_settings(
    image_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get generation settings used for an image."""
    result = await db.execute(
        select(Image).where(
            Image.id == image_id,
            Image.user_id == current_user.id
        )
    )
    image = result.scalar_one_or_none()
    
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
    
    return image.params or {}
