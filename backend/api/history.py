"""Prompt history API endpoints."""
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import get_db, User, PromptHistory
from backend.api.auth import get_current_user

router = APIRouter()


class PromptHistoryResponse(BaseModel):
    """Prompt history response model."""
    id: int
    prompt_ko: Optional[str] = None
    prompt_en: Optional[str] = None
    prompt_enhanced: Optional[str] = None
    created_at: datetime
    is_favorite: bool

    class Config:
        from_attributes = True


class PromptHistoryCreate(BaseModel):
    """Create prompt history request."""
    prompt_ko: Optional[str] = None
    prompt_en: Optional[str] = None
    prompt_enhanced: Optional[str] = None


@router.get("/", response_model=List[PromptHistoryResponse])
async def get_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    favorite_only: bool = False,
    search: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's prompt history."""
    query = select(PromptHistory).where(PromptHistory.user_id == current_user.id)
    
    if favorite_only:
        query = query.where(PromptHistory.is_favorite == True)
    
    if search:
        search_filter = f"%{search}%"
        query = query.where(
            (PromptHistory.prompt_ko.ilike(search_filter)) |
            (PromptHistory.prompt_en.ilike(search_filter)) |
            (PromptHistory.prompt_enhanced.ilike(search_filter))
        )
    
    offset = (page - 1) * page_size
    query = query.order_by(desc(PromptHistory.created_at)).offset(offset).limit(page_size)
    
    result = await db.execute(query)
    history = result.scalars().all()
    
    return history


@router.post("/", response_model=PromptHistoryResponse)
async def create_history(
    prompt_data: PromptHistoryCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new prompt history entry."""
    history = PromptHistory(
        user_id=current_user.id,
        prompt_ko=prompt_data.prompt_ko,
        prompt_en=prompt_data.prompt_en,
        prompt_enhanced=prompt_data.prompt_enhanced,
    )
    
    db.add(history)
    await db.commit()
    await db.refresh(history)
    
    return history


@router.delete("/{history_id}")
async def delete_history(
    history_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a prompt history entry."""
    result = await db.execute(
        select(PromptHistory).where(
            PromptHistory.id == history_id,
            PromptHistory.user_id == current_user.id
        )
    )
    history = result.scalar_one_or_none()
    
    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="History entry not found"
        )
    
    await db.delete(history)
    await db.commit()
    
    return {"message": "History entry deleted"}


@router.post("/{history_id}/favorite")
async def toggle_favorite(
    history_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Toggle favorite status of a history entry."""
    result = await db.execute(
        select(PromptHistory).where(
            PromptHistory.id == history_id,
            PromptHistory.user_id == current_user.id
        )
    )
    history = result.scalar_one_or_none()
    
    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="History entry not found"
        )
    
    history.is_favorite = not history.is_favorite
    await db.commit()
    
    return {"is_favorite": history.is_favorite}
