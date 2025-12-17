"""Workflow API endpoints."""
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import get_db, User, Workflow
from backend.api.auth import get_current_user

router = APIRouter()


class WorkflowCreate(BaseModel):
    """Create workflow request."""
    name: str
    description: Optional[str] = None
    nodes_data: Optional[dict] = None


class WorkflowUpdate(BaseModel):
    """Update workflow request."""
    name: Optional[str] = None
    description: Optional[str] = None
    nodes_data: Optional[dict] = None
    is_favorite: Optional[bool] = None


class WorkflowResponse(BaseModel):
    """Workflow response model."""
    id: int
    name: str
    description: Optional[str] = None
    thumbnail_path: Optional[str] = None
    nodes_data: Optional[dict] = None
    created_at: datetime
    updated_at: datetime
    is_favorite: bool

    class Config:
        from_attributes = True


@router.get("/", response_model=List[WorkflowResponse])
async def get_workflows(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's workflows."""
    result = await db.execute(
        select(Workflow)
        .where(Workflow.user_id == current_user.id)
        .order_by(desc(Workflow.updated_at))
    )
    workflows = result.scalars().all()
    return workflows


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific workflow."""
    result = await db.execute(
        select(Workflow).where(
            Workflow.id == workflow_id,
            Workflow.user_id == current_user.id
        )
    )
    workflow = result.scalar_one_or_none()
    
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found"
        )
    
    return workflow


@router.post("/", response_model=WorkflowResponse, status_code=status.HTTP_201_CREATED)
async def create_workflow(
    workflow_data: WorkflowCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new workflow."""
    workflow = Workflow(
        user_id=current_user.id,
        name=workflow_data.name,
        description=workflow_data.description,
        nodes_data=workflow_data.nodes_data or {},
    )
    
    db.add(workflow)
    await db.commit()
    await db.refresh(workflow)
    
    return workflow


@router.put("/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(
    workflow_id: int,
    workflow_data: WorkflowUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update a workflow."""
    result = await db.execute(
        select(Workflow).where(
            Workflow.id == workflow_id,
            Workflow.user_id == current_user.id
        )
    )
    workflow = result.scalar_one_or_none()
    
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found"
        )
    
    if workflow_data.name is not None:
        workflow.name = workflow_data.name
    if workflow_data.description is not None:
        workflow.description = workflow_data.description
    if workflow_data.nodes_data is not None:
        workflow.nodes_data = workflow_data.nodes_data
    if workflow_data.is_favorite is not None:
        workflow.is_favorite = workflow_data.is_favorite
    
    workflow.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(workflow)
    
    return workflow


@router.delete("/{workflow_id}")
async def delete_workflow(
    workflow_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a workflow."""
    result = await db.execute(
        select(Workflow).where(
            Workflow.id == workflow_id,
            Workflow.user_id == current_user.id
        )
    )
    workflow = result.scalar_one_or_none()
    
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found"
        )
    
    await db.delete(workflow)
    await db.commit()
    
    return {"message": "Workflow deleted successfully"}


@router.post("/{workflow_id}/duplicate", response_model=WorkflowResponse)
async def duplicate_workflow(
    workflow_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Duplicate a workflow."""
    result = await db.execute(
        select(Workflow).where(
            Workflow.id == workflow_id,
            Workflow.user_id == current_user.id
        )
    )
    workflow = result.scalar_one_or_none()
    
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found"
        )
    
    new_workflow = Workflow(
        user_id=current_user.id,
        name=f"{workflow.name} (Copy)",
        description=workflow.description,
        nodes_data=workflow.nodes_data,
    )
    
    db.add(new_workflow)
    await db.commit()
    await db.refresh(new_workflow)
    
    return new_workflow
