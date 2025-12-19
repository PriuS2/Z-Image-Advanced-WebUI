"""Queue manager for batch image generation."""
import asyncio
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import TaskQueue, Image, PromptHistory
from backend.services.image_generator import get_generator, GenerationParams
from backend.websocket.handlers import (
    send_generation_progress,
    send_generation_complete,
    send_generation_error,
    send_preview_image
)


class TaskStatus(str, Enum):
    """Task status enum."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QueuedTask:
    """Represents a queued generation task."""
    task_id: int
    user_id: int
    params: Dict[str, Any]


class QueueManager:
    """Manager for the image generation queue."""
    
    def __init__(self):
        self._queue: asyncio.Queue[QueuedTask] = asyncio.Queue()
        self._current_task: Optional[QueuedTask] = None
        self._is_running = False
        self._cancel_requested = False
    
    async def add_task(self, task: QueuedTask):
        """Add a task to the queue."""
        await self._queue.put(task)
    
    async def cancel_task(self, task_id: int) -> bool:
        """Cancel a specific task.
        
        Returns True if the task was cancelled.
        """
        if self._current_task and self._current_task.task_id == task_id:
            self._cancel_requested = True
            return True
        
        # TODO: Remove from queue if pending
        return False
    
    async def start_worker(self, db: AsyncSession):
        """Start the queue worker."""
        if self._is_running:
            return
        
        self._is_running = True
        
        while self._is_running:
            try:
                # Wait for a task with timeout
                task = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0
                )
                
                self._current_task = task
                self._cancel_requested = False
                
                await self._process_task(task, db)
                
                self._current_task = None
                self._queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Queue worker error: {e}")
    
    async def stop_worker(self):
        """Stop the queue worker."""
        self._is_running = False
    
    async def _process_task(self, task: QueuedTask, db: AsyncSession):
        """Process a single task."""
        start_time = time.time()
        
        try:
            # Update task status to running
            await db.execute(
                update(TaskQueue)
                .where(TaskQueue.id == task.task_id)
                .values(status=TaskStatus.RUNNING, started_at=datetime.utcnow())
            )
            await db.commit()
            
            # Get generator
            generator = get_generator()
            
            # Load control image if path is provided
            control_image = None
            control_image_path = task.params.get("control_image_path")
            if control_image_path:
                try:
                    from PIL import Image as PILImage
                    control_image = PILImage.open(control_image_path).convert('RGB')
                except Exception as e:
                    print(f"Failed to load control image: {e}")
            
            # Load mask image if path is provided
            mask_image = None
            mask_image_path = task.params.get("mask_image_path")
            if mask_image_path:
                try:
                    from PIL import Image as PILImage
                    mask_image = PILImage.open(mask_image_path).convert('L')
                except Exception as e:
                    print(f"Failed to load mask image: {e}")
            
            # Load original image for inpainting if path is provided
            original_image = None
            original_image_path = task.params.get("original_image_path")
            if original_image_path:
                try:
                    from PIL import Image as PILImage
                    original_image = PILImage.open(original_image_path).convert('RGB')
                except Exception as e:
                    print(f"Failed to load original image: {e}")
            
            # Create generation params with control support
            params = GenerationParams(
                prompt=task.params.get("prompt", ""),
                width=task.params.get("width", 1024),
                height=task.params.get("height", 1024),
                num_inference_steps=task.params.get("num_inference_steps", 25),
                guidance_scale=task.params.get("guidance_scale", 0.0),
                control_context_scale=task.params.get("control_context_scale", 0.75),
                seed=task.params.get("seed"),
                sampler=task.params.get("sampler", "Flow"),
                control_type=task.params.get("control_type"),
                control_image=control_image,
                control_image_path=control_image_path,
                mask_image=mask_image,
                original_image=original_image,
            )
            
            # Progress callback
            async def progress_callback(step: int, total_steps: int, node_name: str):
                if self._cancel_requested:
                    raise asyncio.CancelledError("Task cancelled by user")
                
                progress = int((step / total_steps) * 100)
                elapsed = time.time() - start_time
                estimated_remaining = (elapsed / step) * (total_steps - step) if step > 0 else 0
                
                # Update database
                await db.execute(
                    update(TaskQueue)
                    .where(TaskQueue.id == task.task_id)
                    .values(progress=progress)
                )
                await db.commit()
                
                # Send WebSocket update
                await send_generation_progress(
                    user_id=task.user_id,
                    task_id=task.task_id,
                    progress=progress,
                    current_step=step,
                    total_steps=total_steps,
                    elapsed_time=elapsed,
                    estimated_remaining=estimated_remaining,
                    current_node=node_name
                )
            
            # Generate image
            image = await generator.generate(params, progress_callback)
            
            if image:
                # Save image
                image_path = generator.save_image(image)
                thumbnail_path = None
                
                # Create thumbnail
                thumbnail = generator.create_thumbnail(image)
                if thumbnail:
                    thumb_filename = f"thumb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    thumbnail_path = generator.save_image(thumbnail, thumb_filename)
                
                # Update task as completed
                await db.execute(
                    update(TaskQueue)
                    .where(TaskQueue.id == task.task_id)
                    .values(
                        status=TaskStatus.COMPLETED,
                        result_path=image_path,
                        progress=100,
                        completed_at=datetime.utcnow()
                    )
                )
                
                # Save to gallery
                gallery_image = Image(
                    user_id=task.user_id,
                    path=image_path,
                    thumbnail_path=thumbnail_path,
                    prompt=params.prompt,
                    params=task.params,
                    generation_info={
                        "prompts": {
                            "english": params.prompt,
                            "korean": task.params.get("prompt_ko"),
                        },
                        "params": task.params,
                        "generation_time_seconds": time.time() - start_time,
                    }
                )
                db.add(gallery_image)
                
                # Save to prompt history
                history = PromptHistory(
                    user_id=task.user_id,
                    prompt_ko=task.params.get("prompt_ko"),
                    prompt_en=params.prompt,
                )
                db.add(history)
                
                await db.commit()
                
                # Send completion notification
                await send_generation_complete(
                    user_id=task.user_id,
                    task_id=task.task_id,
                    image_path=image_path
                )
            
        except asyncio.CancelledError:
            # Task was cancelled
            await db.execute(
                update(TaskQueue)
                .where(TaskQueue.id == task.task_id)
                .values(
                    status=TaskStatus.CANCELLED,
                    error_message="Cancelled by user",
                    completed_at=datetime.utcnow()
                )
            )
            await db.commit()
            
        except Exception as e:
            # Task failed
            error_msg = str(e)
            await db.execute(
                update(TaskQueue)
                .where(TaskQueue.id == task.task_id)
                .values(
                    status=TaskStatus.FAILED,
                    error_message=error_msg,
                    completed_at=datetime.utcnow()
                )
            )
            await db.commit()
            
            await send_generation_error(
                user_id=task.user_id,
                task_id=task.task_id,
                error=error_msg
            )


# Global instance
_queue_manager: Optional[QueueManager] = None


def get_queue_manager() -> QueueManager:
    """Get the global queue manager instance."""
    global _queue_manager
    if _queue_manager is None:
        _queue_manager = QueueManager()
    return _queue_manager
