"""Z-Image Advanced WebUI - FastAPI Backend."""
import os
import sys
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.config import get_config
from backend.api import api_router
from backend.db import init_db, AsyncSessionLocal
from backend.websocket import websocket_endpoint
from backend.services.queue_manager import get_queue_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print("[START] Starting Z-Image Advanced WebUI...")
    
    # Initialize database
    await init_db()
    print("[OK] Database initialized")
    
    # Create output directories
    config = get_config()
    os.makedirs(config.models.outputs_path, exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("controls", exist_ok=True)
    os.makedirs("masks", exist_ok=True)
    print("[OK] Output directories created")
    
    # Start queue worker
    queue_manager = get_queue_manager()
    worker_task = asyncio.create_task(start_queue_worker(queue_manager))
    print("[OK] Queue worker started")
    
    yield
    
    # Shutdown
    print("[END] Shutting down Z-Image Advanced WebUI...")
    await queue_manager.stop_worker()
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


async def start_queue_worker(queue_manager):
    """Start the queue worker with a new database session."""
    async with AsyncSessionLocal() as db:
        await queue_manager.start_worker(db)


# Create FastAPI app
config = get_config()
app = FastAPI(
    title="Z-Image Advanced WebUI",
    description="AI Image Generation WebUI based on Z-Image-Turbo-Fun-Controlnet-Union-2.0",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api")

# WebSocket endpoint
@app.websocket("/ws/{user_id}")
async def websocket_route(websocket: WebSocket, user_id: int):
    """WebSocket endpoint for real-time updates."""
    await websocket_endpoint(websocket, user_id)

# Mount static files for outputs
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Z-Image Advanced WebUI",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.debug,
    )
