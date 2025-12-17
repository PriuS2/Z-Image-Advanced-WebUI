"""WebSocket handlers for real-time updates."""
import asyncio
import json
from typing import Dict, Set
from fastapi import WebSocket, WebSocketDisconnect

# Store active connections per user
active_connections: Dict[int, Set[WebSocket]] = {}


class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[int, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: int):
        """Accept and store a new connection."""
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        self.active_connections[user_id].add(websocket)
    
    def disconnect(self, websocket: WebSocket, user_id: int):
        """Remove a connection."""
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
    
    async def send_personal_message(self, message: dict, user_id: int):
        """Send a message to all connections of a specific user."""
        if user_id in self.active_connections:
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass  # Connection might be closed
    
    async def broadcast(self, message: dict):
        """Broadcast a message to all connected users."""
        for user_id, connections in self.active_connections.items():
            for connection in connections:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass


# Global connection manager
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket, user_id: int = 0):
    """WebSocket endpoint for real-time updates.
    
    Message types:
    - generation_progress: Progress updates during image generation
    - generation_complete: Generation completed
    - generation_error: Generation failed
    - preview_image: Step-by-step preview image (base64)
    - node_status: Current active node in the flow
    """
    await manager.connect(websocket, user_id)
    
    try:
        while True:
            # Receive messages from client (e.g., ping, cancel requests)
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get("type", "")
                
                if message_type == "ping":
                    await websocket.send_json({"type": "pong"})
                
                elif message_type == "cancel":
                    task_id = message.get("task_id")
                    # TODO: Handle task cancellation
                    await websocket.send_json({
                        "type": "cancel_ack",
                        "task_id": task_id
                    })
                
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON"
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)


async def send_generation_progress(
    user_id: int,
    task_id: int,
    progress: int,
    current_step: int,
    total_steps: int,
    elapsed_time: float,
    estimated_remaining: float,
    current_node: str = "generate"
):
    """Send generation progress update to user."""
    await manager.send_personal_message({
        "type": "generation_progress",
        "task_id": task_id,
        "progress": progress,
        "current_step": current_step,
        "total_steps": total_steps,
        "elapsed_time": elapsed_time,
        "estimated_remaining": estimated_remaining,
        "current_node": current_node
    }, user_id)


async def send_preview_image(user_id: int, task_id: int, image_base64: str):
    """Send preview image during generation."""
    await manager.send_personal_message({
        "type": "preview_image",
        "task_id": task_id,
        "image": image_base64
    }, user_id)


async def send_generation_complete(user_id: int, task_id: int, image_path: str):
    """Send generation complete notification."""
    await manager.send_personal_message({
        "type": "generation_complete",
        "task_id": task_id,
        "image_path": image_path
    }, user_id)


async def send_generation_error(user_id: int, task_id: int, error: str):
    """Send generation error notification."""
    await manager.send_personal_message({
        "type": "generation_error",
        "task_id": task_id,
        "error": error
    }, user_id)
