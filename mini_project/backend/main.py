from fastapi import FastAPI, HTTPException, Body
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from models import MachineState, WashConfig, AIInsight, Notification
from state_manager import StateManager
import socket
import requests
import asyncio

# --- Backend State Manager --- #
manager = StateManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initial broadcast
    manager._broadcast_state()
    
    # Startup: Auto-Discovery
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        url = "https://ai-washing-machine-default-rtdb.asia-southeast1.firebasedatabase.app/backend_ip.json"
        
        # Async to_thread to prevent blocking lifespan
        await asyncio.to_thread(requests.put, url, json=ip, timeout=5)
        print(f"Auto-Discovery: Published Laptop IP ({ip}) to Firebase")
    except Exception as e:
        print(f"Auto-Discovery failed: {e}")
    yield
    # Shutdown logic (if any) could go here

app = FastAPI(lifespan=lifespan)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Smart Washing Machine API is running"}

@app.get("/api/state", response_model=MachineState)
async def get_state():
    return manager.get_state()

@app.post("/api/config")
async def update_config(config: WashConfig):
    try:
        manager.update_config(config)
        return manager.get_state()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/start")
async def start_wash(params: Optional[dict] = Body(None)):
    manager.start_wash(params)
    return manager.get_state()

@app.post("/api/pause")
async def pause_wash():
    manager.pause_wash()
    return manager.get_state()

@app.post("/api/stop")
async def stop_wash():
    manager.stop_wash()
    return manager.get_state()

@app.get("/api/ai/insight", response_model=AIInsight)
async def get_ai_insight():
    return manager.get_ai_insight()

@app.get("/api/notifications", response_model=list[Notification])
async def get_notifications():
    return manager.get_notifications()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
