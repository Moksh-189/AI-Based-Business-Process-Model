from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json
import os
import subprocess
import asyncio
from typing import List
from pathlib import Path

# --- Chatbot & Context ---
from chatbot import ProcessChatbot

app = FastAPI(title="AI.BPI - Business Process Intelligence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global State ---
optimization_state = {
    "cycle_time_red": 0,
    "throughput_inc": 0,
    "opex_red": 0,
    "is_training": False
}

# --- WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()
chatbot = None

@app.on_event("startup")
async def startup_event():
    global chatbot
    if os.getenv("GEMINI_API_KEY"):
        try:
            chatbot = ProcessChatbot()
            print("Chatbot initialized.")
        except Exception as e:
            print(f"Chatbot init failed: {e}")

# --- Endpoints ---

@app.get("/api/topology")
async def get_topology():
    """Returns static topology (simulating DB fetch)."""
    # ... (existing logic, simplified for brevity here, assume existing structure)
    # Re-using the structure from before but inline for clarity or importing if needed.
    # For now, keeping the manual structure to ensure it works.
    return {
            "nodes": [
                {"id": "1", "type": "input", "data": {"label": "Start: Create Purchase Req"}, "position": {"x": 250, "y": 0}, "style": {"background": "#1e293b", "color": "#fff", "border": "1px solid #06b6d4", "width": 180}},
                {"id": "2", "data": {"label": "Approve Requisition"}, "position": {"x": 250, "y": 100}, "style": {"background": "#1e293b", "color": "#fff", "border": "1px solid #fff"}},
                {"id": "3", "data": {"label": "Create PO"}, "position": {"x": 250, "y": 200}, "style": {"background": "#1e293b", "color": "#fff", "border": "1px solid #fff"}},
                {"id": "4", "data": {"label": "Receive Goods"}, "position": {"x": 100, "y": 300}, "style": {"background": "#1e293b", "color": "#fff", "border": "1px solid #fff"}},
                {"id": "5", "data": {"label": "Scan Invoice"}, "position": {"x": 400, "y": 300}, "style": {"background": "#1e293b", "color": "#fff", "border": "1px solid #fff"}},
                {"id": "6", "data": {"label": "Clear Invoice (BOTTLENECK: 43.7d)"}, "position": {"x": 250, "y": 400}, "style": {"background": "#3f1a1a", "color": "#f87171", "border": "2px solid #ef4444", "width": 200, "boxShadow": "0 0 15px rgba(239, 68, 68, 0.4)"}, "className": "animate-pulse"},
                {"id": "7", "type": "output", "data": {"label": "End: Pay Vendor"}, "position": {"x": 250, "y": 500}, "style": {"background": "#1e293b", "color": "#fff", "border": "1px solid #10b981"}}
            ],
            "edges": [
                {"id": "e1-2", "source": "1", "target": "2", "animated": True, "style": {"stroke": "#64748b"}},
                {"id": "e2-3", "source": "2", "target": "3", "animated": True, "style": {"stroke": "#64748b"}},
                {"id": "e3-4", "source": "3", "target": "4", "animated": True, "style": {"stroke": "#64748b"}},
                {"id": "e3-5", "source": "3", "target": "5", "animated": True, "style": {"stroke": "#64748b"}},
                {"id": "e4-6", "source": "4", "target": "6", "animated": True, "style": {"stroke": "#ef4444", "strokeWidth": 2}},
                {"id": "e5-6", "source": "5", "target": "6", "animated": True, "style": {"stroke": "#ef4444", "strokeWidth": 2}},
                {"id": "e6-7", "source": "6", "target": "7", "animated": True, "style": {"stroke": "#10b981"}}
            ]
        }

@app.get("/api/telemetry")
async def get_telemetry():
    global optimization_state
    # Baseline
    b_cycle, b_thru, b_opex = 45, 120, 85
    
    return [
        {"name": "Cycle Time (Days)", "Baseline": b_cycle, "Optimized": round(b_cycle - optimization_state["cycle_time_red"], 1)},
        {"name": "Throughput (/mo)", "Baseline": b_thru, "Optimized": round(b_thru + optimization_state["throughput_inc"], 1)},
        {"name": "OpEx ($k/mo)", "Baseline": b_opex, "Optimized": round(b_opex - optimization_state["opex_red"], 1)}
    ]

class Employee(BaseModel):
    id: str
    name: str
    role: str
    efficiency: int

class OptimizeRequest(BaseModel):
    assigned: List[Employee]

@app.post("/api/simulate")
async def simulate_optimization(request: OptimizeRequest):
    """Simulate Digital Twin results based on workforce."""
    global optimization_state
    
    total_eff = sum(e.efficiency for e in request.assigned)
    role_bonus = sum(1.5 if "Approver" in e.role else (1.2 if "Analyst" in e.role else 0) for e in request.assigned)
    
    impact = (total_eff * (1 + role_bonus)) / 1000.0
    
    optimization_state["cycle_time_red"] = min(25, 45 * impact)
    optimization_state["throughput_inc"] = min(100, 120 * impact)
    optimization_state["opex_red"] = min(30, 85 * (impact * 0.5))
    
    return {"status": "simulated", "state": optimization_state}

@app.post("/api/optimize")
async def trigger_training():
    """Trigger the actual RL training script."""
    global optimization_state
    if optimization_state["is_training"]:
        return {"status": "already_training"}
        
    try:
        # Note: In production, use Celery/Redis. Here, subprocess is okay for demo.
        subprocess.Popen(["python", "train_gnn_agent.py"])
        optimization_state["is_training"] = True
        return {"status": "started"}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Error processing chat request.")

@app.post("/api/chat/reload")
async def reload_chat_context():
    """Reloads the chatbot context to include new optimization results."""
    global chatbot
    if chatbot:
        chatbot.reload_context()
        return {"status": "reloaded"}
    else:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

@app.websocket("/ws/chat")
async def chat_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            
            # 1. Check Chatbot availability
            global chatbot
            if not chatbot:
                 if os.getenv("GEMINI_API_KEY"):
                     chatbot = ProcessChatbot()
                 else:
                     await manager.send_personal_message("System: API Key missing. Chat disabled.", websocket)
                     continue

            # 2. Get Response
            response = chatbot.ask(data)
            await manager.send_personal_message(response, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# --- Static File Serving (Production / Docker) ---
FRONTEND_DIR = Path(__file__).parent / "frontend" / "dist"

if FRONTEND_DIR.exists():
    # Serve static assets (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Catch-all: serve index.html for SPA client-side routing."""
        file_path = FRONTEND_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_DIR / "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
