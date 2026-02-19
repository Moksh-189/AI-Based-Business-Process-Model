from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json
import os
import subprocess
import asyncio
import threading
from typing import List, Optional
from pathlib import Path

# --- Chatbot & Context ---
# --- Chatbot & Context ---
from chatbot import ProcessChatbot

# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Load from environment variable for security


app = FastAPI(title="AI.BPI - Business Process Intelligence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "https://bizopt.netlify.app", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
@app.head("/")
async def root():
    return {"status": "ok", "message": "AI.BPI Backend Running"}

# --- Global State ---
optimization_state = {
    "cycle_time_red": 0,
    "throughput_inc": 0,
    "opex_red": 0,
    "is_training": False
}

# Training progress shared state
training_progress = {
    "active": False,
    "messages": [],   # list of JSON-serializable dicts
    "complete": False,
    "results": None,
}

# --- WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

# Separate manager for training WebSocket
class TrainingWSManager:
    def __init__(self):
        self.connections: List[WebSocket] = []
    
    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)
    
    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)
    
    async def broadcast(self, message: dict):
        dead = []
        for ws in self.connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

training_manager = TrainingWSManager()

chatbot = None

@app.on_event("startup")
async def startup_event():
    global chatbot
    # Priority: Hardcoded Key > Environment Variable
    api_key = GEMINI_API_KEY if GEMINI_API_KEY and not GEMINI_API_KEY.startswith("PASTE_") else os.getenv("GEMINI_API_KEY")
    
    if api_key:
        try:
            chatbot = ProcessChatbot(api_key=api_key)
            print("Chatbot initialized with API Key.")
        except Exception as e:
            print(f"Chatbot init failed: {e}")

# --- Endpoints ---

@app.get("/api/topology")
async def get_topology():
    """Returns process topology built from real process mining data."""
    print("API: /api/topology called")
    try:
        with open('bottleneck_report.json', 'r') as f:
            bottleneck_data = json.load(f)
        with open('dfg_data.json', 'r') as f:
            dfg_data = json.load(f)
    except FileNotFoundError:
        return {"nodes": [], "edges": []}
    
    activities = bottleneck_data.get('bottlenecks', [])
    dfg_edges = dfg_data.get('edges', [])
    
    # Take top 8 activities by frequency for a clean graph
    top_activities = sorted(activities, key=lambda a: a['frequency'], reverse=True)[:8]
    activity_names = {a['activity'] for a in top_activities}
    
    # Build nodes with a clean vertical flow layout
    nodes = []
    BOTTLENECK_THRESHOLD = 0.5
    CENTER_X = 300
    
    for i, act in enumerate(top_activities):
        is_bn = act.get('bottleneck_score', 0) >= BOTTLENECK_THRESHOLD
        avg_dur = round(act['avg_duration_hours'], 1)
        
        # Vertical flow: slight left/right offset for visual interest
        offset = -120 if i % 2 == 0 else 120
        x = CENTER_X + offset
        y = i * 110
        
        # Styling
        if is_bn:
            style = {
                "background": "#3f1a1a", "color": "#f87171",
                "border": "2px solid #ef4444", "width": 220,
                "boxShadow": "0 0 15px rgba(239, 68, 68, 0.4)"
            }
        elif i == 0:
            style = {"background": "#1e293b", "color": "#fff", "border": "1px solid #06b6d4", "width": 220}
        elif i == len(top_activities) - 1:
            style = {"background": "#1e293b", "color": "#fff", "border": "1px solid #10b981", "width": 220}
        else:
            style = {"background": "#1e293b", "color": "#fff", "border": "1px solid #64748b", "width": 220}
        
        node = {
            "id": str(i + 1),
            "data": {
                "label": act['activity'],
                "isBottleneck": is_bn,
                "avgDuration": avg_dur,
                "frequency": act['frequency'],
                "bottleneckScore": round(act.get('bottleneck_score', 0), 3),
            },
            "position": {"x": x, "y": y},
            "style": style,
        }
        if i == 0:
            node["type"] = "input"
        elif i == len(top_activities) - 1:
            node["type"] = "output"
        if is_bn:
            node["className"] = "animate-pulse"
        
        nodes.append(node)
    
    # Build edges: only top 12 edges by frequency between selected activities
    name_to_id = {act['activity']: str(i + 1) for i, act in enumerate(top_activities)}
    candidate_edges = []
    seen = set()
    for edge in dfg_edges:
        src, tgt = edge['source'], edge['target']
        if src in name_to_id and tgt in name_to_id:
            pair = (name_to_id[src], name_to_id[tgt])
            if pair not in seen and pair[0] != pair[1]:
                seen.add(pair)
                candidate_edges.append({**edge, "_src_id": pair[0], "_tgt_id": pair[1]})
    
    # Sort by frequency and take top 12
    candidate_edges.sort(key=lambda e: e.get('frequency', 0), reverse=True)
    top_edges = candidate_edges[:12]
    
    max_freq = max((e.get('frequency', 1) for e in top_edges), default=1)
    edges = []
    for edge in top_edges:
        freq = edge.get('frequency', 0)
        is_heavy = freq > max_freq * 0.5
        edges.append({
            "id": f"e{edge['_src_id']}-{edge['_tgt_id']}",
            "source": edge['_src_id'], "target": edge['_tgt_id'],
            "animated": True,
            "label": f"{freq:,}" if is_heavy else "",
            "style": {
                "stroke": "#ef4444" if is_heavy else "#64748b",
                "strokeWidth": 2 if is_heavy else 1,
            }
        })
    
    print(f"API: Returning {len(nodes)} nodes, {len(edges)} edges")
    return {"nodes": nodes, "edges": edges}

@app.get("/api/telemetry")
async def get_telemetry():
    global optimization_state
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

class SimulateRequest(BaseModel):
    assigned: List[Employee]

class SuggestRequest(BaseModel):
    process_id: str
    process_label: str
    assigned: List[Employee]

@app.post("/api/simulate")
async def simulate_optimization(request: SimulateRequest):
    """Simulate Digital Twin results based on workforce."""
    global optimization_state
    
    total_eff = sum(e.efficiency for e in request.assigned)
    role_bonus = sum(1.5 if "Approver" in e.role else (1.2 if "Analyst" in e.role else 0) for e in request.assigned)
    
    impact = (total_eff * (1 + role_bonus)) / 1000.0
    
    optimization_state["cycle_time_red"] = min(25, 45 * impact)
    optimization_state["throughput_inc"] = min(100, 120 * impact)
    optimization_state["opex_red"] = min(30, 85 * (impact * 0.5))
    
    return {"status": "simulated", "state": optimization_state}

@app.post("/api/suggest")
async def get_ai_suggestion(request: SuggestRequest):
    """Simulate + get AI suggestion for employee assignment to a process."""
    global optimization_state, chatbot
    
    # Run simulation
    total_eff = sum(e.efficiency for e in request.assigned)
    num_assigned = len(request.assigned)
    role_bonus = sum(
        1.5 if "Approver" in e.role else (1.2 if "Analyst" in e.role else (1.0 if "Engineer" in e.role else 0.5))
        for e in request.assigned
    )
    
    impact = (total_eff * (1 + role_bonus)) / 1000.0
    
    # Calculate simulation results for this specific process
    is_bottleneck = request.process_label.lower() in ["clear invoice"]
    base_duration = 43.7 if is_bottleneck else 10.0
    
    cycle_reduction = min(35, base_duration * impact * (1.5 if is_bottleneck else 0.8))
    throughput_gain = min(80, 120 * impact * 0.6)
    opex_change = total_eff * num_assigned * 0.15  # cost of additional staff
    
    simulation = {
        "cycle_time_before": base_duration,
        "cycle_time_after": round(max(base_duration - cycle_reduction, base_duration * 0.3), 1),
        "cycle_reduction_pct": round((cycle_reduction / base_duration) * 100, 1),
        "throughput_gain_pct": round(throughput_gain, 1),
        "opex_increase": round(opex_change, 1),
        "is_bottleneck": is_bottleneck,
        "impact_score": round(impact * 100, 1),
    }
    
    # Update global state too
    optimization_state["cycle_time_red"] = min(25, 45 * impact)
    optimization_state["throughput_inc"] = min(100, 120 * impact)
    optimization_state["opex_red"] = min(30, 85 * (impact * 0.5))
    
    # Get AI suggestion from Gemini
    ai_suggestion = None
    if chatbot:
        try:
            employee_desc = ", ".join([f"{e.name} ({e.role}, {e.efficiency}% efficiency)" for e in request.assigned])
            prompt = (
                f"An operator assigned {employee_desc} to the '{request.process_label}' process step. "
                f"{'This is a critical BOTTLENECK with {:.1f} day avg duration. '.format(base_duration) if is_bottleneck else ''}"
                f"The Digital Twin simulation predicts: cycle time reduction of {simulation['cycle_reduction_pct']:.0f}%, "
                f"throughput gain of {simulation['throughput_gain_pct']:.0f}%. "
                f"Give a concise 3-4 sentence analysis of this assignment. "
                f"Cite the projected improvements (Cycle Time -{simulation['cycle_reduction_pct']:.0f}%, Throughput +{simulation['throughput_gain_pct']:.0f}%) "
                f"and briefly explain why this employee fits."
            )
            ai_suggestion = chatbot.ask(prompt)
        except Exception as e:
            print(f"AI suggestion error: {e}")
            ai_suggestion = None
    
    if not ai_suggestion:
        # Provide a rule-based fallback suggestion
        if is_bottleneck and total_eff > 85:
            ai_suggestion = (
                f"Good assignment. Deploying {num_assigned} resource(s) with avg {total_eff//num_assigned}% efficiency "
                f"to the bottleneck '{request.process_label}' should reduce cycle time by ~{simulation['cycle_reduction_pct']:.0f}%. "
                f"Consider adding a senior analyst for maximum impact on invoice clearance throughput."
            )
        elif is_bottleneck:
            ai_suggestion = (
                f"This bottleneck needs high-efficiency resources. Current assignment has avg "
                f"{total_eff//max(num_assigned,1)}% efficiency. Consider swapping in senior staff to clear the "
                f"'{request.process_label}' backlog faster."
            )
        else:
            ai_suggestion = (
                f"Assignment to '{request.process_label}' looks reasonable. Predicted {simulation['cycle_reduction_pct']:.0f}% "
                f"cycle time improvement. Focus high-efficiency resources on bottleneck processes for maximum ROI."
            )
    
    return {
        "simulation": simulation,
        "ai_suggestion": ai_suggestion,
    }


def _run_training_thread(progress_file: str):
    """Run training in a background thread, writing progress to a file."""
    global training_progress
    training_progress["active"] = True
    training_progress["complete"] = False
    training_progress["messages"] = []
    training_progress["results"] = None
    
    try:
        proc = subprocess.Popen(
            ["python", "train_gnn_agent.py", "--progress-file", progress_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            
            msg = {"type": "log", "text": line}
            
            # Parse progress from PPO output or custom markers
            if line.startswith("PROGRESS:"):
                try:
                    parts = line.split("PROGRESS:")[1].strip()
                    data = json.loads(parts)
                    msg = {"type": "progress", **data}
                except Exception:
                    pass
            elif "total_timesteps" in line.lower() or "| time/" in line:
                msg["type"] = "training_log"
            
            training_progress["messages"].append(msg)
        
        proc.wait()
        
        # Read final results if progress file exists
        results = None
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    results = json.load(f)
            except Exception:
                pass
        
        training_progress["results"] = results
        training_progress["complete"] = True
        training_progress["messages"].append({
            "type": "complete",
            "text": "Training complete!",
            "results": results
        })
        
        # === UPDATE TELEMETRY based on training results ===
        if results:
            # Use the agent's performance to compute improvements
            avg_reward = results.get('gelu_results', {}).get('avg_reward', 0)
            random_reward = results.get('random_baseline', {}).get('avg_reward', -10)
            
            # Calculate improvement ratio (how much better than random)
            if random_reward != 0:
                improvement = max(0, (avg_reward - random_reward) / abs(random_reward))
            else:
                improvement = max(0, avg_reward / 10)
            
            # Clamp to reasonable range (0 to 1)
            improvement = min(improvement, 1.0)
            
            # Apply improvements to telemetry
            optimization_state["cycle_time_red"] = round(improvement * 20, 1)      # Up to 20 days reduction
            optimization_state["throughput_inc"] = round(improvement * 60, 1)       # Up to +60/mo
            optimization_state["opex_red"] = round(improvement * 15, 1)             # Up to $15k/mo reduction
            
            print(f"[OK] Telemetry updated: cycle-{optimization_state['cycle_time_red']}d, "
                  f"thru+{optimization_state['throughput_inc']}/mo, "
                  f"opex-{optimization_state['opex_red']}k", flush=True)
        
        # Reload chatbot context so it picks up latest agent_comparison.json
        if chatbot:
            try:
                chatbot.reload_context()
                print("[OK] Chatbot context reloaded with new training results.", flush=True)
            except Exception as e:
                print(f"[WARN] Chatbot reload failed: {e}", flush=True)
        
    except Exception as e:
        training_progress["messages"].append({
            "type": "error",
            "text": f"Training error: {str(e)}"
        })
        training_progress["complete"] = True
    finally:
        training_progress["active"] = False
        optimization_state["is_training"] = False


@app.post("/api/optimize")
async def trigger_training():
    """Trigger the RL training and enable WebSocket progress streaming."""
    global optimization_state, training_progress
    
    if optimization_state["is_training"]:
        return {"status": "already_training"}
    
    progress_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_progress.json")
    
    # Clean up old progress
    if os.path.exists(progress_file):
        os.remove(progress_file)
    
    training_progress["active"] = True
    training_progress["complete"] = False
    training_progress["messages"] = []
    training_progress["results"] = None
    optimization_state["is_training"] = True
    
    # Start training in background thread
    thread = threading.Thread(target=_run_training_thread, args=(progress_file,), daemon=True)
    thread.start()
    
    return {"status": "started"}


@app.websocket("/ws/training")
async def training_ws(websocket: WebSocket):
    """WebSocket endpoint for streaming training progress to the frontend."""
    await training_manager.connect(websocket)
    last_idx = 0
    
    try:
        while True:
            # Send any new messages
            messages = training_progress["messages"]
            if last_idx < len(messages):
                for msg in messages[last_idx:]:
                    await websocket.send_json(msg)
                last_idx = len(messages)
            
            # If training is complete and all messages sent, send final and close
            if training_progress["complete"] and last_idx >= len(training_progress["messages"]):
                break
            
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    finally:
        training_manager.disconnect(websocket)


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
            
            global chatbot
            if not chatbot:
                # Check if the message looks like an API key (starts with AIza for Gemini usually, or just assume it is if we are waiting)
                if data.startswith("AIza") or len(data) > 20: 
                    try:
                        chatbot = ProcessChatbot(api_key=data)
                        await manager.send_personal_message("✅ API Key accepted. Chatbot initialized.", websocket)
                        continue
                    except Exception as e:
                        await manager.send_personal_message(f"❌ Invalid API Key: {str(e)}", websocket)
                        continue
                
                # If no chatbot and not a key, check if we have a hardcoded key we missed
                api_key = GEMINI_API_KEY if GEMINI_API_KEY and not GEMINI_API_KEY.startswith("PASTE_") else os.getenv("GEMINI_API_KEY")
                
                if api_key:
                    try:
                        chatbot = ProcessChatbot(api_key=api_key)
                        await manager.send_personal_message("✅ Chatbot ready.", websocket)
                        # Process the pending message as a query
                        response = chatbot.ask(data)
                        await manager.send_personal_message(response, websocket)
                        continue
                    except Exception as e:
                        await manager.send_personal_message(f"❌ Error initializing: {str(e)}", websocket)
                        continue
                else:
                    await manager.send_personal_message("⚠️ API Key needed. Please paste it in server.py or here:", websocket)
                    continue

            response = chatbot.ask(data)
            await manager.send_personal_message(response, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# --- Static File Serving (Production / Docker) ---
FRONTEND_DIR = Path(__file__).parent / "frontend" / "dist"

if FRONTEND_DIR.exists():
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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
