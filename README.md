# 🧠 AI.BPI — AI-Powered Business Process Intelligence

A full-stack **Digital Twin** platform that combines **Process Mining**, **Graph Neural Networks (GNN)**, **Reinforcement Learning (RL)**, and a **Gemini-powered AI Chatbot** to analyze, simulate, and optimize business processes.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────┐
│                   React Frontend                 │
│  (Process Topology, Workforce Allocation,        │
│   Telemetry Dashboard, AI Chatbot)               │
├──────────────────────────────────────────────────┤
│              FastAPI Backend (server.py)          │
│  REST: /api/topology, /api/telemetry,            │
│        /api/simulate, /api/optimize              │
│  WebSocket: /ws/chat                             │
├──────────────────────────────────────────────────┤
│                  AI / ML Layer                   │
│  GNN (GAT): Bottleneck Prediction                │
│  RL (PPO):  Resource Optimization                │
│  Gemini:    Natural Language Analysis            │
├──────────────────────────────────────────────────┤
│              Digital Twin (SimPy)                 │
│  Discrete Event Simulation of SAP Processes      │
└──────────────────────────────────────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---|---|
| **Process Topology** | Interactive graph visualization (React Flow) of the procure-to-pay process with animated edges and bottleneck highlighting. |
| **Workforce Allocation** | Drag-and-drop employee assignment with real-time Digital Twin simulation. |
| **Live Telemetry** | Bar charts comparing Baseline vs. Optimized KPIs (Cycle Time, Throughput, OpEx). |
| **Auto-Optimize** | One-click GNN+RL agent training to discover optimal resource allocation strategies. |
| **AI Chatbot** | Gemini 2.5 Flash-powered assistant with RAG context from process stats, bottleneck reports, and optimization results. |
| **Toast Notifications** | Real-time feedback for all async operations (optimization, simulation). |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **Google Gemini API Key** (optional, for chatbot)

### 1. Clone & Setup

```bash
git clone https://github.com/Moksh-189/AI-Based-Business-Process-Model.git
cd AI-Based-Business-Process-Model

# Python environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev    # Starts Vite dev server on http://localhost:5173
```

### 3. Backend Setup

```bash
# From project root
# Optional: Set Gemini API key for chatbot
set GEMINI_API_KEY=your_key_here       # Windows
# export GEMINI_API_KEY=your_key_here  # Linux/Mac

python server.py    # Starts FastAPI on http://localhost:8000
```

### 4. Open App

Navigate to **http://localhost:5173** (dev mode).

---

## 🐳 Docker

```bash
# Build
docker build -t ai-bpi .

# Run
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key ai-bpi

# Open http://localhost:8000
```

> **Note:** Large data files (`.csv`, `.xes`) are excluded from the Docker image. Mount them as volumes if needed for full process mining.

---

## 📁 Project Structure

```
├── server.py                # FastAPI backend (REST + WebSocket + static serving)
├── chatbot.py               # Gemini AI chatbot with RAG context
├── digital_twin.py          # SimPy discrete event simulation
├── gnn_model.py             # Graph Attention Network (GAT + GLU)
├── gnn_env.py               # Gymnasium RL environment with GNN embeddings
├── train_gnn_agent.py       # PPO agent training (GLU vs GELU comparison)
├── process_mining.py        # SAP XES/CSV process mining & analysis
├── graph_builder.py         # PyTorch Geometric graph construction
├── worker_data.py           # Employee/resource data generation
├── requirements.txt         # Python dependencies
├── Dockerfile               # Multi-stage Docker build
├── frontend/                # React + TypeScript + Vite
│   ├── src/
│   │   ├── components/      # ProcessTopology, TelemetryPanel, WorkforceAllocation, etc.
│   │   ├── context/         # ToastContext (global notifications)
│   │   ├── layouts/         # Layout with Sidebar + FloatingChatbot
│   │   └── pages/           # Home, Topology
│   └── tests/               # Playwright E2E tests
└── *.json                   # Process data (stats, bottlenecks, DFG, topology)
```

---

## 🧪 Testing

```bash
# Backend integration tests
cd tests
python -m pytest

# Frontend E2E tests (requires both servers running)
cd frontend
npx playwright test
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
