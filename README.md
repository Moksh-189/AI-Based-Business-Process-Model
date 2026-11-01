# AI-Based Business Process Intelligence

> A prescriptive optimization engine that combines **Graph Neural Networks (GAT+GLU)**, **Reinforcement Learning (PPO)**, and **Digital Twin simulation** to autonomously detect and resolve bottlenecks in enterprise workflows — going beyond prediction to real-time intervention.

---

## Problem Statement

Modern enterprises suffer from **"Operational Blindness"** caused by fragmented data ecosystems. Financial transactions live in SAP, engineering workflows in Jira, and communications in Teams — all semantically disconnected. This fragmentation creates three critical failures:

1. **Flat Event Logs Lose Graph Topology** — Traditional process mining tools (PM4Py, Celonis) flatten rich many-to-many relationships into linear sequences, obscuring the root causes of delays and revenue leakage.

2. **Existing AI in BPM Is Purely Predictive** — State-of-the-art approaches (LSTMs, basic GNNs) can forecast that a bottleneck *will* occur, but cannot autonomously intervene to *prevent* it.

3. **Static Heuristics Can't Handle Complexity** — Rule-based systems (FIFO, priority queues) fail when process topology is dynamic and interdependent.

This project builds a **Prescriptive Intelligence Layer** that unifies cross-departmental data into a process graph, learns its structural dynamics via GNNs, and deploys an RL agent to make real-time optimization decisions.

---

## Why This Is Better Than Existing Solutions

| Dimension | Existing Approaches | This Project |
|-----------|-------------------|--------------|
| **State Representation** | Flat tabular vectors (Silvander 2019, Żbikowski 2021) — lose structural context | **Graph Neural Network** encodes full process topology; the RL agent "sees" bottleneck dependencies |
| **Prediction vs. Action** | GNN-BPM models (Venugopal 2023, Lischka 2025) only forecast delays | **Closes the loop** — GNN predictions feed directly into a PPO agent that autonomously reallocates resources |
| **Sequence vs. Structure** | LSTMs (Tax 2017) treat processes as linear text — miss parallelism and loops | **Graph Attention Networks** naturally handle non-linear, cyclic process flows |
| **Optimization Speed** | Digital Twin + Genetic Algorithms (Ivanov 2020) require thousands of simulations per decision | RL agent trains offline on SimPy Twin, then makes **sub-millisecond decisions** in production |
| **Reward Signal** | Sparse rewards (only at case completion) | **Dense, GNN-informed reward shaping** — penalizes bottleneck congestion, rewards high-value throughput |
| **LLM Integration** | GPT-based BPM tools generate static reports (Mendling 2023) | **RAG-grounded chatbot** (Gemini 2.5 Flash) with real-time access to quantitative process artifacts |

### Key Results

| Metric | Value |
|--------|-------|
| **Reward Optimization** | **+796%** lift vs. random baseline |
| **Bottleneck Prediction MAE** | **0.078** (vs. SOTA ~0.12) |
| **GNN Prediction Accuracy** | **96.3%** |
| **Decision Latency** | Sub-millisecond (trained PPO agent) |

---

## Architecture Overview

The system operates in four phases:

```
Phase 1: Process Mining
  SAP Event Log (BPI 2019, 600k+ events)
  + Synthetic Jira Tickets (131k+)
  + Synthetic Teams Logs
  → PM4Py → DFG, Bottleneck Scores, Conformance, Resource Profiles

Phase 2: Graph Construction + GNN Training
  → PyG Heterogeneous Graph (42 Activity Nodes + 628 Resource Nodes)
  → GAT + GLU Model → Bottleneck Prediction (0.078 MAE)
  → Node Embeddings (670×32) for RL state encoding

Phase 3: Reinforcement Learning
  → GNN-Enhanced RL Environment (embeddings + ticket features)
  → PPO Agent Training (4 parallel envs, GELU/GLU comparison)
  → Best Agent Selection → ppo_gnn_best.zip

Phase 4: Digital Twin + AI Chatbot
  → SimPy Discrete-Event Simulation (what-if scenarios)
  → Gemini 2.5 Flash Chatbot with RAG context injection
```

For the full architecture diagram and detailed data flow, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **AI/ML** | PyTorch, PyTorch Geometric (PyG), Stable-Baselines3, PettingZoo |
| **Process Mining** | PM4Py |
| **Simulation** | SimPy |
| **Backend** | FastAPI, Uvicorn, WebSockets |
| **Frontend** | React, Vite, TailwindCSS, Chart.js |
| **LLM** | Google Gemini 2.5 Flash (RAG-based chatbot) |
| **Deployment** | Docker, Render, Netlify |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- (Optional) CUDA-compatible GPU for training

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/Moksh-189/AI-Based-Business-Process-Model.git
cd AI-Based-Business-Process-Model

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
pip install torch torch-geometric pm4py

# Start the API server
python server.py
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Training the Models

```bash
# Step 1: Process mining + graph construction
python process_mining.py
python graph_builder.py

# Step 2: Train GNN
python train_gnn.py

# Step 3: Train RL agent
python train_gnn_agent.py
```

---

## Project Structure

```
├── server.py                  # FastAPI backend (REST + WebSocket)
├── process_mining.py          # PM4Py: DFG, bottlenecks, conformance
├── graph_builder.py           # PyG heterogeneous graph construction
├── gnn_model.py               # GAT + GAT+GLU architectures
├── hetero_gnn_model.py        # Heterogeneous GNN variant
├── tgn_model.py               # Temporal Graph Network
├── train_gnn.py               # GNN training + model comparison
├── gnn_env.py                 # GNN-enhanced RL environment
├── custom_env.py              # Base RL environment
├── marl_env.py                # Multi-agent RL environment
├── train_gnn_agent.py         # PPO agent training pipeline
├── train_marl_agent.py        # Multi-agent training
├── digital_twin.py            # SimPy discrete-event simulation
├── simulation_engine.py       # Company simulation engine
├── chatbot.py                 # Gemini 2.5 Flash RAG chatbot
├── app.py                     # Streamlit dashboard
├── frontend/                  # React + Vite frontend
├── ARCHITECTURE.md            # Full system architecture diagrams
├── RESEARCH_COMPARISON.md     # Gap analysis vs. 7 SOTA approaches
├── RESEARCH_SURVEY.md         # Research methodology documentation
├── CHANGELOG.md               # Version history
└── requirements.txt           # Python dependencies
```

---

## Live Demo

- **Dashboard**: [bizopt.netlify.app](https://bizopt.netlify.app/)
- **Source Code**: [github.com/Moksh-189/AI-Based-Business-Process-Model](https://github.com/Moksh-189/AI-Based-Business-Process-Model)

---

## License

This project is open source. See the repository for license details.
