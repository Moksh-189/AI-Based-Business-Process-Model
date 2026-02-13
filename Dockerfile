# ============================================
# Stage 1: Build React Frontend
# ============================================
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ============================================
# Stage 2: Python Backend + Serve Frontend
# ============================================
FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY server.py chatbot.py digital_twin.py gnn_model.py gnn_env.py \
     train_gnn_agent.py train_agent.py train_gnn.py custom_env.py \
     worker_data.py simulation_engine.py graph_builder.py \
     process_mining.py dependency.py ./

# Copy data files (JSON only, CSVs are too large - mount as volume)
COPY *.json ./

# Copy built frontend from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/telemetry')" || exit 1

# Run
CMD ["python", "server.py"]
