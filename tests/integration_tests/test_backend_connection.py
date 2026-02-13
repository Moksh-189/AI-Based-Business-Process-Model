
import pytest
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)

def test_read_topology():
    response = client.get("/api/topology")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) > 0

def test_telemetry():
    response = client.get("/api/telemetry")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 3
    assert data[0]["name"] == "Cycle Time (Days)"

def test_simulation():
    payload = {
        "assigned": [
            {"id": "1", "name": "Test User", "role": "Snr. Analyst", "efficiency": 90}
        ]
    }
    response = client.post("/api/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "simulated"
    assert "state" in data
    assert "cycle_time_red" in data["state"]

def test_websocket_chat():
    with client.websocket_connect("/ws/chat") as websocket:
        websocket.send_text("Hello")
        # Since chatbot might not be init (no key), it sends a system message or response
        # We just check we get *something* back
        data = websocket.receive_text()
        assert isinstance(data, str)
        assert len(data) > 0
