"""
backend/main.py
---------------
FastAPI backend — WebSocket live feed + REST API.

Run with:
    .venv\\Scripts\\python.exe -m uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware

from backend import alert_store, device_registry
from capture.simulator import FlowSimulator, ATTACK_SCENARIO_LABELS

app = FastAPI(
    title="IoMT Shield API",
    description="Real-time hospital IoMT intrusion detection backend",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# WebSocket — live feed
# ---------------------------------------------------------------------------
@app.websocket("/ws/live-feed")
async def live_feed(
    ws: WebSocket,
    scenario: str = Query(default="__MIXED__"),
    n: int = Query(default=60),
    delay: float = Query(default=0.4),
):
    await ws.accept()
    try:
        sim = FlowSimulator(scenario=scenario, delay=delay)
        async for event in sim.stream(n=n):
            await ws.send_json(event)
            if event["prediction"] == "Attack":
                alert_store.save_alert(event)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        await ws.send_json({"error": str(exc)})


# ---------------------------------------------------------------------------
# REST — alerts
# ---------------------------------------------------------------------------
@app.get("/api/alerts")
def get_alerts(limit: int = 200, attacks_only: bool = False):
    return alert_store.get_recent(limit=limit, attacks_only=attacks_only)


@app.get("/api/stats")
def get_stats():
    return alert_store.get_summary()


@app.post("/api/alerts/{alert_id}/acknowledge")
def ack_alert(alert_id: int):
    alert_store.acknowledge(alert_id)
    return {"status": "acknowledged"}


@app.delete("/api/alerts")
def clear_alerts():
    alert_store.clear_all()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# REST — devices
# ---------------------------------------------------------------------------
@app.get("/api/devices")
def get_devices():
    return device_registry.all_devices()


# ---------------------------------------------------------------------------
# REST — scenarios
# ---------------------------------------------------------------------------
@app.get("/api/scenarios")
def get_scenarios():
    return ATTACK_SCENARIO_LABELS


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}

