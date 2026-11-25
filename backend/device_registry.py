"""
backend/device_registry.py
---------------------------
Loads and queries the hospital device registry JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

REGISTRY_PATH = Path("data/device_registry.json")

_registry: dict | None = None


def _load() -> dict:
    global _registry
    if _registry is None:
        with open(REGISTRY_PATH) as f:
            _registry = json.load(f)
    return _registry


def lookup(ip: str) -> dict:
    reg = _load()
    return reg.get(ip, {
        "name":        f"Unknown Device ({ip})",
        "ward":        "Unknown",
        "type":        "Unknown",
        "criticality": "MEDIUM",
        "x": 0.5, "y": 0.5,
    })


def all_devices() -> dict:
    return _load()


def all_ips() -> list[str]:
    return list(_load().keys())

