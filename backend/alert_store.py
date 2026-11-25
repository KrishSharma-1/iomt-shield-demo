"""
backend/alert_store.py
-----------------------
SQLite-backed alert log.  Thread-safe for Streamlit's multithreaded runtime.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

DB_PATH = Path("data/alerts.db")
_lock   = threading.Lock()


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        con = _conn()
        con.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT    NOT NULL,
                src_ip       TEXT,
                device_name  TEXT,
                ward         TEXT,
                criticality  TEXT,
                prediction   TEXT    NOT NULL,
                attack_detail TEXT,
                confidence   REAL,
                attack_risk  REAL,
                severity     TEXT,
                acknowledged INTEGER DEFAULT 0
            )
        """)
        con.commit()
        con.close()


def save_alert(event: dict):
    with _lock:
        con = _conn()
        con.execute("""
            INSERT INTO alerts
              (timestamp, src_ip, device_name, ward, criticality, prediction,
               attack_detail, confidence, attack_risk, severity)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            event.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
            event.get("src_ip"),
            event.get("device_name"),
            event.get("ward"),
            event.get("criticality"),
            event.get("prediction"),
            event.get("attack_detail"),
            event.get("confidence"),
            event.get("attack_risk"),
            event.get("severity"),
        ))
        con.commit()
        con.close()


def get_recent(limit: int = 200, attacks_only: bool = False) -> list[dict]:
    with _lock:
        con  = _conn()
        where = "WHERE prediction = 'Attack'" if attacks_only else ""
        rows = con.execute(
            f"SELECT * FROM alerts {where} ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        con.close()
    return [dict(r) for r in rows]


def get_summary() -> dict:
    with _lock:
        con = _conn()
        total   = con.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        attacks = con.execute("SELECT COUNT(*) FROM alerts WHERE prediction='Attack'").fetchone()[0]
        benign  = total - attacks
        by_ward = dict(con.execute(
            "SELECT ward, COUNT(*) FROM alerts WHERE prediction='Attack' GROUP BY ward"
        ).fetchall())
        by_severity = dict(con.execute(
            "SELECT severity, COUNT(*) FROM alerts WHERE prediction='Attack' GROUP BY severity"
        ).fetchall())
        con.close()
    return {
        "total":       total,
        "attacks":     attacks,
        "benign":      benign,
        "by_ward":     by_ward,
        "by_severity": by_severity,
    }


def acknowledge(alert_id: int):
    with _lock:
        con = _conn()
        con.execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (alert_id,))
        con.commit()
        con.close()


def clear_all():
    with _lock:
        con = _conn()
        con.execute("DELETE FROM alerts")
        con.commit()
        con.close()


# Auto-init on import
init_db()

