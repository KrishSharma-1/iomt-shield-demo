"""
backend/inference.py
--------------------
Loads the ML bundle once at import time and exposes a `score()` function.
Shared by both the FastAPI backend and the standalone Streamlit dashboard.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path("models/iomt_rf_model.joblib")

_bundle: dict | None = None


def _get_bundle() -> dict:
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def score(row: dict) -> dict:
    """
    Score a single flow feature dict.

    Returns a dict with keys:
        prediction, confidence, attack_risk, feature_cols, model_version
    """
    b             = _get_bundle()
    model         = b["model"]
    feature_cols  = b["feature_cols"]
    medians       = b["medians"]

    X: dict = {}
    for col in feature_cols:
        val = row.get(col, medians.get(col, 0.0))
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = 0.0
        X[col] = val if np.isfinite(val) else float(medians.get(col, 0.0))

    x_df    = pd.DataFrame([X])
    pred    = str(model.predict(x_df)[0])
    classes = list(model.classes_)

    if hasattr(model, "predict_proba"):
        probs    = model.predict_proba(x_df)[0]
        conf     = float(probs.max() * 100)
        atk_risk = float(probs[classes.index("Attack")] * 100) if "Attack" in classes else (100.0 if pred == "Attack" else 0.0)
    else:
        conf     = 100.0
        atk_risk = 100.0 if pred == "Attack" else 0.0

    return {
        "prediction":   pred,
        "confidence":   round(conf, 2),
        "attack_risk":  round(atk_risk, 2),
        "feature_cols": feature_cols,
        "model_version": b.get("model_version", "1.0.0"),
    }


def feature_importances() -> list[dict]:
    b   = _get_bundle()
    m   = b["model"]
    if not hasattr(m, "feature_importances_"):
        return []
    return [
        {"feature": f, "importance": round(float(i), 6)}
        for f, i in zip(b["feature_cols"], m.feature_importances_)
    ]

