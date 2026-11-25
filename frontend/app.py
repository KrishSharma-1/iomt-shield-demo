"""
frontend/app.py
---------------
IoMT Shield v2 — Full hospital cybersecurity dashboard.
Runs standalone (no backend needed) using the built-in demo simulator.

Launch:
    .venv\\Scripts\\python.exe -m streamlit run frontend/app.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend import alert_store
from capture.simulator import FlowSimulator, ATTACK_SCENARIO_LABELS

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🏥 IoMT Shield — Hospital Cyber Defence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH    = Path("models/iomt_rf_model.joblib")
REGISTRY_PATH = Path("data/device_registry.json")

SEVERITY_COLORS = {
    "CRITICAL": "#FF2D2D",
    "HIGH":     "#FF8C00",
    "MEDIUM":   "#FFD700",
    "INFO":     "#2ECC71",
}
SEVERITY_ICONS = {
    "CRITICAL": "🔴",
    "HIGH":     "🟠",
    "MEDIUM":   "🟡",
    "INFO":     "🟢",
}

# ---------------------------------------------------------------------------
# Load artifacts (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        st.error("Model not found. Run `python train.py` first.")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_registry():
    with open(REGISTRY_PATH) as f:
        return json.load(f)


bundle   = load_bundle()
registry = load_registry()
model    = bundle["model"]
feat_cols = bundle["feature_cols"]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hospital.png", width=60)
    st.title("IoMT Shield v2")
    st.caption("Hospital Network Intrusion Detection")
    st.divider()

    scenario_key = st.selectbox(
        "🎯 Attack Scenario",
        options=list(ATTACK_SCENARIO_LABELS.keys()),
        format_func=lambda k: ATTACK_SCENARIO_LABELS[k],
        index=2,   # default: __MIXED__
    )
    num_events = st.slider("Events to stream", 20, 150, 50)
    delay_sec  = st.slider("Stream interval (s)", 0.1, 1.5, 0.35, 0.05)

    st.divider()
    if st.button("🗑️ Clear Alert Log"):
        alert_store.clear_all()
        st.success("Alert log cleared.")

    st.divider()
    st.markdown("**Dataset:** CICIoMT2024 (UNB)")
    st.markdown("**Model:** Random Forest (80 trees)")
    st.markdown(f"**Features:** {len(feat_cols)}")

# ---------------------------------------------------------------------------
# Top KPI bar
# ---------------------------------------------------------------------------
st.markdown("## 🏥 IoMT Shield — Hospital Cyber Defence Centre")
st.caption("Real-time medical device network intrusion detection • CICIoMT2024 dataset • Random Forest classifier")

stats = alert_store.get_summary()
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Events Logged",   stats["total"])
k2.metric("🚨 Attacks Detected",   stats["attacks"])
k3.metric("✅ Benign Flows",        stats["benign"])
k4.metric("Devices Monitored",     len(registry))
k5.metric("Model Features",        len(feat_cols))

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_live, tab_scenarios, tab_map, tab_alerts, tab_features = st.tabs([
    "📡 Live Feed",
    "⚡ Attack Scenarios",
    "🗺️ Device Map",
    "⚠️ Alert Log",
    "📊 Feature Drivers",
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Feed
# ══════════════════════════════════════════════════════════════════════════
with tab_live:
    st.subheader("📡 Live Hospital Network Telemetry Stream")
    st.info(
        f"**Scenario:** {ATTACK_SCENARIO_LABELS[scenario_key]}  |  "
        f"**Events:** {num_events}  |  **Interval:** {delay_sec}s"
    )

    start_btn = st.button("▶ Start Live Stream", type="primary", key="live_start")

    status_ph  = st.empty()
    kpi_ph     = st.empty()
    feed_ph    = st.empty()

    if start_btn:
        sim      = FlowSimulator(scenario=scenario_key, delay=delay_sec)
        log_rows = []
        counts   = {"attacks": 0, "benign": 0}

        async def _run():
            async for ev in sim.stream(n=num_events):
                is_attack = ev["prediction"] == "Attack"
                if is_attack:
                    counts["attacks"] += 1
                    alert_store.save_alert(ev)
                else:
                    counts["benign"] += 1

                sev  = ev["severity"]
                icon = SEVERITY_ICONS.get(sev, "⚪")

                log_rows.insert(0, {
                    "⏰ Time":        ev["timestamp"],
                    "Status":        "🚨 ATTACK" if is_attack else "✅ BENIGN",
                    "Severity":      f"{icon} {sev}",
                    "Device":        ev["device_name"],
                    "Ward":          ev["ward"],
                    "Scenario":      ev["attack_detail"],
                    "Confidence %":  ev["confidence"],
                    "Attack Risk %": ev["attack_risk"],
                })

                status_ph.info(f"⏳ Processing event **{ev['step']}** of {num_events} — `{ev['device_name']}`")

                with kpi_ph.container():
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Flows Processed", ev["step"])
                    c2.metric("🚨 Attacks", counts["attacks"],
                              delta=1 if is_attack else 0, delta_color="inverse")
                    c3.metric("✅ Benign",  counts["benign"],
                              delta=0 if is_attack else 1)

                feed_ph.dataframe(
                    pd.DataFrame(log_rows),
                    use_container_width=True,
                    hide_index=True,
                )

        asyncio.run(_run())
        attack_pct = round(counts["attacks"] / num_events * 100, 1)
        status_ph.success(
            f"✅ Stream complete — **{counts['attacks']} attacks** flagged ({attack_pct}%) "
            f"out of {num_events} flows. Alert log updated."
        )


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Attack Scenarios
# ══════════════════════════════════════════════════════════════════════════
with tab_scenarios:
    st.subheader("⚡ Attack Scenario Launcher")
    st.markdown(
        "Select an attack type below and click **Launch Attack** to simulate "
        "a targeted cyberattack on a hospital IoMT device in real time."
    )

    col_btns = st.columns(2)
    scenario_launched = None

    with col_btns[0]:
        st.markdown("#### 🔴 Critical Attacks")
        if st.button("💥 DDoS — Distributed UDP Flood", use_container_width=True):
            scenario_launched = "TCP_IP-DDoS-UDP2_train"
        if st.button("🌊 DoS — UDP Flood Attack",       use_container_width=True):
            scenario_launched = "TCP_IP-DoS-UDP_test"

    with col_btns[1]:
        st.markdown("#### 🟡 Mixed / Benign")
        if st.button("📊 Realistic Mix (70% Benign)",   use_container_width=True):
            scenario_launched = "__MIXED__"
        if st.button("✅ Benign Baseline Only",           use_container_width=True):
            scenario_launched = "__BENIGN__"

    if scenario_launched:
        label = ATTACK_SCENARIO_LABELS.get(scenario_launched, scenario_launched)
        st.warning(f"🚀 Launching: **{label}** — streaming 30 events…")

        atk_status = st.empty()
        atk_feed   = st.empty()

        sim2   = FlowSimulator(scenario=scenario_launched, delay=0.3)
        log2   = []
        cnt2   = {"attacks": 0}

        async def _run2():
            async for ev in sim2.stream(n=30):
                is_attack = ev["prediction"] == "Attack"
                if is_attack:
                    cnt2["attacks"] += 1
                    alert_store.save_alert(ev)
                sev  = ev["severity"]
                icon = SEVERITY_ICONS.get(sev, "⚪")
                log2.insert(0, {
                    "⏰": ev["timestamp"],
                    "Status":   "🚨 ATTACK" if is_attack else "✅ BENIGN",
                    "Sev":      f"{icon} {sev}",
                    "Device":   ev["device_name"],
                    "Ward":     ev["ward"],
                    "Scenario": ev["attack_detail"],
                    "Conf %":   ev["confidence"],
                    "Risk %":   ev["attack_risk"],
                })
                atk_status.info(f"Event {ev['step']}/30 — {ev['device_name']}")
                atk_feed.dataframe(pd.DataFrame(log2), use_container_width=True, hide_index=True)

        asyncio.run(_run2())
        atk_status.success(f"✅ Done — **{cnt2['attacks']} attacks** detected in this scenario.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — Device Map
# ══════════════════════════════════════════════════════════════════════════
with tab_map:
    st.subheader("🗺️ Hospital IoMT Device Status Map")
    st.caption("Devices turn red when they have recent attack alerts in the log.")

    # Determine which IPs have recent alerts
    recent = alert_store.get_recent(limit=500, attacks_only=True)
    attacked_ips = {r["src_ip"] for r in recent}

    xs, ys, colors, texts, sizes, symbols = [], [], [], [], [], []
    crit_colors = {
        "CRITICAL": "#FF2D2D",
        "HIGH":     "#FF8C00",
        "MEDIUM":   "#FFD700",
    }

    for ip, dev in registry.items():
        is_attacked = ip in attacked_ips
        color  = "#FF2D2D" if is_attacked else "#2ECC71"
        symbol = "x" if is_attacked else "circle"
        size   = 22 if dev["criticality"] == "CRITICAL" else 16
        status = "⚠️ COMPROMISED" if is_attacked else "✅ SAFE"
        label  = (
            f"<b>{dev['name']}</b><br>"
            f"IP: {ip}<br>"
            f"Ward: {dev['ward']}<br>"
            f"Type: {dev['type']}<br>"
            f"Criticality: {dev['criticality']}<br>"
            f"Status: {status}"
        )
        xs.append(dev["x"])
        ys.append(dev["y"])
        colors.append(color)
        texts.append(label)
        sizes.append(size)
        symbols.append(symbol)

    fig = go.Figure()

    # Hospital floor background zones
    zones = [
        dict(x0=0.0, y0=0.6, x1=0.4, y1=1.0, label="ICU",        color="rgba(255,100,100,0.08)"),
        dict(x0=0.6, y0=0.5, x1=1.0, y1=1.0, label="Radiology",  color="rgba(100,100,255,0.08)"),
        dict(x0=0.3, y0=0.5, x1=0.7, y1=1.0, label="Cardiology", color="rgba(255,165,0,0.08)"),
        dict(x0=0.0, y0=0.1, x1=0.5, y1=0.5, label="Endocrine",  color="rgba(100,200,100,0.08)"),
        dict(x0=0.5, y0=0.1, x1=1.0, y1=0.5, label="General",    color="rgba(200,200,100,0.08)"),
    ]
    for z in zones:
        fig.add_shape(type="rect", x0=z["x0"], y0=z["y0"], x1=z["x1"], y1=z["y1"],
                      fillcolor=z["color"], line=dict(color="rgba(0,0,0,0.1)", width=1))
        fig.add_annotation(x=(z["x0"]+z["x1"])/2, y=z["y1"]-0.03,
                           text=f"<b>{z['label']}</b>", showarrow=False,
                           font=dict(size=11, color="gray"))

    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        marker=dict(color=colors, size=sizes, symbol=symbols,
                    line=dict(color="white", width=2)),
        hovertext=texts,
        hoverinfo="text",
        text=[d["name"].split("#")[0].strip()[:18] for d in registry.values()],
        textposition="top center",
        textfont=dict(size=8),
    ))

    fig.update_layout(
        height=480,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        showlegend=False,
    )

    # Legend
    l1, l2, l3 = st.columns(3)
    l1.markdown("🟢 **Safe device**")
    l2.markdown("🔴 **Compromised device** (attack detected)")
    l3.markdown("⬤ Large = CRITICAL criticality")

    st.plotly_chart(fig, use_container_width=True)

    # Device table
    st.markdown("#### Device Registry")
    dev_rows = []
    for ip, dev in registry.items():
        dev_rows.append({
            "IP Address":  ip,
            "Device":      dev["name"],
            "Ward":        dev["ward"],
            "Type":        dev["type"],
            "Criticality": dev["criticality"],
            "Status":      "⚠️ ATTACKED" if ip in attacked_ips else "✅ Safe",
        })
    st.dataframe(pd.DataFrame(dev_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — Alert Log
# ══════════════════════════════════════════════════════════════════════════
with tab_alerts:
    st.subheader("⚠️ Persistent Alert Log")
    st.caption("All attack events are stored in SQLite and persist across sessions.")

    col_f1, col_f2 = st.columns([3, 1])
    with col_f1:
        filter_ward = st.selectbox(
            "Filter by ward",
            ["All"] + sorted({r["ward"] for r in alert_store.get_recent(1000) if r["ward"]}),
        )
    with col_f2:
        attacks_only = st.checkbox("Attacks only", value=True)

    alerts = alert_store.get_recent(limit=300, attacks_only=attacks_only)
    if filter_ward != "All":
        alerts = [a for a in alerts if a.get("ward") == filter_ward]

    if alerts:
        df_alerts = pd.DataFrame(alerts)
        display_cols = ["timestamp", "severity", "device_name", "ward",
                        "prediction", "attack_detail", "confidence", "attack_risk", "src_ip"]
        df_display = df_alerts[[c for c in display_cols if c in df_alerts.columns]]
        df_display = df_display.rename(columns={
            "timestamp":    "Time",
            "severity":     "Severity",
            "device_name":  "Device",
            "ward":         "Ward",
            "prediction":   "Prediction",
            "attack_detail": "Attack Type",
            "confidence":   "Confidence %",
            "attack_risk":  "Risk %",
            "src_ip":       "Source IP",
        })
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # Summary by severity
        st.markdown("#### Severity Breakdown")
        sev_counts = df_alerts["severity"].value_counts()
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("🔴 CRITICAL", int(sev_counts.get("CRITICAL", 0)))
        sc2.metric("🟠 HIGH",     int(sev_counts.get("HIGH",     0)))
        sc3.metric("🟡 MEDIUM",   int(sev_counts.get("MEDIUM",   0)))
        sc4.metric("🟢 INFO",     int(sev_counts.get("INFO",     0)))

        # CSV download for HIPAA compliance demo
        csv_bytes = df_display.to_csv(index=False).encode()
        st.download_button(
            "📥 Export Alert Log (CSV — HIPAA Audit)",
            data=csv_bytes,
            file_name=f"iomt_alert_log_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    else:
        st.info("No alerts yet. Run the Live Feed or an Attack Scenario first.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — Feature Drivers
# ══════════════════════════════════════════════════════════════════════════
with tab_features:
    st.subheader("📊 Top Network Features Driving Intrusion Decisions")

    if hasattr(model, "feature_importances_") and model.feature_importances_.sum() > 0:
        imp_df = (
            pd.DataFrame({"Feature": feat_cols, "Importance": model.feature_importances_})
            .sort_values("Importance", ascending=False)
            .head(15)
        )
        st.bar_chart(imp_df.set_index("Feature"))
        st.dataframe(imp_df, use_container_width=True, hide_index=True)
    else:
        st.warning(
            "⚠️ The current pre-trained model was trained on single-class data "
            "(Attack only), so all feature importances are 0.  "
            "**Run `python train.py` with the full CICIoMT2024 dataset** "
            "(put CSVs in `raw_csv/`) to get a properly trained model with both classes."
        )
        st.markdown(
            "The 45 features the model uses are:"
        )
        st.write(feat_cols)

