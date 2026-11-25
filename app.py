import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="IoMT Shield — Healthcare Network Intrusion Demo",
    page_icon="🛡️",
    layout="wide",
)

MODEL_PATH = Path("models/iomt_rf_model.joblib")
DATA_PATH = Path("data/demo_samples.csv")


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found at {MODEL_PATH}. Run `python train.py` first.")
        st.stop()
    if not DATA_PATH.exists():
        st.error(
            f"Demo dataset not found at {DATA_PATH}. Run `python train.py` first.")
        st.stop()

    bundle = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    return bundle, df


def score_rows(bundle: dict, rows_df: pd.DataFrame) -> pd.DataFrame:
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    medians = bundle["medians"]

    X = pd.DataFrame(index=rows_df.index)
    for col in feature_cols:
        if col in rows_df.columns:
            series = pd.to_numeric(rows_df[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan)
            X[col] = series.fillna(medians.get(col, 0.0))
        else:
            X[col] = medians.get(col, 0.0)

    preds = model.predict(X)
    classes = list(model.classes_)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        max_conf = probs.max(axis=1)
        attack_prob = (
            probs[:, classes.index("Attack")]
            if "Attack" in classes
            else np.where(preds == "Attack", max_conf, 1.0 - max_conf)
        )
    else:
        max_conf = np.ones(len(X))
        attack_prob = np.where(preds == "Attack", 1.0, 0.0)

    result = rows_df.copy()
    result["prediction"] = preds
    result["confidence"] = np.round(max_conf * 100, 2)
    result["attack_risk_pct"] = np.round(attack_prob * 100, 2)
    return result


bundle, demo_df = load_artifacts()
model = bundle["model"]
feature_cols = bundle["feature_cols"]

st.title("🛡️ IoMT Shield — Medical Device Intrusion Triage Prototype")
st.caption("Built on CICIoMT2024 (WiFi & MQTT) • Random Forest Flow Classifier")

# Top KPI summary
total_samples = len(demo_df)
actual_attacks = int((demo_df["binary_target"] == "Attack").sum(
)) if "binary_target" in demo_df.columns else 0
actual_benign = int((demo_df["binary_target"] == "Benign").sum(
)) if "binary_target" in demo_df.columns else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Demo Flow Pool", f"{total_samples}")
c2.metric("Attack Flows in Pool", f"{actual_attacks}")
c3.metric("Benign Flows in Pool", f"{actual_benign}")
c4.metric("Model Features", f"{len(feature_cols)}")

tab1, tab2, tab3 = st.tabs(
    ["🔍 Single Flow Inspector", "📡 Live Telemetry Replay", "📊 Model Feature Drivers"])

with tab1:
    st.subheader("Inspect an Individual Medical Network Flow")
    row_idx = st.slider("Select Sample Row Index", min_value=0,
                        max_value=max(0, len(demo_df) - 1), value=0)
    selected_row = demo_df.iloc[[row_idx]]
    scored = score_rows(bundle, selected_row).iloc[0]

    pred_label = scored["prediction"]
    conf_pct = scored["confidence"]
    detail_label = scored.get("attack_detail", "N/A")
    ground_truth = scored.get("binary_target", "N/A")

    left, right = st.columns([1, 2])
    with left:
        if pred_label == "Attack":
            st.error(f"🚨 ALERT: {pred_label.upper()} ({conf_pct}% confidence)")
        else:
            st.success(
                f"✅ NORMAL: {pred_label.upper()} ({conf_pct}% confidence)")

        st.write(f"**Dataset Scenario / Label:** `{detail_label}`")
        st.write(f"**Ground Truth Class:** `{ground_truth}`")
        st.write(f"**Estimated Attack Risk:** `{scored['attack_risk_pct']}%`")

    with right:
        top_features = feature_cols[:12]
        preview_vals = {col: scored[col]
                        for col in top_features if col in scored}
        st.write("**Key Flow Telemetry Snapshot (First 12 Features)**")
        st.dataframe(pd.DataFrame([preview_vals]).T.rename(
            columns={0: "Value"}), use_container_width=True)

with tab2:
    st.subheader("Simulate Live Hospital IoMT Telemetry Stream")
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    num_events = col_ctrl1.slider(
        "Events to Stream", min_value=10, max_value=min(150, len(demo_df)), value=30)
    delay_sec = col_ctrl2.slider(
        "Interval (seconds)", min_value=0.1, max_value=1.5, value=0.35, step=0.05)
    start_btn = col_ctrl3.button("▶ Start Live Stream", type="primary")

    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    log_placeholder = st.empty()

    if start_btn:
        stream_slice = demo_df.sample(n=num_events, random_state=int(
            time.time()) % 10000).reset_index(drop=True)
        alerts = 0
        benign_count = 0
        logs = []

        for step in range(num_events):
            one_row = stream_slice.iloc[[step]]
            scored_row = score_rows(bundle, one_row).iloc[0]
            is_attack = scored_row["prediction"] == "Attack"
            if is_attack:
                alerts += 1
            else:
                benign_count += 1

            logs.insert(
                0,
                {
                    "Event #": step + 1,
                    "Status": "🚨 ATTACK" if is_attack else "✅ BENIGN",
                    "Scenario": scored_row.get("attack_detail", "Unknown"),
                    "Confidence (%)": scored_row["confidence"],
                    "Attack Risk (%)": scored_row["attack_risk_pct"],
                },
            )

            status_placeholder.info(
                f"Streaming medical device flow {step + 1} of {num_events}...")
            with metrics_placeholder.container():
                m1, m2, m3 = st.columns(3)
                m1.metric("Flows Processed", f"{step + 1}")
                m2.metric("Attacks Flagged", f"{alerts}")
                m3.metric("Benign Flows", f"{benign_count}")

            log_placeholder.dataframe(pd.DataFrame(
                logs), use_container_width=True, hide_index=True)
            time.sleep(delay_sec)

        status_placeholder.success(
            f"Stream complete: {alerts} attacks flagged out of {num_events} flows.")

with tab3:
    st.subheader("Top Features Driving Random Forest Decisions")
    if hasattr(model, "feature_importances_"):
        imp_df = pd.DataFrame(
            {"Feature": feature_cols, "Importance": model.feature_importances_}
        ).sort_values("Importance", ascending=False).head(15)

        st.bar_chart(imp_df.set_index("Feature"))
        st.dataframe(imp_df, use_container_width=True, hide_index=True)
    else:
        st.info("Feature importances are not available for this estimator.")
