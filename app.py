
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

# ---------------------------------------------------------------------
# Project paths / imports
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import config
from dataset import make_loaders
from model import build_model

st.set_page_config(
    page_title="Network World Model IDS",
    page_icon="🛡️",
    layout="wide",
)

CKPT = Path(config.CHECKPOINT_DIR) / "world_model_v2_best.pt"
THRESH = Path(config.OUTPUT_DIR) / "alert_threshold.json"
FEATURES = Path(config.PROCESSED_DATA_DIR) / "feature_cols.txt"
ALERT_LOG = Path(config.OUTPUT_DIR) / "alerts_log.json"
REPORT = Path(config.OUTPUT_DIR) / "test_classification_report.txt"
HISTORY = Path(config.OUTPUT_DIR) / "training_history.json"


# ---------------------------------------------------------------------
# Load project artifacts
# ---------------------------------------------------------------------
@st.cache_resource
def load_model():
    ckpt = torch.load(CKPT, map_location=config.DEVICE)

    if not isinstance(ckpt, dict):
        raise RuntimeError("Checkpoint is not a dictionary.")

    # Current V2 checkpoint format
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        raise RuntimeError(
            f"Unknown checkpoint format. Keys found: {list(ckpt.keys())}"
        )

    if "num_features" in ckpt:
        num_features = int(ckpt["num_features"])
    elif FEATURES.exists():
        num_features = len(
            [x.strip() for x in FEATURES.read_text().splitlines() if x.strip()]
        )
    else:
        raise RuntimeError("Could not determine number of input features.")

    model = build_model(num_features)
    model.load_state_dict(state_dict)
    model.eval()
    return model, num_features, ckpt


@st.cache_data
def load_project_data():
    # V2 interface returns (loaders, num_features, train_counts).
    result = make_loaders()

    if len(result) == 3:
        loaders, num_features, train_counts = result
    elif len(result) == 2:
        loaders, num_features = result
        train_counts = None
    else:
        raise RuntimeError(f"Unexpected make_loaders() return length: {len(result)}")

    ds = loaders["test"].dataset

    if FEATURES.exists():
        feature_names = [
            x.strip() for x in FEATURES.read_text().splitlines() if x.strip()
        ]
    else:
        feature_names = [f"Feature {i+1}" for i in range(int(num_features))]

    threshold = None
    if THRESH.exists():
        raw = json.loads(THRESH.read_text())
        threshold = float(raw["forecast_error_threshold"])

    return ds, feature_names, threshold, train_counts


def load_alerts():
    if not ALERT_LOG.exists():
        return []

    try:
        data = json.loads(ALERT_LOG.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def parse_test_report():
    """Parse the project's saved sklearn classification report."""
    if not REPORT.exists():
        return None

    text = REPORT.read_text(errors="ignore")
    metrics = {}

    # Lines such as:
    # Benign 0.96 0.96 0.96 685356
    for stage in config.STAGES:
        pattern = rf"^\s*{re.escape(stage)}\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([\d,]+)"
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            metrics[stage] = {
                "precision": float(match.group(1)),
                "recall": float(match.group(2)),
                "f1": float(match.group(3)),
                "support": int(match.group(4).replace(",", "")),
            }

    macro = re.search(
        r"^\s*macro avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([\d,]+)",
        text,
        re.MULTILINE,
    )
    weighted = re.search(
        r"^\s*weighted avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([\d,]+)",
        text,
        re.MULTILINE,
    )
    accuracy = re.search(r"^\s*accuracy\s+([0-9.]+)", text, re.MULTILINE)

    return {
        "metrics": metrics,
        "macro_f1": float(macro.group(3)) if macro else None,
        "macro_precision": float(macro.group(1)) if macro else None,
        "macro_recall": float(macro.group(2)) if macro else None,
        "weighted_f1": float(weighted.group(3)) if weighted else None,
        "accuracy": float(accuracy.group(1)) if accuracy else None,
        "raw": text,
    }


def load_history():
    if not HISTORY.exists():
        return []
    try:
        data = json.loads(HISTORY.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


# ---------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------
def explain_prediction(model, x, pred_idx):
    z = (
        x.unsqueeze(0)
        .to(config.DEVICE)
        .detach()
        .requires_grad_(True)
    )

    model.zero_grad(set_to_none=True)
    _, logits = model(z)
    logits[0, pred_idx].backward()

    attribution = (
        z.grad[0].abs() * z[0].detach().abs()
    ).sum(0).cpu().numpy()

    with torch.no_grad():
        _, _, layers = model(
            x.unsqueeze(0).to(config.DEVICE),
            return_attention=True,
        )

    attention = torch.stack(layers).mean(0)[0, -1].cpu().numpy()
    return attribution, attention


def predict_sequence(model, x, y_next, y_stage):
    with torch.no_grad():
        forecast, logits = model(
            x.unsqueeze(0).to(config.DEVICE)
        )
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(probs.argmax())
    pred_stage = config.STAGES[pred_idx]
    true_stage = config.STAGES[int(y_stage)]

    forecast_error = float(
        ((forecast[0].cpu().numpy() - y_next.numpy()) ** 2).mean()
    )

    confidence = float(probs[pred_idx])

    benign_idx = config.STAGE_TO_IDX["Benign"]
    stage_alert = (
        pred_idx != benign_idx
        and confidence >= config.ALERT_STAGE_PROB_THRESHOLD
    )

    novelty_alert = (
        threshold_value is not None
        and forecast_error >= threshold_value
    )

    return {
        "forecast": forecast,
        "probs": probs,
        "pred_idx": pred_idx,
        "pred_stage": pred_stage,
        "true_stage": true_stage,
        "confidence": confidence,
        "forecast_error": forecast_error,
        "stage_alert": stage_alert,
        "novelty_alert": novelty_alert,
    }


# ---------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------
if not CKPT.exists():
    st.error(f"Checkpoint not found: {CKPT}")
    st.stop()

try:
    (model, num_features, checkpoint), = [load_model()]
    ds, feature_names, threshold_value, train_counts = load_project_data()
except Exception as exc:
    st.error(f"Could not load project data/model: {exc}")
    st.stop()

if len(feature_names) != num_features:
    feature_names = feature_names[:num_features]
    if len(feature_names) < num_features:
        feature_names += [
            f"Feature {i+1}" for i in range(len(feature_names), num_features)
        ]

alerts = load_alerts()
report = parse_test_report()
history = load_history()

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.sidebar.title("🛡️ World Model IDS")
st.sidebar.caption("Predictive + Explainable Intrusion Early Warning")

st.sidebar.success(f"Device: {config.DEVICE}")

page = st.sidebar.radio(
    "Dashboard",
    [
        "🔎 Detection & XAI",
        "📊 Performance",
        "🚨 Alert History",
        "ℹ️ System Overview",
    ],
)

if page == "🔎 Detection & XAI":
    max_idx = max(0, len(ds) - 1)
    default_idx = min(45, max_idx)

    idx = st.sidebar.number_input(
        "Held-out test sequence",
        min_value=0,
        max_value=max_idx,
        value=default_idx,
        step=1,
    )

    topk = st.sidebar.slider(
        "Top features",
        min_value=5,
        max_value=min(15, len(feature_names)),
        value=min(10, len(feature_names)),
    )
else:
    idx = 45 if len(ds) > 45 else 0
    topk = 10


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("🛡️ Network World Model IDS")
st.caption("Predict → Forecast → Identify attack stage → Explain → Early warning")

# ---------------------------------------------------------------------
# Detection & XAI
# ---------------------------------------------------------------------
if page == "🔎 Detection & XAI":
    x, y_next, y_stage = ds[int(idx)]
    result = predict_sequence(model, x, y_next, y_stage)

    alert_active = result["stage_alert"] or result["novelty_alert"]

    if alert_active:
        st.error("🚨 EARLY WARNING ALERT")
    else:
        st.success("✅ No alert for this sequence")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Stage", result["pred_stage"])
    c2.metric("Confidence", f"{result['confidence'] * 100:.2f}%")
    c3.metric("Forecast Error", f"{result['forecast_error']:.6f}")
    c4.metric("True Stage", result["true_stage"])

    a, b = st.columns(2)

    with a:
        st.subheader("Stage Alert")
        if result["stage_alert"]:
            st.error("Triggered")
        else:
            st.success("Not triggered")
        st.caption(
            f"Non-Benign probability threshold: "
            f"{config.ALERT_STAGE_PROB_THRESHOLD:.2f}"
        )

    with b:
        st.subheader("Novelty Alert")
        if result["novelty_alert"]:
            st.error("Triggered")
        else:
            st.success("Not triggered")
        if threshold_value is not None:
            st.caption(
                f"Forecast-error threshold: {threshold_value:.6f}"
            )

    st.subheader("Attack-Stage Probabilities")
    prob_df = pd.DataFrame(
        {"Probability": result["probs"]},
        index=config.STAGES,
    )
    st.bar_chart(prob_df)

    st.divider()
    st.subheader("Why did the model make this prediction?")

    try:
        attribution, attention = explain_prediction(
            model, x, result["pred_idx"]
        )

        order = np.argsort(attribution)[::-1][:topk]

        feature_df = pd.DataFrame(
            {
                "Feature": [feature_names[i] for i in order],
                "Attribution": [float(attribution[i]) for i in order],
            }
        )

        attention_df = pd.DataFrame(
            {"Attention": attention},
            index=[f"Timestep {i + 1}" for i in range(len(attention))],
        )

        left, right = st.columns(2)

        with left:
            st.markdown("### 🔬 Feature Attribution")
            st.bar_chart(feature_df.set_index("Feature"))
            st.dataframe(
                feature_df,
                hide_index=True,
                use_container_width=True,
            )

        with right:
            st.markdown("### ⏱️ Temporal Attention")
            st.bar_chart(attention_df)
            st.caption(
                "Higher values indicate greater attention from the final "
                "timestep in this explanation."
            )

    except Exception as exc:
        st.warning(f"Explainability could not be calculated: {exc}")

    with st.expander("View input sequence"):
        st.dataframe(
            pd.DataFrame(
                x.cpu().numpy(),
                columns=feature_names,
            ),
            use_container_width=True,
        )

    with st.expander("Technical details"):
        st.json(
            {
                "sequence_index": int(idx),
                "true_stage": result["true_stage"],
                "predicted_stage": result["pred_stage"],
                "confidence": result["confidence"],
                "forecast_error": result["forecast_error"],
                "stage_alert": result["stage_alert"],
                "novelty_alert": result["novelty_alert"],
                "device": str(config.DEVICE),
                "window_size": int(x.shape[0]),
                "num_features": int(x.shape[1]),
                "parameters": sum(
                    p.numel()
                    for p in model.parameters()
                    if p.requires_grad
                ),
                "checkpoint_epoch": checkpoint.get("epoch"),
            }
        )

    st.info(
        "Interpretation note: feature attribution describes model "
        "behavior for this prediction; it is not proof that a feature "
        "caused an attack."
    )


# ---------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------
elif page == "📊 Performance":
    st.header("📊 Model Performance")

    if report is None:
        st.warning(
            "No saved test classification report was found. "
            "Run the project's training/evaluation script first."
        )
    else:
        metrics = report["metrics"]

        # Only count the primary four stages as the project's sufficiently
        # represented evaluation group. This matches the current project
        # interpretation rather than pretending the rare exploratory
        # classes have reliable support.
        primary_stages = [
            s for s in
            ["Benign", "Reconnaissance", "C2", "ActionsOnObjectives"]
            if s in metrics
        ]

        if primary_stages:
            primary_f1 = float(
                np.mean([metrics[s]["f1"] for s in primary_stages])
            )
        else:
            primary_f1 = None

        p1, p2, p3, p4 = st.columns(4)
        p1.metric(
            "Primary Macro-F1",
            f"{primary_f1:.4f}" if primary_f1 is not None else "—",
        )
        p2.metric(
            "Overall Accuracy",
            f"{report['accuracy']:.2%}"
            if report["accuracy"] is not None
            else "—",
        )
        p3.metric(
            "Test Sequences",
            f"{sum(v['support'] for v in metrics.values()):,}"
            if metrics
            else "—",
        )
        p4.metric(
            "Input Features",
            str(num_features),
        )

        st.subheader("Per-Stage Results")

        if metrics:
            rows = []
            for stage in config.STAGES:
                if stage in metrics:
                    rows.append(
                        {
                            "Stage": stage,
                            "Precision": metrics[stage]["precision"],
                            "Recall": metrics[stage]["recall"],
                            "F1": metrics[stage]["f1"],
                            "Support": metrics[stage]["support"],
                        }
                    )

            metric_df = pd.DataFrame(rows)
            st.dataframe(
                metric_df.style.format(
                    {
                        "Precision": "{:.2f}",
                        "Recall": "{:.2f}",
                        "F1": "{:.2f}",
                        "Support": "{:,}",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

            st.subheader("F1 by Attack Stage")
            st.bar_chart(
                metric_df.set_index("Stage")[["F1"]]
            )

            rare = [
                (s, metrics[s]["support"])
                for s in ["Delivery", "Exploitation"]
                if s in metrics
            ]

            if rare:
                rare_text = ", ".join(
                    f"{s}: {n:,} test samples" for s, n in rare
                )
                st.warning(
                    "Exploratory rare-stage coverage: " + rare_text +
                    ". These stages should not be presented as having "
                    "reliable six-class performance from this test split."
                )

        if history:
            st.subheader("Training History")
            hist_df = pd.DataFrame(history)

            if "epoch" in hist_df.columns:
                hist_df = hist_df.set_index("epoch")

            available = [
                c for c in
                ["train_loss", "val_loss", "train_macro_f1", "val_macro_f1"]
                if c in hist_df.columns
            ]

            if available:
                st.line_chart(hist_df[available])

        with st.expander("Raw saved classification report"):
            st.code(report["raw"], language="text")


# ---------------------------------------------------------------------
# Alert History
# ---------------------------------------------------------------------
elif page == "🚨 Alert History":
    st.header("🚨 Early-Warning Alert History")

    if not alerts:
        st.warning(
            "No alerts_log.json found. Run: "
            "`python src/inference.py --limit 200`"
        )
    else:
        alert_df = pd.DataFrame(alerts)

        total = len(alert_df)
        attack_alerts = int(
            (alert_df["true_stage"] != "Benign").sum()
        ) if "true_stage" in alert_df else 0

        stage_alert_count = 0
        novelty_alert_count = 0

        if "stage_alert" in alert_df.columns:
            stage_alert_count = int(
                alert_df["stage_alert"].fillna(False).astype(bool).sum()
            )

        if "novelty_alert" in alert_df.columns:
            novelty_alert_count = int(
                alert_df["novelty_alert"].fillna(False).astype(bool).sum()
            )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Alerts", total)
        c2.metric("Alerts on Non-Benign Truth", attack_alerts)
        c3.metric("Stage Alerts", stage_alert_count)
        c4.metric("Novelty Alerts", novelty_alert_count)

        if "predicted_stage" in alert_df.columns:
            st.subheader("Predicted Stage Distribution")
            st.bar_chart(
                alert_df["predicted_stage"].value_counts()
            )

        st.subheader("Alert Log")

        display_cols = [
            c for c in
            [
                "sequence_index",
                "predicted_stage",
                "confidence",
                "true_stage",
                "forecast_error",
                "reasons",
            ]
            if c in alert_df.columns
        ]

        display_df = alert_df[display_cols].copy()

        if "confidence" in display_df.columns:
            display_df["confidence"] = (
                display_df["confidence"] * 100
            ).round(2)

        if "forecast_error" in display_df.columns:
            display_df["forecast_error"] = (
                display_df["forecast_error"].round(6)
            )

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "The alert log is generated by the project's inference "
            "script from held-out test sequences; it is a simulation "
            "of streaming early-warning behavior, not live packet capture."
        )


# ---------------------------------------------------------------------
# System Overview
# ---------------------------------------------------------------------
else:
    st.header("ℹ️ System Overview")

    st.markdown(
        """
### What the system does

The Network World Model IDS learns temporal patterns in network-flow
sequences and combines two tasks:

1. **Future-state forecasting** — predict the next network state.
2. **Attack-stage classification** — identify the likely stage of activity.

The forecast error is then used as a novelty signal, while the
classification probability is used for stage-based alerting.

### Pipeline

**CIC-IDS data → preprocessing → temporal sequences → Transformer World Model →**

- Future-state forecast
- Attack-stage prediction
- Forecast-error novelty detection
- Feature attribution
- Temporal attention
- Early-warning alert
- Streamlit dashboard

### Six project stages

"""
    )

    stage_df = pd.DataFrame(
        {
            "Stage": config.STAGES,
            "Index": list(range(len(config.STAGES))),
        }
    )
    st.dataframe(stage_df, hide_index=True, use_container_width=True)

    st.subheader("Current Model")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Parameters", f"{sum(p.numel() for p in model.parameters()):,}")
    c2.metric("Window Size", str(config.WINDOW_SIZE))
    c3.metric("Forecast Horizon", str(config.FORECAST_HORIZON))
    c4.metric("Features", str(num_features))

    st.subheader("Alert Logic")

    st.markdown(
        f"""
**Stage alert**

A non-Benign stage is flagged when its predicted probability is at
least **{config.ALERT_STAGE_PROB_THRESHOLD:.2f}**.

**Novelty alert**

A sequence is flagged when its forecast error is at least the
calibrated threshold:

**{threshold_value:.6f}**
"""
        if threshold_value is not None
        else
        f"""
**Stage alert**

A non-Benign stage is flagged when its predicted probability is at
least **{config.ALERT_STAGE_PROB_THRESHOLD:.2f}**.

**Novelty threshold**

No saved threshold was found.
"""
    )

    st.subheader("Demo Sequences")

    demo_df = pd.DataFrame(
        [
            {
                "Sequence": 45,
                "Purpose": "Known C2 detection + explanation",
            },
            {
                "Sequence": 34,
                "Purpose": "Novelty-only alert demonstration",
            },
        ]
    )
    st.dataframe(demo_df, hide_index=True, use_container_width=True)

    st.info(
        "Use sequence 45 for the main C2 demonstration and sequence 34 "
        "to demonstrate novelty detection."
    )

st.divider()
st.caption(
    "Network World Model IDS • Predictive + Explainable Intrusion Early Warning"
)
