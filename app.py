"""Streamlit dashboard for evaluating the saved EV adoption classifiers."""

from __future__ import annotations

import html
import io
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
SAMPLE_PATH = ROOT / "test_data.csv"

METRIC_KEYS = ["accuracy", "auc", "precision", "recall", "f1", "mcc"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "auc": "ROC AUC",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 Score",
    "mcc": "MCC",
}
METRIC_HINTS = {
    "accuracy": "Share of all buyers classified correctly",
    "auc": "Ranking quality across every threshold",
    "precision": "Predicted EV buyers who actually bought",
    "recall": "Actual EV buyers the model caught",
    "f1": "Balance of precision and recall",
    "mcc": "Correlation on imbalanced classes (-1 to 1)",
}
MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "K-Nearest Neighbors",
    "gaussian_nb": "Gaussian Naive Bayes",
    "svm": "Support Vector Machine",
}
MODEL_BLURBS = {
    "logistic_regression": "Linear model on scaled and one-hot encoded features. Fast and easy to interpret.",
    "decision_tree": "Rule-based splits on the raw features. Captures non-linear interactions.",
    "knn": "Classifies each buyer by majority vote of the most similar buyers in training.",
    "gaussian_nb": "Probabilistic model that assumes features are independent given the class.",
    "svm": "Finds the maximum-margin boundary between buyers and non-buyers.",
}
INK = "#0B1220"
MUTED = "#5B6577"
ACCENT = "#14B8A6"
NO_COLOR = "#94A3B8"
SERIES_COLORS = ["#14B8A6", "#6366F1", "#F59E0B", "#EC4899", "#0EA5E9"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');
html, body, [class*="css"], .stApp, button, input, textarea, select {
    font-family: 'Plus Jakarta Sans', Inter, system-ui, sans-serif;
}
.stApp { background: #F5F7FB; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1220 0%, #0F1F2E 55%, #0D2B2C 100%);
}
section[data-testid="stSidebar"] * { color: #E2E8F0; }
.hero { background: linear-gradient(135deg, #0B1220, #12363A); color: white;
    border-radius: 22px; padding: 2rem 2.2rem; margin-bottom: 1.4rem; }
.hero h1 { margin: 0 0 .45rem; font-size: 2.2rem; }
.hero p { color: #C8D6DF; margin: 0; max-width: 760px; }
.model-card { background: white; border: 1px solid #E5EAF0; border-radius: 16px;
    padding: 1rem 1.1rem; height: 100%; }
.model-card h3 { margin: 0 0 .4rem; color: #0B1220; font-size: 1rem; }
.model-card p { color: #5B6577; font-size: .82rem; line-height: 1.45; }
.eyebrow { color: #14B8A6; font-size: .75rem; font-weight: 800; letter-spacing: .12em;
    text-transform: uppercase; margin-bottom: .35rem; }
</style>
"""


@st.cache_data
def load_metadata() -> dict[str, Any]:
    return json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))


@st.cache_data
def load_saved_metrics() -> dict[str, Any]:
    return json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))


@st.cache_resource
def load_model(model_key: str) -> Any:
    return joblib.load(MODEL_DIR / f"{model_key}.joblib")


def positive_scores(model: Any, features: pd.DataFrame) -> Any:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        classes = list(model.classes_)
        return probabilities[:, classes.index(1)]
    return model.decision_function(features)


def calculate_metrics(
    model: Any, features: pd.DataFrame, target: pd.Series
) -> tuple[dict[str, float], list[list[int]], dict[str, dict[str, float]], Any, Any]:
    predictions = model.predict(features)
    scores = positive_scores(model, features)
    metrics = {
        "accuracy": float(accuracy_score(target, predictions)),
        "auc": float(roc_auc_score(target, scores)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(target, predictions)),
    }
    matrix = confusion_matrix(target, predictions, labels=[0, 1]).tolist()
    report = classification_report(
        target,
        predictions,
        labels=[0, 1],
        target_names=["No", "Yes"],
        output_dict=True,
        zero_division=0,
    )
    return metrics, matrix, report, predictions, scores


def validate_upload(
    data: pd.DataFrame, metadata: dict[str, Any]
) -> tuple[pd.DataFrame, pd.Series]:
    required = metadata["feature_columns"] + [metadata["target"]]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if data.empty:
        raise ValueError("The uploaded CSV is empty.")
    target = data[metadata["target"]].map(
        {metadata["negative_label"]: 0, metadata["positive_label"]: 1}
    )
    if target.isna().any():
        labels = f"'{metadata['negative_label']}' and '{metadata['positive_label']}'"
        raise ValueError(f"The target column must contain only {labels}.")
    return data[metadata["feature_columns"]].copy(), target.astype(int)


def plot_confusion_matrix(matrix: list[list[int]], model_name: str) -> go.Figure:
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=["Predicted No", "Predicted Yes"],
            y=["Actual No", "Actual Yes"],
            text=matrix,
            texttemplate="%{text}",
            colorscale=[[0, "#E6FFFB"], [1, ACCENT]],
            showscale=False,
        )
    )
    figure.update_layout(
        title=f"{model_name} confusion matrix",
        height=360,
        margin=dict(l=20, r=20, t=55, b=20),
        font=dict(color=INK),
    )
    return figure


def plot_roc_curves(
    curves: dict[str, tuple[Any, Any]], selected_key: str
) -> go.Figure:
    figure = go.Figure()
    for index, (key, (false_positive, true_positive)) in enumerate(curves.items()):
        selected = key == selected_key
        figure.add_trace(
            go.Scatter(
                x=false_positive,
                y=true_positive,
                mode="lines",
                name=MODEL_NAMES[key],
                line=dict(
                    color=SERIES_COLORS[index],
                    width=3 if selected else 1.5,
                ),
                opacity=1 if selected else 0.65,
            )
        )
    figure.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random",
            line=dict(color="#CBD5E1", dash="dash"),
        )
    )
    figure.update_layout(
        title="ROC curves",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        height=400,
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation="h", y=-0.2),
    )
    return figure


def render_metric_cards(metrics: dict[str, float]) -> None:
    columns = st.columns(len(METRIC_KEYS))
    for column, key in zip(columns, METRIC_KEYS):
        column.metric(METRIC_LABELS[key], f"{metrics[key]:.4f}")
        column.caption(METRIC_HINTS[key])


def main() -> None:
    st.set_page_config(
        page_title="EV Adoption Classifier",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="hero"><div class="eyebrow">Saved-model evaluation lab</div>'
        "<h1>⚡ EV adoption, made measurable.</h1>"
        "<p>Compare five pre-trained classifiers on labeled buyer data. "
        "Models are loaded from disk and never retrained in this dashboard.</p></div>",
        unsafe_allow_html=True,
    )

    try:
        metadata = load_metadata()
        saved_metrics = load_saved_metrics()
    except (OSError, json.JSONDecodeError, KeyError) as error:
        st.error(f"Model metadata could not be loaded: {error}")
        st.stop()

    with st.sidebar:
        st.markdown("## Evaluation setup")
        model_key = st.selectbox(
            "Choose a classifier",
            list(MODEL_NAMES),
            format_func=lambda key: MODEL_NAMES[key],
        )
        upload = st.file_uploader(
            "Upload labeled test CSV",
            type=["csv"],
            help="The file must include all input features and Will_Buy_EV.",
        )
        if SAMPLE_PATH.exists():
            st.download_button(
                "Download sample test data",
                SAMPLE_PATH.read_bytes(),
                file_name=SAMPLE_PATH.name,
                mime="text/csv",
            )
        st.caption(
            f"Bundled holdout: {metadata['test_rows']:,} rows · "
            f"random seed: {metadata['random_state']}"
        )

    st.markdown("### Choose a model")
    model_columns = st.columns(len(MODEL_NAMES))
    for column, key in zip(model_columns, MODEL_NAMES):
        baseline = saved_metrics.get(key, {}).get("metrics", {})
        with column:
            st.markdown(
                f'<div class="model-card"><h3>{html.escape(MODEL_NAMES[key])}</h3>'
                f"<p>{html.escape(MODEL_BLURBS[key])}</p>"
                f'<p><b>Holdout AUC:</b> {baseline.get("auc", 0):.3f}</p></div>',
                unsafe_allow_html=True,
            )

    if upload is None:
        st.info("Upload `test_data.csv` in the sidebar to begin evaluation.")
        return

    try:
        uploaded_data = pd.read_csv(io.BytesIO(upload.getvalue()))
        features, target = validate_upload(uploaded_data, metadata)
        model = load_model(model_key)
        metrics, matrix, report, predictions, scores = calculate_metrics(
            model, features, target
        )
    except (ValueError, KeyError, OSError, TypeError) as error:
        st.error(f"Evaluation could not be completed: {error}")
        return

    st.markdown(f"### {MODEL_NAMES[model_key]} results")
    render_metric_cards(metrics)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            plot_confusion_matrix(matrix, MODEL_NAMES[model_key]),
            use_container_width=True,
        )
    with right:
        st.markdown("#### Classification report")
        st.dataframe(
            pd.DataFrame(report).transpose().round(4),
            use_container_width=True,
        )

    curves: dict[str, tuple[Any, Any]] = {}
    for key in MODEL_NAMES:
        try:
            candidate = model if key == model_key else load_model(key)
            candidate_scores = positive_scores(candidate, features)
            false_positive, true_positive, _ = roc_curve(target, candidate_scores)
            curves[key] = (false_positive, true_positive)
        except (OSError, ValueError):
            continue
    if curves:
        st.plotly_chart(
            plot_roc_curves(curves, model_key), use_container_width=True
        )

    result = uploaded_data.copy()
    result["Predicted_Will_Buy_EV"] = [
        metadata["positive_label"] if value else metadata["negative_label"]
        for value in predictions
    ]
    result["Positive_Class_Score"] = scores
    st.download_button(
        "Download predictions",
        result.to_csv(index=False).encode("utf-8"),
        file_name=f"{model_key}_predictions.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
