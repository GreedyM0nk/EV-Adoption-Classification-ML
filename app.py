"""Streamlit dashboard for evaluating the saved EV adoption classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
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
)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
METRIC_KEYS = ["accuracy", "auc", "precision", "recall", "f1", "mcc"]
MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "K-Nearest Neighbors",
    "gaussian_nb": "Gaussian Naive Bayes",
    "svm": "Support Vector Machine",
}


@st.cache_data
def load_metadata() -> dict:
    return json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))


@st.cache_resource
def load_model(model_key: str):
    return joblib.load(MODEL_DIR / f"{model_key}.joblib")


def calculate_metrics(model, features: pd.DataFrame, target: pd.Series) -> tuple[dict, list[list[int]], dict]:
    predictions = model.predict(features)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        positive_index = list(model.classes_).index(1)
        scores = probabilities[:, positive_index]
    else:
        scores = model.decision_function(features)
    metrics = {
        "accuracy": accuracy_score(target, predictions),
        "auc": roc_auc_score(target, scores),
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1": f1_score(target, predictions, zero_division=0),
        "mcc": matthews_corrcoef(target, predictions),
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
    return metrics, matrix, report


def validate_upload(data: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame, pd.Series]:
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
        raise ValueError("The target column must contain only 'Yes' and 'No'.")
    features = data[metadata["feature_columns"]].copy()
    return features, target.astype(int)


def render_confusion_matrix(matrix: list[list[int]], model_name: str) -> None:
    figure, axis = plt.subplots(figsize=(4.5, 3.5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        xlabel="Predicted label",
        ylabel="Actual label",
        title=f"{model_name} confusion matrix",
    )
    for row in range(2):
        for column in range(2):
            axis.text(column, row, matrix[row][column], ha="center", va="center")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    st.pyplot(figure, clear_figure=True)


def main() -> None:
    st.set_page_config(page_title="EV Adoption Classifier", page_icon="⚡", layout="wide")
    st.title("⚡ EV Adoption Classifier")
    st.write(
        "Evaluate whether buyers will purchase an electric vehicle using one of five "
        "pre-trained classifiers. Upload labeled test data to compare predictions with "
        "the known outcome; the app never retrains models."
    )

    try:
        metadata = load_metadata()
    except (OSError, json.JSONDecodeError) as error:
        st.error(f"Model metadata could not be loaded: {error}")
        st.stop()

    with st.sidebar:
        st.header("Evaluation setup")
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
        st.caption(
            f"Expected: {metadata['test_rows']} rows in the bundled holdout; "
            f"fixed random seed {metadata['random_state']}."
        )

    if upload is None:
        st.info("Upload `test_data.csv` to begin evaluation.")
        return

    try:
        uploaded_data = pd.read_csv(upload)
        features, target = validate_upload(uploaded_data, metadata)
        model = load_model(model_key)
        metrics, matrix, report = calculate_metrics(model, features, target)
    except (ValueError, KeyError, OSError) as error:
        st.error(f"Evaluation could not be completed: {error}")
        return

    st.subheader(f"{MODEL_NAMES[model_key]} results")
    columns = st.columns(6)
    labels = {
        "accuracy": "Accuracy",
        "auc": "AUC",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "mcc": "MCC",
    }
    for column, key in zip(columns, METRIC_KEYS):
        column.metric(labels[key], f"{metrics[key]:.4f}")

    left, right = st.columns(2)
    with left:
        render_confusion_matrix(matrix, MODEL_NAMES[model_key])
    with right:
        st.subheader("Classification report")
        report_frame = pd.DataFrame(report).transpose()
        st.dataframe(report_frame.round(4), use_container_width=True)

    predictions = model.predict(features)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(features)[:, list(model.classes_).index(1)]
    else:
        scores = model.decision_function(features)
    result = uploaded_data.copy()
    result["Predicted_Will_Buy_EV"] = ["Yes" if value else "No" for value in predictions]
    result["Positive_Class_Score"] = scores
    st.download_button(
        "Download predictions",
        result.to_csv(index=False).encode("utf-8"),
        file_name=f"{model_key}_predictions.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
