"""Train, evaluate, and persist the five required EV adoption classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET = "Will_Buy_EV"
POSITIVE_LABEL = "Yes"
NEGATIVE_LABEL = "No"
DROP_COLUMNS = ["Buyer_ID"]
MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "K-Nearest Neighbors",
    "gaussian_nb": "Gaussian Naive Bayes",
    "svm": "Support Vector Machine",
}


def build_preprocessor(
    numeric_columns: list[str], categorical_columns: list[str]
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def build_searches(
    numeric_columns: list[str], categorical_columns: list[str]
) -> dict[str, GridSearchCV]:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    common = build_preprocessor(numeric_columns, categorical_columns)
    definitions: dict[str, tuple[Any, dict[str, list[Any]]]] = {
        "logistic_regression": (
            LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {"classifier__C": [0.1, 1.0, 10.0]},
        ),
        "decision_tree": (
            DecisionTreeClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {
                "classifier__max_depth": [None, 5, 10],
                "classifier__min_samples_leaf": [1, 5],
            },
        ),
        "knn": (
            KNeighborsClassifier(),
            {
                "classifier__n_neighbors": [5, 11, 21],
                "classifier__weights": ["uniform", "distance"],
            },
        ),
        "gaussian_nb": (
            GaussianNB(),
            {"classifier__var_smoothing": [1e-10, 1e-9, 1e-8]},
        ),
        "svm": (
            SVC(
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {
                "classifier__C": [0.5, 1.0, 2.0],
                "classifier__kernel": ["rbf", "linear"],
            },
        ),
    }
    searches: dict[str, GridSearchCV] = {}
    for key, (classifier, parameters) in definitions.items():
        pipeline = Pipeline(
            [
                ("preprocessor", common),
                ("classifier", classifier),
            ]
        )
        searches[key] = GridSearchCV(
            pipeline,
            parameters,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
    return searches


def positive_scores(model: Any, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        classes = list(model.classes_)
        return probabilities[:, classes.index(1)]
    scores = model.decision_function(features)
    return np.asarray(scores, dtype=float)


def evaluate_model(
    model: Any, features: pd.DataFrame, target: pd.Series
) -> tuple[dict[str, float], list[list[int]], dict[str, dict[str, float]]]:
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
    report = {
        key: {
            metric: float(value)
            for metric, value in values.items()
            if metric in {"precision", "recall", "f1-score", "support"}
        }
        for key, values in classification_report(
            target,
            predictions,
            labels=[0, 1],
            target_names=["No", "Yes"],
            output_dict=True,
            zero_division=0,
        ).items()
        if key in {"No", "Yes", "macro avg", "weighted avg"}
    }
    return metrics, matrix, report


def train_and_save(
    data_path: Path, output_dir: Path, test_data_path: Path
) -> dict[str, Any]:
    data = pd.read_csv(data_path)
    required = set([TARGET, *DROP_COLUMNS])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if data[TARGET].isna().any():
        raise ValueError("The target column contains missing values.")

    data[TARGET] = data[TARGET].map({NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1})
    if data[TARGET].isna().any():
        raise ValueError("Target values must be exactly 'Yes' or 'No'.")

    feature_columns = [c for c in data.columns if c not in [TARGET, *DROP_COLUMNS]]
    numeric_columns = data[feature_columns].select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [c for c in feature_columns if c not in numeric_columns]
    x_train, x_test, y_train, y_test = train_test_split(
        data[feature_columns],
        data[TARGET].astype(int),
        test_size=0.2,
        stratify=data[TARGET],
        random_state=RANDOM_STATE,
    )
    test_indices = x_test.index

    output_dir.mkdir(parents=True, exist_ok=True)
    searches = build_searches(numeric_columns, categorical_columns)
    metrics_by_model: dict[str, Any] = {}
    for key, search in searches.items():
        search.fit(x_train, y_train)
        metrics, matrix, report = evaluate_model(search.best_estimator_, x_test, y_test)
        joblib.dump(search.best_estimator_, output_dir / f"{key}.joblib")
        metrics_by_model[key] = {
            "display_name": MODEL_NAMES[key],
            "metrics": metrics,
            "confusion_matrix": matrix,
            "classification_report": report,
            "best_params": search.best_params_,
            "cv_best_auc": float(search.best_score_),
        }

    original_test = data.loc[test_indices].copy()
    original_test[TARGET] = original_test[TARGET].map({0: NEGATIVE_LABEL, 1: POSITIVE_LABEL})
    original_test.to_csv(test_data_path, index=False)
    metadata = {
        "target": TARGET,
        "positive_label": POSITIVE_LABEL,
        "negative_label": NEGATIVE_LABEL,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "dropped_columns": DROP_COLUMNS,
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "target_distribution": {
            "No": int((data[TARGET] == 0).sum()),
            "Yes": int((data[TARGET] == 1).sum()),
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_by_model, indent=2), encoding="utf-8"
    )
    return metrics_by_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("EV_Adoption_and_Range_Anxiety_Dataset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("model"))
    parser.add_argument("--test-data", type=Path, default=Path("test_data.csv"))
    args = parser.parse_args()
    results = train_and_save(args.data, args.output_dir, args.test_data)
    print(json.dumps({key: value["metrics"] for key, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
