"""Logistic Regression module for placement prediction."""

import os
from pathlib import Path
import tempfile

# This lets Matplotlib create its temporary files without permission errors.
MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "placement-dashboard-mpl"
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.data_loader import load_data
from ml.preprocessing import create_preprocessor, get_model_features


IS_IMPLEMENTED = True
CHARTS_DIR = Path(__file__).resolve().parent.parent / "static" / "charts"


def save_logistic_charts(actual_values, predicted_values, predicted_probabilities):
    """Save simple charts used on the Logistic Regression page."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # Confusion matrix chart
    matrix = confusion_matrix(actual_values, predicted_values)
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title("Confusion matrix")
    axis.set_xlabel("Predicted placement status")
    axis.set_ylabel("Actual placement status")
    axis.set_xticks([0, 1], ["Not placed", "Placed"])
    axis.set_yticks([0, 1], ["Not placed", "Placed"])
    for row in range(2):
        for column in range(2):
            axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center", fontsize=12)
    figure.colorbar(image, ax=axis, fraction=0.045, pad=0.04)
    figure.tight_layout()
    figure.savefig(CHARTS_DIR / "logistic_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    # ROC curve chart
    false_positive_rate, true_positive_rate, _ = roc_curve(actual_values, predicted_probabilities)
    auc_score = roc_auc_score(actual_values, predicted_probabilities)
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    axis.plot(false_positive_rate, true_positive_rate, color="#2764d7", label=f"AUC = {auc_score:.3f}")
    axis.plot([0, 1], [0, 1], "--", color="#c93b4a", label="Random model")
    axis.set_title("ROC curve")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(CHARTS_DIR / "logistic_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    return matrix, "charts/logistic_confusion_matrix.png", "charts/logistic_roc_curve.png"


def run_logistic_regression():
    """Train a simple Logistic Regression model to predict placement."""
    data = load_data()

    # PlacementStatus is the target. The helper removes ID, target, salary, and anomaly columns.
    features = get_model_features(data)
    target = data["PlacementStatus"]

    # Split before preprocessing so the test set does not affect the training transformations.
    train_features, test_features, train_target, test_target = train_test_split(
        features, target, test_size=0.20, random_state=42, stratify=target
    )

    model_pipeline = Pipeline(
        [
            ("preprocessor", create_preprocessor(features)),
            ("model", LogisticRegression(max_iter=1000, solver="liblinear")),
        ]
    )
    model_pipeline.fit(train_features, train_target)

    predictions = model_pipeline.predict(test_features)
    probabilities = model_pipeline.predict_proba(test_features)[:, 1]
    matrix, confusion_chart, roc_chart = save_logistic_charts(
        test_target, predictions, probabilities
    )

    return {
        "source_rows": len(data),
        "train_rows": len(train_features),
        "test_rows": len(test_features),
        "accuracy": round(accuracy_score(test_target, predictions), 3),
        "precision": round(precision_score(test_target, predictions), 3),
        "recall": round(recall_score(test_target, predictions), 3),
        "auc": round(roc_auc_score(test_target, probabilities), 3),
        "true_negative": int(matrix[0, 0]),
        "false_positive": int(matrix[0, 1]),
        "false_negative": int(matrix[1, 0]),
        "true_positive": int(matrix[1, 1]),
        "confusion_chart": confusion_chart,
        "roc_chart": roc_chart,
    }
