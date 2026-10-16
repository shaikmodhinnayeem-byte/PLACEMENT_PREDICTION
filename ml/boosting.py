"""Boosting Algorithms module comparing AdaBoost and Gradient Boosting for placement prediction."""

from functools import lru_cache
import os
from pathlib import Path
import tempfile

MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "placement-dashboard-mpl"
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from ml.data_loader import load_data
from ml.preprocessing import create_preprocessor, get_model_features


IS_IMPLEMENTED = True
CHARTS_DIR = Path(__file__).resolve().parent.parent / "static" / "charts"


def _save_roc_curves(y_test, ada_probs, gb_probs, ada_auc, gb_auc) -> str:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fpr_ada, tpr_ada, _ = roc_curve(y_test, ada_probs)
    fpr_gb, tpr_gb, _ = roc_curve(y_test, gb_probs)

    fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=150)
    ax.plot(fpr_gb, tpr_gb, color="#1b9b6e", linewidth=2, label=f"Gradient Boosting (AUC = {gb_auc:.3f})")
    ax.plot(fpr_ada, tpr_ada, color="#2764d7", linewidth=2, linestyle="--", label=f"AdaBoost (AUC = {ada_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle=":", color="#888888", label="Random Chance")
    ax.set_title("Comparative ROC Curves: AdaBoost vs Gradient Boosting", fontsize=11, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    chart_path = "charts/boosting_roc_comparison.png"
    fig.savefig(CHARTS_DIR / "boosting_roc_comparison.png", bbox_inches="tight")
    plt.close(fig)
    return chart_path


def _save_staged_curves(y_test, ada_staged, gb_staged) -> str:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=150)
    stages = range(1, len(ada_staged) + 1)
    ax.plot(stages, gb_staged, color="#1b9b6e", linewidth=2, label="Gradient Boosting Test Acc")
    ax.plot(stages, ada_staged, color="#2764d7", linewidth=2, linestyle="--", label="AdaBoost Test Acc")
    ax.set_title("Boosting Progression: Accuracy vs Iterations / Estimators", fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of Estimators / Stages")
    ax.set_ylabel("Test Accuracy")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    chart_path = "charts/boosting_learning_curve.png"
    fig.savefig(CHARTS_DIR / "boosting_learning_curve.png", bbox_inches="tight")
    plt.close(fig)
    return chart_path


def _save_feature_importance_comparison(feature_names, ada_imp, gb_imp) -> str:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"feature": feature_names, "ada": ada_imp, "gb": gb_imp})
    df["max_imp"] = df[["ada", "gb"]].max(axis=1)
    df = df.sort_values(by="max_imp", ascending=True).tail(8)

    y = np.arange(len(df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=150)
    ax.barh(y - width/2, df["gb"], width, label="Gradient Boosting", color="#1b9b6e")
    ax.barh(y + width/2, df["ada"], width, label="AdaBoost", color="#2764d7")
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"])
    ax.set_title("Feature Importance: AdaBoost vs Gradient Boosting", fontsize=11, fontweight="bold")
    ax.set_xlabel("Relative Importance")
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    chart_path = "charts/boosting_feature_importance.png"
    fig.savefig(CHARTS_DIR / "boosting_feature_importance.png", bbox_inches="tight")
    plt.close(fig)
    return chart_path


@lru_cache(maxsize=1)
def run_boosting() -> dict:
    """Train and compare AdaBoost and Gradient Boosting models."""
    data = load_data()
    features = get_model_features(data)
    target = data["PlacementStatus"]

    train_features, test_features, train_target, test_target = train_test_split(
        features, target, test_size=0.20, random_state=42, stratify=target
    )

    preprocessor = create_preprocessor(features, scaler_type="standard")
    X_train = preprocessor.fit_transform(train_features)
    X_test = preprocessor.transform(test_features)
    feature_names = [f.replace("numeric__", "").replace("categorical__", "") for f in preprocessor.get_feature_names_out()]

    # 1. AdaBoost Classifier
    adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)
    adaboost.fit(X_train, train_target)
    ada_pred = adaboost.predict(X_test)
    ada_prob = adaboost.predict_proba(X_test)[:, 1]

    ada_staged = [accuracy_score(test_target, p) for p in adaboost.staged_predict(X_test)]

    # 2. Gradient Boosting Classifier
    gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
    gb.fit(X_train, train_target)
    gb_pred = gb.predict(X_test)
    gb_prob = gb.predict_proba(X_test)[:, 1]

    gb_staged = [accuracy_score(test_target, p) for p in gb.staged_predict(X_test)]

    # Metrics
    ada_acc = accuracy_score(test_target, ada_pred)
    ada_prec = precision_score(test_target, ada_pred)
    ada_rec = recall_score(test_target, ada_pred)
    ada_f1 = f1_score(test_target, ada_pred)
    ada_auc = roc_auc_score(test_target, ada_prob)

    gb_acc = accuracy_score(test_target, gb_pred)
    gb_prec = precision_score(test_target, gb_pred)
    gb_rec = recall_score(test_target, gb_pred)
    gb_f1 = f1_score(test_target, gb_pred)
    gb_auc = roc_auc_score(test_target, gb_prob)

    # Charts
    roc_chart = _save_roc_curves(test_target, ada_prob, gb_prob, ada_auc, gb_auc)
    staged_chart = _save_staged_curves(test_target, ada_staged, gb_staged)
    imp_chart = _save_feature_importance_comparison(feature_names, adaboost.feature_importances_, gb.feature_importances_)

    return {
        "train_rows": len(train_features),
        "test_rows": len(test_features),
        "adaboost": {
            "name": "AdaBoost (Adaptive Boosting)",
            "accuracy": round(float(ada_acc) * 100, 2),
            "precision": round(float(ada_prec) * 100, 2),
            "recall": round(float(ada_rec) * 100, 2),
            "f1": round(float(ada_f1) * 100, 2),
            "auc": round(float(ada_auc), 3),
            "n_estimators": 50,
            "mechanism": "Sequential weight updates targeting misclassified instances.",
        },
        "gradient_boosting": {
            "name": "Gradient Boosting (GBM)",
            "accuracy": round(float(gb_acc) * 100, 2),
            "precision": round(float(gb_prec) * 100, 2),
            "recall": round(float(gb_rec) * 100, 2),
            "f1": round(float(gb_f1) * 100, 2),
            "auc": round(float(gb_auc), 3),
            "n_estimators": 50,
            "learning_rate": 0.1,
            "mechanism": "Sequential tree fitting on negative gradients (pseudo-residuals).",
        },
        "roc_chart": roc_chart,
        "staged_chart": staged_chart,
        "imp_chart": imp_chart,
    }
