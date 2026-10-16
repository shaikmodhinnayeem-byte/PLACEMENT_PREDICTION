"""Ensemble Learning module comparing Bagging (Random Forest), Voting, and Stacking classifiers."""

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
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ml.data_loader import load_data
from ml.preprocessing import create_preprocessor, get_model_features


IS_IMPLEMENTED = True
CHARTS_DIR = Path(__file__).resolve().parent.parent / "static" / "charts"


def _save_ensemble_comparison_chart(model_names, accuracies, aucs) -> str:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    bars1 = ax.bar(x - width/2, accuracies, width, label="Test Accuracy (%)", color="#2764d7")
    bars2 = ax.bar(x + width/2, [a * 100 for a in aucs], width, label="ROC-AUC (×100)", color="#1b9b6e")

    ax.set_ylabel("Score (%)")
    ax.set_title("Model Benchmarking: Individual Models vs Ensemble Paradigms", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim([max(0, min(accuracies) - 10), 100])
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.6, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    chart_path = "charts/ensemble_comparison.png"
    fig.savefig(CHARTS_DIR / "ensemble_comparison.png", bbox_inches="tight")
    plt.close(fig)
    return chart_path


def _save_agreement_matrix(model_names, predictions_dict) -> str:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    n = len(model_names)
    matrix = np.zeros((n, n))
    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            agreement = np.mean(predictions_dict[name1] == predictions_dict[name2]) * 100
            matrix[i, j] = agreement

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    img = ax.imshow(matrix, cmap="Blues", vmin=70, vmax=100)
    ax.set_title("Prediction Agreement Matrix (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(model_names, rotation=25, ha="right")
    ax.set_yticklabels(model_names)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i, j]:.1f}%", ha="center", va="center", fontsize=9,
                    color="white" if matrix[i, j] > 88 else "black")

    fig.colorbar(img, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    chart_path = "charts/ensemble_agreement.png"
    fig.savefig(CHARTS_DIR / "ensemble_agreement.png", bbox_inches="tight")
    plt.close(fig)
    return chart_path


@lru_cache(maxsize=1)
def run_ensemble_learning() -> dict:
    """Train and benchmark Bagging, Voting, and Stacking against single model baselines."""
    data = load_data()
    features = get_model_features(data)
    target = data["PlacementStatus"]

    train_features, test_features, train_target, test_target = train_test_split(
        features, target, test_size=0.20, random_state=42, stratify=target
    )

    preprocessor = create_preprocessor(features, scaler_type="standard")
    X_train = preprocessor.fit_transform(train_features)
    X_test = preprocessor.transform(test_features)

    # Base estimators
    lr_base = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
    dt_base = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
    gb_base = GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=42)

    # 1. Random Forest (Bagging Ensemble)
    rf = RandomForestClassifier(n_estimators=60, max_depth=8, random_state=42, n_jobs=-1)

    # 2. Voting Classifier (Soft Voting Ensemble)
    voting = VotingClassifier(
        estimators=[("lr", lr_base), ("dt", dt_base), ("gb", gb_base)],
        voting="soft",
    )

    # 3. Stacking Classifier (Stacking Ensemble)
    stacking = StackingClassifier(
        estimators=[("rf", RandomForestClassifier(n_estimators=30, max_depth=6, random_state=42)), ("gb", gb_base)],
        final_estimator=LogisticRegression(),
        cv=3,
    )

    models = {
        "Logistic Reg (Base)": lr_base,
        "Decision Tree (Base)": dt_base,
        "Random Forest (Bagging)": rf,
        "Voting Ensemble": voting,
        "Stacking Ensemble": stacking,
    }

    results = []
    predictions_dict = {}
    model_names = list(models.keys())
    accuracies = []
    aucs = []

    for name, model in models.items():
        model.fit(X_train, train_target)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
        predictions_dict[name] = preds

        acc = accuracy_score(test_target, preds) * 100
        prec = precision_score(test_target, preds) * 100
        rec = recall_score(test_target, preds) * 100
        f1 = f1_score(test_target, preds) * 100
        auc = roc_auc_score(test_target, probs)

        accuracies.append(round(acc, 2))
        aucs.append(round(auc, 3))

        results.append({
            "name": name,
            "category": "Ensemble" if "Ensemble" in name or "Random Forest" in name else "Individual",
            "accuracy": round(acc, 2),
            "precision": round(prec, 2),
            "recall": round(rec, 2),
            "f1": round(f1, 2),
            "auc": round(auc, 3),
        })

    comp_chart = _save_ensemble_comparison_chart(model_names, accuracies, aucs)
    agree_chart = _save_agreement_matrix(model_names, predictions_dict)

    best_model = max(results, key=lambda r: r["accuracy"])
    best_ensemble = max([m for m in results if m["category"] == "Ensemble"], key=lambda r: r["accuracy"])

    return {
        "train_rows": len(train_features),
        "test_rows": len(test_features),
        "models": results,
        "best_model": best_model,
        "best_ensemble": best_ensemble,
        "comp_chart": comp_chart,
        "agree_chart": agree_chart,
    }
