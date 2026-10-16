"""ID3 Decision Tree module for placement prediction using Entropy and Information Gain."""

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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from ml.data_loader import load_data
from ml.preprocessing import create_preprocessor, get_model_features


IS_IMPLEMENTED = True
CHARTS_DIR = Path(__file__).resolve().parent.parent / "static" / "charts"


def _compute_entropy(target_series: pd.Series) -> float:
    """Compute Shannon Entropy: H(S) = -sum(p * log2(p))."""
    counts = target_series.value_counts(normalize=True)
    return float(-sum(p * np.log2(p) for p in counts if p > 0))


def _save_tree_diagram(tree_model, feature_names) -> str:
    """Save graphical decision tree layout up to depth 3 for readability."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 7.5), dpi=150)
    plot_tree(
        tree_model,
        max_depth=3,
        feature_names=feature_names,
        class_names=["Not Placed", "Placed"],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
        proportion=True,
        precision=2,
    )
    ax.set_title("ID3 Decision Tree Structure (Criterion: Entropy / Information Gain)", fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    chart_path = "charts/id3_tree_structure.png"
    fig.savefig(CHARTS_DIR / "id3_tree_structure.png", bbox_inches="tight")
    plt.close(fig)
    return chart_path


def _save_confusion_matrix(actual, predicted) -> str:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(actual, predicted)
    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=150)
    img = ax.imshow(matrix, cmap="Purples")
    ax.set_title("ID3 Decision Tree Confusion Matrix", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_xticks([0, 1], ["Not placed", "Placed"])
    ax.set_yticks([0, 1], ["Not placed", "Placed"])
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{matrix[row, col]:,}", ha="center", va="center", fontsize=12, color="white" if matrix[row, col] > matrix.max()/2 else "black")
    fig.colorbar(img, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    chart_path = "charts/id3_confusion_matrix.png"
    fig.savefig(CHARTS_DIR / "id3_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)
    return chart_path


def _save_feature_importance(feature_names, importances) -> str:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    df_imp = df_imp.sort_values(by="importance", ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.barh(df_imp["feature"], df_imp["importance"], color="#5d4fa8", edgecolor="none")
    ax.set_title("Top Features by Information Gain (Entropy Reduction)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Information Gain / Relative Importance")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    chart_path = "charts/id3_feature_importance.png"
    fig.savefig(CHARTS_DIR / "id3_feature_importance.png", bbox_inches="tight")
    plt.close(fig)
    return chart_path


@lru_cache(maxsize=1)
def run_id3_decision_tree() -> dict:
    """Train and evaluate an ID3 (Entropy criterion) Decision Tree."""
    data = load_data()
    features = get_model_features(data)
    target = data["PlacementStatus"]

    root_entropy = _compute_entropy(target)

    train_features, test_features, train_target, test_target = train_test_split(
        features, target, test_size=0.20, random_state=42, stratify=target
    )

    preprocessor = create_preprocessor(features, scaler_type="standard")
    X_train = preprocessor.fit_transform(train_features)
    X_test = preprocessor.transform(test_features)
    feature_names = [f.replace("numeric__", "").replace("categorical__", "") for f in preprocessor.get_feature_names_out()]

    # ID3 uses Entropy / Information Gain
    tree = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=5,
        min_samples_leaf=15,
        random_state=42
    )
    tree.fit(X_train, train_target)

    predictions = tree.predict(X_test)
    probabilities = tree.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(test_target, predictions)
    prec = precision_score(test_target, predictions)
    rec = recall_score(test_target, predictions)
    f1 = f1_score(test_target, predictions)
    auc = roc_auc_score(test_target, probabilities)

    # Charts
    tree_chart = _save_tree_diagram(tree, feature_names)
    conf_chart = _save_confusion_matrix(test_target, predictions)
    imp_chart = _save_feature_importance(feature_names, tree.feature_importances_)

    # Extract top human-readable rules
    raw_rules = export_text(tree, feature_names=feature_names, max_depth=2).split("\n")
    sample_rules = [r for r in raw_rules if r.strip()][:12]

    # Top features
    imp_series = pd.Series(tree.feature_importances_, index=feature_names).sort_values(ascending=False).head(8)
    top_features = [{"feature": f, "gain": round(float(v), 4)} for f, v in imp_series.items()]

    return {
        "train_rows": len(train_features),
        "test_rows": len(test_features),
        "tree_depth": tree.get_depth(),
        "leaf_count": tree.get_n_leaves(),
        "root_entropy": round(root_entropy, 4),
        "accuracy": round(float(acc) * 100, 2),
        "precision": round(float(prec) * 100, 2),
        "recall": round(float(rec) * 100, 2),
        "f1": round(float(f1) * 100, 2),
        "auc": round(float(auc), 3),
        "tree_chart": tree_chart,
        "conf_chart": conf_chart,
        "imp_chart": imp_chart,
        "sample_rules": sample_rules,
        "top_features": top_features,
    }
