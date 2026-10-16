"""Salary prediction for placed students using linear regression with comprehensive accuracy metrics."""

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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.data_loader import load_data
from ml.preprocessing import create_preprocessor, get_model_features


IS_IMPLEMENTED = True
CHARTS_DIR = Path(__file__).resolve().parent.parent / "static" / "charts"


def _save_actual_vs_predicted(actual, predicted) -> str:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(actual, predicted, alpha=0.22, s=14, color="#2764d7", edgecolors="none", label="Predicted students")
    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())
    
    # 45-degree perfect prediction line
    axis.plot([lower, upper], [lower, upper], linestyle="-", color="#111827", linewidth=1.5, label="Perfect prediction")
    
    # +/- 15% tolerance margin bands
    line_x = np.linspace(lower, upper, 100)
    axis.plot(line_x, line_x * 1.15, linestyle="--", color="#c93b4a", alpha=0.7, label="+15% Margin")
    axis.plot(line_x, line_x * 0.85, linestyle="--", color="#c93b4a", alpha=0.7, label="-15% Margin")
    axis.fill_between(line_x, line_x * 0.85, line_x * 1.15, color="#2764d7", alpha=0.06, label="Accurate zone (±15%)")

    axis.set_title("Actual vs Predicted Salary with ±15% Accuracy Tolerance Band", fontsize=11, fontweight="bold")
    axis.set_xlabel("Actual Salary Package (LPA)")
    axis.set_ylabel("Predicted Salary Package (LPA)")
    axis.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=8)
    axis.grid(True, linestyle=":", alpha=0.4)
    figure.tight_layout()
    figure.savefig(CHARTS_DIR / "linear_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return "charts/linear_actual_vs_predicted.png"


@lru_cache(maxsize=1)
def run_linear_regression() -> dict:
    """Fit and evaluate a salary model with accuracy and error metrics."""
    data = load_data()
    placed_data = data.loc[data["PlacementStatus"].eq(1)].copy()
    features = get_model_features(placed_data)
    target = placed_data["Salary Package"]
    train_features, test_features, train_target, test_target = train_test_split(
        features, target, test_size=0.20, random_state=42
    )

    pipeline = Pipeline(
        [("preprocessor", create_preprocessor(features, scaler_type="standard")), ("model", LinearRegression())]
    )
    pipeline.fit(train_features, train_target)
    predictions = pipeline.predict(test_features)

    # Standard error metrics
    r2 = float(r2_score(test_target, predictions))
    mae = float(mean_absolute_error(test_target, predictions))
    rmse = float(root_mean_squared_error(test_target, predictions))

    # Regression Accuracy Metrics
    # 1. Tolerance Accuracies: predictions within percentage bounds of actual value
    actual_arr = np.array(test_target)
    pred_arr = np.array(predictions)
    abs_errors = np.abs(actual_arr - pred_arr)
    
    tol_10_acc = float(np.mean(abs_errors <= (0.10 * actual_arr)) * 100)
    tol_15_acc = float(np.mean(abs_errors <= (0.15 * actual_arr)) * 100)
    tol_20_acc = float(np.mean(abs_errors <= (0.20 * actual_arr)) * 100)
    tol_1lpa_acc = float(np.mean(abs_errors <= 1.0) * 100)

    # 2. Mean Absolute Percentage Error (MAPE) and Derived Accuracy
    non_zero_mask = actual_arr > 0
    mape = float(np.mean(abs_errors[non_zero_mask] / actual_arr[non_zero_mask]) * 100)
    mape_accuracy = max(0.0, 100.0 - mape)

    # Feature coefficients
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pd.DataFrame(
        {"feature": feature_names, "coefficient": pipeline.named_steps["model"].coef_}
    )
    coefficients["feature"] = (
        coefficients["feature"].str.replace("numeric__", "", regex=False).str.replace(
            "categorical__", "", regex=False
        )
    )
    top_coefficients = coefficients.reindex(
        coefficients["coefficient"].abs().sort_values(ascending=False).index
    ).head(12)

    return {
        "placed_rows": len(placed_data),
        "train_rows": len(train_features),
        "test_rows": len(test_features),
        "r2": round(r2, 3),
        "r2_percent": round(max(0.0, r2 * 100), 1),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "mape": round(mape, 2),
        "mape_accuracy": round(mape_accuracy, 2),
        "tol_10_acc": round(tol_10_acc, 2),
        "tol_15_acc": round(tol_15_acc, 2),
        "tol_20_acc": round(tol_20_acc, 2),
        "tol_1lpa_acc": round(tol_1lpa_acc, 2),
        "actual_mean": round(float(test_target.mean()), 2),
        "predicted_mean": round(float(predictions.mean()), 2),
        "chart": _save_actual_vs_predicted(test_target, predictions),
        "coefficients": [
            {"feature": row.feature, "coefficient": round(float(row.coefficient), 3)}
            for row in top_coefficients.itertuples(index=False)
        ],
    }
