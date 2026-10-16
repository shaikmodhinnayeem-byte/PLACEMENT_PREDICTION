"""Exploratory analysis summaries and chart generation."""

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

from ml.data_loader import load_data


IS_IMPLEMENTED = True
CHARTS_DIR = Path(__file__).resolve().parent.parent / "static" / "charts"


def _save_figure(figure, filename: str) -> str:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(CHARTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return f"charts/{filename}"


@lru_cache(maxsize=1)
def get_eda_results() -> dict:
    """Create the EDA statistics and static charts for the dashboard."""
    data = load_data()
    numeric_columns = data.select_dtypes(include="number").columns.tolist()
    missing = data.isna().sum().sort_values(ascending=False)
    placement_counts = data["PlacementStatus"].value_counts().sort_index()
    placement_total = int(placement_counts.sum())

    summary_frame = data[numeric_columns].describe().T.round(2).reset_index()
    summary_frame = summary_frame.rename(columns={"index": "column"})
    correlations = data[numeric_columns].corr(numeric_only=True)
    placement_correlations = (
        correlations["PlacementStatus"]
        .drop(labels=["PlacementStatus"], errors="ignore")
        .sort_values(key=lambda values: values.abs(), ascending=False)
        .head(10)
        .round(3)
    )

    figure, axis = plt.subplots(figsize=(6, 4))
    labels = ["Not placed (0)", "Placed (1)"]
    values = [int(placement_counts.get(0, 0)), int(placement_counts.get(1, 0))]
    bars = axis.bar(labels, values, color=["#dc941d", "#1b9b6e"], width=0.55)
    axis.set_title("Placement class balance")
    axis.set_ylabel("Students")
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom")
    class_chart = _save_figure(figure, "placement_class_balance.png")

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(data["CGPA"].dropna(), bins=30, color="#2764d7", edgecolor="white")
    axis.set_title("CGPA distribution")
    axis.set_xlabel("CGPA")
    axis.set_ylabel("Students")
    cgpa_chart = _save_figure(figure, "cgpa_distribution.png")

    correlation_columns = [column for column in numeric_columns if column not in {"StudentID", "IsAnomaly"}]
    figure, axis = plt.subplots(figsize=(13, 10))
    image = axis.imshow(data[correlation_columns].corr(), cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_title("Numeric feature correlation matrix")
    axis.set_xticks(range(len(correlation_columns)), correlation_columns, rotation=90, fontsize=7)
    axis.set_yticks(range(len(correlation_columns)), correlation_columns, fontsize=7)
    figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    correlation_chart = _save_figure(figure, "correlation_matrix.png")

    return {
        "row_count": len(data),
        "column_count": len(data.columns),
        "total_missing": int(missing.sum()),
        "placed_count": int(placement_counts.get(1, 0)),
        "placed_percent": round(100 * placement_counts.get(1, 0) / placement_total, 1),
        "missing_columns": [
            {"name": column, "missing": int(count)} for column, count in missing.items() if count
        ],
        "summary_rows": summary_frame.to_dict(orient="records"),
        "placement_correlations": [
            {"feature": feature, "correlation": float(value)}
            for feature, value in placement_correlations.items()
        ],
        "charts": {
            "class_balance": class_chart,
            "cgpa_distribution": cgpa_chart,
            "correlation_matrix": correlation_chart,
        },
    }
