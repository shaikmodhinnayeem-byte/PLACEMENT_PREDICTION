"""Leakage-safe feature preparation shared by the model modules."""

from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from ml.data_loader import load_data


IS_IMPLEMENTED = True
EXCLUDED_FEATURES = ["StudentID", "PlacementStatus", "Salary Package", "IsAnomaly"]


def get_model_features(data: pd.DataFrame) -> pd.DataFrame:
    """Remove IDs, targets, and planted anomalies from the feature set."""
    return data.drop(columns=EXCLUDED_FEATURES, errors="ignore").copy()


def get_feature_groups(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    numerical = features.select_dtypes(include="number").columns.tolist()
    categorical = features.select_dtypes(exclude="number").columns.tolist()
    return numerical, categorical


def create_preprocessor(features: pd.DataFrame, scaler_type: str = "standard") -> ColumnTransformer:
    """Build a transformer that imputes, encodes, and scales without leakage.

    Parameters:
        features: DataFrame containing input features.
        scaler_type: 'standard' for StandardScaler (Z-score) or 'minmax' for MinMaxScaler ([0, 1]).
    """
    numerical, categorical = get_feature_groups(features)
    scaler = MinMaxScaler(feature_range=(0, 1)) if scaler_type.lower() == "minmax" else StandardScaler()

    numerical_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", scaler)]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numerical_pipeline, numerical), ("categorical", categorical_pipeline, categorical)],
        remainder="drop",
    )


@lru_cache(maxsize=1)
def get_preprocessing_summary() -> dict:
    """Return the preprocessing plan and resulting train/test feature dimensions."""
    data = load_data()
    features = get_model_features(data)
    numerical, categorical = get_feature_groups(features)
    target = data["PlacementStatus"]
    train_features, test_features, train_target, test_target = train_test_split(
        features, target, test_size=0.20, random_state=42, stratify=target
    )

    # Standard preprocessor
    standard_preprocessor = create_preprocessor(features, scaler_type="standard")
    transformed_train = standard_preprocessor.fit_transform(train_features)
    transformed_test = standard_preprocessor.transform(test_features)

    # Min-Max preprocessor
    minmax_preprocessor = create_preprocessor(features, scaler_type="minmax")
    minmax_train = minmax_preprocessor.fit_transform(train_features)

    missing = features.isna().sum().sort_values(ascending=False)

    # Min-Max Normalization feature sample comparison (select key numerical columns)
    sample_cols = ["CGPA", "AptitudeTestScore", "CodingTestScore", "SoftSkillsRating", "AttendancePercent"]
    sample_cols = [c for c in sample_cols if c in numerical]

    minmax_comparisons = []
    for col in sample_cols:
        orig = features[col].dropna()
        col_min = float(orig.min())
        col_max = float(orig.max())
        col_mean = float(orig.mean())
        # Normalized formula: (X - min) / (max - min)
        norm_mean = (col_mean - col_min) / (col_max - col_min) if col_max > col_min else 0.0
        minmax_comparisons.append({
            "feature": col,
            "orig_min": round(col_min, 2),
            "orig_max": round(col_max, 2),
            "orig_mean": round(col_mean, 2),
            "norm_min": 0.0,
            "norm_max": 1.0,
            "norm_mean": round(norm_mean, 3),
        })

    return {
        "source_rows": len(data),
        "feature_count": len(features.columns),
        "numerical_count": len(numerical),
        "categorical_count": len(categorical),
        "train_rows": len(train_features),
        "test_rows": len(test_features),
        "transformed_feature_count": int(transformed_train.shape[1]),
        "train_shape": f"{transformed_train.shape[0]:,} × {transformed_train.shape[1]:,}",
        "test_shape": f"{transformed_test.shape[0]:,} × {transformed_test.shape[1]:,}",
        "missing_columns": [
            {"name": column, "missing": int(count)} for column, count in missing.items() if count
        ],
        "numerical_columns": numerical,
        "categorical_columns": categorical,
        "excluded_columns": EXCLUDED_FEATURES,
        "placed_train": int(train_target.sum()),
        "placed_test": int(test_target.sum()),
        "minmax_comparisons": minmax_comparisons,
    }
