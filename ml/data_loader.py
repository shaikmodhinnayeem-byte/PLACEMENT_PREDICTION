from pathlib import Path

import pandas as pd


IS_IMPLEMENTED = True
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "placement_data.csv"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the placement dataset from the standard project location."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Add placement_data.csv there or run generate_data.py for a synthetic fallback."
        )
    return pd.read_csv(path)


def get_data_summary(data: pd.DataFrame) -> dict:
    """Return the values displayed by the Data Loading page."""
    missing = data.isna().sum()
    columns = [
        {"name": column, "dtype": str(data[column].dtype), "missing": int(missing[column])}
        for column in data.columns
    ]
    return {
        "row_count": len(data),
        "column_count": len(data.columns),
        "columns": columns,
        "preview_columns": list(data.columns),
        "preview_rows": data.head(10).fillna("—").to_dict(orient="records"),
    }
