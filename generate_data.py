"""Create a small synthetic dataset for local dashboard testing only."""

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_PATH = Path(__file__).parent / "data" / "placement_data.csv"


def generate_data(rows: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    placement = rng.binomial(1, 0.66, rows)
    frame = pd.DataFrame({
        "StudentID": range(1, rows + 1),
        "Gender": rng.choice(["Female", "Male"], rows),
        "City": rng.choice(["Bengaluru", "Chennai", "Delhi", "Mumbai"], rows),
        "CollegeTier": rng.choice(["Tier 1", "Tier 2", "Tier 3"], rows),
        "Stream": rng.choice(["CSE", "ECE", "IT", "Mechanical"], rows),
        "Specialisation": rng.choice(["AI", "Data Science", "General", "Networks"], rows),
        "Hostel": rng.choice(["Yes", "No"], rows),
        "HistoryOfBacklogs": rng.choice(["Yes", "No"], rows, p=[0.18, 0.82]),
        **{f"SGPA_Sem{semester}": rng.uniform(5.5, 10, rows).round(2) for semester in range(1, 9)},
        "CGPA": rng.uniform(5.5, 10, rows).round(2),
        "AttendancePercent": rng.uniform(55, 100, rows).round(1),
        "Internships": rng.integers(0, 4, rows),
        "Projects": rng.integers(0, 6, rows),
        "Workshops": rng.integers(0, 6, rows).astype(float),
        "Certifications": rng.integers(0, 8, rows),
        "Publications": rng.integers(0, 3, rows),
        "AptitudeTestScore": rng.uniform(30, 100, rows).round(1),
        "SoftSkillsRating": rng.uniform(1, 10, rows).round(1),
        "CodingTestScore": rng.uniform(20, 100, rows).round(1),
        "MockInterviewScore": rng.uniform(1, 10, rows).round(1),
        "ExtraCurricular": rng.integers(0, 2, rows),
        "IsAnomaly": rng.choice([0, 1], rows, p=[0.98, 0.02]),
        "PlacementStatus": placement,
        "Salary Package": np.where(placement == 1, rng.uniform(3, 18, rows).round(2), 0.0),
    })
    frame["CGPA_Tier"] = pd.cut(frame["CGPA"], [0, 6.5, 8, 10], labels=["Low", "Mid", "High"])
    for column in ["AptitudeTestScore", "SoftSkillsRating", "CodingTestScore", "MockInterviewScore", "Workshops"]:
        frame.loc[rng.random(rows) < 0.08, column] = np.nan
    return frame


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    generate_data().to_csv(OUTPUT_PATH, index=False)
    print(f"Synthetic dataset written to {OUTPUT_PATH}")
