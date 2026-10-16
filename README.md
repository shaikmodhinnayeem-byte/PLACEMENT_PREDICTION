# PLACEMENT_PREDICTION

A comprehensive machine learning platform and web dashboard for student placement prediction and salary estimation.

## 🚀 Overview

The **Placement & Salary Intelligence Platform** provides an end-to-end ML pipeline with an interactive Flask web application. It includes modules for:
- **Data Loading & Inspection**: Schema, data types, and missing value analysis.
- **Exploratory Data Analysis (EDA)**: Statistical distributions, cross-feature correlations, and class balance.
- **Data Preprocessing**: Imputation, categorical encoding, and Min-Max [0, 1] normalization.
- **Linear Regression**: Continuous salary package (LPA) prediction with tolerance intervals (±10%, ±15%, ±20%), MAPE, and R².
- **Logistic Regression**: Binary classification for placement status with odds ratio and ROC analysis.
- **ID3 Decision Tree**: Entropy and Information Gain-based tree classifier.
- **Boosting Algorithms**: Benchmark comparing AdaBoost and Gradient Boosting (GBM).
- **Ensemble Learning**: Bagging (Random Forest), Soft Voting, and Stacking meta-models.
- **K-Means Clustering**: Unsupervised student persona segmentation with Elbow Method and Silhouette analysis.
- **DBSCAN Clustering**: Density-based spatial grouping for natural cohorts and anomaly isolation.

---

## 🛠️ Tech Stack

- **Backend**: Python 3, Flask
- **Machine Learning**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Frontend**: HTML5, Vanilla CSS

---

## 💻 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/shaikmodhinnayeem-byte/PLACEMENT_PREDICTION.git
cd PLACEMENT_PREDICTION
```

### 2. Set up virtual environment
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000`.
