# PLACEMENT_PREDICTION

A comprehensive machine learning platform and web dashboard for student placement prediction, continuous salary package estimation, and unsupervised cohort clustering.

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-black?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6+-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Pipeline-10%2F10%20Modules%20Live-brightgreen)](#)
[![Documentation PDF](https://img.shields.io/badge/Documentation-PDF%20Report-red?logo=adobeacrobatreader&logoColor=white)](DOCUMENTATION.pdf)

> 📄 **Complete Technical Report**: A 14-page comprehensive technical and statistical report with embedded visualizations is available in [DOCUMENTATION.pdf](DOCUMENTATION.pdf).

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Summary of Model Performance](#-summary-of-model-performance)
- [Pipeline Outputs & Visualizations](#-pipeline-outputs--visualizations)
  - [1. Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
  - [2. Linear Regression (Salary Prediction)](#2-linear-regression-salary-package-prediction)
  - [3. Logistic Regression (Placement Classification)](#3-logistic-regression-placement-classification)
  - [4. ID3 Decision Tree Classifier](#4-id3-decision-tree-classifier)
  - [5. Boosting Algorithms (AdaBoost vs GBM)](#5-boosting-algorithms-adaboost-vs-gradient-boosting)
  - [6. Ensemble Learning (Bagging, Voting, Stacking)](#6-ensemble-learning-benchmark)
  - [7. K-Means Clustering (Student Personas)](#7-k-means-clustering-student-personas)
  - [8. DBSCAN Clustering (Anomaly & Cohort Detection)](#8-dbscan-clustering-density-analysis)
- [Installation & Running Locally](#-installation--running-locally)

---

## 🚀 Overview

The **Placement & Salary Intelligence Platform** provides an end-to-end ML pipeline with an interactive Flask web application built on 50,000 real-world simulated student academic and skill records.

- **Data Source**: 50,000 student records across 32 features (CGPA, internships, projects, soft skills, aptitude test scores, certifications, etc.)
- **Binary Classification**: Predict placement outcome (`PlacementStatus`: 0 = Unplaced, 1 = Placed).
- **Continuous Regression**: Predict salary package (LPA) exclusively for placed candidates.
- **Unsupervised Learning**: Discover distinct student skill cohorts (K-Means) and isolate anomaly profiles (DBSCAN).

---

## 📊 Summary of Model Performance

| Model / Architecture | Task Type | Accuracy / Metric | AUC / R² | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | Binary Classification | **92.03%** | **0.979** (AUC) | 92.70% | 95.40% |
| **Gradient Boosting (GBM)** | Boosting Classification | **91.81%** | **0.978** (AUC) | 92.40% | 95.30% |
| **Voting Ensemble** | Soft Voting | **91.76%** | **0.978** (AUC) | 92.30% | 95.30% |
| **Stacking Meta-Model** | Stacking (LogReg Blender)| **91.64%** | **0.978** (AUC) | 92.20% | 95.20% |
| **AdaBoost** | Sequential Boosting | **91.35%** | **0.976** (AUC) | 92.00% | 94.90% |
| **Random Forest** | Bagging Ensemble | **91.30%** | **0.975** (AUC) | 91.90% | 95.00% |
| **ID3 Decision Tree** | Interpretable Tree | **89.21%** | **0.960** (AUC) | 88.33% | 96.30% |
| **Linear Regression** | Salary Prediction (LPA) | **88.35%** (MAPE Acc) | **0.919** (R²) | MAE: 1.58 LPA | ±15% Tol: 74.3% |

---

## 🖼️ Pipeline Outputs & Visualizations

### 1. Exploratory Data Analysis (EDA)

Analysis of class balance (65.7% placed vs 34.3% unplaced), CGPA distribution, and correlation heatmaps across candidate attributes.

| Placement Class Balance | CGPA Distribution by Status |
| :---: | :---: |
| ![Class Balance](static/charts/placement_class_balance.png) | ![CGPA Distribution](static/charts/cgpa_distribution.png) |

| Feature Correlation Heatmap |
| :---: |
| ![Correlation Heatmap](static/charts/correlation_matrix.png) |

---

### 2. Linear Regression (Salary Package Prediction)

Predicts continuous compensation package (LPA) for placed candidates with strict accuracy tolerance evaluation.
- **R² Score**: `0.919` (91.9% variance explained)
- **Mean Absolute Error (MAE)**: `1.58 LPA`
- **MAPE Accuracy**: `88.35%`
- **Tolerance Accuracy (±15% band)**: `74.3%`

| Actual vs Predicted Salary with ±15% Margin Band |
| :---: |
| ![Actual vs Predicted Salary](static/charts/linear_actual_vs_predicted.png) |

---

### 3. Logistic Regression (Placement Classification)

Predicts binary placement outcomes using odds ratio analysis and ROC curves.
- **Overall Accuracy**: `92.03%`
- **ROC AUC**: `0.979`
- **Precision**: `92.70%` | **Recall**: `95.40%`

| ROC Curve (AUC = 0.979) | Confusion Matrix |
| :---: | :---: |
| ![Logistic ROC Curve](static/charts/logistic_roc_curve.png) | ![Logistic Confusion Matrix](static/charts/logistic_confusion_matrix.png) |

---

### 4. ID3 Decision Tree Classifier

Information Gain and Entropy-driven decision tree providing interpretable human-readable decision rules.
- **Accuracy**: `89.21%` | **Recall**: `96.30%` | **AUC**: `0.960`

| Visual Decision Tree Hierarchy | Feature Importances | Confusion Matrix |
| :---: | :---: | :---: |
| ![Tree Structure](static/charts/id3_tree_structure.png) | ![ID3 Feature Importance](static/charts/id3_feature_importance.png) | ![ID3 Confusion Matrix](static/charts/id3_confusion_matrix.png) |

---

### 5. Boosting Algorithms (AdaBoost vs Gradient Boosting)

Comparative benchmark of sequential boosting algorithms. Gradient Boosting achieves superior performance with tighter residual convergence.
- **Gradient Boosting (GBM)**: Accuracy: `91.81%` | AUC: `0.978`
- **AdaBoost**: Accuracy: `91.35%` | AUC: `0.976`

| Comparative ROC Curves | Staged Error Convergence | Feature Importance |
| :---: | :---: | :---: |
| ![Boosting ROC](static/charts/boosting_roc_comparison.png) | ![Boosting Learning Curve](static/charts/boosting_learning_curve.png) | ![Boosting Feature Importance](static/charts/boosting_feature_importance.png) |

---

### 6. Ensemble Learning Benchmark

Evaluates diverse ensemble strategies: Random Forest (Bagging), Soft Voting Classifier, and Stacking meta-models.

| Ensemble Comparison & Leaderboard | Model Agreement Heatmap |
| :---: | :---: |
| ![Ensemble Comparison](static/charts/ensemble_comparison.png) | ![Ensemble Agreement](static/charts/ensemble_agreement.png) |

---

### 7. K-Means Clustering (Student Personas)

Unsupervised student segmentation using the Elbow Method and Silhouette analysis to identify natural candidate cohorts.
- **Optimal Clusters**: `K = 3`
- **Average Silhouette Score**: `0.35`

| Elbow & Silhouette Curves | 2D PCA Cluster Distribution | Persona Feature Profiles |
| :---: | :---: | :---: |
| ![K-Means Elbow](static/charts/kmeans_elbow_silhouette.png) | ![K-Means PCA](static/charts/kmeans_clusters_pca.png) | ![Cluster Profiles](static/charts/kmeans_cluster_profiles.png) |

---

### 8. DBSCAN Clustering (Density Analysis)

Density-Based Spatial Clustering of Applications with Noise (DBSCAN) identifies core candidate clusters and isolates statistical outlier anomalies without requiring predefined cluster counts.

| K-Distance Knee Plot | PCA Clusters & Outlier Noise |
| :---: | :---: |
| ![DBSCAN K-Distance](static/charts/dbscan_k_distance.png) | ![DBSCAN PCA](static/charts/dbscan_clusters_pca.png) |

---

## 💻 Installation & Running Locally

### 1. Clone the Repository
```bash
git clone https://github.com/shaikmodhinnayeem-byte/PLACEMENT_PREDICTION.git
cd PLACEMENT_PREDICTION
```

### 2. Set Up Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run the Web Application
```powershell
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser to interact with the live dashboard.
