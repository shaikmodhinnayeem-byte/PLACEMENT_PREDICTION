"""DBSCAN Clustering module for density-based grouping and noise/anomaly detection."""

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
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ml.data_loader import load_data


IS_IMPLEMENTED = True
CHARTS_DIR = Path(__file__).resolve().parent.parent / "static" / "charts"


def _save_k_distance_plot(X_scaled, k_neighbors=8) -> str:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, k_neighbors - 1])

    fig, ax = plt.subplots(figsize=(6.5, 4.6), dpi=150)
    ax.plot(k_distances, color="#2764d7", linewidth=2)
    ax.set_title(f"K-Distance Graph (k={k_neighbors}) for Optimal ε (Eps) Selection", fontsize=11, fontweight="bold")
    ax.set_xlabel("Sorted Student Samples")
    ax.set_ylabel(f"Distance to {k_neighbors}-th Nearest Neighbor")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    chart_path = "charts/dbscan_k_distance.png"
    fig.savefig(CHARTS_DIR / "dbscan_k_distance.png", bbox_inches="tight")
    plt.close(fig)
    return chart_path


def _save_dbscan_pca(pca_2d, labels) -> str:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150)
    unique_labels = set(labels)

    # Palette
    colors = ["#2764d7", "#1b9b6e", "#e67e22", "#9b59b6", "#17a2b8"]

    for k in unique_labels:
        mask = labels == k
        if k == -1:
            # Noise points
            ax.scatter(
                pca_2d[mask, 0],
                pca_2d[mask, 1],
                c="#c93b4a",
                s=20,
                alpha=0.6,
                marker="x",
                label="Noise / Anomalies (-1)",
                zorder=4,
            )
        else:
            col = colors[k % len(colors)]
            ax.scatter(
                pca_2d[mask, 0],
                pca_2d[mask, 1],
                c=col,
                s=16,
                alpha=0.25,
                edgecolors="none",
                label=f"Cluster {k}",
            )

    ax.set_title("DBSCAN Density Clusters & Detected Outliers (PCA 2D)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    chart_path = "charts/dbscan_clusters_pca.png"
    fig.savefig(CHARTS_DIR / "dbscan_clusters_pca.png", bbox_inches="tight")
    plt.close(fig)
    return chart_path


@lru_cache(maxsize=1)
def run_dbscan_clustering() -> dict:
    """Execute DBSCAN clustering and anomaly/outlier detection."""
    data = load_data().copy()
    feature_cols = [
        "CGPA",
        "CodingTestScore",
        "AptitudeTestScore",
        "SoftSkillsRating",
        "AttendancePercent",
        "Internships",
        "Projects",
    ]
    df_clean = data.dropna(subset=feature_cols).copy()

    # Representative sample for responsiveness (DBSCAN is O(N^2))
    sample_size = min(3500, len(df_clean))
    df_sample = df_clean.sample(n=sample_size, random_state=42).copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_sample[feature_cols])

    # K-distance graph
    k_distance_chart = _save_k_distance_plot(X_scaled, k_neighbors=8)

    # DBSCAN parameters: eps and min_samples tuned to identify dense cores and distinct noise
    eps_val = 1.15
    min_samples_val = 10
    dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
    labels = dbscan.fit_predict(X_scaled)
    df_sample["DBSCAN_Cluster"] = labels

    # PCA 2D projection
    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(X_scaled)
    pca_chart = _save_dbscan_pca(pca_2d, labels)

    # Cluster summary
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    noise_ratio = round(float(n_noise / len(df_sample) * 100), 2)

    # Compare with dataset IsAnomaly if present
    anomaly_match_rate = None
    if "IsAnomaly" in df_sample.columns:
        planted_anomalies = df_sample["IsAnomaly"] == 1
        dbscan_anomalies = df_sample["DBSCAN_Cluster"] == -1
        # Overlap
        overlap = (planted_anomalies & dbscan_anomalies).sum()
        anomaly_match_rate = round(float(overlap / max(1, planted_anomalies.sum()) * 100), 1)

    cluster_details = []
    for c_id in sorted(set(labels)):
        subset = df_sample[df_sample["DBSCAN_Cluster"] == c_id]
        placement_rate = float(subset["PlacementStatus"].mean() * 100)
        c_name = "Outliers / Noise Points" if c_id == -1 else f"Dense Cluster {c_id}"
        cluster_details.append({
            "id": c_id,
            "name": c_name,
            "is_noise": c_id == -1,
            "count": len(subset),
            "percentage": round(len(subset) / len(df_sample) * 100, 1),
            "placement_rate": round(placement_rate, 1),
            "avg_cgpa": round(float(subset["CGPA"].mean()), 2),
            "avg_coding": round(float(subset["CodingTestScore"].mean()), 2),
        })

    return {
        "sample_size": sample_size,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": noise_ratio,
        "eps": eps_val,
        "min_samples": min_samples_val,
        "anomaly_match_rate": anomaly_match_rate,
        "k_distance_chart": k_distance_chart,
        "pca_chart": pca_chart,
        "cluster_details": cluster_details,
    }
