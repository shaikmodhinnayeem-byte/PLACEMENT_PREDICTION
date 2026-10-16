"""K-Means Clustering module for unsupervised student profiling and persona discovery."""

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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

from ml.data_loader import load_data


IS_IMPLEMENTED = True
CHARTS_DIR = Path(__file__).resolve().parent.parent / "static" / "charts"


def _save_elbow_and_silhouette(k_range, inertias, silhouettes) -> str:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    # Elbow plot
    ax1.plot(k_range, inertias, marker="o", color="#2764d7", linewidth=2)
    ax1.set_title("Elbow Method: Inertia vs K", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Silhouette plot
    ax2.plot(k_range, silhouettes, marker="s", color="#1b9b6e", linewidth=2)
    ax2.set_title("Silhouette Score vs K", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Average Silhouette Coefficient")
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    chart_path = "charts/kmeans_elbow_silhouette.png"
    fig.savefig(CHARTS_DIR / "kmeans_elbow_silhouette.png", bbox_inches="tight")
    plt.close(fig)
    return chart_path


def _save_pca_clusters(pca_2d, labels, centroids_pca) -> str:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150)
    colors = ["#2764d7", "#1b9b6e", "#e67e22", "#9b59b6"]

    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        color = colors[cluster_id % len(colors)]
        ax.scatter(
            pca_2d[mask, 0],
            pca_2d[mask, 1],
            alpha=0.25,
            s=16,
            color=color,
            edgecolors="none",
            label=f"Cluster {cluster_id}",
        )

    # Mark centroids
    ax.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        marker="X",
        s=140,
        c="#111827",
        edgecolors="white",
        linewidths=1.5,
        label="Cluster Centroids",
        zorder=5,
    )

    ax.set_title("K-Means Student Clusters (PCA 2D Projection)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Principal Component 1 (Primary Academic & Skill Axis)")
    ax.set_ylabel("Principal Component 2 (Secondary Variance Axis)")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    chart_path = "charts/kmeans_clusters_pca.png"
    fig.savefig(CHARTS_DIR / "kmeans_clusters_pca.png", bbox_inches="tight")
    plt.close(fig)
    return chart_path


def _save_cluster_profiles_chart(cluster_profiles, feature_names) -> str:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    x = np.arange(len(feature_names))
    n_clusters = len(cluster_profiles)
    width = 0.8 / n_clusters
    colors = ["#2764d7", "#1b9b6e", "#e67e22", "#9b59b6"]

    for idx, prof in enumerate(cluster_profiles):
        values = [prof["means"][f] for f in feature_names]
        offset = (idx - n_clusters / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=prof["name"], color=colors[idx % len(colors)])

    ax.set_title("Cluster Skill & Academic Profile Comparison (Original Scale)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=20, ha="right")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    chart_path = "charts/kmeans_cluster_profiles.png"
    fig.savefig(CHARTS_DIR / "kmeans_cluster_profiles.png", bbox_inches="tight")
    plt.close(fig)
    return chart_path


@lru_cache(maxsize=1)
def run_kmeans_clustering() -> dict:
    """Execute K-Means clustering with Min-Max normalization, elbow evaluation, and student profiling."""
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

    # Use stratified sample for quick silhouette computation if large
    sample_size = min(4000, len(df_clean))
    df_sample = df_clean.sample(n=sample_size, random_state=42).copy()

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df_sample[feature_cols])

    # Elbow and Silhouette evaluation for K = 2 to 6
    k_range = list(range(2, 7))
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        preds = km.fit_predict(X_scaled)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(X_scaled, preds)))

    # Optimal K (k=3 delivers well-separated academic tiers)
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df_sample["Cluster"] = cluster_labels

    # PCA 2D projection
    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(X_scaled)
    centroids_pca = pca.transform(kmeans.cluster_centers_)

    # Charts
    elbow_chart = _save_elbow_and_silhouette(k_range, inertias, silhouettes)
    pca_chart = _save_pca_clusters(pca_2d, cluster_labels, centroids_pca)

    # Cluster personas
    cluster_profiles = []
    persona_names = {
        0: "Balanced Performers",
        1: "Tech Achievers / High Potential",
        2: "Developing Students / At-Risk",
    }

    # Sort clusters by mean CGPA to make persona assignment intuitive
    cluster_cgpas = df_sample.groupby("Cluster")["CGPA"].mean().sort_values()
    sorted_cluster_order = cluster_cgpas.index.tolist()
    
    # Lowest CGPA -> Developing, Mid -> Balanced, Highest -> High Achievers
    persona_mapping = {
        sorted_cluster_order[0]: "Developing Students / At-Risk",
        sorted_cluster_order[1]: "Balanced Candidates",
        sorted_cluster_order[2]: "Tech Achievers / High Potential",
    }

    for c_id in range(optimal_k):
        subset = df_sample[df_sample["Cluster"] == c_id]
        placement_rate = float(subset["PlacementStatus"].mean() * 100)
        avg_salary = float(subset.loc[subset["PlacementStatus"] == 1, "Salary Package"].mean())
        
        means = {f: round(float(subset[f].mean()), 2) for f in feature_cols}
        cluster_profiles.append({
            "id": c_id,
            "name": persona_mapping.get(c_id, f"Cluster {c_id}"),
            "size": len(subset),
            "percentage": round(len(subset) / len(df_sample) * 100, 1),
            "placement_rate": round(placement_rate, 1),
            "avg_salary": round(avg_salary, 2) if not np.isnan(avg_salary) else 0.0,
            "means": means,
        })

    display_features = ["CGPA", "CodingTestScore", "AptitudeTestScore", "SoftSkillsRating"]
    profile_chart = _save_cluster_profiles_chart(cluster_profiles, display_features)

    return {
        "sample_size": sample_size,
        "optimal_k": optimal_k,
        "silhouette_score": round(silhouettes[k_range.index(optimal_k)], 3),
        "inertia": round(inertias[k_range.index(optimal_k)], 1),
        "elbow_chart": elbow_chart,
        "pca_chart": pca_chart,
        "profile_chart": profile_chart,
        "profiles": cluster_profiles,
        "feature_cols": display_features,
    }
