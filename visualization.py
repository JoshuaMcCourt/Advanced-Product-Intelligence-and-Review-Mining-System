# src/visualization.py

import os
import matplotlib.pyplot as plt
import pandas as pd


# Risk Distribution Plot

def plot_risk_distribution(df: pd.DataFrame, save_path: str) -> str:

    if "risk_score" not in df.columns:
        raise ValueError("Missing required column: risk_score")

    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.hist(df["risk_score"].dropna(), bins=50, alpha=0.75)

    plt.title("Risk Score Distribution")
    plt.xlabel("Risk Score")
    plt.ylabel("Frequency")

    output_file = os.path.join(save_path, "risk_distribution.png")

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    return output_file


# Cluster Distribution Plot

def plot_cluster_distribution(df: pd.DataFrame, save_path: str) -> str:

    if "cluster" not in df.columns:
        raise ValueError("Missing required column: cluster")

    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(8, 5))

    cluster_counts = df["cluster"].value_counts().sort_index()

    if cluster_counts.empty:
        raise ValueError("Cluster column is empty or invalid")

    cluster_counts.plot(kind="bar")

    plt.title("Cluster Distribution")
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of Reviews")

    output_file = os.path.join(save_path, "cluster_distribution.png")

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    return output_file


# Risk vs Cluster View

def plot_risk_by_cluster(df: pd.DataFrame, save_path: str) -> str:

    if "cluster" not in df.columns or "risk_score" not in df.columns:
        raise ValueError("Missing required columns: cluster, risk_score")

    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(10, 6))

    df.groupby("cluster")["risk_score"].mean().plot(kind="bar")

    plt.title("Average Risk Score by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Avg Risk Score")

    output_file = os.path.join(save_path, "risk_by_cluster.png")

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    return output_file