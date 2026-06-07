from pathlib import Path

import matplotlib.pyplot as plt


def plot_clusters(pca_data, labels, output_path="outputs/cluster_plot.png"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        pca_data[:, 0],
        pca_data[:, 1],
        c=labels,
        cmap="viridis",
        s=65,
        alpha=0.85,
    )
    plt.title("Customer Segments")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pca_projection(pca_data, labels, pca_model, output_path="outputs/pca_plot.png"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        pca_data[:, 0],
        pca_data[:, 1],
        c=labels,
        cmap="viridis",
        s=65,
        alpha=0.85,
    )
    plt.title(
        "PCA Projection "
        f"({pca_model.explained_variance_ratio_[0]:.1%}, {pca_model.explained_variance_ratio_[1]:.1%} variance)"
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()