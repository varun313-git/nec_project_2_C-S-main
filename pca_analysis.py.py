from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def visualize_clusters(X_scaled, clusters):

    pca = PCA(n_components=2)

    pca_data = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10,6))

    plt.scatter(
        pca_data[:,0],
        pca_data[:,1],
        c=clusters,
        cmap='viridis',
        s=100
    )

    plt.title("Customer Segments (PCA Visualization)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label="Cluster")

    plt.show()