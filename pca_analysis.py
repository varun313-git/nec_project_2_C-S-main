from sklearn.decomposition import PCA


def apply_pca(data):
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(data)
    return transformed, pca