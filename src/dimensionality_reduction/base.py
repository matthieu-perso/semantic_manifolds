import umap
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
import numpy as np

def apply_umap(embeddings, n_components=10, metric='cosine'):
    umap_reducer = umap.UMAP(n_components=n_components, metric=metric)
    return umap_reducer.fit_transform(embeddings)

def apply_pca(embeddings, n_components=80):
    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(embeddings)
    variance_retained = np.sum(pca.explained_variance_ratio_)
    variance_lost = 1 - variance_retained
    print(f"Variance retained: {variance_retained:.4f}")
    print(f"Variance lost: {variance_lost:.4f}")
    return pca_embeddings

def apply_isomap(embeddings, n_components=10):
    isomap = Isomap(n_components=n_components)
    return isomap.fit_transform(embeddings)

def calculate_reconstruction_error(original_embeddings, reduced_embeddings):
    nbrs = NearestNeighbors(n_neighbors=1).fit(reduced_embeddings)
    distances, indices = nbrs.kneighbors(reduced_embeddings)
    reconstructed_embeddings = original_embeddings[indices.flatten()]
    reconstruction_error = mean_squared_error(original_embeddings, reconstructed_embeddings)
    print(f"Isomap Reconstruction Error: {reconstruction_error:.4f}")
    return reconstruction_error

if __name__ == "__main__":
    embeddings = np.load("embeddings.pickle")
    umap_embeddings = apply_umap(embeddings)
    pca_embeddings = apply_pca(embeddings)
    isomap_embeddings = apply_isomap(embeddings)
    # Calculate reconstruction error for Isomap
    calculate_reconstruction_error(embeddings, isomap_embeddings)
