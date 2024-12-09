import numpy as np

def cartesian_to_polar(embeddings):
    n_points, n_dim = embeddings.shape
    polar_coords = np.zeros((n_points, n_dim - 1))
    
    for i in range(n_points):
        r = np.linalg.norm(embeddings[i])
        polar_coords[i, 0] = np.arccos(embeddings[i, 0] / r)
        
        for j in range(1, n_dim - 1):
            norm_proj = np.linalg.norm(embeddings[i, j:])
            polar_coords[i, j] = np.arccos(embeddings[i, j] / norm_proj)
    return polar_coords


# polar_embeddings = cartesian_to_polar(embeddings)
