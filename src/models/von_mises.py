import numpy as np

def von_mises_kernel(theta, theta_i, kappa):
    # Compute cosine similarity
    cosine_similarity = np.dot(theta, theta_i.T)
    return np.exp(kappa * cosine_similarity)

def kda_density_polar(polar_embeddings, query_points, kappa):
    densities = []
    for query in query_points:
        kernel_values = von_mises_kernel(polar_embeddings, query, kappa)
        densities.append(np.mean(kernel_values))
    return np.array(densities)
