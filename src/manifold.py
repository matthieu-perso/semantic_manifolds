# Define manifold by density threshold
import numpy as np
from src.utils import cartesian_to_polar
from src.parametric_models.von_mises import kda_density_polar

def define_manifold(densities, threshold, query_points):
    manifold_points = query_points[densities >= threshold]
    return manifold_points

def check_membership(embedding, polar_embeddings, kappa, threshold):
    polar_embedding = cartesian_to_polar(embedding[np.newaxis, :])
    density = kda_density_polar(polar_embeddings, polar_embedding, kappa)
    return density >= threshold







