# Import statements
import numpy as np

def sample_shadowing_2d(query_points, sigma_sq, L=50, seed=None):
    # Initialise the modern random number generator
    rng = np.random.default_rng(seed)
    N = len(query_points)
    
    # query_points shape: (N, 2)
    
    # Compute pairwise distance matrix for 2D
    diff = query_points[:, np.newaxis, :] - query_points[np.newaxis, :, :]

    # Square the differences, sum along the last axis, then sqrt
    d = np.sqrt(np.sum(diff**2, axis=-1))

    # Apply kernel to get covariance matrix Sigma
    Sigma = sigma_sq * np.exp(-(d**2) / (2 * L**2))     # TODO: check parameter and function call
    Sigma += 1e-6 * np.eye(len(query_points)) # Numerical stability

    # Cholesky-decompose Sigma
    L_matrix = np.linalg.cholesky(Sigma)

    # Sample standard normals
    z = rng.normal(size=len(query_points))

    return L_matrix @ z

