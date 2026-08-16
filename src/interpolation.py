from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay, cKDTree


def neighbor_weights(
    reference_xy: np.ndarray,
    query_xy: np.ndarray,
    mode: str = "idw",
    k: int = 4,
    distance_power: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local reference rows, convex weights, and Euclidean distances."""
    reference_xy = np.asarray(reference_xy, dtype=np.float64)
    query_xy = np.asarray(query_xy, dtype=np.float64)
    if mode == "idw":
        distances, local = cKDTree(reference_xy).query(query_xy, k=k)
        distances = np.asarray(distances)
        local = np.asarray(local)
        if distances.ndim == 1:
            distances = distances[:, None]
            local = local[:, None]
        weights = 1.0 / np.maximum(distances, 0.25) ** distance_power
        weights /= weights.sum(axis=1, keepdims=True)
        return local, weights, distances
    if mode != "delaunay":
        raise ValueError(f"unknown interpolation mode: {mode}")

    triangulation = Delaunay(reference_xy)
    simplex = triangulation.find_simplex(query_xy)
    local = np.empty((len(query_xy), 3), dtype=np.int64)
    weights = np.empty((len(query_xy), 3), dtype=np.float64)
    distances = np.empty((len(query_xy), 3), dtype=np.float64)
    outside = simplex < 0

    for row in np.flatnonzero(~outside):
        transform = triangulation.transform[simplex[row]]
        first = transform[:2] @ (query_xy[row] - transform[2])
        barycentric = np.r_[first, 1.0 - first.sum()]
        vertices = triangulation.simplices[simplex[row]]
        local[row] = vertices
        # Numerical noise at triangle edges can create tiny negative values.
        weights[row] = np.maximum(barycentric, 0.0)
        weights[row] /= weights[row].sum()
        distances[row] = np.linalg.norm(reference_xy[vertices] - query_xy[row], axis=1)

    if np.any(outside):
        fallback_local, fallback_weights, fallback_distances = neighbor_weights(
            reference_xy,
            query_xy[outside],
            mode="idw",
            k=3,
            distance_power=distance_power,
        )
        local[outside] = fallback_local
        weights[outside] = fallback_weights
        distances[outside] = fallback_distances
    return local, weights, distances
