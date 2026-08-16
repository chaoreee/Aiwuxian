from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def circular_delta(a: np.ndarray | float, b: np.ndarray | float, size: int) -> np.ndarray:
    """Shortest signed displacement from circular coordinate ``b`` to ``a``."""
    return (np.asarray(a) - np.asarray(b) + size / 2.0) % size - size / 2.0


def fractional_roll(power: np.ndarray, shift: float, axis: int) -> np.ndarray:
    """Non-negative, energy-preserving linear circular shift."""
    lower = int(np.floor(shift))
    fraction = float(shift - lower)
    if fraction < 1e-7:
        return np.roll(power, lower, axis=axis)
    return (
        (1.0 - fraction) * np.roll(power, lower, axis=axis)
        + fraction * np.roll(power, lower + 1, axis=axis)
    )


def local_linear_value(
    query_xy: np.ndarray,
    reference_xy: np.ndarray,
    values: np.ndarray,
    distances: np.ndarray,
    circular_size: int | None,
    distance_power: float = 1.0,
    ridge: float = 0.15,
) -> float:
    """Predict a spectral coordinate at a spatial hole using a local affine plane.

    Coordinates are centered on the query, so the fitted intercept is the desired
    prediction. Circular targets are unwrapped around the closest reference.
    """
    values = np.asarray(values, dtype=np.float64)
    if circular_size is not None:
        anchor = float(values[0])
        values = anchor + circular_delta(values, anchor, circular_size)
    scale = max(float(np.median(distances)), 1.0)
    design = np.column_stack(
        [np.ones(len(reference_xy)), (np.asarray(reference_xy) - query_xy) / scale]
    )
    weights = 1.0 / np.maximum(np.asarray(distances), 0.5) ** distance_power
    normal = design.T @ (weights[:, None] * design)
    normal += np.diag([1e-8, ridge, ridge])
    rhs = design.T @ (weights * values)
    prediction = float(np.linalg.solve(normal, rhs)[0])
    return prediction % circular_size if circular_size is not None else prediction


class LocalSpectralAligner:
    """Estimate query angle/delay centroids from nearby training summaries."""

    def __init__(
        self,
        positions: np.ndarray,
        indices: np.ndarray,
        summaries: dict[str, np.ndarray],
        k: int = 24,
        ridge: float = 0.15,
        distance_power: float = 1.0,
    ) -> None:
        positions = np.asarray(positions)
        indices = np.asarray(indices, dtype=np.int64)
        usable = summaries["total_power"][indices] > 0
        self.indices = indices[usable]
        self.positions = positions[self.indices, :2]
        self.tree = cKDTree(self.positions)
        self.summaries = summaries
        self.k = min(int(k), len(self.indices))
        self.ridge = float(ridge)
        self.distance_power = float(distance_power)

    def predict(self, query_position: np.ndarray) -> tuple[float, float, float]:
        distances, local = self.tree.query(np.asarray(query_position)[:2], k=self.k)
        distances = np.atleast_1d(distances)
        local = np.atleast_1d(local)
        indices = self.indices[local]
        reference_xy = self.positions[local]
        kwargs = {
            "query_xy": np.asarray(query_position)[:2],
            "reference_xy": reference_xy,
            "distances": distances,
            "distance_power": self.distance_power,
            "ridge": self.ridge,
        }
        h = local_linear_value(
            values=self.summaries["h_centroid"][indices], circular_size=16, **kwargs
        )
        v = local_linear_value(
            values=self.summaries["v_centroid"][indices], circular_size=8, **kwargs
        )
        # Delay energy is concentrated near zero; unwrap the circular centroid
        # locally, but do not wrap the returned intercept before differencing.
        d = local_linear_value(
            values=self.summaries["delay_centroid"][indices], circular_size=192, **kwargs
        )
        return h, v, d


def align_upa_spectra(
    angle_power: np.ndarray,
    delay_power: np.ndarray,
    source_coordinates: tuple[float, float, float],
    target_coordinates: tuple[float, float, float],
    components: str = "hvd",
) -> tuple[np.ndarray, np.ndarray]:
    """Shift a neighbor's UPA-PAS/PDP power to the predicted query coordinates."""
    source_h, source_v, source_d = source_coordinates
    target_h, target_v, target_d = target_coordinates
    if "h" in components:
        angle_power = fractional_roll(
            angle_power, float(circular_delta(target_h, source_h, 16)), axis=1
        )
    if "v" in components:
        angle_power = fractional_roll(
            angle_power, float(circular_delta(target_v, source_v, 8)), axis=2
        )
    if "d" in components:
        delay_power = fractional_roll(
            delay_power, float(circular_delta(target_d, source_d, 192)), axis=4
        )
    return angle_power, delay_power
