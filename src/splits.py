from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def spatial_hole_split(
    data_dir: str | Path,
    radius: float = 14.0,
    holes_per_region: int = 6,
    seed: int = 20260811,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make connected validation holes with test-like neighbor distances."""
    data_dir = Path(data_dir)
    pos = np.load(data_dir / "Round2_Train_Pos.npy")[:, :2]
    test = np.load(data_dir / "Round2_Test_Pos.npy")[:, :2]
    tree, test_tree = cKDTree(pos), cKDTree(test)
    rng = np.random.default_rng(seed)
    centers: list[int] = []

    for region_mask in (pos[:, 1] < 0, pos[:, 1] > 0):
        candidates = np.flatnonzero(region_mask)
        rng.shuffle(candidates)
        candidates = np.asarray(
            [
                i
                for i in candidates
                if 15 <= len(tree.query_ball_point(pos[i], radius)) <= 80
                and test_tree.query(pos[i])[0] > radius + 10.0
            ],
            dtype=np.int64,
        )
        if len(candidates) == 0:
            raise RuntimeError("no validation-hole candidates found")
        selected = [int(candidates[0])]
        while len(selected) < holes_per_region:
            eligible = [
                int(i)
                for i in candidates
                if min(np.linalg.norm(pos[i] - pos[j]) for j in selected) > 2.8 * radius
            ]
            if not eligible:
                raise RuntimeError("could not place all validation holes")
            scores = []
            for i in eligible:
                min_distance = min(np.linalg.norm(pos[i] - pos[j]) for j in selected)
                density_penalty = 0.2 * abs(len(tree.query_ball_point(pos[i], radius)) - 40)
                scores.append(min_distance - density_penalty)
            selected.append(eligible[int(np.argmax(scores))])
        centers.extend(selected)

    val_mask = np.zeros(len(pos), dtype=bool)
    for center in centers:
        val_mask[tree.query_ball_point(pos[center], radius)] = True
    val_idx = np.flatnonzero(val_mask)
    train_idx = np.flatnonzero(~val_mask)
    return train_idx, val_idx, np.asarray(centers, dtype=np.int64)
