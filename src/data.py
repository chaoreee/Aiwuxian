from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.fft import fftn, ifft
from scipy.spatial import cKDTree
from torch.utils.data import Dataset


BS_POSITIONS = np.asarray([[-18.413, -65.881, 25.0], [52.0, 35.0, 22.0]], dtype=np.float64)


def load_setup(data_dir: str | Path) -> dict:
    with (Path(data_dir) / "Round2_Setup.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def serving_bs(pos: np.ndarray) -> np.ndarray:
    """The two site clouds are separated by a wide gap on the y axis."""
    pos = np.asarray(pos)
    return (pos[:, 1] > 0.0).astype(np.int64)


def physical_features(pos: np.ndarray) -> np.ndarray:
    """Map coordinates to stable, dimensionless transmitter/receiver features."""
    pos = np.asarray(pos, dtype=np.float64)
    bs_id = serving_bs(pos)
    bs = BS_POSITIONS[bs_id]
    delta = pos - bs
    horizontal = np.linalg.norm(delta[:, :2], axis=1)
    distance = np.linalg.norm(delta, axis=1)
    unit_xy = delta[:, :2] / np.maximum(horizontal[:, None], 1e-8)
    one_hot = np.eye(2, dtype=np.float64)[bs_id]
    features = np.column_stack(
        [
            pos[:, 0] / 200.0,
            pos[:, 1] / 250.0,
            delta[:, 0] / 200.0,
            delta[:, 1] / 200.0,
            delta[:, 2] / 25.0,
            horizontal / 200.0,
            distance / 200.0,
            unit_xy,
            one_hot,
        ]
    )
    return features.astype(np.float32)


def load_ply_vertices(path: str | Path) -> np.ndarray:
    path = Path(path)
    with path.open("r", encoding="ascii") as f:
        vertex_count = None
        for line in f:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            if line.strip() == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"PLY vertex count missing in {path}")
        return np.loadtxt(f, dtype=np.float32, max_rows=vertex_count, usecols=(0, 1, 2))


def map_geometry_features(pos: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Multi-scale local point-cloud descriptors consumed by the neural field."""
    pos = np.asarray(pos, dtype=np.float64)
    tree = cKDTree(np.asarray(vertices[:, :2], dtype=np.float64))
    nearest_distance, nearest_idx = tree.query(pos[:, :2], k=1)
    rows = []
    radii = (1.0, 2.0, 5.0, 10.0, 20.0)
    for i, point in enumerate(pos):
        row = [nearest_distance[i] / 20.0, float(vertices[nearest_idx[i], 2]) / 50.0]
        for radius in radii:
            ids = tree.query_ball_point(point[:2], radius)
            heights = vertices[ids, 2] if ids else np.zeros(1, dtype=np.float32)
            row.extend(
                [
                    np.log1p(len(ids)) / 5.0,
                    float(np.max(heights)) / 50.0,
                    float(np.mean(heights)) / 30.0,
                ]
            )
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def build_map_feature_cache(
    data_dir: str | Path,
    cache_dir: str | Path,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_path = cache_dir / "round2_train_map_features.npy"
    test_path = cache_dir / "round2_test_map_features.npy"
    if train_path.exists() and test_path.exists() and not overwrite:
        return train_path, test_path
    vertices = load_ply_vertices(data_dir / "Round2_Map.ply")
    train = np.load(data_dir / "Round2_Train_Pos.npy")
    test = np.load(data_dir / "Round2_Test_Pos.npy")
    np.save(train_path, map_geometry_features(train, vertices))
    np.save(test_path, map_geometry_features(test, vertices))
    return train_path, test_path


def position_features(
    data_dir: str | Path,
    cache_dir: str | Path,
    split: str,
    use_map_features: bool,
) -> np.ndarray:
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    pos = np.load(data_dir / f"Round2_{split.title()}_Pos.npy")
    base = physical_features(pos)
    if not use_map_features:
        return base
    geometry = np.load(cache_dir / f"round2_{split}_map_features.npy")
    return np.concatenate([base, geometry.astype(np.float32)], axis=1)


def build_angle_delay_cache(
    data_dir: str | Path,
    cache_dir: str | Path,
    delay_bins: int = 16,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Create a compact, unitary angle-delay representation.

    The antenna-port order is P-H-V.  The cache layout is
    [point, M_P, N, M_H, M_V, delay].
    """
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_path = cache_dir / f"round2_angle_delay_d{delay_bins}.npy"
    stats_path = cache_dir / f"round2_angle_delay_d{delay_bins}_stats.npz"
    if target_path.exists() and stats_path.exists() and not overwrite:
        return target_path, stats_path

    setup = load_setup(data_dir)
    channels = np.load(data_dir / "Round2_Train_Channel.npy", mmap_mode="r")
    expected = (setup["P_Train"], setup["M"], setup["N"], setup["S"])
    if channels.shape != expected:
        raise ValueError(f"unexpected channel shape {channels.shape}, expected {expected}")
    if not 1 <= delay_bins <= setup["S"]:
        raise ValueError("delay_bins must be in [1, S]")

    shape = (
        setup["P_Train"],
        setup["M_P"],
        setup["N"],
        setup["M_H"],
        setup["M_V"],
        delay_bins,
    )
    cache = np.lib.format.open_memmap(target_path, mode="w+", dtype=np.complex64, shape=shape)
    crop_rms = np.zeros(setup["P_Train"], dtype=np.float64)
    retained = np.zeros(setup["P_Train"], dtype=np.float64)

    for i in range(setup["P_Train"]):
        x = np.asarray(channels[i]).reshape(
            setup["M_P"], setup["M_H"], setup["M_V"], setup["N"], setup["S"]
        )
        delay = ifft(x, axis=4, norm="ortho")
        angle_delay = fftn(delay, axes=(1, 2), norm="ortho")
        crop = angle_delay[..., :delay_bins].transpose(0, 3, 1, 2, 4)
        cache[i] = crop.astype(np.complex64, copy=False)
        crop_power = np.sum(np.abs(crop).astype(np.float64) ** 2, dtype=np.float64)
        full_power = np.sum(np.abs(x).astype(np.float64) ** 2, dtype=np.float64)
        crop_rms[i] = np.sqrt(crop_power / crop.size) if crop_power > 0 else 0.0
        retained[i] = crop_power / full_power if full_power > 0 else 0.0
        if (i + 1) % 250 == 0:
            cache.flush()
            print(f"cached {i + 1}/{setup['P_Train']}", flush=True)
    cache.flush()
    np.savez(
        stats_path,
        crop_rms=crop_rms,
        retained_energy=retained,
        nonzero=crop_rms > 0,
        delay_bins=np.asarray(delay_bins),
    )
    return target_path, stats_path


class AngleDelayDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        cache_dir: str | Path,
        indices: Sequence[int],
        delay_bins: int = 16,
        use_map_features: bool = False,
    ) -> None:
        data_dir, cache_dir = Path(data_dir), Path(cache_dir)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.features = position_features(data_dir, cache_dir, "train", use_map_features)
        self.targets = np.load(cache_dir / f"round2_angle_delay_d{delay_bins}.npy", mmap_mode="r")
        stats = np.load(cache_dir / f"round2_angle_delay_d{delay_bins}_stats.npz")
        self.rms = stats["crop_rms"].astype(np.float32)
        self.nonzero = stats["nonzero"].astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[item])
        target = np.array(self.targets[idx], copy=True)
        rms = float(self.rms[idx])
        if rms > 0:
            target /= rms
        target_ri = np.concatenate([target.real.reshape(8, *target.shape[-3:]), target.imag.reshape(8, *target.shape[-3:])])
        return {
            "features": torch.from_numpy(self.features[idx]),
            "target": torch.from_numpy(target_ri.astype(np.float32, copy=False)),
            "log_rms": torch.tensor(np.log10(max(rms, 1e-12)), dtype=torch.float32),
            "nonzero": torch.tensor(self.nonzero[idx], dtype=torch.float32),
            "index": torch.tensor(idx, dtype=torch.int64),
        }


def real_to_complex(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] != 16:
        raise ValueError(f"expected 16 real channels, got {x.shape}")
    return torch.complex(x[:, :8], x[:, 8:]).reshape(x.shape[0], 2, 4, *x.shape[-3:])


def reconstruct_frequency_channel(angle_delay: np.ndarray, s: int = 192) -> np.ndarray:
    """Invert a [P, N, H, V, D] angle-delay tensor to [M, N, S]."""
    p, n, mh, mv, d = angle_delay.shape
    full = np.zeros((p, n, mh, mv, s), dtype=np.complex64)
    full[..., :d] = angle_delay
    antenna_delay = np.fft.ifftn(full, axes=(2, 3), norm="ortho")
    frequency = np.fft.fft(antenna_delay, axis=4, norm="ortho")
    return frequency.transpose(0, 2, 3, 1, 4).reshape(p * mh * mv, n, s).astype(np.complex64)
