from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from src.metrics import channel_metrics, competition_score
from src.splits import spatial_hole_split


def unit_power(x: np.ndarray) -> np.ndarray:
    norm = np.sqrt(np.sum(np.abs(x.astype(np.complex128)) ** 2, dtype=np.float64))
    return x / norm if norm > 0 else x


def alternating_spectrum_projection(neighbors: list[np.ndarray], weights: np.ndarray, iterations: int = 4) -> np.ndarray:
    normalized = [unit_power(x) for x in neighbors]
    angle_powers = []
    delay_powers = []
    for x in normalized:
        shaped = x.reshape(2, 16, 8, 4, 192)
        angle_powers.append(np.abs(np.fft.fftn(shaped, axes=(1, 2), norm="ortho")) ** 2)
        delay_powers.append(np.abs(np.fft.ifft(shaped, axis=4, norm="ortho")) ** 2)
    target_angle = np.sqrt(sum(w * p for w, p in zip(weights, angle_powers)))
    target_delay = np.sqrt(sum(w * p for w, p in zip(weights, delay_powers)))
    x = normalized[0].reshape(2, 16, 8, 4, 192).copy()
    for _ in range(iterations):
        angle = np.fft.fftn(x, axes=(1, 2), norm="ortho")
        angle = target_angle * np.exp(1j * np.angle(angle))
        x = np.fft.ifftn(angle, axes=(1, 2), norm="ortho")
        delay = np.fft.ifft(x, axis=4, norm="ortho")
        delay = target_delay * np.exp(1j * np.angle(delay))
        x = np.fft.fft(delay, axis=4, norm="ortho")
    return x.reshape(256, 4, 192).astype(np.complex64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="Round2_Map")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    train_idx, val_idx, _ = spatial_hole_split(data_dir, seed=args.seed)
    pos = np.load(data_dir / "Round2_Train_Pos.npy")
    channels = np.load(data_dir / "Round2_Train_Channel.npy", mmap_mode="r")
    distances, local = cKDTree(pos[train_idx, :2]).query(pos[val_idx, :2], k=args.k)
    neighbor_idx = train_idx[local]
    if args.k == 1:
        distances, neighbor_idx = distances[:, None], neighbor_idx[:, None]

    totals = defaultdict(lambda: {"upa": 0.0, "flat": 0.0, "pdp": 0.0, "se": 0.0, "pp": 0.0, "inner": 0.0j})
    target_power = 0.0
    for row, target_idx in enumerate(val_idx):
        target = np.asarray(channels[target_idx])
        neighbors = [np.asarray(channels[i]) for i in neighbor_idx[row]]
        weights = 1.0 / np.maximum(distances[row], 0.25) ** 2
        weights /= weights.sum()
        normalized = [unit_power(x) for x in neighbors]
        idw_norm = sum(w * x for w, x in zip(weights, normalized)).astype(np.complex64)

        reference = normalized[0]
        aligned = []
        for x in normalized:
            inner = np.vdot(x.astype(np.complex128), reference.astype(np.complex128))
            aligned.append(x * (inner / abs(inner) if abs(inner) > 0 else 1.0))
        phase_aligned = sum(w * x for w, x in zip(weights, aligned)).astype(np.complex64)
        projected = alternating_spectrum_projection(neighbors, weights)

        candidates = {
            "nearest": neighbors[0],
            "idw_normalized": idw_norm,
            "phase_aligned": phase_aligned,
            "spectrum_projection": projected,
        }
        tp = np.sum(np.abs(target.astype(np.complex128)) ** 2, dtype=np.float64)
        target_power += tp
        for name, pred in candidates.items():
            upa, flat, pdp, se = channel_metrics(target, pred)
            info = totals[name]
            info["upa"] += upa
            info["flat"] += flat
            info["pdp"] += pdp
            info["se"] += se
            info["pp"] += np.sum(np.abs(pred.astype(np.complex128)) ** 2, dtype=np.float64)
            info["inner"] += np.vdot(pred.astype(np.complex128), target.astype(np.complex128))
        if (row + 1) % 40 == 0:
            print(f"evaluated {row + 1}/{len(val_idx)}", flush=True)

    report = {}
    for name, info in totals.items():
        upa, flat, pdp = info["upa"] / len(val_idx), info["flat"] / len(val_idx), info["pdp"] / len(val_idx)
        raw_nmse = info["se"] / target_power
        calibrated_nmse = (target_power - abs(info["inner"]) ** 2 / info["pp"]) / target_power if info["pp"] > 0 else 1.0
        report[name] = {
            "upa_pas": upa,
            "flat_pas": flat,
            "pdp": pdp,
            "raw_nmse": raw_nmse,
            "calibrated_nmse": calibrated_nmse,
            "calibrated_score_upa": competition_score(upa, pdp, calibrated_nmse),
            "calibrated_score_flat": competition_score(flat, pdp, calibrated_nmse),
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
