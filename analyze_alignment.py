from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.fft import fftn, ifft

from src.data import BS_POSITIONS, serving_bs


def circular_mean(power: np.ndarray, axis: int) -> np.ndarray:
    """Return a fractional FFT-bin centroid in [0, axis_size)."""
    size = power.shape[axis]
    phase = np.exp(2j * np.pi * np.arange(size) / size)
    shape = [1] * power.ndim
    shape[axis] = size
    moment = np.sum(power * phase.reshape(shape), axis=axis)
    return np.mod(np.angle(moment) * size / (2.0 * np.pi), size)


def signed_bin(value: np.ndarray, size: int) -> np.ndarray:
    return (np.asarray(value) + size / 2.0) % size - size / 2.0


def extract_summaries(data_dir: Path, cache_path: Path, overwrite: bool) -> dict[str, np.ndarray]:
    if cache_path.exists() and not overwrite:
        return dict(np.load(cache_path))

    channels = np.load(data_dir / "Round2_Train_Channel.npy", mmap_mode="r")
    count = len(channels)
    angle_power = np.zeros((count, 16, 8), dtype=np.float32)
    delay_power = np.zeros((count, 192), dtype=np.float32)
    total_power = np.zeros(count, dtype=np.float64)

    for index in range(count):
        channel = np.asarray(channels[index]).reshape(2, 16, 8, 4, 192)
        angle = fftn(channel, axes=(1, 2), norm="ortho")
        delay = ifft(channel, axis=4, norm="ortho")
        ap = np.sum(np.abs(angle).astype(np.float64) ** 2, axis=(0, 3, 4))
        dp = np.sum(np.abs(delay).astype(np.float64) ** 2, axis=(0, 1, 2, 3))
        total = float(ap.sum())
        if total > 0:
            angle_power[index] = (ap / total).astype(np.float32)
            delay_power[index] = (dp / dp.sum()).astype(np.float32)
        total_power[index] = total
        if (index + 1) % 200 == 0:
            print(f"summarized {index + 1}/{count}", flush=True)

    h_marginal = angle_power.sum(axis=2)
    v_marginal = angle_power.sum(axis=1)
    result = {
        "angle_power": angle_power,
        "delay_power": delay_power,
        "h_peak": np.argmax(h_marginal, axis=1).astype(np.int16),
        "v_peak": np.argmax(v_marginal, axis=1).astype(np.int16),
        "delay_peak": np.argmax(delay_power, axis=1).astype(np.int16),
        "h_centroid": circular_mean(h_marginal, axis=1).astype(np.float32),
        "v_centroid": circular_mean(v_marginal, axis=1).astype(np.float32),
        "delay_centroid": circular_mean(delay_power, axis=1).astype(np.float32),
        "total_power": total_power,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **result)
    return result


def linear_report(target: np.ndarray, features: np.ndarray, valid: np.ndarray) -> dict[str, float | list[float]]:
    x = np.column_stack([np.ones(valid.sum()), features[valid]])
    y = target[valid]
    coefficient = np.linalg.lstsq(x, y, rcond=None)[0]
    prediction = x @ coefficient
    residual = y - prediction
    denominator = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - np.sum(residual**2) / denominator if denominator > 0 else 0.0
    return {
        "r2": float(r2),
        "mae_bins": float(np.mean(np.abs(residual))),
        "coefficient": coefficient.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="Round2_Map")
    parser.add_argument("--cache", default="cache/alignment_summaries.npz")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    summary = extract_summaries(data_dir, Path(args.cache), args.overwrite)
    pos = np.load(data_dir / "Round2_Train_Pos.npy").astype(np.float64)
    bs_id = serving_bs(pos)
    delta = pos - BS_POSITIONS[bs_id]
    distance = np.linalg.norm(delta, axis=1)
    direction = delta / np.maximum(distance[:, None], 1e-12)
    azimuth = np.arctan2(delta[:, 1], delta[:, 0])
    features = np.column_stack(
        [direction, np.sin(azimuth), np.cos(azimuth), distance / 200.0, pos[:, 2] / 30.0]
    )
    nonzero = summary["total_power"] > 0

    report: dict[str, object] = {
        "nonzero": int(nonzero.sum()),
        "zero": int((~nonzero).sum()),
        "feature_order": ["ux", "uy", "uz", "sin_az", "cos_az", "distance/200", "z/30"],
        "sites": {},
    }
    for site in (0, 1):
        valid = nonzero & (bs_id == site)
        report["sites"][str(site)] = {
            "count": int(valid.sum()),
            "h_peak": linear_report(signed_bin(summary["h_peak"], 16), features, valid),
            "v_peak": linear_report(signed_bin(summary["v_peak"], 8), features, valid),
            "h_centroid": linear_report(signed_bin(summary["h_centroid"], 16), features, valid),
            "v_centroid": linear_report(signed_bin(summary["v_centroid"], 8), features, valid),
            "delay_peak": linear_report(summary["delay_peak"].astype(np.float64), features, valid),
            "delay_centroid": linear_report(signed_bin(summary["delay_centroid"], 192), features, valid),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
