from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.fft import fft, fftn, ifft, ifftn
from scipy.spatial import cKDTree

from src.data import position_features, reconstruct_frequency_channel
from src.metrics import channel_metrics, competition_score
from src.model import PhysicalChannelField
from src.splits import spatial_hole_split


def normalize(x: np.ndarray) -> np.ndarray:
    power = np.sum(np.abs(x.astype(np.complex128)) ** 2, dtype=np.float64)
    return x / np.sqrt(power) if power > 0 else x


def spectra(x: np.ndarray, layout: str) -> tuple[np.ndarray, np.ndarray]:
    if layout == "upa":
        shaped = x.reshape(2, 16, 8, 4, 192)
        angle = np.abs(fftn(shaped, axes=(1, 2), norm="ortho")) ** 2
        delay = np.abs(ifft(shaped, axis=4, norm="ortho")) ** 2
    else:
        angle = np.abs(fft(x, axis=0, norm="ortho")) ** 2
        delay = np.abs(ifft(x, axis=2, norm="ortho")) ** 2
    return angle.astype(np.float32), delay.astype(np.float32)


def project(
    initial: np.ndarray,
    angle_mag: np.ndarray,
    delay_mag: np.ndarray,
    iterations: int = 4,
    layout: str = "upa",
) -> np.ndarray:
    x = initial.reshape(2, 16, 8, 4, 192).copy() if layout == "upa" else initial.copy()
    for _ in range(iterations):
        angle = fftn(x, axes=(1, 2), norm="ortho") if layout == "upa" else fft(x, axis=0, norm="ortho")
        angle *= angle_mag / np.maximum(np.abs(angle), 1e-20)
        x = ifftn(angle, axes=(1, 2), norm="ortho") if layout == "upa" else ifft(angle, axis=0, norm="ortho")
        delay_axis = 4 if layout == "upa" else 2
        delay = ifft(x, axis=delay_axis, norm="ortho")
        delay *= delay_mag / np.maximum(np.abs(delay), 1e-20)
        x = fft(delay, axis=delay_axis, norm="ortho")
    return x.reshape(256, 4, 192).astype(np.complex64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="Round2_Map")
    parser.add_argument("--checkpoint", default="outputs_smooth/best.pt")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--pas-layout", choices=("upa", "flat"), default="upa")
    parser.add_argument("--alphas", type=float, nargs="+", default=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--cache-dir", default="cache")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    train_idx, val_idx, _ = spatial_hole_split(data_dir, seed=args.seed)
    pos = np.load(data_dir / "Round2_Train_Pos.npy")
    channels = np.load(data_dir / "Round2_Train_Channel.npy", mmap_mode="r")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["args"]
    feature_array = position_features(
        data_dir, args.cache_dir, "train", bool(cfg.get("use_map_features", False))
    )
    model = PhysicalChannelField(
        feature_dim=feature_array.shape[1],
        width=int(cfg["width"]),
        delay_bins=int(cfg["delay_bins"]),
        fourier_levels=int(cfg.get("fourier_levels", 8)),
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    features = torch.from_numpy(feature_array[val_idx])
    ai_channels = []
    ai_scales = []
    with torch.inference_mode():
        for start in range(0, len(features), 32):
            batch_out = model(features[start : start + 32].to(device))
            out = batch_out["shape"].cpu().numpy()
            rms = np.power(10.0, batch_out["log_rms"].cpu().numpy())
            coverage = torch.sigmoid(batch_out["coverage_logit"]).cpu().numpy()
            for j, ri in enumerate(out):
                shape = (ri[:8] + 1j * ri[8:]).reshape(2, 4, 16, 8, -1)
                ai_channels.append(normalize(reconstruct_frequency_channel(shape)))
                crop_size = 2 * 4 * 16 * 8 * int(cfg["delay_bins"])
                ai_scales.append(float(rms[j] * np.sqrt(crop_size) * coverage[j]))

    distances, local = cKDTree(pos[train_idx, :2]).query(pos[val_idx, :2], k=args.k)
    neighbor_idx = train_idx[local]
    alphas = tuple(args.alphas)
    totals = defaultdict(
        lambda: {
            "upa": 0.0,
            "flat": 0.0,
            "pdp": 0.0,
            "pp": 0.0,
            "inner": 0.0j,
            "scaled_pp": 0.0,
            "scaled_inner": 0.0j,
            "scaled_se": 0.0,
        }
    )
    target_power = 0.0

    for row, target_idx in enumerate(val_idx):
        target = np.asarray(channels[target_idx])
        neighbors = [normalize(np.asarray(channels[i])) for i in neighbor_idx[row]]
        weights = 1.0 / np.maximum(distances[row], 0.25) ** 2
        weights /= weights.sum()
        reference = neighbors[0]
        aligned = []
        for x in neighbors:
            inner = np.vdot(x.astype(np.complex128), reference.astype(np.complex128))
            aligned.append(x * (inner / abs(inner) if abs(inner) > 0 else 1.0))
        initial = normalize(sum(w * x for w, x in zip(weights, aligned)).astype(np.complex64))

        neighbor_angle = 0.0
        neighbor_delay = 0.0
        for w, x in zip(weights, neighbors):
            angle, delay = spectra(x, args.pas_layout)
            neighbor_angle = neighbor_angle + w * angle
            neighbor_delay = neighbor_delay + w * delay
        ai_angle, ai_delay = spectra(ai_channels[row], args.pas_layout)
        tp = np.sum(np.abs(target.astype(np.complex128)) ** 2, dtype=np.float64)
        target_power += tp

        for alpha in alphas:
            angle_mag = np.sqrt((1.0 - alpha) * neighbor_angle + alpha * ai_angle)
            delay_mag = np.sqrt((1.0 - alpha) * neighbor_delay + alpha * ai_delay)
            pred = project(initial, angle_mag, delay_mag, args.iterations, args.pas_layout)
            upa, flat, pdp, _ = channel_metrics(target, pred)
            info = totals[alpha]
            info["upa"] += upa
            info["flat"] += flat
            info["pdp"] += pdp
            info["pp"] += np.sum(np.abs(pred.astype(np.complex128)) ** 2, dtype=np.float64)
            info["inner"] += np.vdot(pred.astype(np.complex128), target.astype(np.complex128))
            pred_scaled = pred * ai_scales[row]
            info["scaled_pp"] += np.sum(np.abs(pred_scaled.astype(np.complex128)) ** 2, dtype=np.float64)
            info["scaled_inner"] += np.vdot(pred_scaled.astype(np.complex128), target.astype(np.complex128))
            info["scaled_se"] += np.sum(
                np.abs(pred_scaled.astype(np.complex128) - target.astype(np.complex128)) ** 2,
                dtype=np.float64,
            )
        if (row + 1) % 40 == 0:
            print(f"evaluated {row + 1}/{len(val_idx)}", flush=True)

    report = {}
    for alpha, info in totals.items():
        upa, flat, pdp = info["upa"] / len(val_idx), info["flat"] / len(val_idx), info["pdp"] / len(val_idx)
        nmse = (target_power - abs(info["inner"]) ** 2 / info["pp"]) / target_power
        scale_beta = info["scaled_inner"] / info["scaled_pp"] if info["scaled_pp"] > 0 else 0.0j
        scaled_nmse = info["scaled_se"] / target_power
        scaled_calibrated_nmse = (
            target_power - abs(info["scaled_inner"]) ** 2 / info["scaled_pp"]
        ) / target_power
        report[str(alpha)] = {
            "upa_pas": upa,
            "flat_pas": flat,
            "pdp": pdp,
            "calibrated_nmse": nmse,
            "beta_real": float((info["inner"] / info["pp"]).real),
            "beta_imag": float((info["inner"] / info["pp"]).imag),
            "learned_scale_raw_nmse": scaled_nmse,
            "learned_scale_beta_real": float(scale_beta.real),
            "learned_scale_beta_imag": float(scale_beta.imag),
            "learned_scale_calibrated_nmse": scaled_calibrated_nmse,
            "score_upa": competition_score(upa, pdp, nmse),
            "score_flat": competition_score(flat, pdp, nmse),
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
