from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.fft import fft, fftn, ifft, ifftn
from scipy.spatial import cKDTree

from src.alignment import LocalSpectralAligner, align_upa_spectra
from src.data import position_features, reconstruct_frequency_channel
from src.interpolation import neighbor_weights
from src.metrics import channel_metrics_many, competition_score
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


def normalize_spectral_slices(
    angle: np.ndarray,
    delay: np.ndarray,
    layout: str,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "none":
        return angle, delay
    angle_axes = (0, 1, 2) if layout == "upa" else (0,)
    delay_axis = 4 if layout == "upa" else 2
    if mode == "l1":
        angle_norm = np.sum(angle, axis=angle_axes, keepdims=True)
        delay_norm = np.sum(delay, axis=delay_axis, keepdims=True)
    elif mode == "l2":
        angle_norm = np.sqrt(np.sum(angle.astype(np.float64) ** 2, axis=angle_axes, keepdims=True))
        delay_norm = np.sqrt(np.sum(delay.astype(np.float64) ** 2, axis=delay_axis, keepdims=True))
    else:
        raise ValueError(f"unknown spectral slice normalization: {mode}")
    return (
        angle / np.maximum(angle_norm, 1e-30),
        delay / np.maximum(delay_norm, 1e-30),
    )


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


def project_many(
    initial: np.ndarray,
    angle_magnitudes: np.ndarray,
    delay_magnitudes: np.ndarray,
    iterations: int,
    layout: str,
    final_angle: bool = False,
) -> np.ndarray:
    """Project every alpha candidate together to avoid repeated large FFTs."""
    count = len(angle_magnitudes)
    if layout == "upa":
        x = np.broadcast_to(initial.reshape(2, 16, 8, 4, 192), (count, 2, 16, 8, 4, 192)).copy()
        angle_axes = (2, 3)
        delay_axis = 5
    else:
        x = np.broadcast_to(initial, (count, 256, 4, 192)).copy()
        angle_axes = (1,)
        delay_axis = 3
    for _ in range(iterations):
        angle = fftn(x, axes=angle_axes, norm="ortho") if layout == "upa" else fft(x, axis=1, norm="ortho")
        angle *= angle_magnitudes / np.maximum(np.abs(angle), 1e-20)
        x = ifftn(angle, axes=angle_axes, norm="ortho") if layout == "upa" else ifft(angle, axis=1, norm="ortho")
        delay = ifft(x, axis=delay_axis, norm="ortho")
        delay *= delay_magnitudes / np.maximum(np.abs(delay), 1e-20)
        x = fft(delay, axis=delay_axis, norm="ortho")
    if final_angle:
        angle = fftn(x, axes=angle_axes, norm="ortho") if layout == "upa" else fft(x, axis=1, norm="ortho")
        angle *= angle_magnitudes / np.maximum(np.abs(angle), 1e-20)
        x = ifftn(angle, axes=angle_axes, norm="ortho") if layout == "upa" else ifft(angle, axis=1, norm="ortho")
    return x.reshape(count, 256, 4, 192).astype(np.complex64)


def infer_field_channels(
    checkpoint_path: str,
    data_dir: Path,
    cache_dir: str,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[list[np.ndarray], list[float]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["args"]
    feature_array = position_features(
        data_dir, cache_dir, "train", bool(cfg.get("use_map_features", False))
    )
    model = PhysicalChannelField(
        feature_dim=feature_array.shape[1],
        width=int(cfg["width"]),
        delay_bins=int(cfg["delay_bins"]),
        fourier_levels=int(cfg.get("fourier_levels", 8)),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    features = torch.from_numpy(feature_array[indices])
    channels: list[np.ndarray] = []
    scales: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(features), 32):
            batch_out = model(features[start : start + 32].to(device))
            output = batch_out["shape"].cpu().numpy()
            rms = np.power(10.0, batch_out["log_rms"].cpu().numpy())
            coverage = torch.sigmoid(batch_out["coverage_logit"]).cpu().numpy()
            for row, ri in enumerate(output):
                shape = (ri[:8] + 1j * ri[8:]).reshape(2, 4, 16, 8, -1)
                channels.append(normalize(reconstruct_frequency_channel(shape)))
                crop_size = 2 * 4 * 16 * 8 * int(cfg["delay_bins"])
                scales.append(float(rms[row] * np.sqrt(crop_size) * coverage[row]))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return channels, scales


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="Round2_Map")
    parser.add_argument("--checkpoint", default="outputs_smooth/best.pt")
    parser.add_argument("--angle-checkpoint-extra", nargs="*", default=())
    parser.add_argument("--delay-checkpoint", default=None)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--final-angle", action="store_true")
    parser.add_argument("--pas-layout", choices=("upa", "flat"), default="upa")
    parser.add_argument("--alphas", type=float, nargs="+", default=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--alignment", choices=("none", "local"), default="none")
    parser.add_argument("--alignment-k", type=int, default=24)
    parser.add_argument("--alignment-ridge", type=float, default=0.15)
    parser.add_argument("--alignment-distance-power", type=float, default=1.0)
    parser.add_argument("--align-components", choices=("h", "v", "d", "hv", "hd", "vd", "hvd"), default="hvd")
    parser.add_argument("--neighbor-policy", choices=("all", "nonzero"), default="all")
    parser.add_argument("--distance-power", type=float, default=2.0)
    parser.add_argument("--weight-mode", choices=("idw", "delaunay"), default="idw")
    parser.add_argument(
        "--neighbor-power-mean",
        type=float,
        default=1.0,
        help="generalized mean exponent for neighbor powers; 1 is arithmetic mean",
    )
    parser.add_argument("--slice-normalization", choices=("none", "l1", "l2"), default="none")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    train_idx, val_idx, _ = spatial_hole_split(data_dir, seed=args.seed)
    pos = np.load(data_dir / "Round2_Train_Pos.npy")
    channels = np.load(data_dir / "Round2_Train_Channel.npy", mmap_mode="r")
    summaries = dict(np.load(Path(args.cache_dir) / "alignment_summaries.npz"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ai_channels, ai_scales = infer_field_channels(
        args.checkpoint, data_dir, args.cache_dir, val_idx, device
    )
    angle_ai_channel_sets = [ai_channels]
    for extra_checkpoint in args.angle_checkpoint_extra:
        extra_channels, _ = infer_field_channels(
            extra_checkpoint, data_dir, args.cache_dir, val_idx, device
        )
        angle_ai_channel_sets.append(extra_channels)
    delay_ai_channels = ai_channels
    if args.delay_checkpoint is not None:
        delay_ai_channels, _ = infer_field_channels(
            args.delay_checkpoint, data_dir, args.cache_dir, val_idx, device
        )

    if args.neighbor_policy == "nonzero":
        available_idx = train_idx[summaries["total_power"][train_idx] > 0]
    else:
        available_idx = train_idx
    local, interpolation_weights, distances = neighbor_weights(
        pos[available_idx, :2],
        pos[val_idx, :2],
        mode=args.weight_mode,
        k=args.k,
        distance_power=args.distance_power,
    )
    neighbor_idx = available_idx[local]
    aligner = None
    if args.alignment == "local":
        if args.pas_layout != "upa":
            parser.error("local angle-delay alignment currently requires --pas-layout upa")
        aligner = LocalSpectralAligner(
            pos,
            train_idx,
            summaries,
            k=args.alignment_k,
            ridge=args.alignment_ridge,
            distance_power=args.alignment_distance_power,
        )
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
        weights = interpolation_weights[row]
        reference = neighbors[0]
        aligned = []
        for x in neighbors:
            inner = np.vdot(x.astype(np.complex128), reference.astype(np.complex128))
            aligned.append(x * (inner / abs(inner) if abs(inner) > 0 else 1.0))
        initial = normalize(sum(w * x for w, x in zip(weights, aligned)).astype(np.complex64))

        neighbor_angle = 0.0
        neighbor_delay = 0.0
        mean_power = args.neighbor_power_mean
        if mean_power <= 0:
            parser.error("--neighbor-power-mean must be positive")
        target_coordinates = aligner.predict(pos[target_idx]) if aligner is not None else None
        for w, x, source_idx in zip(weights, neighbors, neighbor_idx[row]):
            angle, delay = spectra(x, args.pas_layout)
            angle, delay = normalize_spectral_slices(
                angle, delay, args.pas_layout, args.slice_normalization
            )
            if target_coordinates is not None:
                source_coordinates = (
                    float(summaries["h_centroid"][source_idx]),
                    float(summaries["v_centroid"][source_idx]),
                    float(summaries["delay_centroid"][source_idx]),
                )
                angle, delay = align_upa_spectra(
                    angle,
                    delay,
                    source_coordinates,
                    target_coordinates,
                    args.align_components,
                )
            neighbor_angle = neighbor_angle + w * np.power(np.maximum(angle, 1e-30), mean_power)
            neighbor_delay = neighbor_delay + w * np.power(np.maximum(delay, 1e-30), mean_power)
        if mean_power != 1.0:
            neighbor_angle = np.power(neighbor_angle, 1.0 / mean_power)
            neighbor_delay = np.power(neighbor_delay, 1.0 / mean_power)
        ai_angles = [spectra(items[row], args.pas_layout)[0] for items in angle_ai_channel_sets]
        ai_angle = np.mean(ai_angles, axis=0)
        _, ai_delay = spectra(delay_ai_channels[row], args.pas_layout)
        ai_angle, ai_delay = normalize_spectral_slices(
            ai_angle, ai_delay, args.pas_layout, args.slice_normalization
        )
        tp = np.sum(np.abs(target.astype(np.complex128)) ** 2, dtype=np.float64)
        target_power += tp

        alpha_array = np.asarray(alphas, dtype=np.float32)
        alpha_shape = (len(alphas),) + (1,) * neighbor_angle.ndim
        alpha_view = alpha_array.reshape(alpha_shape)
        angle_mags = np.sqrt((1.0 - alpha_view) * neighbor_angle + alpha_view * ai_angle)
        delay_mags = np.sqrt((1.0 - alpha_view) * neighbor_delay + alpha_view * ai_delay)
        candidates = project_many(
            initial,
            angle_mags,
            delay_mags,
            args.iterations,
            args.pas_layout,
            final_angle=args.final_angle,
        )

        upa_rows, flat_rows, pdp_rows, _ = channel_metrics_many(target, candidates)
        for candidate_i, (alpha, pred) in enumerate(zip(alphas, candidates)):
            upa = float(upa_rows[candidate_i])
            flat = float(flat_rows[candidate_i])
            pdp = float(pdp_rows[candidate_i])
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
