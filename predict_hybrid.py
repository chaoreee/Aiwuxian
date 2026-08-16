from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.fft import fft, fftn, ifft, ifftn
from scipy.spatial import cKDTree

from src.data import load_setup, position_features, reconstruct_frequency_channel
from src.interpolation import neighbor_weights
from src.model import PhysicalChannelField


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
    return angle / np.maximum(angle_norm, 1e-30), delay / np.maximum(delay_norm, 1e-30)


def project(
    initial: np.ndarray,
    angle_mag: np.ndarray,
    delay_mag: np.ndarray,
    layout: str,
    iterations: int,
    final_angle: bool = False,
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
    if final_angle:
        angle = fftn(x, axes=(1, 2), norm="ortho") if layout == "upa" else fft(x, axis=0, norm="ortho")
        angle *= angle_mag / np.maximum(np.abs(angle), 1e-20)
        x = ifftn(angle, axes=(1, 2), norm="ortho") if layout == "upa" else ifft(angle, axis=0, norm="ortho")
    return normalize(x.reshape(256, 4, 192).astype(np.complex64))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="Round2_Map")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--checkpoint", default="outputs_final/best.pt")
    parser.add_argument("--angle-checkpoint-extra", nargs="*", default=())
    parser.add_argument("--delay-checkpoint", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pas-layout", choices=("upa", "flat"), required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--final-angle", action="store_true")
    parser.add_argument("--neighbor-policy", choices=("all", "nonzero"), default="nonzero")
    parser.add_argument("--distance-power", type=float, default=2.0)
    parser.add_argument("--weight-mode", choices=("idw", "delaunay"), default="idw")
    parser.add_argument("--neighbor-power-mean", type=float, default=1.0)
    parser.add_argument("--slice-normalization", choices=("none", "l1", "l2"), default="none")
    parser.add_argument("--calibration-real", type=float, default=None)
    parser.add_argument("--calibration-imag", type=float, default=None)
    parser.add_argument(
        "--zero-mask-reference",
        default=None,
        help="optional prior AI prediction whose exactly-zero rows preserve coverage decisions",
    )
    args = parser.parse_args()
    default_calibration = {
        "upa": complex(0.03364188373439833, -0.04210538480648865),
        "flat": complex(0.03590192111741986, -0.033391479199905275),
    }
    if args.calibration_real is None and args.calibration_imag is None:
        calibration = default_calibration[args.pas_layout]
    elif args.calibration_real is not None and args.calibration_imag is not None:
        calibration = complex(args.calibration_real, args.calibration_imag)
    else:
        raise ValueError("calibration-real and calibration-imag must be supplied together")

    data_dir = Path(args.data_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["args"]
    test_features = position_features(
        data_dir, args.cache_dir, "test", bool(cfg.get("use_map_features", False))
    )
    model = PhysicalChannelField(
        feature_dim=test_features.shape[1],
        width=int(cfg["width"]),
        delay_bins=int(cfg["delay_bins"]),
        fourier_levels=int(cfg.get("fourier_levels", 8)),
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    extra_angle_models: list[tuple[PhysicalChannelField, np.ndarray]] = []
    for extra_path in args.angle_checkpoint_extra:
        extra_checkpoint = torch.load(extra_path, map_location="cpu", weights_only=False)
        extra_cfg = extra_checkpoint["args"]
        extra_features = position_features(
            data_dir, args.cache_dir, "test", bool(extra_cfg.get("use_map_features", False))
        )
        extra_model = PhysicalChannelField(
            feature_dim=extra_features.shape[1],
            width=int(extra_cfg["width"]),
            delay_bins=int(extra_cfg["delay_bins"]),
            fourier_levels=int(extra_cfg.get("fourier_levels", 8)),
        )
        extra_model.load_state_dict(extra_checkpoint["model"])
        extra_model.to(device).eval()
        extra_angle_models.append((extra_model, extra_features))
    delay_model = None
    delay_features = test_features
    if args.delay_checkpoint is not None:
        delay_checkpoint = torch.load(args.delay_checkpoint, map_location="cpu", weights_only=False)
        delay_cfg = delay_checkpoint["args"]
        delay_features = position_features(
            data_dir, args.cache_dir, "test", bool(delay_cfg.get("use_map_features", False))
        )
        delay_model = PhysicalChannelField(
            feature_dim=delay_features.shape[1],
            width=int(delay_cfg["width"]),
            delay_bins=int(delay_cfg["delay_bins"]),
            fourier_levels=int(delay_cfg.get("fourier_levels", 8)),
        )
        delay_model.load_state_dict(delay_checkpoint["model"])
        delay_model.to(device).eval()

    train_pos = np.load(data_dir / "Round2_Train_Pos.npy")
    test_pos = np.load(data_dir / "Round2_Test_Pos.npy")
    train_channels = np.load(data_dir / "Round2_Train_Channel.npy", mmap_mode="r")
    summaries = dict(np.load(Path(args.cache_dir) / "alignment_summaries.npz"))
    if args.neighbor_policy == "nonzero":
        available_idx = np.flatnonzero(summaries["total_power"] > 0)
    else:
        available_idx = np.arange(len(train_pos), dtype=np.int64)
    local, interpolation_weights, distances = neighbor_weights(
        train_pos[available_idx, :2],
        test_pos[:, :2],
        mode=args.weight_mode,
        k=args.k,
        distance_power=args.distance_power,
    )
    neighbor_idx = available_idx[local]
    setup = load_setup(data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.complex64,
        shape=(setup["P_Test"], setup["M"], setup["N"], setup["S"]),
    )
    zero_mask = np.zeros(setup["P_Test"], dtype=bool)
    if args.zero_mask_reference is not None:
        zero_reference = np.load(args.zero_mask_reference, mmap_mode="r")
        if zero_reference.shape != result.shape:
            raise ValueError(
                f"zero-mask reference shape {zero_reference.shape} does not match {result.shape}"
            )
        zero_mask = np.asarray(
            [not np.any(np.asarray(zero_reference[row])) for row in range(len(zero_reference))]
        )
        print(f"preserving {int(zero_mask.sum())} exact-zero AI coverage decisions", flush=True)

    features = torch.from_numpy(test_features)
    with torch.inference_mode():
        for start in range(0, len(features), 32):
            out = model(features[start : start + 32].to(device))
            extra_angle_outputs = [
                extra_model(torch.from_numpy(extra_features[start : start + 32]).to(device))
                for extra_model, extra_features in extra_angle_models
            ]
            extra_angle_shapes = [item["shape"].cpu().numpy() for item in extra_angle_outputs]
            delay_out = (
                delay_model(
                    torch.from_numpy(delay_features[start : start + 32]).to(device)
                )
                if delay_model is not None
                else out
            )
            shapes = out["shape"].cpu().numpy()
            delay_shapes = delay_out["shape"].cpu().numpy()
            crop_rms = np.power(10.0, out["log_rms"].cpu().numpy())
            coverage = torch.sigmoid(out["coverage_logit"]).cpu().numpy()
            for j, ri in enumerate(shapes):
                row = start + j
                ai_shape = (ri[:8] + 1j * ri[8:]).reshape(2, 4, 16, 8, -1)
                ai_channel = normalize(reconstruct_frequency_channel(ai_shape))
                angle_ai_channels = [ai_channel]
                for extra_shapes in extra_angle_shapes:
                    extra_ri = extra_shapes[j]
                    extra_shape = (extra_ri[:8] + 1j * extra_ri[8:]).reshape(2, 4, 16, 8, -1)
                    angle_ai_channels.append(
                        normalize(reconstruct_frequency_channel(extra_shape))
                    )
                delay_ri = delay_shapes[j]
                delay_shape = (delay_ri[:8] + 1j * delay_ri[8:]).reshape(2, 4, 16, 8, -1)
                delay_ai_channel = normalize(reconstruct_frequency_channel(delay_shape))
                neighbors = [normalize(np.asarray(train_channels[i])) for i in neighbor_idx[row]]
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
                for w, x in zip(weights, neighbors):
                    angle, delay = spectra(x, args.pas_layout)
                    angle, delay = normalize_spectral_slices(
                        angle, delay, args.pas_layout, args.slice_normalization
                    )
                    neighbor_angle = neighbor_angle + w * np.power(np.maximum(angle, 1e-30), mean_power)
                    neighbor_delay = neighbor_delay + w * np.power(np.maximum(delay, 1e-30), mean_power)
                if mean_power != 1.0:
                    neighbor_angle = np.power(neighbor_angle, 1.0 / mean_power)
                    neighbor_delay = np.power(neighbor_delay, 1.0 / mean_power)
                ai_angle = np.mean(
                    [spectra(channel, args.pas_layout)[0] for channel in angle_ai_channels],
                    axis=0,
                )
                _, ai_delay = spectra(delay_ai_channel, args.pas_layout)
                ai_angle, ai_delay = normalize_spectral_slices(
                    ai_angle, ai_delay, args.pas_layout, args.slice_normalization
                )
                angle_mag = np.sqrt((1.0 - args.alpha) * neighbor_angle + args.alpha * ai_angle)
                delay_mag = np.sqrt((1.0 - args.alpha) * neighbor_delay + args.alpha * ai_delay)
                pred = project(
                    initial,
                    angle_mag,
                    delay_mag,
                    args.pas_layout,
                    args.iterations,
                    final_angle=args.final_angle,
                )

                # The learned crop RMS restores a realistic per-point channel norm.
                crop_size = 2 * 4 * 16 * 8 * int(cfg["delay_bins"])
                predicted_norm = crop_rms[j] * np.sqrt(crop_size) * coverage[j]
                if zero_mask[row]:
                    result[row] = 0.0
                else:
                    result[row] = (pred * predicted_norm * calibration).astype(np.complex64)
            result.flush()
            print(f"predicted {min(start + len(shapes), len(features))}/{len(features)}", flush=True)
    print(f"saved {output_path} shape={result.shape} dtype={result.dtype} calibration={calibration}")


if __name__ == "__main__":
    main()
