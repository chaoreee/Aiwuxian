from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.fft import fft, fftn, ifft, ifftn
from scipy.spatial import cKDTree

from src.data import load_setup, position_features, reconstruct_frequency_channel
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


def project(initial: np.ndarray, angle_mag: np.ndarray, delay_mag: np.ndarray, layout: str, iterations: int) -> np.ndarray:
    x = initial.reshape(2, 16, 8, 4, 192).copy() if layout == "upa" else initial.copy()
    for _ in range(iterations):
        angle = fftn(x, axes=(1, 2), norm="ortho") if layout == "upa" else fft(x, axis=0, norm="ortho")
        angle *= angle_mag / np.maximum(np.abs(angle), 1e-20)
        x = ifftn(angle, axes=(1, 2), norm="ortho") if layout == "upa" else ifft(angle, axis=0, norm="ortho")
        delay_axis = 4 if layout == "upa" else 2
        delay = ifft(x, axis=delay_axis, norm="ortho")
        delay *= delay_mag / np.maximum(np.abs(delay), 1e-20)
        x = fft(delay, axis=delay_axis, norm="ortho")
    return normalize(x.reshape(256, 4, 192).astype(np.complex64))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="Round2_Map")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--checkpoint", default="outputs_final/best.pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pas-layout", choices=("upa", "flat"), required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--calibration-real", type=float, default=None)
    parser.add_argument("--calibration-imag", type=float, default=None)
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

    train_pos = np.load(data_dir / "Round2_Train_Pos.npy")
    test_pos = np.load(data_dir / "Round2_Test_Pos.npy")
    train_channels = np.load(data_dir / "Round2_Train_Channel.npy", mmap_mode="r")
    distances, neighbor_idx = cKDTree(train_pos[:, :2]).query(test_pos[:, :2], k=args.k)
    setup = load_setup(data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.complex64,
        shape=(setup["P_Test"], setup["M"], setup["N"], setup["S"]),
    )

    features = torch.from_numpy(test_features)
    with torch.inference_mode():
        for start in range(0, len(features), 32):
            out = model(features[start : start + 32].to(device))
            shapes = out["shape"].cpu().numpy()
            crop_rms = np.power(10.0, out["log_rms"].cpu().numpy())
            coverage = torch.sigmoid(out["coverage_logit"]).cpu().numpy()
            for j, ri in enumerate(shapes):
                row = start + j
                ai_shape = (ri[:8] + 1j * ri[8:]).reshape(2, 4, 16, 8, -1)
                ai_channel = normalize(reconstruct_frequency_channel(ai_shape))
                neighbors = [normalize(np.asarray(train_channels[i])) for i in neighbor_idx[row]]
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
                ai_angle, ai_delay = spectra(ai_channel, args.pas_layout)
                angle_mag = np.sqrt((1.0 - args.alpha) * neighbor_angle + args.alpha * ai_angle)
                delay_mag = np.sqrt((1.0 - args.alpha) * neighbor_delay + args.alpha * ai_delay)
                pred = project(initial, angle_mag, delay_mag, args.pas_layout, args.iterations)

                # The learned crop RMS restores a realistic per-point channel norm.
                crop_size = 2 * 4 * 16 * 8 * int(cfg["delay_bins"])
                predicted_norm = crop_rms[j] * np.sqrt(crop_size) * coverage[j]
                result[row] = (pred * predicted_norm * calibration).astype(np.complex64)
            result.flush()
            print(f"predicted {min(start + len(shapes), len(features))}/{len(features)}", flush=True)
    print(f"saved {output_path} shape={result.shape} dtype={result.dtype} calibration={calibration}")


if __name__ == "__main__":
    main()
