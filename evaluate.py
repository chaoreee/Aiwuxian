from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data import position_features, reconstruct_frequency_channel
from src.metrics import channel_metrics, competition_score
from src.model import PhysicalChannelField
from src.splits import spatial_hole_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="Round2_Map")
    parser.add_argument("--checkpoint", default="outputs/best.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    features_np = position_features(
        data_dir, args.cache_dir, "train", bool(train_args.get("use_map_features", False))
    )
    model = PhysicalChannelField(
        feature_dim=features_np.shape[1],
        width=int(train_args["width"]),
        delay_bins=int(train_args["delay_bins"]),
        fourier_levels=int(train_args.get("fourier_levels", 8)),
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    _, val_idx, _ = spatial_hole_split(data_dir, seed=args.seed)
    positions = np.load(data_dir / "Round2_Train_Pos.npy")
    channels = np.load(data_dir / "Round2_Train_Channel.npy", mmap_mode="r")
    features = torch.from_numpy(features_np[val_idx])

    predictions: list[tuple[np.ndarray, float, float]] = []
    with torch.inference_mode():
        for start in range(0, len(features), args.batch_size):
            output = model(features[start : start + args.batch_size].to(device))
            ri = output["shape"].cpu().numpy()
            rms = np.power(10.0, output["log_rms"].cpu().numpy())
            coverage = torch.sigmoid(output["coverage_logit"]).cpu().numpy()
            for j in range(len(ri)):
                shape = (ri[j, :8] + 1j * ri[j, 8:]).reshape(2, 4, 16, 8, -1)
                predictions.append((shape.astype(np.complex64), float(rms[j]), float(coverage[j])))

    target_power = 0.0
    rows = []
    inner = 0.0j
    pred_power = 0.0
    target_nonzero = []
    for local_i, data_i in enumerate(val_idx):
        target = np.asarray(channels[data_i])
        shape, rms, coverage = predictions[local_i]
        pred = reconstruct_frequency_channel(shape * (rms * coverage))
        upa, flat, pdp, se = channel_metrics(target, pred)
        tp = np.sum(np.abs(target.astype(np.complex128)) ** 2, dtype=np.float64)
        pp = np.sum(np.abs(pred.astype(np.complex128)) ** 2, dtype=np.float64)
        inner += np.vdot(pred.astype(np.complex128), target.astype(np.complex128))
        pred_power += pp
        target_power += tp
        target_nonzero.append(tp > 0)
        rows.append((upa, flat, pdp, se, coverage, tp, pp))

    rows = np.asarray(rows, dtype=np.float64)
    beta = inner / pred_power if pred_power > 0 else 0.0j
    calibrated_error = target_power - (abs(inner) ** 2 / pred_power if pred_power > 0 else 0.0)
    raw_nmse = rows[:, 3].sum() / target_power
    calibrated_nmse = calibrated_error / target_power
    target_nonzero = np.asarray(target_nonzero)

    report = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_proxy": float(checkpoint["proxy"]),
        "val_points": int(len(val_idx)),
        "target_zero": int((~target_nonzero).sum()),
        "coverage_mean_nonzero": float(rows[target_nonzero, 4].mean()),
        "coverage_mean_zero": float(rows[~target_nonzero, 4].mean()) if np.any(~target_nonzero) else None,
        "upa_pas": float(rows[:, 0].mean()),
        "flat_pas": float(rows[:, 1].mean()),
        "pdp": float(rows[:, 2].mean()),
        "raw_nmse": float(raw_nmse),
        "calibrated_complex_beta_real": float(beta.real),
        "calibrated_complex_beta_imag": float(beta.imag),
        "calibrated_nmse": float(calibrated_nmse),
        "raw_score_upa": competition_score(rows[:, 0].mean(), rows[:, 2].mean(), raw_nmse),
        "calibrated_score_upa": competition_score(rows[:, 0].mean(), rows[:, 2].mean(), calibrated_nmse),
        "calibrated_score_flat": competition_score(rows[:, 1].mean(), rows[:, 2].mean(), calibrated_nmse),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    print("coverage threshold sweep (PAS/PDP zero handling only):")
    for threshold in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9):
        pred_nonzero = rows[:, 4] >= threshold
        upa = rows[:, 0].copy()
        flat = rows[:, 1].copy()
        pdp = rows[:, 2].copy()
        suppressed = ~pred_nonzero
        upa[suppressed] = (~target_nonzero[suppressed]).astype(np.float64)
        flat[suppressed] = (~target_nonzero[suppressed]).astype(np.float64)
        pdp[suppressed] = (~target_nonzero[suppressed]).astype(np.float64)
        print(threshold, "pred_zero", int(suppressed.sum()), "UPA", upa.mean(), "flat", flat.mean(), "PDP", pdp.mean())


if __name__ == "__main__":
    main()
