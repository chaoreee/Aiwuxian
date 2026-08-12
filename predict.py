from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data import load_setup, position_features, reconstruct_frequency_channel
from src.model import PhysicalChannelField


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="Round2_Map")
    parser.add_argument("--checkpoint", default="outputs/best.pt")
    parser.add_argument("--output", default="outputs/Round2_Test_Channel.npy")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cache-dir", default="cache")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]
    delay_bins = int(train_args["delay_bins"])
    feature_array = position_features(
        data_dir, args.cache_dir, "test", bool(train_args.get("use_map_features", False))
    )
    model = PhysicalChannelField(
        feature_dim=feature_array.shape[1],
        width=int(train_args["width"]),
        delay_bins=delay_bins,
        fourier_levels=int(train_args.get("fourier_levels", 8)),
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    pos = np.load(data_dir / "Round2_Test_Pos.npy")
    features = torch.from_numpy(feature_array)
    setup = load_setup(data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.complex64,
        shape=(setup["P_Test"], setup["M"], setup["N"], setup["S"]),
    )

    with torch.inference_mode():
        for start in range(0, len(features), args.batch_size):
            output = model(features[start : start + args.batch_size].to(device))
            shape_ri = output["shape"].cpu().numpy()
            scale = np.power(10.0, output["log_rms"].cpu().numpy())
            coverage = torch.sigmoid(output["coverage_logit"]).cpu().numpy()
            for j in range(len(shape_ri)):
                complex_shape = (shape_ri[j, :8] + 1j * shape_ri[j, 8:]).reshape(2, 4, 16, 8, delay_bins)
                angle_delay = complex_shape * (scale[j] * coverage[j])
                result[start + j] = reconstruct_frequency_channel(angle_delay, setup["S"])
            result.flush()
            print(f"predicted {min(start + args.batch_size, len(features))}/{len(features)}", flush=True)
    print(f"saved {output_path} shape={result.shape} dtype={result.dtype}")


if __name__ == "__main__":
    main()
