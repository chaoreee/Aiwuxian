from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


SOURCE_FILES = (
    "README.md",
    "requirements.txt",
    "prepare_cache.py",
    "train.py",
    "predict.py",
    "predict_hybrid.py",
    "evaluate.py",
    "evaluate_baselines.py",
    "evaluate_hybrid.py",
    "src/__init__.py",
    "src/data.py",
    "src/losses.py",
    "src/metrics.py",
    "src/model.py",
    "src/splits.py",
    "training_summary.json",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", required=True)
    parser.add_argument("--checkpoint", default="outputs_final/best.pt")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", allowZip64=True) as zf:
        zf.write(args.channel, "Round2_Test_Channel.npy", compress_type=zipfile.ZIP_STORED)
        zf.write(args.checkpoint, "outputs_final/best.pt", compress_type=zipfile.ZIP_STORED)
        for source in SOURCE_FILES:
            zf.write(source, source.replace("\\", "/"), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    print(f"created {destination} ({destination.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
