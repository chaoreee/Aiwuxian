from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data import AngleDelayDataset
from src.losses import channel_loss
from src.model import PhysicalChannelField
from src.splits import spatial_hole_split


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    for row in rows:
        for key, value in row.items():
            sums[key] += value
    return {key: value / max(len(rows), 1) for key, value in sums.items()}


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    pas_mode: str,
    complex_weight: float,
) -> dict[str, float]:
    model.eval()
    rows = []
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        output = model(batch["features"])
        _, metrics = channel_loss(
            output,
            batch["target"],
            batch["log_rms"],
            batch["nonzero"],
            pas_mode=pas_mode,
            complex_weight=complex_weight,
        )
        rows.append(metrics)
    return average_metrics(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="Round2_Map")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--delay-bins", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--fourier-levels", type=int, default=8)
    parser.add_argument("--pas-mode", choices=("upa", "flat", "both"), default="upa")
    parser.add_argument("--use-map-features", action="store_true")
    parser.add_argument("--complex-weight", type=float, default=0.50)
    parser.add_argument("--train-all", action="store_true")
    parser.add_argument("--scheduler-tmax", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("warning: CUDA is unavailable; training will be slow")

    train_idx, val_idx, centers = spatial_hole_split(args.data_dir, seed=args.seed)
    if args.train_all:
        train_idx = np.arange(len(np.load(Path(args.data_dir) / "Round2_Train_Pos.npy")), dtype=np.int64)
    print(f"spatial split: train={len(train_idx)} val={len(val_idx)} centers={centers.tolist()}")
    train_ds = AngleDelayDataset(
        args.data_dir, args.cache_dir, train_idx, args.delay_bins, args.use_map_features
    )
    val_ds = AngleDelayDataset(
        args.data_dir, args.cache_dir, val_idx, args.delay_bins, args.use_map_features
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = PhysicalChannelField(
        feature_dim=train_ds.features.shape[1],
        width=args.width,
        delay_bins=args.delay_bins,
        fourier_levels=args.fourier_levels,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.scheduler_tmax if args.scheduler_tmax > 0 else args.epochs,
        eta_min=args.lr * 0.05,
    )
    best = -float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_rows = []
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["features"])
            loss, metrics = channel_loss(
                output,
                batch["target"],
                batch["log_rms"],
                batch["nonzero"],
                pas_mode=args.pas_mode,
                complex_weight=args.complex_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_rows.append(metrics)
        scheduler.step()
        train_metrics = average_metrics(train_rows)
        val_metrics = validate(model, val_loader, device, args.pas_mode, args.complex_weight)
        proxy = 0.4 * val_metrics.get("pas_cos", 0.0) + 0.4 * val_metrics.get("pdp_cos", 0.0) + 0.2 / (1.0 + val_metrics.get("complex_mse", 99.0))
        row = {"epoch": epoch, "proxy": proxy, "lr": scheduler.get_last_lr()[0], "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if proxy > best:
            best = proxy
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "epoch": epoch, "proxy": proxy},
                output_dir / "best.pt",
            )
        with (output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
