from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import real_to_complex


def _cosine_last(a: torch.Tensor, b: torch.Tensor, dim: int, eps: float = 1e-10) -> torch.Tensor:
    numerator = torch.sum(a * b, dim=dim)
    denominator = torch.sqrt(torch.sum(a.square(), dim=dim) * torch.sum(b.square(), dim=dim)).clamp_min(eps)
    return (numerator / denominator).mean()


def channel_loss(
    output: dict[str, torch.Tensor],
    target_ri: torch.Tensor,
    target_log_rms: torch.Tensor,
    nonzero: torch.Tensor,
    full_subcarriers: int = 192,
    pas_mode: str = "upa",
    complex_weight: float = 0.50,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_ri = output["shape"]
    valid = nonzero > 0.5
    bce = F.binary_cross_entropy_with_logits(output["coverage_logit"], nonzero)
    if not torch.any(valid):
        return bce, {"coverage_bce": float(bce.detach())}

    pred_ri_v = pred_ri[valid]
    target_ri_v = target_ri[valid]
    pred = real_to_complex(pred_ri_v)
    target = real_to_complex(target_ri_v)

    complex_mse = F.mse_loss(pred_ri_v, target_ri_v)
    scale_loss = F.smooth_l1_loss(output["log_rms"][valid], target_log_rms[valid], beta=0.25)

    # PDP: inverse spatial FFT, power cosine along delay for every antenna pair.
    pred_delay = torch.fft.ifftn(pred, dim=(-3, -2), norm="ortho")
    target_delay = torch.fft.ifftn(target, dim=(-3, -2), norm="ortho")
    pred_pdp = pred_delay.abs().square()
    target_pdp = target_delay.abs().square()
    pdp_cos = _cosine_last(pred_pdp, target_pdp, dim=-1)

    # PAS: restore frequency samples and compare the angle-power vectors.
    pred_af = torch.fft.fft(pred, n=full_subcarriers, dim=-1, norm="ortho")
    target_af = torch.fft.fft(target, n=full_subcarriers, dim=-1, norm="ortho")
    pred_pas = pred_af.abs().square().permute(0, 2, 5, 1, 3, 4).flatten(3)
    target_pas = target_af.abs().square().permute(0, 2, 5, 1, 3, 4).flatten(3)
    upa_pas_cos = _cosine_last(pred_pas, target_pas, dim=-1)

    pred_h = torch.fft.fft(pred_delay, n=full_subcarriers, dim=-1, norm="ortho")
    target_h = torch.fft.fft(target_delay, n=full_subcarriers, dim=-1, norm="ortho")
    pred_h = pred_h.permute(0, 1, 3, 4, 2, 5).flatten(1, 3)
    target_h = target_h.permute(0, 1, 3, 4, 2, 5).flatten(1, 3)
    pred_flat = torch.fft.fft(pred_h, dim=1, norm="ortho").abs().square().permute(0, 2, 3, 1)
    target_flat = torch.fft.fft(target_h, dim=1, norm="ortho").abs().square().permute(0, 2, 3, 1)
    flat_pas_cos = _cosine_last(pred_flat, target_flat, dim=-1)
    if pas_mode == "upa":
        pas_cos = upa_pas_cos
    elif pas_mode == "flat":
        pas_cos = flat_pas_cos
    elif pas_mode == "both":
        pas_cos = 0.5 * (upa_pas_cos + flat_pas_cos)
    else:
        raise ValueError(f"unknown pas_mode: {pas_mode}")

    loss = (
        complex_weight * complex_mse
        + 0.45 * (1.0 - pas_cos)
        + 0.45 * (1.0 - pdp_cos)
        + 0.12 * scale_loss
        + 0.10 * bce
    )
    metrics = {
        "loss": float(loss.detach()),
        "complex_mse": float(complex_mse.detach()),
        "pas_cos": float(pas_cos.detach()),
        "upa_pas_cos": float(upa_pas_cos.detach()),
        "flat_pas_cos": float(flat_pas_cos.detach()),
        "pdp_cos": float(pdp_cos.detach()),
        "scale_loss": float(scale_loss.detach()),
        "coverage_bce": float(bce.detach()),
    }
    return loss, metrics
