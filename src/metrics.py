from __future__ import annotations

import numpy as np
from scipy.fft import fft, fftn, ifft


def _cosine(a: np.ndarray, b: np.ndarray, axis: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    numerator = np.sum(a * b, axis=axis, dtype=np.float64)
    aa = np.sum(a * a, axis=axis, dtype=np.float64)
    bb = np.sum(b * b, axis=axis, dtype=np.float64)
    denominator = np.sqrt(aa * bb)
    result = np.zeros_like(numerator, dtype=np.float64)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    result[(aa == 0) & (bb == 0)] = 1.0
    return result


def channel_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    m_p: int = 2,
    m_h: int = 16,
    m_v: int = 8,
) -> tuple[float, float, float, float]:
    """Return UPA-PAS, flat-port PAS, PDP similarities and squared error."""
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    n, s = target.shape[1:]

    target_angle = fftn(target.reshape(m_p, m_h, m_v, n, s), axes=(1, 2), norm="ortho")
    pred_angle = fftn(prediction.reshape(m_p, m_h, m_v, n, s), axes=(1, 2), norm="ortho")
    target_pas = np.abs(target_angle).astype(np.float64) ** 2
    pred_pas = np.abs(pred_angle).astype(np.float64) ** 2
    upa_pas = _cosine(
        target_pas.transpose(3, 4, 0, 1, 2).reshape(n, s, -1),
        pred_pas.transpose(3, 4, 0, 1, 2).reshape(n, s, -1),
        axis=-1,
    ).mean()

    target_flat = np.abs(fft(target, axis=0, norm="ortho")).astype(np.float64) ** 2
    pred_flat = np.abs(fft(prediction, axis=0, norm="ortho")).astype(np.float64) ** 2
    flat_pas = _cosine(target_flat, pred_flat, axis=0).mean()

    target_delay = np.abs(ifft(target, axis=2, norm="ortho")).astype(np.float64) ** 2
    pred_delay = np.abs(ifft(prediction, axis=2, norm="ortho")).astype(np.float64) ** 2
    pdp = _cosine(target_delay, pred_delay, axis=2).mean()
    squared_error = np.sum(
        np.abs(prediction.astype(np.complex128) - target.astype(np.complex128)) ** 2,
        dtype=np.float64,
    )
    return float(upa_pas), float(flat_pas), float(pdp), float(squared_error)


def channel_metrics_many(
    target: np.ndarray,
    predictions: np.ndarray,
    m_p: int = 2,
    m_h: int = 16,
    m_v: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized ``channel_metrics`` for predictions shaped [K, M, N, S]."""
    target = np.asarray(target)
    predictions = np.asarray(predictions)
    if predictions.ndim != 4 or predictions.shape[1:] != target.shape:
        raise ValueError(
            f"expected predictions [K, {target.shape}], got {predictions.shape}"
        )
    n, s = target.shape[1:]

    target_angle = fftn(target.reshape(m_p, m_h, m_v, n, s), axes=(1, 2), norm="ortho")
    pred_angle = fftn(
        predictions.reshape(-1, m_p, m_h, m_v, n, s), axes=(2, 3), norm="ortho"
    )
    target_pas = np.abs(target_angle).astype(np.float64) ** 2
    pred_pas = np.abs(pred_angle).astype(np.float64) ** 2
    upa_pas = _cosine(
        target_pas.transpose(3, 4, 0, 1, 2).reshape(n, s, -1),
        pred_pas.transpose(0, 4, 5, 1, 2, 3).reshape(len(predictions), n, s, -1),
        axis=-1,
    ).mean(axis=(1, 2))

    target_flat = np.abs(fft(target, axis=0, norm="ortho")).astype(np.float64) ** 2
    pred_flat = np.abs(fft(predictions, axis=1, norm="ortho")).astype(np.float64) ** 2
    flat_pas = _cosine(target_flat[None, ...], pred_flat, axis=1).mean(axis=(1, 2))

    target_delay = np.abs(ifft(target, axis=2, norm="ortho")).astype(np.float64) ** 2
    pred_delay = np.abs(ifft(predictions, axis=3, norm="ortho")).astype(np.float64) ** 2
    pdp = _cosine(target_delay[None, ...], pred_delay, axis=3).mean(axis=(1, 2))
    squared_error = np.sum(
        np.abs(
            predictions.astype(np.complex128) - target.astype(np.complex128)[None, ...]
        )
        ** 2,
        axis=(1, 2, 3),
        dtype=np.float64,
    )
    return upa_pas, flat_pas, pdp, squared_error


def competition_score(pas: float, pdp: float, nmse: float) -> float:
    return 0.4 * pas + 0.4 * pdp + 0.2 / (1.0 + nmse)
