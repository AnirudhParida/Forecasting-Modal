"""Forecast metrics. All masked — missing actuals never count as a hit or a miss."""

from __future__ import annotations

import numpy as np


def _apply(mask: np.ndarray, err: np.ndarray) -> float:
    total = mask.sum()
    return float((err * mask).sum() / total) if total > 0 else float("nan")


def mae(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    return _apply(mask, np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(_apply(mask, (y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    """Symmetric MAPE in percent. Symmetric because capacity metrics sit near
    zero often enough that plain MAPE divides by ~0 and explodes."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    ratio = np.where(denom > 1e-9, np.abs(y_true - y_pred) / np.maximum(denom, 1e-9), 0.0)
    return _apply(mask, ratio) * 100.0


def pinball(
    y_true: np.ndarray, y_pred_q: np.ndarray, mask: np.ndarray, quantiles: list[float]
) -> float:
    """y_pred_q: (..., Q)."""
    q = np.array(quantiles).reshape((1,) * y_true.ndim + (-1,))
    err = y_true[..., None] - y_pred_q
    loss = np.maximum(q * err, (q - 1) * err)
    m = mask[..., None]
    total = m.sum() * len(quantiles)
    return float((loss * m).sum() / total) if total > 0 else float("nan")


def interval_coverage(
    y_true: np.ndarray,
    y_pred_q: np.ndarray,
    mask: np.ndarray,
    quantiles: list[float],
    lower: float = 0.1,
    upper: float = 0.9,
) -> float:
    """Fraction of actuals inside the predicted interval.

    A well-calibrated P10-P90 band covers ~80%. Far below means the intervals
    are too tight to plan capacity against; far above means they are too wide
    to be useful.
    """
    if lower not in quantiles or upper not in quantiles:
        return float("nan")
    lo = y_pred_q[..., quantiles.index(lower)]
    hi = y_pred_q[..., quantiles.index(upper)]
    inside = ((y_true >= lo) & (y_true <= hi)).astype("float64")
    return _apply(mask, inside) * 100.0


def evaluate(
    y_true: np.ndarray,
    y_pred_q: np.ndarray,
    mask: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    median_q: float = 0.5,
) -> dict:
    """Overall and per-horizon metrics.

    y_true (S, NT, H); y_pred_q (S, NT, H, Q) or (S, NT, H) for point forecasts.
    """
    point_only = y_pred_q.ndim == y_true.ndim
    if point_only:
        y_point = y_pred_q
        y_q = y_pred_q[..., None]
        qs = [median_q]
    else:
        qi = quantiles.index(median_q) if median_q in quantiles else len(quantiles) // 2
        y_point = y_pred_q[..., qi]
        y_q = y_pred_q
        qs = quantiles

    out = {
        "mae": mae(y_true, y_point, mask),
        "rmse": rmse(y_true, y_point, mask),
        "smape": smape(y_true, y_point, mask),
        "pinball": pinball(y_true, y_q, mask, qs),
        "coverage_p10_p90": interval_coverage(y_true, y_q, mask, qs),
        "n_observations": int(mask.sum()),
        "per_horizon": {},
    }
    for j, h in enumerate(horizons):
        out["per_horizon"][str(h)] = {
            "mae": mae(y_true[:, :, j], y_point[:, :, j], mask[:, :, j]),
            "rmse": rmse(y_true[:, :, j], y_point[:, :, j], mask[:, :, j]),
            "smape": smape(y_true[:, :, j], y_point[:, :, j], mask[:, :, j]),
        }
    return out


def per_target(
    y_true: np.ndarray,
    y_point: np.ndarray,
    mask: np.ndarray,
    target_ids: list[str],
) -> dict[str, dict]:
    return {
        tid: {
            "mae": mae(y_true[:, i], y_point[:, i], mask[:, i]),
            "rmse": rmse(y_true[:, i], y_point[:, i], mask[:, i]),
            "smape": smape(y_true[:, i], y_point[:, i], mask[:, i]),
        }
        for i, tid in enumerate(target_ids)
    }
