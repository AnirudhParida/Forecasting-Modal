"""Baselines the ST-GNN has to beat before it is worth shipping.

Ordered by how embarrassing it is to lose to them:

  persistence     last observed value, held flat
  seasonal_naive  the value one season ago (weekly on a daily grid)
  drift           linear extrapolation of the last window (Theil-Sen, so a
                  couple of spikes cannot set the slope)
  climatology     the training-period median

Capacity forecasting at a quarter horizon is exactly where a trend line is hard
to beat. If these win, that is the finding, and it belongs in the report rather
than being quietly dropped.
"""

from __future__ import annotations

import numpy as np


def _last_observed(window: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Most recent observed value per node. window: (T, N), mask: (T, N)."""
    T, N = window.shape
    out = np.zeros(N, dtype="float32")
    for n in range(N):
        obs = np.nonzero(mask[:, n])[0]
        out[n] = window[obs[-1], n] if len(obs) else 0.0
    return out


def persistence(window: np.ndarray, mask: np.ndarray, horizons: list[int]) -> np.ndarray:
    last = _last_observed(window, mask)
    return np.repeat(last[:, None], len(horizons), axis=1)


def seasonal_naive(
    window: np.ndarray, mask: np.ndarray, horizons: list[int], season: int
) -> np.ndarray:
    T, N = window.shape
    out = np.zeros((N, len(horizons)), dtype="float32")
    last = _last_observed(window, mask)
    for j, h in enumerate(horizons):
        # Step back whole seasons from the forecast point into the window.
        offset = ((h - 1) % season) + 1
        idx = T - offset
        if 0 <= idx < T:
            col = window[idx].copy()
            col[mask[idx] == 0] = last[mask[idx] == 0]
            out[:, j] = col
        else:
            out[:, j] = last
    return out


def _theil_sen_slope(y: np.ndarray, max_pairs: int = 400) -> float:
    n = len(y)
    if n < 3:
        return 0.0
    x = np.arange(n)
    if n * (n - 1) // 2 > max_pairs:
        rng = np.random.default_rng(0)
        i = rng.integers(0, n, max_pairs)
        j = rng.integers(0, n, max_pairs)
        keep = i != j
        i, j = i[keep], j[keep]
    else:
        i, j = np.triu_indices(n, k=1)
    dx = x[j] - x[i]
    ok = dx != 0
    if not ok.any():
        return 0.0
    return float(np.median((y[j][ok] - y[i][ok]) / dx[ok]))


def drift(window: np.ndarray, mask: np.ndarray, horizons: list[int]) -> np.ndarray:
    T, N = window.shape
    out = np.zeros((N, len(horizons)), dtype="float32")
    for n in range(N):
        obs = np.nonzero(mask[:, n])[0]
        if len(obs) < 3:
            out[n, :] = _last_observed(window, mask)[n]
            continue
        y = window[obs, n]
        slope = _theil_sen_slope(y)
        anchor = y[-1]
        last_t = obs[-1]
        for j, h in enumerate(horizons):
            out[n, j] = anchor + slope * (T - 1 - last_t + h)
    return out


def climatology(train_values: np.ndarray, n_nodes: int, horizons: list[int]) -> np.ndarray:
    med = np.nanmedian(train_values, axis=0)
    med = np.nan_to_num(med)
    return np.repeat(med[:, None], len(horizons), axis=1)


def run_all(
    values: np.ndarray,
    mask: np.ndarray,
    starts: np.ndarray,
    input_steps: int,
    horizons: list[int],
    target_idx: list[int],
    season: int,
    train_values: np.ndarray,
) -> dict[str, np.ndarray]:
    """Predictions per baseline, each shaped (S, N_target, H)."""
    S, H = len(starts), len(horizons)
    NT = len(target_idx)
    out = {
        name: np.zeros((S, NT, H), dtype="float32")
        for name in ("persistence", "seasonal_naive", "drift", "climatology")
    }

    clim = climatology(train_values, values.shape[1], horizons)[target_idx]
    filled = np.nan_to_num(values)

    for i, s in enumerate(starts):
        w = filled[s : s + input_steps]
        m = mask[s : s + input_steps]
        out["persistence"][i] = persistence(w, m, horizons)[target_idx]
        out["seasonal_naive"][i] = seasonal_naive(w, m, horizons, season)[target_idx]
        out["drift"][i] = drift(w, m, horizons)[target_idx]
        out["climatology"][i] = clim

    return out
