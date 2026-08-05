"""Per-node transforms, scaling, calendar features and windowing.

Scaler statistics are fitted on the training slice only and persisted. Fitting
on the whole panel would leak future distribution information into the
backtest and quietly inflate every score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .panel import Panel


# --------------------------------------------------------------- transforms
def apply_node_transforms(panel: Panel) -> Panel:
    """Apply each node's declared transform (abs / log1p / diff / none).

    `abs` on host_mem_usage is the approved interim handling for blocker B5 —
    the -941.98 percent values. It is recorded per node so the decision stays
    visible in the artefacts rather than disappearing into the numbers.
    """
    values = panel.values.copy()
    mask = panel.mask.copy()

    for _, node in panel.nodes.iterrows():
        nid, transform = node["node_id"], node["transform"]
        if nid not in values.columns or transform == "none":
            continue

        col = values[nid]
        if transform == "abs":
            values[nid] = col.abs()
        elif transform == "log1p":
            values[nid] = np.log1p(col.clip(lower=0))
        elif transform == "diff":
            values[nid] = col.diff()
            mask.iloc[0, mask.columns.get_loc(nid)] = 0.0

    return Panel(
        values=values, mask=mask, nodes=panel.nodes, grid=panel.grid, freq=panel.freq
    )


def clip_outliers(panel: Panel, lower_q: float = 0.001, upper_q: float = 0.999) -> Panel:
    """Winsorise. Dynatrace counters occasionally emit single absurd spikes on
    agent restart; one such point dominates a correlation over a short panel."""
    values = panel.values.copy()
    lo = values.quantile(lower_q)
    hi = values.quantile(upper_q)
    values = values.clip(lower=lo, upper=hi, axis=1)
    return Panel(
        values=values, mask=panel.mask, nodes=panel.nodes, grid=panel.grid, freq=panel.freq
    )


# ------------------------------------------------------------------- scaling
@dataclass
class RobustScaler:
    """Median / IQR scaling — resistant to the spikes that survive clipping."""

    center: dict[str, float]
    scale: dict[str, float]

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "RobustScaler":
        center, scale = {}, {}
        for col in df.columns:
            series = df[col].dropna()
            if series.empty:
                center[col], scale[col] = 0.0, 1.0
                continue
            med = float(series.median())
            iqr = float(series.quantile(0.75) - series.quantile(0.25))
            if iqr <= 1e-9:
                iqr = float(series.std()) or 1.0
            center[col], scale[col] = med, iqr
        return cls(center=center, scale=scale)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in out.columns:
            out[col] = (out[col] - self.center.get(col, 0.0)) / self.scale.get(col, 1.0)
        return out

    def inverse_transform_array(self, arr: np.ndarray, columns: list[str]) -> np.ndarray:
        """Invert scaling on a raw array whose last axis is `columns`."""
        centers = np.array([self.center.get(c, 0.0) for c in columns])
        scales = np.array([self.scale.get(c, 1.0) for c in columns])
        return arr * scales + centers

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"center": self.center, "scale": self.scale}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "RobustScaler":
        d = json.loads(Path(path).read_text())
        return cls(center=d["center"], scale=d["scale"])


# ---------------------------------------------------------------- calendar
def calendar_features(index: pd.DatetimeIndex, freq_seconds: int) -> np.ndarray:
    """Cyclical time encodings, shaped (T, C).

    Daily terms are only meaningful when a step is shorter than a day, so they
    are omitted on the daily grid where they would be constant.
    """
    dow = index.dayofweek.to_numpy()
    doy = index.dayofyear.to_numpy()
    feats = [
        np.sin(2 * np.pi * dow / 7),
        np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * (index.day.to_numpy() - 1) / 31),
        np.cos(2 * np.pi * (index.day.to_numpy() - 1) / 31),
        # Annual cycle: at a 90-day horizon the seasonal-naive baseline wins on
        # exactly the series with yearly shape; without these terms the model
        # cannot even see where in the year the forecast lands.
        np.sin(2 * np.pi * (doy - 1) / 365.25),
        np.cos(2 * np.pi * (doy - 1) / 365.25),
    ]
    if freq_seconds < 86_400:
        seconds = (
            index.hour.to_numpy() * 3600
            + index.minute.to_numpy() * 60
            + index.second.to_numpy()
        )
        feats = [
            np.sin(2 * np.pi * seconds / 86_400),
            np.cos(2 * np.pi * seconds / 86_400),
        ] + feats
    return np.stack(feats, axis=-1).astype("float32")


# --------------------------------------------------------------- windowing
def make_windows(
    values: np.ndarray,
    mask: np.ndarray,
    calendar: np.ndarray,
    input_steps: int,
    horizons: list[int],
    target_idx: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slide windows over the panel.

    Returns
        X       (S, N, T_in, C)  C = [value, mask, *calendar]
        Y       (S, N_target, H)
        Y_mask  (S, N_target, H)
        starts  (S,) index of each window's first input step
    """
    T, N = values.shape
    max_h = max(horizons)
    n_cal = calendar.shape[1]

    starts = np.arange(0, T - input_steps - max_h + 1)
    if len(starts) == 0:
        raise ValueError(
            f"panel has {T} steps but a window needs {input_steps + max_h} "
            f"(input_steps={input_steps} + max horizon={max_h}). "
            "Collect more history or shorten the horizon."
        )

    S, H, NT = len(starts), len(horizons), len(target_idx)
    X = np.zeros((S, N, input_steps, 2 + n_cal), dtype="float32")
    Y = np.zeros((S, NT, H), dtype="float32")
    Y_mask = np.zeros((S, NT, H), dtype="float32")

    filled = np.nan_to_num(values, nan=0.0)

    for i, s in enumerate(starts):
        e = s + input_steps
        X[i, :, :, 0] = filled[s:e].T
        X[i, :, :, 1] = mask[s:e].T
        for c in range(n_cal):
            X[i, :, :, 2 + c] = calendar[s:e, c][None, :]

        for j, h in enumerate(horizons):
            t = e + h - 1
            Y[i, :, j] = filled[t, target_idx]
            Y_mask[i, :, j] = mask[t, target_idx]

    return X, Y, Y_mask, starts


def chronological_split(
    n_samples: int, train: float = 0.70, val: float = 0.15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contiguous, ordered splits. Never shuffle a time series."""
    i_train = int(n_samples * train)
    i_val = int(n_samples * (train + val))
    idx = np.arange(n_samples)
    return idx[:i_train], idx[i_train:i_val], idx[i_val:]
