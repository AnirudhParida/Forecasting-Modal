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


# --------------------------------------------------------------- trend removal
@dataclass
class LinearDetrend:
    """Per-column linear trend removal, fitted on the training slice only.

    Fitting is restricted to the training portion to prevent future information
    from leaking into the scaler's center/scale. For stationary metrics the
    fitted slope is near-zero (safe no-op). For monotonically growing metrics
    (JVM heap bytes, memory usage during a long ramp-up) the subtracted trend
    makes the residuals stationary, which dramatically improves both the
    RobustScaler's IQR estimate and the model's ability to extrapolate.

    Usage
    -----
        detrend = LinearDetrend.fit(df, train_end=280)
        df_flat  = detrend.transform(df)
        scaler   = RobustScaler.fit(df_flat.iloc[:280])
        ...
        # At inference — add trend back to model output
        actuals  = detrend.inverse_transform_array(pred_arr, cols, t_start, horizons)
    """

    slopes: dict[str, float]      # units-per-timestep, per column
    intercepts: dict[str, float]  # value at t=0, per column (training mean line)

    @classmethod
    def fit(cls, df: pd.DataFrame, train_end: int) -> "LinearDetrend":
        """Fit linear slope on the first `train_end` rows per column."""
        slopes, intercepts = {}, {}
        t_all = np.arange(len(df), dtype=float)

        for col in df.columns:
            train_series = df[col].iloc[:train_end]
            valid_mask   = train_series.notna().values
            n_valid      = int(valid_mask.sum())

            if n_valid < 10:
                # Not enough data to fit a meaningful trend — no-op
                slopes[col], intercepts[col] = 0.0, 0.0
                continue

            t_valid = t_all[:train_end][valid_mask]
            y_valid = train_series.values[valid_mask]

            # OLS: slope and intercept
            slope, intercept = np.polyfit(t_valid, y_valid, 1)

            # Sanity guard: if the slope × full panel length is more than 3× the
            # observed training range, the trend is almost certainly an anomaly
            # (e.g. a memory metric that Dynatrace reported as 0%→430% due to a
            # bad agent baseline). A genuine long-term trend should be proportionate
            # to the historical spread — an extreme slope would corrupt the scaler
            # center/IQR and produce physically impossible extrapolations at inference.
            train_range = float(y_valid.max() - y_valid.min()) if len(y_valid) > 1 else 0.0
            extrapolated_swing = abs(slope) * len(df)
            if train_range > 1e-9 and extrapolated_swing > 3.0 * train_range:
                # Anomalous slope — fall back to a zero slope at the training mean
                slope     = 0.0
                intercept = float(np.median(y_valid))

            slopes[col]     = float(slope)
            intercepts[col] = float(intercept)

        return cls(slopes=slopes, intercepts=intercepts)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Subtract fitted linear trend from every column."""
        out = df.copy()
        t   = np.arange(len(df), dtype=float)

        for col in out.columns:
            s = self.slopes.get(col, 0.0)
            b = self.intercepts.get(col, 0.0)
            if abs(s) > 1e-12 or abs(b) > 1e-12:
                out[col] = out[col] - (s * t + b)
        return out

    def inverse_transform_array(
        self,
        arr: np.ndarray,
        columns: list[str],
        panel_end_step: int,
        horizons: list[int],
    ) -> np.ndarray:
        """Re-add the trend to a forecast array.

        Parameters
        ----------
        arr             : (..., N_cols, H) predicted values (de-trended + scaled)
        columns         : list of column names matching N_cols axis
        panel_end_step  : the absolute time-step index at which the panel ends
                          (i.e. the last input day; t=0 is the panel start)
        horizons        : list of horizon offsets (e.g. [7, 30, 60, 90])
        """
        result = arr.copy()
        h_arr  = np.array(horizons, dtype=float)
        for i, col in enumerate(columns):
            s = self.slopes.get(col, 0.0)
            b = self.intercepts.get(col, 0.0)
            if abs(s) < 1e-12 and abs(b) < 1e-12:
                continue
            # t for each horizon: panel_end_step + h
            t_forecast = panel_end_step + h_arr           # shape (H,)
            trend_vals = s * t_forecast + b               # shape (H,)
            result[..., i, :] += trend_vals
        return result

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"slopes": self.slopes, "intercepts": self.intercepts}, indent=2)
        )

    @classmethod
    def load(cls, path: Path) -> "LinearDetrend":
        d = json.loads(Path(path).read_text())
        return cls(slopes=d["slopes"], intercepts=d["intercepts"])


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
