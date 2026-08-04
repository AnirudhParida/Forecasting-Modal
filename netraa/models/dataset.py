"""Panel -> supervised windows, with leak-free scaling.

The scaler is fitted only on timesteps that fall strictly before the first
validation window. Fitting on the full panel would put the test period's median
and IQR into the training transform — a small leak that reliably makes backtest
numbers look better than the model is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.panel import Panel
from ..features.transforms import (
    RobustScaler,
    apply_node_transforms,
    calendar_features,
    chronological_split,
    clip_outliers,
    make_windows,
)


@dataclass
class Dataset:
    X: np.ndarray                # (S, N, T_in, C)
    Y: np.ndarray                # (S, N_target, H)
    Y_mask: np.ndarray           # (S, N_target, H)
    starts: np.ndarray
    node_ids: list[str]
    target_ids: list[str]
    target_idx: list[int]
    scaler: RobustScaler
    horizons: list[int]
    input_steps: int
    index: pd.DatetimeIndex
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    values_scaled: np.ndarray
    mask: np.ndarray

    @property
    def n_channels(self) -> int:
        return self.X.shape[3]

    def window_time(self, i: int) -> pd.Timestamp:
        """Timestamp of the last input step of window i — its forecast origin."""
        return self.index[self.starts[i] + self.input_steps - 1]

    def summary(self) -> str:
        return (
            f"windows: {len(self.starts)} "
            f"(train {len(self.train_idx)} / val {len(self.val_idx)} / test {len(self.test_idx)})\n"
            f"nodes: {len(self.node_ids)}, targets: {len(self.target_ids)}\n"
            f"input_steps: {self.input_steps}, horizons: {self.horizons}\n"
            f"channels: {self.n_channels} [value, mask, calendar x {self.n_channels - 2}]"
        )


def prepare(
    panel: Panel,
    input_steps: int,
    horizons: list[int],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    clip: bool = True,
) -> Dataset:
    panel = apply_node_transforms(panel)
    if clip:
        panel = clip_outliers(panel)

    node_ids = panel.node_ids
    target_ids = panel.targets()
    if not target_ids:
        raise ValueError(
            "no target nodes survived panel construction — nothing to forecast. "
            "Check `role: target` entries in metrics_registry.yaml and the "
            "coverage floor in build_panel()."
        )
    target_idx = [node_ids.index(t) for t in target_ids]

    T = panel.n_steps
    max_h = max(horizons)
    n_windows = T - input_steps - max_h + 1
    if n_windows < 3:
        raise ValueError(
            f"panel has {T} steps; a window needs {input_steps + max_h} and at "
            f"least 3 windows are required to split. Collect more history, "
            f"shorten forecast.input_steps, or reduce the horizon."
        )

    tr, va, te = chronological_split(n_windows, train_frac, val_frac)

    # Last timestep any training window can see or predict.
    train_end = int(tr[-1] + input_steps + max_h) if len(tr) else input_steps
    scaler = RobustScaler.fit(panel.values.iloc[:train_end])
    scaled = scaler.transform(panel.values)

    values = scaled.to_numpy(dtype="float32")
    mask = panel.mask.to_numpy(dtype="float32")
    freq_seconds = int(pd.Timedelta(panel.freq).total_seconds())
    calendar = calendar_features(panel.values.index, freq_seconds)

    X, Y, Y_mask, starts = make_windows(
        values, mask, calendar, input_steps, horizons, target_idx
    )

    return Dataset(
        X=X,
        Y=Y,
        Y_mask=Y_mask,
        starts=starts,
        node_ids=node_ids,
        target_ids=target_ids,
        target_idx=target_idx,
        scaler=scaler,
        horizons=horizons,
        input_steps=input_steps,
        index=panel.values.index,
        train_idx=tr,
        val_idx=va,
        test_idx=te,
        values_scaled=values,
        mask=mask,
    )


def inverse_scale(
    arr: np.ndarray, target_ids: list[str], scaler: RobustScaler
) -> np.ndarray:
    """Undo scaling on an array whose axis 1 indexes target nodes."""
    centers = np.array([scaler.center.get(t, 0.0) for t in target_ids], dtype="float32")
    scales = np.array([scaler.scale.get(t, 1.0) for t in target_ids], dtype="float32")
    shape = [1] * arr.ndim
    shape[1] = len(target_ids)
    return arr * scales.reshape(shape) + centers.reshape(shape)
