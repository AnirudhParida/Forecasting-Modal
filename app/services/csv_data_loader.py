"""
app/services/csv_data_loader.py
================================
Unified CSV-based forecast data loader for models that produce forecast
output as flat CSV files:
  - CHRONOS      -> outputs/forecasts/chronos/csv/{HOST}_chronos_forecast.csv
  - HOLT_WINTERS -> outputs/forecasts/holt_winters/csv/forecast_hw_{HOST}_{N}d.csv
  - TIMESFM      -> outputs/forecasts/timesfm/csv/forecast_tfm_{HOST}_{N}d.csv

All three CSV formats share the same column layout:
  ds                                          - date (YYYY-MM-DD)
  {metric}_yhat                               - P50 median forecast
  {metric}_yhat_lower or {metric}_lower       - P10 lower bound
  {metric}_yhat_upper or {metric}_upper       - P90 upper bound

Historical data and last-recorded-value queries are delegated to
NPDataLoader (same Excel ingestion pipeline as NeuralProphet).

Public interface (mirrors NPDataLoader / DataLoader):
  CSVDataLoader(model)                          - instantiate for a model
  .get_forecast_series(metric, host, st, et)    - List[(date, p50, p10, p90)]
  .get_historical_series(metric, host, st, et)  - List[(date, value)]
  .get_last_recorded_value(metric, host)        - float
  .get_as_of_date(host)                         - str  (YYYY-MM-DD)
  .get_confidence_and_risk(metric, host)        - (float, str)
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from app.services.np_data_loader import NPDataLoader, NP_METRIC_MAP

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PHROPHET_ROOT = Path(
    "/home/anirudh.parida@apmosys.mahape/Documents/Phrophet_forecast"
)

_FORECAST_ROOTS: Dict[str, Path] = {
    "CHRONOS":      _PHROPHET_ROOT / "outputs" / "forecasts" / "chronos" / "csv",
    "HOLT_WINTERS": _PHROPHET_ROOT / "outputs" / "forecasts" / "holt_winters" / "csv",
    "TIMESFM":      _PHROPHET_ROOT / "outputs" / "forecasts" / "timesfm" / "csv",
}

# Supported horizon files (days) for models that split by horizon.
# Chronos uses a single per-host file so it is not listed here.
_HORIZON_OPTIONS = (7, 30, 60, 90)


# ---------------------------------------------------------------------------
# Column-name normalisation helpers
# ---------------------------------------------------------------------------

def _col_lower(df: pd.DataFrame, metric: str) -> Optional[str]:
    """Return the P10 lower-bound column for *metric* (handles both naming conventions)."""
    for candidate in (f"{metric}_yhat_lower", f"{metric}_lower"):
        if candidate in df.columns:
            return candidate
    return None


def _col_upper(df: pd.DataFrame, metric: str) -> Optional[str]:
    """Return the P90 upper-bound column for *metric*."""
    for candidate in (f"{metric}_yhat_upper", f"{metric}_upper"):
        if candidate in df.columns:
            return candidate
    return None


def _col_yhat(df: pd.DataFrame, metric: str) -> Optional[str]:
    """Return the P50 median forecast column for *metric*."""
    for candidate in (f"{metric}_yhat", f"{metric}"):
        if candidate in df.columns:
            return candidate
    return None


# ---------------------------------------------------------------------------
# File-resolution helpers
# ---------------------------------------------------------------------------

def _chronos_file(host: str) -> Path:
    """Single per-host 90-day Chronos CSV."""
    return _FORECAST_ROOTS["CHRONOS"] / f"{host}_chronos_forecast.csv"


def _hw_file(host: str, days: int) -> Path:
    """Holt-Winters: pick the best-matching horizon file that exists."""
    root = _FORECAST_ROOTS["HOLT_WINTERS"]
    ordered = sorted(_HORIZON_OPTIONS, key=lambda h: (abs(h - days), -h))
    for h in ordered:
        candidate = root / f"forecast_hw_{host}_{h}d.csv"
        if candidate.exists():
            return candidate
    return root / f"forecast_hw_{host}_90d.csv"


def _tfm_file(host: str, days: int) -> Path:
    """TimesFM: pick the best-matching horizon file that exists."""
    root = _FORECAST_ROOTS["TIMESFM"]
    ordered = sorted(_HORIZON_OPTIONS, key=lambda h: (abs(h - days), -h))
    for h in ordered:
        candidate = root / f"forecast_tfm_{host}_{h}d.csv"
        if candidate.exists():
            return candidate
    return root / f"forecast_tfm_{host}_90d.csv"


def _resolve_csv_path(model: str, host: str, days: int) -> Path:
    """Return the CSV path for the given model / host / horizon."""
    m = model.upper()
    if m == "CHRONOS":
        return _chronos_file(host)
    elif m == "HOLT_WINTERS":
        return _hw_file(host, days)
    elif m == "TIMESFM":
        return _tfm_file(host, days)
    else:
        raise ValueError(f"CSVDataLoader: unsupported model '{model}'")


# ---------------------------------------------------------------------------
# DataFrame cache  (keyed by absolute path string)
# ---------------------------------------------------------------------------

_DF_CACHE: Dict[str, pd.DataFrame] = {}


def _load_df(path: Path) -> pd.DataFrame:
    """Read and cache a forecast CSV. Returns empty DataFrame on failure."""
    key = str(path)
    if key not in _DF_CACHE:
        if path.exists():
            try:
                df = pd.read_csv(path, parse_dates=["ds"])
                _DF_CACHE[key] = df
            except Exception as exc:
                logger.warning("CSVDataLoader: failed to read %s - %s", path, exc)
                _DF_CACHE[key] = pd.DataFrame()
        else:
            logger.warning("CSVDataLoader: forecast file not found - %s", path)
            _DF_CACHE[key] = pd.DataFrame()
    return _DF_CACHE[key]


# ---------------------------------------------------------------------------
# Risk / confidence heuristic  (mirrors NPDataLoader logic)
# ---------------------------------------------------------------------------

def _risk_for_value(metric: str, val: float) -> str:
    """Derive a risk level from the last observed value."""
    m = metric.lower()
    if "disk_read" in m:
        if val > 50.0:
            return "HIGH"
        elif val > 20.0:
            return "MEDIUM"
        return "LOW"
    elif "disk_write" in m:
        if val > 5_242_880:
            return "HIGH"
        elif val > 1_048_576:
            return "MEDIUM"
        return "LOW"
    elif "mem" in m or "memory" in m:
        # Lower is worse for available %
        if val < 20.0:
            return "HIGH"
        elif val < 40.0:
            return "MEDIUM"
        return "LOW"
    else:
        # CPU, Disk % - higher is worse
        if val > 80.0:
            return "HIGH"
        elif val > 60.0:
            return "MEDIUM"
        return "LOW"


# ---------------------------------------------------------------------------
# CSVDataLoader
# ---------------------------------------------------------------------------

class CSVDataLoader:
    """
    Data loader for CSV-based forecast models (CHRONOS, HOLT_WINTERS, TIMESFM).

    Instantiate with the model name::

        loader = CSVDataLoader("CHRONOS")
        loader = CSVDataLoader("HOLT_WINTERS")
        loader = CSVDataLoader("TIMESFM")
    """

    def __init__(self, model: str) -> None:
        self.model = model.upper()
        if self.model not in _FORECAST_ROOTS:
            raise ValueError(
                f"CSVDataLoader: model must be one of {list(_FORECAST_ROOTS)}, got '{model}'"
            )

    # ------------------------------------------------------------------
    # Core forecast series
    # ------------------------------------------------------------------

    def get_forecast_series(
        self,
        metric: str,
        host: str,
        st: "datetime.date",
        et: "datetime.date",
    ) -> List[Tuple[str, float, float, float]]:
        """
        Return List of (date_str, p50, p10, p90) for the [st, et] window.
        Falls back to last-observed-value stub when the CSV is missing or
        the metric column is absent.
        """
        days = (et - st).days + 1
        path = _resolve_csv_path(self.model, host, days)
        df = _load_df(path)

        np_metric = NP_METRIC_MAP.get(metric, metric)

        if df.empty:
            return self._stub_series(metric, host, st, et)

        # Try np_metric name first (e.g. "cpu_pct"), then fall back to raw metric
        yhat_col = _col_yhat(df, np_metric) or _col_yhat(df, metric)
        if yhat_col is None:
            logger.warning(
                "CSVDataLoader[%s]: no column found for metric '%s' in %s",
                self.model, metric, path.name,
            )
            return self._stub_series(metric, host, st, et)

        lower_col = _col_lower(df, np_metric) or _col_lower(df, metric)
        upper_col = _col_upper(df, np_metric) or _col_upper(df, metric)

        result: List[Tuple[str, float, float, float]] = []
        for _, row in df.iterrows():
            try:
                row_date = pd.to_datetime(row["ds"]).date()
            except Exception:
                continue
            if not (st <= row_date <= et):
                continue

            p50 = float(row[yhat_col]) if pd.notna(row[yhat_col]) else 0.0
            p10 = float(row[lower_col]) if (lower_col and pd.notna(row[lower_col])) else p50 * 0.95
            p90 = float(row[upper_col]) if (upper_col and pd.notna(row[upper_col])) else p50 * 1.05

            # Enforce p10 <= p50 <= p90
            p10 = min(p10, p50)
            p90 = max(p90, p50)

            result.append((row_date.isoformat(), round(p50, 4), round(p10, 4), round(p90, 4)))

        if not result:
            return self._stub_series(metric, host, st, et)

        return result

    def _stub_series(
        self,
        metric: str,
        host: str,
        st: "datetime.date",
        et: "datetime.date",
    ) -> List[Tuple[str, float, float, float]]:
        """Return a linear-drift stub when forecast data is unavailable."""
        base = NPDataLoader.get_last_recorded_value(metric, host)
        days = (et - st).days + 1
        result = []
        cur = st
        for i in range(days):
            p50 = round(base * (1.0 + 0.002 * i), 4)
            p10 = round(p50 * 0.95, 4)
            p90 = round(p50 * 1.05, 4)
            result.append((cur.isoformat(), p50, p10, p90))
            cur += timedelta(days=1)
        return result

    # ------------------------------------------------------------------
    # Historical series (delegate to NPDataLoader - same raw data source)
    # ------------------------------------------------------------------

    def get_historical_series(
        self,
        metric: str,
        host: str,
        st: "datetime.date",
        et: "datetime.date",
    ) -> List[Tuple[str, float]]:
        """Historical actuals come from the same Excel ingestion pipeline."""
        return NPDataLoader.get_historical_series(metric, host, st, et)

    # ------------------------------------------------------------------
    # Last recorded value
    # ------------------------------------------------------------------

    def get_last_recorded_value(self, metric: str, host: str) -> float:
        return NPDataLoader.get_last_recorded_value(metric, host)

    # ------------------------------------------------------------------
    # As-of date (latest ds in the 90d forecast file)
    # ------------------------------------------------------------------

    def get_as_of_date(self, host: str) -> str:
        """Return the last date covered by the forecast file for *host*."""
        path = _resolve_csv_path(self.model, host, 90)
        df = _load_df(path)
        if not df.empty and "ds" in df.columns:
            try:
                last_ds = pd.to_datetime(df["ds"]).max()
                return last_ds.date().isoformat()
            except Exception:
                pass
        return NPDataLoader.get_as_of_date(host)

    # ------------------------------------------------------------------
    # Confidence & Risk
    # ------------------------------------------------------------------

    def get_confidence_and_risk(self, metric: str, host: str) -> Tuple[float, str]:
        """
        Confidence is fixed at a model-specific baseline (no val_loss available
        for zero-shot models). Risk is derived from the last observed value.
        """
        confidence_defaults = {
            "CHRONOS":      85.0,
            "HOLT_WINTERS": 78.0,
            "TIMESFM":      82.0,
        }
        confidence = confidence_defaults.get(self.model, 75.0)
        last_val = self.get_last_recorded_value(metric, host)
        risk = _risk_for_value(metric, last_val)
        return confidence, risk
