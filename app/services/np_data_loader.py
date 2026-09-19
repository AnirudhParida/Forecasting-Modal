"""
app/services/np_data_loader.py
==============================
NeuralProphet Data Adapter.
Reads from `artifacts_np` pre-computed JSON files.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

import sys
NP_PIPELINE_ROOT = Path("/home/anirudh.parida@apmosys.mahape/Documents/Phrophet_forecast")
if str(NP_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(NP_PIPELINE_ROOT))

# pyrefly: ignore [missing-import]
from pipeline.ingestion import load_and_preprocess  # noqa: E402
# pyrefly: ignore [missing-import]
from config.settings import HOSTS                   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = ROOT / "artifacts_np"

NP_METRIC_MAP = {
    "host_cpu_usage": "cpu_pct",
    "cpu_pct": "cpu_pct",
    "host_mem_avail_pct": "memory_pct",
    "memory_pct": "memory_pct",
    "host_disk_avail_pct": "disk_pct",
    "disk_pct": "disk_pct",
    "host_disk_read_ops_sec": "disk_read_bytes",
    "disk_read_bytes": "disk_read_bytes",
    "disk_read_ops": "disk_read_ops",
    "host_disk_write_bytes_sec": "disk_write_bytes",
    "disk_write_bytes": "disk_write_bytes",
}


class NPDataLoader:
    _historical_data_cache: Dict[str, pd.DataFrame] = {}

    @classmethod
    def _get_historical_df(cls, host: str) -> pd.DataFrame:
        if host not in cls._historical_data_cache:
            if host in HOSTS:
                all_data = load_and_preprocess()
                if host in all_data:
                    cls._historical_data_cache[host] = all_data[host]
                else:
                    return pd.DataFrame()
            else:
                return pd.DataFrame()
        return cls._historical_data_cache[host]

    @classmethod
    def _resolve_df_col(cls, df: pd.DataFrame, metric: str, np_metric: str) -> Optional[str]:
        if df.empty:
            return None
        if np_metric in df.columns:
            return np_metric
        if metric in df.columns:
            return metric
        return None

    @staticmethod
    def _determine_horizon(st: datetime.date, et: datetime.date) -> int:
        days = (et - st).days + 1
        if days <= 15:
            return 15
        elif days <= 30:
            return 30
        elif days <= 60:
            return 60
        else:
            return 90

    @classmethod
    def _read_cache(cls, host: str, horizon: int) -> dict:
        cache_file = CACHE_DIR / f"{host}_forecast_{horizon}d.json"
        if not cache_file.exists():
            return {}
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    @classmethod
    def get_forecast_series(
        cls, metric: str, host: str, st: datetime.date, et: datetime.date
    ) -> List[Tuple[str, float, float, float]]:
        """Returns List of (date_str, p50, p10, p90)"""
        np_metric = NP_METRIC_MAP.get(metric, metric)
        horizon = cls._determine_horizon(st, et)
        cache_data = cls._read_cache(host, horizon)

        metrics_data = cache_data.get("metrics", {})
        if np_metric not in metrics_data:
            return []

        forecast_list = metrics_data[np_metric].get("forecast", [])
        
        result = []
        for row in forecast_list:
            dt_str = row["date"]
            dt_obj = datetime.strptime(dt_str, "%Y-%m-%d").date()
            if st <= dt_obj <= et:
                p50 = row.get("yhat", 0.0)
                p10 = row.get("yhat_lower")
                if p10 is None:
                    p10 = p50 * 0.95
                p90 = row.get("yhat_upper")
                if p90 is None:
                    p90 = p50 * 1.05
                result.append((dt_str, p50, p10, p90))
        return result

    @classmethod
    def get_historical_series(
        cls, metric: str, host: str, st: datetime.date, et: datetime.date
    ) -> List[Tuple[str, float]]:
        """Returns List of (date_str, value)"""
        np_metric = NP_METRIC_MAP.get(metric, metric)
        df = cls._get_historical_df(host)
        col_name = cls._resolve_df_col(df, metric, np_metric)
        if not col_name:
            return []
        
        mask = (df["ds"].dt.date >= st) & (df["ds"].dt.date <= et)
        filtered = df[mask]
        
        result = []
        for _, row in filtered.iterrows():
            if pd.notna(row[col_name]):
                dt_str = row["ds"].date().isoformat()
                result.append((dt_str, float(row[col_name])))
        return result

    @classmethod
    def get_last_recorded_value(cls, metric: str, host: str) -> float:
        np_metric = NP_METRIC_MAP.get(metric, metric)
        df = cls._get_historical_df(host)
        col_name = cls._resolve_df_col(df, metric, np_metric)
        if not col_name:
            return 0.0
        
        valid_series = df[col_name].dropna()
        if valid_series.empty:
            return 0.0
        return float(valid_series.iloc[-1])

    @classmethod
    def get_confidence_and_risk(cls, metric: str, host: str) -> Tuple[str, str]:
        np_metric = NP_METRIC_MAP.get(metric, metric)
        cache_data = cls._read_cache(host, 30)
        metrics_data = cache_data.get("metrics", {})
        if np_metric not in metrics_data:
            return 70.0, "Medium"
        
        eval_metrics = metrics_data[np_metric].get("evaluation", {})
        mape = eval_metrics.get("mape")
        if mape is None:
            return 70.0, "Medium"
        
        if mape < 10.0:
            confidence = 90.0
            risk = "Low"
        elif mape < 25.0:
            confidence = 70.0
            risk = "Medium"
        else:
            confidence = 40.0
            risk = "High"
            
        return confidence, risk

    @classmethod
    def get_top_drivers(cls, metric: str, host: str, top_k: int = 5) -> List[dict]:
        """
        Return regressors used for this metric as drivers.
        In NeuralProphet pipeline, all other metrics act as regressors.
        """
        np_metric = NP_METRIC_MAP.get(metric, metric)
        all_metrics = ["cpu_pct", "memory_pct", "disk_pct", "disk_read_ops", "disk_write_bytes"]
        drivers = []
        rank = 1
        for m in all_metrics:
            if m != np_metric:
                drivers.append({
                    "rank": rank,
                    "name": m.replace("_", " ").title(),
                    "impact_pct": round(25.0 / rank, 2),
                    "strength": 0.5,
                    "lag": "t-14",
                })
                rank += 1
        return drivers[:top_k]

    @classmethod
    def get_as_of_date(cls, host: str) -> str:
        cache_data = cls._read_cache(host, 30)
        return cache_data.get("as_of_date", datetime.now().strftime("%Y-%m-%d"))

