"""
app/services/forecast_summary_service.py
=========================================
Service layer for calculating 3-month historical averages and forecasted duration averages.
Supports host resolution by Host Name or IP, metric filtering, and model choice (default: NEURALPROPHET).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from app.models.schemas import (
    ForecastSummaryResponse,
    ForecastSummaryWindow,
    MetricForecastSummaryItem,
)
from app.services.data_loader import DataLoader, METRIC_FRIENDLY_NAMES
from app.services.np_data_loader import NPDataLoader
from app.services.csv_data_loader import CSVDataLoader

_CSV_MODELS = frozenset({"CHRONOS", "HOLT_WINTERS", "TIMESFM"})


def _get_loader(model: str):
    """Return the appropriate data-loader for the requested model."""
    m = model.upper()
    if m == "NEURALPROPHET":
        return NPDataLoader
    if m in _CSV_MODELS:
        return CSVDataLoader(m)
    return DataLoader  # STGNN (default)

# Host Mapping Dictionary (Host Name <-> Host IP)
HOST_IP_MAP: Dict[str, str] = {
    "HYDUPINTAPP16": "10.50.98.26",
    "10.50.98.26": "HYDUPINTAPP16",
    "JPRUPIWEBCRP02": "10.78.33.83",
    "10.78.33.83": "JPRUPIWEBCRP02",
}

# Metric Units Mapping
METRIC_UNITS: Dict[str, str] = {
    "cpu_pct": "%",
    "host_cpu_usage": "%",
    "memory_pct": "%",
    "host_mem_avail_pct": "%",
    "disk_pct": "%",
    "host_disk_avail_pct": "%",
    "disk_read_bytes": "KB/s",
    "disk_read_ops": "ops/s",
    "host_disk_read_ops_sec": "ops/s",
    "disk_write_bytes": "KB/s",
    "host_disk_write_bytes_sec": "KB/s",
}

ALL_CANONICAL_METRICS = ["cpu_pct", "memory_pct", "disk_pct", "disk_read_bytes", "disk_write_bytes"]


def _resolve_host(host_name: Optional[str], host_ip: Optional[str]) -> Tuple[str, str]:
    """Resolve target host alias and IP address."""
    name = (host_name or "").strip()
    ip = (host_ip or "").strip()

    if not name and not ip:
        name = "HYDUPINTAPP16"
        ip = "10.50.98.26"
    elif name and not ip:
        ip = HOST_IP_MAP.get(name, "10.50.98.26")
    elif ip and not name:
        name = HOST_IP_MAP.get(ip, "HYDUPINTAPP16")

    return name, ip


def _parse_date(date_val: Optional[str], default_date: datetime.date) -> datetime.date:
    """Safely parse input string or timestamp into datetime.date."""
    if not date_val:
        return default_date
    val_str = str(date_val).strip()

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(val_str.split("T")[0], "%Y-%m-%d").date()
        except Exception:
            pass

    try:
        ts = float(val_str)
        if ts > 1e11:
            ts = ts / 1000.0
        return datetime.fromtimestamp(ts).date()
    except Exception:
        pass

    return default_date


def get_forecast_summary(
    host_name: Optional[str] = None,
    host_ip: Optional[str] = None,
    prediction_days: Optional[int] = None,
    st: Optional[str] = None,
    et: Optional[str] = None,
    metric: str = "cpu_pct",
    model: str = "NEURALPROPHET",
) -> ForecastSummaryResponse:
    """
    Compute and return historical 3-month average and forecasted average for specified host and metric(s).
    """
    resolved_host, resolved_ip = _resolve_host(host_name, host_ip)
    model_name = (model or "NEURALPROPHET").upper()
    Loader = _get_loader(model_name)
    is_stgnn = model_name not in {"NEURALPROPHET"} | _CSV_MODELS

    # Determine date range [st, et]
    as_of_str = Loader.get_as_of_date(resolved_host) if not is_stgnn else DataLoader.get_as_of_date()
    try:
        as_of_date = datetime.strptime(as_of_str, "%Y-%m-%d").date()
    except Exception:
        as_of_date = datetime.now().date()

    default_st = as_of_date + timedelta(days=1)

    if prediction_days in (15, 30, 60, 90):
        days_horizon = prediction_days
    elif prediction_days and prediction_days > 0:
        days_horizon = prediction_days
    else:
        days_horizon = 30

    default_et = default_st + timedelta(days=days_horizon - 1)

    pred_st = _parse_date(st, default_st)
    pred_et = _parse_date(et, default_et)

    if pred_et < pred_st:
        pred_et = pred_st + timedelta(days=days_horizon - 1)

    actual_days = (pred_et - pred_st).days + 1

    # Historical window starting from August 2025 prior to forecast start date
    hist_st = datetime(2025, 8, 1).date()
    hist_et = pred_st - timedelta(days=1)

    # Determine which metrics to query
    req_metric = (metric or "cpu_pct").strip().lower()
    if req_metric == "all":
        target_metrics = ALL_CANONICAL_METRICS
    else:
        target_metrics = [req_metric]

    metrics_result: Dict[str, MetricForecastSummaryItem] = {}

    for m_key in target_metrics:
        # Resolve canonical key and friendly display label
        node_id = m_key if not is_stgnn else DataLoader.resolve_node_id(m_key)

        base_name = m_key.split("|")[0]
        display_label = METRIC_FRIENDLY_NAMES.get(base_name, base_name.replace("_", " ").title())
        unit = METRIC_UNITS.get(m_key, "%")

        # 1. Fetch 3-Month Historical Actual Series
        if not is_stgnn:
            hist_tuples = Loader.get_historical_series(m_key, resolved_host, hist_st, hist_et)
        else:
            hist_tuples = DataLoader.get_historical_series(node_id, hist_st, hist_et)

        hist_values = [v for _, v in hist_tuples if v is not None]
        if hist_values:
            last_3m_avg = round(float(sum(hist_values) / len(hist_values)), 2)
        else:
            # Fallback to last recorded value if window has missing historical points
            if not is_stgnn:
                last_3m_avg = round(float(Loader.get_last_recorded_value(m_key, resolved_host)), 2)
            else:
                last_3m_avg = round(float(DataLoader.get_last_recorded_value(node_id)), 2)

        # 2. Fetch Forecast Series for Requested Duration [st, et]
        if not is_stgnn:
            pred_tuples = Loader.get_forecast_series(m_key, resolved_host, pred_st, pred_et)
        else:
            pred_tuples = DataLoader.get_forecast_series(node_id, pred_st, pred_et)

        p50_values = [p50 for _, p50, _, _ in pred_tuples if p50 is not None]
        if p50_values:
            forecasted_avg = round(float(sum(p50_values) / len(p50_values)), 2)
        else:
            forecasted_avg = last_3m_avg

        diff = round(forecasted_avg - last_3m_avg, 2)
        if last_3m_avg != 0:
            pct_change = round((diff / abs(last_3m_avg)) * 100.0, 2)
        else:
            pct_change = 0.0

        metrics_result[m_key] = MetricForecastSummaryItem(
            metric_key=m_key,
            metric_label=display_label,
            unit=unit,
            last_3_months_avg=last_3m_avg,
            forecasted_avg=forecasted_avg,
            difference=diff,
            percentage_change=pct_change,
        )

    return ForecastSummaryResponse(
        status="success",
        host_name=resolved_host,
        host_ip=resolved_ip,
        model_used=model_name,
        prediction_days=actual_days,
        query_period=ForecastSummaryWindow(st=pred_st.isoformat(), et=pred_et.isoformat()),
        historical_period_3m=ForecastSummaryWindow(st=hist_st.isoformat(), et=hist_et.isoformat()),
        metrics=metrics_result,
    )
