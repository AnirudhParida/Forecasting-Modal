"""
app/services/timeseries_service.py
==================================
Service layer for time series data processing using DataLoader.
Dynamically reads parquet datasets and STGNN forecast artifacts to populate:
1. actual_data: 3 months of past actual data points (timestamp and value only).
2. prediction_data: model forecasts (timestamp, p50, p10, p90).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from app.models.schemas import (
    ActualDataPoint,
    PredictionDataPoint,
    TimeseriesResponse,
    TimeseriesSummaryStats,
    TimeseriesWindow,
)
from app.services.data_loader import DataLoader, METRIC_FRIENDLY_NAMES


def _parse_date(date_val: Optional[str], default_date: datetime.date) -> datetime.date:
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


def get_timeseries_data(
    st: Optional[str] = None,
    et: Optional[str] = None,
    timeframe: Optional[str] = None,
    metric: str = "host_cpu_usage",
    host_name: str = "JPRUPIWEBCRP02",
    host_ip: str = "10.78.33.83",
) -> TimeseriesResponse:
    """Generate timeseries response dynamically from dataset parquet and model artifacts."""
    today = datetime.now().date()

    # Determine default prediction range [st, et]
    default_st = datetime(today.year, today.month, 1).date()
    if today.month == 12:
        default_et = datetime(today.year, 12, 31).date()
    else:
        default_et = (datetime(today.year, today.month + 1, 1) - timedelta(days=1)).date()

    pred_st = _parse_date(st, default_st)
    pred_et = _parse_date(et, default_et)

    if pred_et < pred_st:
        pred_et = pred_st + timedelta(days=29)

    as_of_date_str = DataLoader.get_as_of_date()

    # Resolve metric node ID and display label
    node_id = DataLoader.resolve_node_id(metric)
    base_name = node_id.split("|")[0]
    display_label = METRIC_FRIENDLY_NAMES.get(base_name, base_name.replace("_", " ").title())

    # 1. DYNAMIC ACTUAL DATA (Past 3 months prior to prediction start date)
    hist_st = pred_st - timedelta(days=90)
    hist_et = pred_st - timedelta(days=1)

    hist_tuples = DataLoader.get_historical_series(node_id, hist_st, hist_et)
    actual_data: List[ActualDataPoint] = [
        ActualDataPoint(timestamp=dt_s, value=val) for dt_s, val in hist_tuples
    ]
    actual_values = [pt.value for pt in actual_data]

    # 2. DYNAMIC PREDICTION DATA ([st, et])
    pred_tuples = DataLoader.get_forecast_series(node_id, pred_st, pred_et)
    prediction_data: List[PredictionDataPoint] = [
        PredictionDataPoint(timestamp=dt_s, p50=p50, p10=p10, p90=p90)
        for dt_s, p50, p10, p90 in pred_tuples
    ]
    p50_values = [pt.p50 for pt in prediction_data]

    # Summary Statistics
    historical_avg = round(float(sum(actual_values) / max(len(actual_values), 1)), 2)
    predicted_p50_avg = round(float(sum(p50_values) / max(len(p50_values), 1)), 2)

    summary_stats = TimeseriesSummaryStats(
        historical_avg=historical_avg,
        predicted_p50_avg=predicted_p50_avg,
        min_p50=round(float(min(p50_values)), 2) if p50_values else 0.0,
        max_p50=round(float(max(p50_values)), 2) if p50_values else 0.0,
    )

    return TimeseriesResponse(
        metric=node_id,
        metric_label=display_label,
        host_name=host_name,
        host_ip=host_ip,
        as_of_date=as_of_date_str,
        prediction_window=TimeseriesWindow(st=pred_st.isoformat(), et=pred_et.isoformat()),
        historical_window=TimeseriesWindow(st=hist_st.isoformat(), et=hist_et.isoformat()),
        summary_stats=summary_stats,
        actual_data=actual_data,
        prediction_data=prediction_data,
    )
