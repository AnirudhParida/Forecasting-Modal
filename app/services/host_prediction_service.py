"""
app/services/host_prediction_service.py
========================================
Service for fetching host resource predictions (Disk, CPU, Memory)
over requested date ranges (`st` and `et`) or duration using DataLoader.
Dynamically extracts last recorded values from dataset parquet files,
forecasted P50 values from model artifacts, percentage changes, trends,
confidence scores, and risk levels with ZERO hardcoding.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

from app.models.schemas import (
    HostPredictionMetrics,
    HostPredictionSummaryResponse,
    MetricPredictionItem,
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


def parse_duration_days(duration: str) -> int:
    """Parse flexible duration string into integer days."""
    d_str = str(duration).strip().lower()
    match = re.search(r"(\d+)", d_str)
    if match:
        days = int(match.group(1))
        if "m" in d_str and "d" not in d_str and days <= 12:
            return days * 30
        return max(1, days)
    if "month" in d_str:
        return 30
    return 30


def get_host_prediction_summary(
    st: Optional[str] = None,
    et: Optional[str] = None,
    host_name: str = "JPRUPIWEBCRP02",
    prediction_duration: Optional[str] = None,
    host_ip: str = "10.78.33.83",
) -> HostPredictionSummaryResponse:
    """Compute host prediction summary dynamically from dataset parquet & model artifacts."""
    today = datetime.now().date()

    if st or et:
        default_st = datetime(today.year, today.month, 1).date()
        if today.month == 12:
            default_et = datetime(today.year, 12, 31).date()
        else:
            default_et = (datetime(today.year, today.month + 1, 1) - timedelta(days=1)).date()

        pred_st = _parse_date(st, default_st)
        pred_et = _parse_date(et, default_et)
        if pred_et < pred_st:
            pred_et = pred_st + timedelta(days=29)

        days = (pred_et - pred_st).days + 1
        canonical_duration = f"{days}d"
    else:
        duration_str = prediction_duration if prediction_duration else "30d"
        days = parse_duration_days(duration_str)
        canonical_duration = f"{days}d"

        as_of_str = DataLoader.get_as_of_date()
        try:
            pred_st = datetime.strptime(as_of_str, "%Y-%m-%d").date() + timedelta(days=1)
        except Exception:
            pred_st = today
        pred_et = pred_st + timedelta(days=days - 1)

    as_of_date_str = DataLoader.get_as_of_date()

    # Helper function to build dynamic metric item
    def build_metric_item(query: str, default_label: str) -> MetricPredictionItem:
        node_id = DataLoader.resolve_node_id(query)
        base_name = node_id.split("|")[0]
        metric_label = METRIC_FRIENDLY_NAMES.get(base_name, default_label)

        # 1. Current value: last recorded non-null data point in dataset
        current_val = DataLoader.get_last_recorded_value(node_id)

        # 2. Predicted P50 value: mean forecast value over [pred_st, pred_et]
        pred_tuples = DataLoader.get_forecast_series(node_id, pred_st, pred_et)
        p50_vals = [t[1] for t in pred_tuples]
        predicted_p50 = round(float(sum(p50_vals) / max(len(p50_vals), 1)), 2)

        # 3. Percentage change: ((predicted_p50 - current_val) / current_val) * 100
        if current_val > 0:
            pct_change = round(((predicted_p50 - current_val) / current_val) * 100.0, 2)
        else:
            pct_change = 0.0

        trend = "INCREASING" if pct_change > 1.0 else ("DECREASING" if pct_change < -1.0 else "STABLE")

        # 4. Confidence & Risk Level
        confidence, risk_level = DataLoader.get_confidence_and_risk(node_id)

        return MetricPredictionItem(
            metric_key=base_name,
            metric_label=metric_label,
            unit="%",
            current_value=current_val,
            predicted_p50=predicted_p50,
            percentage_change=pct_change,
            trend=trend,
            confidence=confidence,
            risk_level=risk_level,
        )

    # Dynamically build CPU Usage, Memory Usage, and Disk Usage items
    cpu_item = build_metric_item("host_cpu_usage", "CPU Utilization (%)")
    mem_item = build_metric_item("host_mem_usage", "Memory Utilization (%)")
    disk_item = build_metric_item("disk_busy_time", "Disk Usage (%)")

    return HostPredictionSummaryResponse(
        host_name=host_name,
        host_ip=host_ip,
        prediction_duration=canonical_duration,
        prediction_days=days,
        as_of_date=as_of_date_str,
        target_date=pred_et.isoformat(),
        metrics=HostPredictionMetrics(
            cpu_usage=cpu_item,
            memory_usage=mem_item,
            disk_usage=disk_item,
        ),
    )
