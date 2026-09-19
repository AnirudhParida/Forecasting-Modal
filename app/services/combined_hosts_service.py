"""
app/services/combined_hosts_service.py
======================================
Service layer for computing combined multi-host resource forecast analytics.
Iterates over all hosts and all 5 metrics, providing 3-month historical averages
alongside 15d, 30d, 60d, and 90d forecasted averages.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from app.models.schemas import (
    CombinedHostsSummaryResponse,
    CombinedMetricSummaryItem,
    HostCombinedSummaryItem,
)
from app.services.data_loader import DataLoader, METRIC_FRIENDLY_NAMES
from app.services.np_data_loader import NPDataLoader
from app.services.csv_data_loader import CSVDataLoader
from app.services.forecast_summary_service import HOST_IP_MAP, METRIC_UNITS, ALL_CANONICAL_METRICS, _CSV_MODELS, _get_loader


def _parse_date(date_val: Optional[str], default_date: datetime.date) -> datetime.date:
    if not date_val:
        return default_date
    val_str = str(date_val).strip()

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(val_str.split("T")[0], "%Y-%m-%d").date()
        except Exception:
            pass

    return default_date


def get_combined_hosts_summary(
    st: Optional[str] = None,
    et: Optional[str] = None,
    prediction_duration: Optional[str] = None,
    model: str = "NEURALPROPHET",
) -> CombinedHostsSummaryResponse:
    """
    Compute and return multi-host combined resource forecast summaries.
    """
    model_name = (model or "NEURALPROPHET").upper()
    Loader = _get_loader(model_name)
    is_stgnn = model_name not in {"NEURALPROPHET"} | _CSV_MODELS

    # Canonical host list
    registered_hosts = ["HYDUPINTAPP16", "JPRUPIWEBCRP02"]

    # Determine reference dataset cut-off date
    as_of_str = Loader.get_as_of_date("HYDUPINTAPP16") if not is_stgnn else DataLoader.get_as_of_date()
    try:
        as_of_date = datetime.strptime(as_of_str, "%Y-%m-%d").date()
    except Exception:
        as_of_date = datetime.now().date()

    # Historical 3-month date range up to dataset cutoff
    hist_st = as_of_date - timedelta(days=89)
    hist_et = as_of_date

    host_summaries: List[HostCombinedSummaryItem] = []

    for host_alias in registered_hosts:
        host_ip = HOST_IP_MAP.get(host_alias, "10.50.98.26")
        metrics_dict: Dict[str, CombinedMetricSummaryItem] = {}

        for m_key in ALL_CANONICAL_METRICS:
            node_id = m_key if not is_stgnn else DataLoader.resolve_node_id(m_key)
            base_name = m_key.split("|")[0]
            display_label = METRIC_FRIENDLY_NAMES.get(base_name, base_name.replace("_", " ").title())
            unit = METRIC_UNITS.get(m_key, "%")

            # 1. Past 3 Months Actual Historical Average
            if not is_stgnn:
                hist_tuples = Loader.get_historical_series(m_key, host_alias, hist_st, hist_et)
            else:
                hist_tuples = DataLoader.get_historical_series(node_id, hist_st, hist_et)

            hist_vals = [v for _, v in hist_tuples if v is not None]
            if hist_vals:
                last_3m_avg = round(float(sum(hist_vals) / len(hist_vals)), 2)
            else:
                if not is_stgnn:
                    last_3m_avg = round(float(Loader.get_last_recorded_value(m_key, host_alias)), 2)
                else:
                    last_3m_avg = round(float(DataLoader.get_last_recorded_value(node_id)), 2)

            # 2. Compute Horizon Forecast Averages (15d, 30d, 60d, 90d)
            horizon_averages: Dict[int, float] = {}
            for h in (15, 30, 60, 90):
                p_st = as_of_date + timedelta(days=1)
                p_et = as_of_date + timedelta(days=h)

                if not is_stgnn:
                    pred_tuples = Loader.get_forecast_series(m_key, host_alias, p_st, p_et)
                else:
                    pred_tuples = DataLoader.get_forecast_series(node_id, p_st, p_et)

                p50_vals = [p50 for _, p50, _, _ in pred_tuples if p50 is not None]
                if p50_vals:
                    horizon_averages[h] = round(float(sum(p50_vals) / len(p50_vals)), 2)
                else:
                    horizon_averages[h] = last_3m_avg

            metrics_dict[m_key] = CombinedMetricSummaryItem(
                metric_key=m_key,
                metric_label=display_label,
                unit=unit,
                last_3_months_avg=last_3m_avg,
                avg_15d=horizon_averages[15],
                avg_30d=horizon_averages[30],
                avg_60d=horizon_averages[60],
                avg_90d=horizon_averages[90],
            )

        host_summaries.append(
            HostCombinedSummaryItem(
                host_name=host_alias,
                host_ip=host_ip,
                metrics=metrics_dict,
            )
        )

    return CombinedHostsSummaryResponse(
        status="success",
        model_used=model_name,
        as_of_date=as_of_str,
        hosts_count=len(host_summaries),
        hosts=host_summaries,
    )
