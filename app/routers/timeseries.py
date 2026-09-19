"""
app/routers/timeseries.py
=========================
FastAPI Router for Time Series Data Endpoint (with st and et parameters).
"""
from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Query

from app.models.schemas import TimeseriesResponse
from app.services.timeseries_service import get_timeseries_data

router = APIRouter(prefix="/api/v1", tags=["Timeseries"])


@router.get(
    "/timeseries",
    response_model=TimeseriesResponse,
    summary="Get Timeseries Data: 3-Month Historical Actuals & Forecast Predictions (P10, P50, P90)",
    description=(
        "Retrieves timeseries data using prediction start date (st) and end date (et). "
        "Returns 'actual_data' (3 months of past actual data points prior to st, containing timestamp and value only) "
        "and 'prediction_data' (model forecast points containing timestamp, p50, p10, and p90)."
    ),
)
async def get_timeseries(
    st: Optional[str] = Query(
        "2026-09-01",
        description="Prediction start date (YYYY-MM-DD)",
    ),
    et: Optional[str] = Query(
        "2026-09-30",
        description="Prediction end date (YYYY-MM-DD)",
    ),
    timeframe: Optional[str] = Query(
        None,
        description="Optional timeframe alias (e.g. 1months, 3months)",
    ),
    metric: str = Query(
        "host_cpu_usage",
        description="Target metric key (e.g. host_cpu_usage, host_mem_available, CPU Utilization (%))",
    ),
    host_name: str = Query(
        "JPRUPIWEBCRP02",
        description="Target host machine name",
    ),
    host_ip: str = Query(
        "10.78.33.83",
        description="Target host IP address",
    ),
    model: str = Query(
        "STGNN",
        description="Forecasting model to use (STGNN | NEURALPROPHET | CHRONOS | HOLT_WINTERS | TIMESFM)",
    ),
) -> TimeseriesResponse:
    """FastAPI endpoint handler for timeseries data retrieval with st & et parameters."""
    return get_timeseries_data(
        st=st,
        et=et,
        timeframe=timeframe,
        metric=metric,
        host_name=host_name,
        host_ip=host_ip,
        model=model,
    )
