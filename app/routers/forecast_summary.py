"""
app/routers/forecast_summary.py
================================
FastAPI Router for Metric Forecast Summary API.
Provides 3-month historical averages and forecasted averages over requested durations (15, 30, 60, 90 days).
"""
from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Body, Query
from pydantic import BaseModel, Field

from app.models.schemas import ForecastSummaryResponse
from app.services.forecast_summary_service import get_forecast_summary

router = APIRouter(prefix="/api/v1", tags=["Forecast Summary"])


class ForecastSummaryRequestBody(BaseModel):
    host_name: Optional[str] = Field(None, description="Host machine name (e.g. HYDUPINTAPP16, JPRUPIWEBCRP02)")
    host_ip: Optional[str] = Field(None, description="Host IP address (e.g. 10.50.98.26, 10.78.33.83)")
    prediction_days: Optional[int] = Field(None, description="Prediction duration horizon (15, 30, 60, or 90 days)")
    st: Optional[str] = Field(None, description="Optional prediction start date (YYYY-MM-DD)")
    et: Optional[str] = Field(None, description="Optional prediction end date (YYYY-MM-DD)")
    metric: str = Field("cpu_pct", description="Target metric (cpu_pct, memory_pct, disk_pct, disk_read_bytes, disk_write_bytes, or all)")
    model: str = Field("NEURALPROPHET", description="Forecasting model to use (STGNN | NEURALPROPHET | CHRONOS | HOLT_WINTERS | TIMESFM)")


@router.get(
    "/forecast-summary",
    response_model=ForecastSummaryResponse,
    summary="Get Forecast Summary: 3-Month Historical Average & Forecasted Horizon Average",
    description=(
        "Returns the average actual metric value over the past 3 months (90 days) and the average "
        "forecasted value for the requested prediction horizon (15, 30, 60, or 90 days). "
        "Accepts host identification via Host Name or Host IP, and defaults to NEURALPROPHET model."
    ),
)
async def get_forecast_summary_endpoint(
    host_name: Optional[str] = Query(None, description="Host machine name (e.g. HYDUPINTAPP16)"),
    host_ip: Optional[str] = Query(None, description="Host IP address (e.g. 10.50.98.26)"),
    prediction_days: Optional[int] = Query(None, description="Prediction horizon in days (15, 30, 60, 90)"),
    st: Optional[str] = Query(None, description="Prediction start date (YYYY-MM-DD)"),
    et: Optional[str] = Query(None, description="Prediction end date (YYYY-MM-DD)"),
    metric: str = Query("cpu_pct", description="Target metric key or 'all'"),
    model: str = Query("NEURALPROPHET", description="Forecasting model to use (STGNN | NEURALPROPHET | CHRONOS | HOLT_WINTERS | TIMESFM)"),
) -> ForecastSummaryResponse:
    """GET handler for metric forecast summary."""
    return get_forecast_summary(
        host_name=host_name,
        host_ip=host_ip,
        prediction_days=prediction_days,
        st=st,
        et=et,
        metric=metric,
        model=model,
    )


@router.post(
    "/forecast-summary",
    response_model=ForecastSummaryResponse,
    summary="POST Forecast Summary: 3-Month Historical Average & Forecasted Horizon Average",
    description=(
        "POST payload handler returning 3-month historical averages and forecasted averages for a host metric."
    ),
)
async def post_forecast_summary_endpoint(
    body: ForecastSummaryRequestBody = Body(...),
) -> ForecastSummaryResponse:
    """POST handler for metric forecast summary."""
    return get_forecast_summary(
        host_name=body.host_name,
        host_ip=body.host_ip,
        prediction_days=body.prediction_days,
        st=body.st,
        et=body.et,
        metric=body.metric,
        model=body.model,
    )
