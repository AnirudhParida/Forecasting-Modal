"""
app/routers/host_prediction.py
===============================
FastAPI Router for Host Resource Prediction Summary API (with st & et parameters).
"""
from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Query

from app.models.schemas import HostPredictionSummaryResponse
from app.services.host_prediction_service import get_host_prediction_summary

router = APIRouter(prefix="/api/v1", tags=["Host Prediction"])


@router.get(
    "/host-prediction",
    response_model=HostPredictionSummaryResponse,
    summary="Get Host Resource Predictions (Disk, CPU, Memory) with Confidence & Risk Level",
    description=(
        "Retrieves last recorded dataset values (current_value), forecasted P50 values, "
        "percentage changes, confidence scores, and risk levels for Disk, CPU, and Memory usage "
        "over requested date range (st and et) or duration."
    ),
)
async def get_host_prediction(
    st: Optional[str] = Query(
        "2026-09-01",
        description="Prediction start date (YYYY-MM-DD)",
    ),
    et: Optional[str] = Query(
        "2026-09-30",
        description="Prediction end date (YYYY-MM-DD)",
    ),
    prediction_duration: Optional[str] = Query(
        None,
        description="Optional prediction duration horizon (e.g. 30d, 60d)",
    ),
    host_name: str = Query(
        "JPRUPIWEBCRP02",
        description="Host machine name",
    ),
    host_ip: str = Query(
        "10.78.33.83",
        description="Host IP address",
    ),
) -> HostPredictionSummaryResponse:
    """FastAPI endpoint handler for host resource prediction summary."""
    return get_host_prediction_summary(
        st=st,
        et=et,
        prediction_duration=prediction_duration,
        host_name=host_name,
        host_ip=host_ip,
    )
