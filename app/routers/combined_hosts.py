"""
app/routers/combined_hosts.py
==============================
FastAPI Router for Combined Multi-Host Forecast Summary API.
Delivers 3-month historical averages and 15d, 30d, 60d, 90d forecasted averages for all hosts.
"""
from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Body, Query
from pydantic import BaseModel, Field

from app.models.schemas import CombinedHostsSummaryResponse
from app.services.combined_hosts_service import get_combined_hosts_summary

router = APIRouter(prefix="/api/v1", tags=["Combined Hosts Summary"])


class CombinedHostsRequestBody(BaseModel):
    st: Optional[str] = Field(None, description="Optional start date (YYYY-MM-DD)")
    et: Optional[str] = Field(None, description="Optional end date (YYYY-MM-DD)")
    prediction_duration: Optional[str] = Field(None, description="Optional forecast duration alias (e.g. 15d, 30d, 60d, 90d)")
    model: str = Field("NEURALPROPHET", description="Forecasting model to use (STGNN | NEURALPROPHET | CHRONOS | HOLT_WINTERS | TIMESFM)")


@router.get(
    "/combined-hosts-summary",
    response_model=CombinedHostsSummaryResponse,
    summary="Get Combined Multi-Host Forecast Summary (3-Month Historical Avg + 15d, 30d, 60d, 90d Forecast Averages)",
    description=(
        "Retrieves combined forecast analytics for all registered hosts. For each host, delivers all 5 metrics "
        "(CPU, Memory, Disk %, Disk Read Bytes, Disk Write Bytes) containing the 3-month historical average "
        "and 15-day, 30-day, 60-day, and 90-day forecast averages."
    ),
)
async def get_combined_hosts_summary_endpoint(
    st: Optional[str] = Query(None, description="Optional start date (YYYY-MM-DD)"),
    et: Optional[str] = Query(None, description="Optional end date (YYYY-MM-DD)"),
    prediction_duration: Optional[str] = Query(None, description="Optional duration alias (e.g. 30d)"),
    model: str = Query("NEURALPROPHET", description="Forecasting model to use (STGNN | NEURALPROPHET | CHRONOS | HOLT_WINTERS | TIMESFM)"),
) -> CombinedHostsSummaryResponse:
    """GET handler for combined multi-host forecast summary."""
    return get_combined_hosts_summary(
        st=st,
        et=et,
        prediction_duration=prediction_duration,
        model=model,
    )


@router.post(
    "/combined-hosts-summary",
    response_model=CombinedHostsSummaryResponse,
    summary="POST Combined Multi-Host Forecast Summary",
    description=(
        "POST payload handler returning 3-month historical averages and 15d, 30d, 60d, 90d forecast averages across all hosts."
    ),
)
async def post_combined_hosts_summary_endpoint(
    body: CombinedHostsRequestBody = Body(...),
) -> CombinedHostsSummaryResponse:
    """POST handler for combined multi-host forecast summary."""
    return get_combined_hosts_summary(
        st=body.st,
        et=body.et,
        prediction_duration=body.prediction_duration,
        model=body.model,
    )
