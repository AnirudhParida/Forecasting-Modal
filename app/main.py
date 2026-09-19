"""
app/main.py
===========
Main FastAPI Application Entrypoint.
Provides REST API endpoints for:
1. 3, 6, 9 Month Timeseries Data (with P10, P50, P90 quantiles)
2. Explainability & Driver Interdependency Analysis
3. Host Resource Prediction Summary (Disk, CPU, Memory P50 & % change)
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.combined_hosts import router as combined_hosts_router
from app.routers.explainability import router as explainability_router
from app.routers.forecast_summary import router as forecast_summary_router
from app.routers.host_prediction import router as host_prediction_router
from app.routers.timeseries import router as timeseries_router

app = FastAPI(
    title="Forecasting & Driver Interdependency API — v1.1.0",
    description=(
        "FastAPI REST service delivering time series forecast quantiles (P10/P50/P90), "
        "Explainability and Driver Interdependency Analysis, and Host Resource Prediction Summaries."
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(timeseries_router)
app.include_router(explainability_router)
app.include_router(host_prediction_router)
app.include_router(forecast_summary_router)
app.include_router(combined_hosts_router)


@app.get("/", tags=["Health & Status"])
async def root():
    """Root status endpoint."""
    return {
        "service": "Forecasting & Driver Interdependency API — v1.1.0",
        "status": "online",
        "version": "1.1.0",
        "documentation": "/docs",
        "supported_models": ["STGNN", "NEURALPROPHET", "CHRONOS", "HOLT_WINTERS", "TIMESFM"],
        "endpoints": {
            "timeseries": "/api/v1/timeseries?timeframe=3months",
            "explainability": "/api/v1/explainability",
            "host_prediction": "/api/v1/host-prediction?host_name=JPRUPIWEBCRP02&prediction_duration=30d&model=STGNN",
            "forecast_summary": "/api/v1/forecast-summary?host_name=HYDUPINTAPP16&metric=cpu_pct&prediction_days=30&model=NEURALPROPHET",
            "forecast_summary_chronos": "/api/v1/forecast-summary?host_name=HYDUPINTAPP16&metric=cpu_pct&prediction_days=30&model=CHRONOS",
            "forecast_summary_holt_winters": "/api/v1/forecast-summary?host_name=HYDUPINTAPP16&metric=cpu_pct&prediction_days=30&model=HOLT_WINTERS",
            "forecast_summary_timesfm": "/api/v1/forecast-summary?host_name=HYDUPINTAPP16&metric=cpu_pct&prediction_days=30&model=TIMESFM",
            "combined_hosts_summary": "/api/v1/combined-hosts-summary?model=NEURALPROPHET",
        },
    }




@app.get("/health", tags=["Health & Status"])
async def health_check():
    """Service health check."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

