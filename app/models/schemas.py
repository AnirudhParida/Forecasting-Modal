"""
app/models/schemas.py
=====================
Pydantic schemas for request and response validation for:
1. Timeseries Data API (with st & et query parameters, actual_data, prediction_data)
2. Explainability & Driver Interdependency Analysis API
3. Host Resource Prediction Summary API (with st & et, confidence & risk_level per metric)
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ── Timeseries Schemas ────────────────────────────────────────────────────────

class ActualDataPoint(BaseModel):
    timestamp: str = Field(..., description="Timestamp in ISO 8601 date format")
    value: float = Field(..., description="Historical observed actual data point value")


class PredictionDataPoint(BaseModel):
    timestamp: str = Field(..., description="Timestamp in ISO 8601 date format")
    p50: float = Field(..., description="50th percentile (median) forecasted value")
    p10: float = Field(..., description="10th percentile lower bound forecast")
    p90: float = Field(..., description="90th percentile upper bound forecast")


class TimeseriesWindow(BaseModel):
    st: str = Field(..., description="Start date (YYYY-MM-DD)")
    et: str = Field(..., description="End date (YYYY-MM-DD)")


class TimeseriesSummaryStats(BaseModel):
    historical_avg: float = Field(..., description="3-month historical baseline average")
    predicted_p50_avg: float = Field(..., description="Average predicted P50 value over prediction horizon")
    min_p50: float = Field(..., description="Minimum P50 forecast value")
    max_p50: float = Field(..., description="Maximum P50 forecast value")


class TimeseriesResponse(BaseModel):
    metric: str = Field(..., description="Canonical metric key")
    metric_label: str = Field(..., description="Human-readable metric label")
    host_name: str = Field(..., description="Target host name")
    host_ip: str = Field(..., description="Target host IP address")
    as_of_date: str = Field(..., description="Dataset cutoff date up to which data was recorded")
    prediction_window: TimeseriesWindow = Field(..., description="Requested prediction date range [st, et]")
    historical_window: TimeseriesWindow = Field(..., description="3-month historical date range prior to prediction")
    summary_stats: TimeseriesSummaryStats = Field(..., description="Summary statistics")
    actual_data: List[ActualDataPoint] = Field(..., description="Past 3 months of historical data points (timestamp and value only)")
    prediction_data: List[PredictionDataPoint] = Field(..., description="Forecasted data points (timestamp, p50, p10, p90)")


# ── Explainability Schemas ───────────────────────────────────────────────────

class TargetMetricInfo(BaseModel):
    name: str = Field(..., json_schema_extra={"example": "CPU Utilization (%)"})
    host_name: str = Field(..., json_schema_extra={"example": "JPRUPIWEBCRP02"})
    host_ip: str = Field(..., json_schema_extra={"example": "10.78.33.83"})
    full_label: str = Field(..., json_schema_extra={"example": "CPU Utilization (%) | JPRUPIWEBCRP02 (10.78.33.83)"})


class ForecastDetails(BaseModel):
    p50: float = Field(..., description="50th percentile forecast")
    p5: float = Field(..., description="5th percentile lower bound")
    p95: float = Field(..., description="95th percentile upper bound")
    p10: float = Field(..., description="10th percentile lower bound")
    p90: float = Field(..., description="90th percentile upper bound")
    unit: str = Field("%", description="Metric unit")


class BaselineShiftInfo(BaseModel):
    shift_pct: float = Field(..., description="Percentage shift relative to baseline")
    historical_avg_30d: float = Field(..., description="30-day historical average")
    comparison_period: str = Field("30-day historical average", description="Comparison baseline description")


class DriverInfo(BaseModel):
    rank: int = Field(..., description="Driver rank order")
    name: str = Field(..., description="Interdependency driver metric name")
    impact_pct: float = Field(..., description="Impact percentage on target metric")
    strength: float = Field(..., description="Coupling strength (0.0 to 1.0)")
    lag: str = Field("0m", description="Lag offset in minutes")


class ExplainabilityResponse(BaseModel):
    target_metric: TargetMetricInfo = Field(..., description="Target metric identification details")
    forecast: ForecastDetails = Field(..., description="Forecast quantile predictions")
    baseline_shift: BaselineShiftInfo = Field(..., description="Baseline shift analysis")
    confidence: float = Field(..., description="Model confidence score percentage")
    risk_level: str = Field(..., description="Assessed risk level (e.g. LOW, MEDIUM, HIGH)")
    summary: str = Field(..., description="Human-readable executive summary text")
    top_interdependency_drivers: List[DriverInfo] = Field(..., description="Ranked list of top interdependency drivers")


# ── Host Resource Prediction Summary Schemas ─────────────────────────────────

class MetricPredictionItem(BaseModel):
    metric_key: str = Field(..., description="Canonical metric identifier")
    metric_label: str = Field(..., description="Human-readable metric display name")
    unit: str = Field("%", description="Measurement unit")
    current_value: float = Field(..., description="Last recorded data point value from the dataset")
    predicted_p50: float = Field(..., description="Predicted P50 forecast value for requested duration")
    percentage_change: float = Field(..., description="Percentage change from current_value to predicted_p50")
    trend: str = Field(..., description="Trend direction: INCREASING | DECREASING | STABLE")
    confidence: float = Field(..., description="Model confidence score percentage (e.g. 60.6 for 60.6%)")
    risk_level: str = Field(..., description="Assessed risk level (e.g. LOW, MEDIUM, HIGH)")


class HostPredictionMetrics(BaseModel):
    cpu_usage: MetricPredictionItem = Field(..., description="CPU usage predictions, confidence, and risk score")
    memory_usage: MetricPredictionItem = Field(..., description="Memory usage predictions, confidence, and risk score")
    disk_usage: MetricPredictionItem = Field(..., description="Disk usage predictions, confidence, and risk score")


class HostPredictionSummaryResponse(BaseModel):
    host_name: str = Field(..., description="Host machine name")
    host_ip: str = Field(..., description="Host IP address")
    prediction_duration: str = Field(..., description="Requested prediction duration (e.g. 30d, 60d, 90d)")
    prediction_days: int = Field(..., description="Number of days in prediction horizon")
    as_of_date: str = Field(..., description="Last recorded data point timestamp in dataset")
    target_date: str = Field(..., description="Target forecast date")
    metrics: HostPredictionMetrics = Field(..., description="CPU, Memory, and Disk prediction breakdown")


# ── Forecast Summary API Schemas (Last 3 Months & Forecast Duration Averages) ──

class ForecastSummaryWindow(BaseModel):
    st: str = Field(..., description="Start date (YYYY-MM-DD)")
    et: str = Field(..., description="End date (YYYY-MM-DD)")


class MetricForecastSummaryItem(BaseModel):
    metric_key: str = Field(..., description="Canonical metric identifier")
    metric_label: str = Field(..., description="Human-readable metric display name")
    unit: str = Field("%", description="Measurement unit")
    last_3_months_avg: float = Field(..., description="Average value of actual data over past 3 months (90 days)")
    forecasted_avg: float = Field(..., description="Average value of forecasted metric over queried duration")
    difference: float = Field(..., description="Absolute difference (forecasted_avg - last_3_months_avg)")
    percentage_change: float = Field(..., description="Percentage change relative to last_3_months_avg")


class ForecastSummaryResponse(BaseModel):
    status: str = Field("success", description="Response status")
    host_name: str = Field(..., description="Host machine name")
    host_ip: str = Field(..., description="Host IP address")
    model_used: str = Field("NEURALPROPHET", description="Forecasting model engine used")
    prediction_days: int = Field(..., description="Prediction horizon in days (e.g. 15, 30, 60, 90)")
    query_period: ForecastSummaryWindow = Field(..., description="Forecasted date range [st, et]")
    historical_period_3m: ForecastSummaryWindow = Field(..., description="Past 3 months historical date range [st, et]")
    metrics: Dict[str, MetricForecastSummaryItem] = Field(..., description="Metric summary breakdown dictionary")


# ── Combined Multi-Host Forecast Summary Schemas ──────────────────────────────

class CombinedMetricSummaryItem(BaseModel):
    metric_key: str = Field(..., description="Canonical metric identifier")
    metric_label: str = Field(..., description="Human-readable metric display name")
    unit: str = Field("%", description="Measurement unit")
    last_3_months_avg: float = Field(..., description="Past 3 months actual historical average")
    avg_15d: float = Field(..., description="15 days forecasted average")
    avg_30d: float = Field(..., description="30 days forecasted average")
    avg_60d: float = Field(..., description="60 days forecasted average")
    avg_90d: float = Field(..., description="90 days forecasted average")


class HostCombinedSummaryItem(BaseModel):
    host_name: str = Field(..., description="Host machine name")
    host_ip: str = Field(..., description="Host IP address")
    metrics: Dict[str, CombinedMetricSummaryItem] = Field(..., description="Breakdown of all 5 metrics")


class CombinedHostsSummaryResponse(BaseModel):
    status: str = Field("success", description="Response status")
    model_used: str = Field("NEURALPROPHET", description="Forecasting model engine used")
    as_of_date: str = Field(..., description="Last recorded data point date in dataset")
    hosts_count: int = Field(..., description="Total number of hosts included")
    hosts: List[HostCombinedSummaryItem] = Field(..., description="List of all host combined forecast summaries")


