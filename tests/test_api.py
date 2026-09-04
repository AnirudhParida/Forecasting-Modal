"""
tests/test_api.py
=================
Automated unit, integration, and EDGE CASE test suite for FastAPI endpoints:
1. GET /api/v1/timeseries
2. GET /api/v1/explainability (with Relative Weight Contribution Normalization)
3. GET /api/v1/host-prediction
"""
# pyrefly: ignore [missing-import]
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ── Standard Sanity Tests ─────────────────────────────────────────────────────

def test_root_and_health():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "documentation" in data

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json() == {"status": "healthy"}


def test_timeseries_st_et_dynamic():
    url = "/api/v1/timeseries?st=2026-09-01&et=2026-09-30&metric=host_cpu_usage&host_name=JPRUPIWEBCRP02&host_ip=10.78.33.83"
    response = client.get(url)
    assert response.status_code == 200
    data = response.json()

    assert "host_cpu_usage" in data["metric"]
    assert data["host_name"] == "JPRUPIWEBCRP02"
    assert data["host_ip"] == "10.78.33.83"
    assert data["prediction_window"]["st"] == "2026-09-01"
    assert data["prediction_window"]["et"] == "2026-09-30"

    assert "actual_data" in data
    assert len(data["actual_data"]) == 90
    actual_pt = data["actual_data"][0]
    assert "timestamp" in actual_pt
    assert "value" in actual_pt
    assert "p50" not in actual_pt

    assert "prediction_data" in data
    assert len(data["prediction_data"]) == 30
    pred_pt = data["prediction_data"][0]
    assert "timestamp" in pred_pt
    assert "p50" in pred_pt
    assert "p10" in pred_pt
    assert "p90" in pred_pt


def test_explainability_dynamic():
    response = client.get("/api/v1/explainability?target_metric=host_cpu_usage&top_k=5")
    assert response.status_code == 200
    data = response.json()

    assert "target_metric" in data
    assert "forecast" in data
    assert "baseline_shift" in data
    assert "confidence" in data
    assert "risk_level" in data
    assert "top_interdependency_drivers" in data

    drivers = data["top_interdependency_drivers"]
    assert len(drivers) == 5

    # Verify Relative Weight Contribution Normalization: Sum of absolute shares = 100.0%
    abs_sum = round(sum(abs(d["impact_pct"]) for d in drivers), 1)
    assert abs(abs_sum - 100.0) <= 0.5


def test_host_prediction_summary_dynamic():
    url = "/api/v1/host-prediction?st=2026-09-01&et=2026-09-30&host_name=JPRUPIWEBCRP02&host_ip=10.78.33.83"
    response = client.get(url)
    assert response.status_code == 200
    data = response.json()

    assert data["host_name"] == "JPRUPIWEBCRP02"
    assert data["prediction_days"] == 30
    assert "metrics" in data

    metrics = data["metrics"]
    for m_key in ("cpu_usage", "memory_usage", "disk_usage"):
        assert m_key in metrics
        m = metrics[m_key]
        assert "current_value" in m
        assert "predicted_p50" in m
        assert "percentage_change" in m
        assert "trend" in m
        assert "confidence" in m
        assert "risk_level" in m


# ── Edge Case Tests ───────────────────────────────────────────────────────────

def test_timeseries_edge_case_reversed_dates():
    """Edge Case: st > et (Start date is after End date)."""
    url = "/api/v1/timeseries?st=2026-10-01&et=2026-09-01&metric=host_cpu_usage"
    response = client.get(url)
    assert response.status_code == 200
    data = response.json()
    assert "prediction_data" in data
    assert len(data["prediction_data"]) >= 1


def test_timeseries_edge_case_invalid_date_format():
    """Edge Case: Malformed or unparseable date strings."""
    url = "/api/v1/timeseries?st=invalid-date-string&et=bad-date-999"
    response = client.get(url)
    assert response.status_code == 200
    data = response.json()
    assert "actual_data" in data
    assert "prediction_data" in data


def test_timeseries_edge_case_unix_and_iso_timestamps():
    """Edge Case: Unix epoch timestamps and full ISO 8601 timestamps."""
    url_unix = "/api/v1/timeseries?st=1725148800&et=1727654400"
    res_unix = client.get(url_unix)
    assert res_unix.status_code == 200

    url_iso = "/api/v1/timeseries?st=2026-09-01T00:00:00Z&et=2026-09-30T23:59:59Z"
    res_iso = client.get(url_iso)
    assert res_iso.status_code == 200


def test_timeseries_edge_case_single_day_range():
    """Edge Case: st == et (Single day prediction window)."""
    url = "/api/v1/timeseries?st=2026-09-01&et=2026-09-01"
    response = client.get(url)
    assert response.status_code == 200
    data = response.json()
    assert len(data["prediction_data"]) == 1


def test_timeseries_edge_case_unknown_metric_and_special_chars():
    """Edge Case: Unknown metric query and URL encoded special characters."""
    url_unknown = "/api/v1/timeseries?metric=nonexistent_random_metric_99999"
    res_unk = client.get(url_unknown)
    assert res_unk.status_code == 200

    url_spec = "/api/v1/timeseries?metric=CPU%20Utilization%20(%25)"
    res_spec = client.get(url_spec)
    assert res_spec.status_code == 200
    assert "CPU" in res_spec.json()["metric_label"]


def test_explainability_edge_case_top_k_boundaries():
    """Edge Case: top_k boundary values (top_k=1, top_k=20)."""
    res1 = client.get("/api/v1/explainability?top_k=1")
    assert res1.status_code == 200
    d1 = res1.json()["top_interdependency_drivers"]
    assert len(d1) == 1
    assert abs(abs(d1[0]["impact_pct"]) - 100.0) <= 0.5

    res20 = client.get("/api/v1/explainability?top_k=20")
    assert res20.status_code == 200
    d20 = res20.json()["top_interdependency_drivers"]
    assert len(d20) <= 20
    abs_sum = round(sum(abs(d["impact_pct"]) for d in d20), 1)
    assert abs(abs_sum - 100.0) <= 0.5


def test_explainability_edge_case_unknown_target_and_empty_strings():
    """Edge Case: Unknown target metric query and empty parameters."""
    res_unk = client.get("/api/v1/explainability?target_metric=unknown_metric_xyz")
    assert res_unk.status_code == 200

    res_empty = client.get("/api/v1/explainability?target_metric=&host_name=&host_ip=")
    assert res_empty.status_code == 200
    assert "target_metric" in res_empty.json()


def test_host_prediction_edge_case_duration_parsing():
    """Edge Case: Flexible prediction_duration formats (60d, 90days, 3m)."""
    res60 = client.get("/api/v1/host-prediction?st=&et=&prediction_duration=60d")
    assert res60.status_code == 200
    assert res60.json()["prediction_days"] == 60

    res90 = client.get("/api/v1/host-prediction?st=&et=&prediction_duration=90days")
    assert res90.status_code == 200
    assert res90.json()["prediction_days"] == 90

    res3m = client.get("/api/v1/host-prediction?st=&et=&prediction_duration=3m")
    assert res3m.status_code == 200
    assert res3m.json()["prediction_days"] == 90
