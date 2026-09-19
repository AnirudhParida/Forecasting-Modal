import sys
import json
import logging
import asyncio

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Import the service directly rather than running the full server
from app.services.timeseries_service import get_timeseries_data
from app.services.host_prediction_service import get_host_prediction_summary
from app.services.explainability_service import get_explainability_analysis
from app.services.forecast_summary_service import get_forecast_summary
from app.services.combined_hosts_service import get_combined_hosts_summary

def test_apis():
    # 1. Timeseries
    print("\n--- TEST: Timeseries (STGNN) ---")
    ts_stgnn = get_timeseries_data(metric="host_cpu_usage", model="STGNN")
    print(f"STGNN - Actual pts: {len(ts_stgnn.actual_data)}, Pred pts: {len(ts_stgnn.prediction_data)}")
    
    print("\n--- TEST: Timeseries (NEURALPROPHET) ---")
    ts_np = get_timeseries_data(metric="host_cpu_usage", model="NEURALPROPHET")
    print(f"NP - Actual pts: {len(ts_np.actual_data)}, Pred pts: {len(ts_np.prediction_data)}")
    
    # 2. Host Prediction
    print("\n--- TEST: Host Prediction (STGNN) ---")
    hp_stgnn = get_host_prediction_summary(prediction_duration="30d", model="STGNN")
    print(f"STGNN CPU p50: {hp_stgnn.metrics.cpu_usage.predicted_p50}")
    
    print("\n--- TEST: Host Prediction (NEURALPROPHET) ---")
    hp_np = get_host_prediction_summary(prediction_duration="30d", model="NEURALPROPHET")
    print(f"NP CPU p50: {hp_np.metrics.cpu_usage.predicted_p50}")

    # 3. Explainability
    print("\n--- TEST: Explainability (STGNN) ---")
    ex_stgnn = get_explainability_analysis(target_metric="host_cpu_usage", model="STGNN")
    print(f"STGNN drivers: {[d.name for d in ex_stgnn.top_interdependency_drivers]}")
    
    print("\n--- TEST: Explainability (NEURALPROPHET) ---")
    ex_np = get_explainability_analysis(target_metric="host_cpu_usage", model="NEURALPROPHET")
    print(f"NP drivers: {[d.name for d in ex_np.top_interdependency_drivers]}")

    # 4. Forecast Summary
    print("\n--- TEST: Forecast Summary (NEURALPROPHET by Host Alias & IP) ---")
    fs_np = get_forecast_summary(host_name="HYDUPINTAPP16", metric="cpu_pct", prediction_days=30, model="NEURALPROPHET")
    print(f"Host: {fs_np.host_name} ({fs_np.host_ip}) | Days: {fs_np.prediction_days}")
    for k, item in fs_np.metrics.items():
        print(f"  [{k}] 3M Historical Avg: {item.last_3_months_avg} {item.unit} | Forecasted Avg: {item.forecasted_avg} {item.unit} (Change: {item.percentage_change}%)")

    fs_ip = get_forecast_summary(host_ip="10.78.33.83", metric="all", prediction_days=60, model="NEURALPROPHET")
    print(f"Host: {fs_ip.host_name} ({fs_ip.host_ip}) | Metrics Count: {len(fs_ip.metrics)}")

    # 5. Combined Hosts Summary
    print("\n--- TEST: Combined Hosts Summary (NEURALPROPHET) ---")
    chs = get_combined_hosts_summary(model="NEURALPROPHET")
    print(f"Total Hosts: {chs.hosts_count} | As Of: {chs.as_of_date}")
    for host in chs.hosts:
        print(f"  Host: {host.host_name} ({host.host_ip})")
        for mk, mitem in host.metrics.items():
            print(f"    [{mk}] 3M Avg: {mitem.last_3_months_avg} | 15d: {mitem.avg_15d} | 30d: {mitem.avg_30d} | 60d: {mitem.avg_60d} | 90d: {mitem.avg_90d} {mitem.unit}")

if __name__ == "__main__":
    test_apis()


