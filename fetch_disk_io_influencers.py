"""SUPERSEDED — kept for reference only. See fetch_cpu_influencers.py and README.md.

This file carried the clearest instance of the key-collision defect: service_cpm,
service_instance_cpm and transaction_volume all resolve to
builtin:service.requestCount.total, and because `metric_queries` is keyed by the
generated query string, only transaction_volume survived to be requested.

Use: netraa backfill --grid coarse
"""

import os

import requests
import pandas as pd

# Dynatrace configuration
BASE_URL = "https://nlq64817.live.dynatrace.com/api/v2/metrics/query"

# B7: token was hardcoded here. Treat the original value as compromised and
# rotate it in Dynatrace.
API_TOKEN = os.environ.get("DYNATRACE_API_TOKEN", "")
if not API_TOKEN:
    raise SystemExit(
        "DYNATRACE_API_TOKEN is not set. Copy .env.example to .env and fill it in."
    )

HOST_ID = os.environ.get("NETRAA_HOST_ID", "HOST-D9739223FC540A23")
SERVICE_ID = os.environ.get("NETRAA_SERVICE_ID", "SERVICE-C16E984A4D0792A9")

headers = {
    "accept": "application/json; charset=utf-8",
    "Authorization": f"Api-Token {API_TOKEN}"
}

# 1. Service Metrics
service_metrics = {
    "service_cpm": "builtin:service.requestCount.total",
    "endpoint_cpm": "builtin:service.keyRequest.count.total",
    "service_instance_cpm": "builtin:service.requestCount.total",
    "transaction_volume": "builtin:service.requestCount.total"
}

# 2. Host Disk Metrics
host_metrics = {
    "disk_read_iops": "builtin:host.disk.reads",
    "disk_write_iops": "builtin:host.disk.writes",
    "disk_read_throughput": "builtin:host.disk.bytesRead",
    "disk_write_throughput": "builtin:host.disk.bytesWritten",
    "disk_read_latency": "builtin:host.disk.readTime",
    "disk_write_latency": "builtin:host.disk.writeTime",
    "disk_queue_length": "builtin:host.disk.queueLength",
    "disk_busy_time": "builtin:host.disk.activeTime"
}

metric_queries = {}

# Apply Service filters
for custom_name, m in service_metrics.items():
    query = f'{m}:filter(eq("dt.entity.service","{SERVICE_ID}"))'
    metric_queries[query] = custom_name

# Apply Host filters
for custom_name, m in host_metrics.items():
    query = f'{m}:filter(eq("dt.entity.host","{HOST_ID}"))'
    metric_queries[query] = custom_name

all_data = []

print("Fetching disk I/O-influencing metrics from Dynatrace...")

for query, custom_name in metric_queries.items():
    params = {
        "metricSelector": query,
        "from": "now-10m",
        "to": "now",
        "resolution": "1m"
    }
    
    response = requests.get(BASE_URL, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if 'result' in data and len(data['result']) > 0 and len(data['result'][0]['data']) > 0:
            for series in data['result'][0]['data']:
                
                # Disk metrics are often split by disk name (e.g., /dev/sda, C:).
                # This appends the specific disk identifier to the column name if present.
                dimension_suffix = ""
                if 'dimensions' in series and len(series['dimensions']) > 0:
                     # Clean the dimension string to avoid special characters in CSV headers
                     clean_dim = str(series['dimensions'][0]).replace("/", "_").replace("\\", "_").replace(":", "")
                     dimension_suffix = f"_{clean_dim}"
                     
                timestamps = series.get('timestamps', [])
                values = series.get('values', [])
                
                for t, v in zip(timestamps, values):
                    ts_str = pd.to_datetime(t, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
                    all_data.append({
                        "timestamp": ts_str,
                        "metric": f"{custom_name}{dimension_suffix}",
                        "value": v if v is not None else 0
                    })
    else:
        print(f"Failed to fetch {custom_name} | Error: {response.status_code}")

df = pd.DataFrame(all_data)

if not df.empty:
    pivot_df = df.pivot_table(index="timestamp", columns="metric", values="value", aggfunc="mean")
    output_file = "disk_io_influencers_last_10m.csv"
    pivot_df.to_csv(output_file)
    print(f"\nSuccess! Saved to {output_file}")
else:
    print("\nNo data retrieved.")
