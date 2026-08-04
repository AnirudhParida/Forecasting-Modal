"""SUPERSEDED — kept for reference only.

Replaced by the `netraa` pipeline (see README.md). Known defects in this script,
all fixed there:
  * `from=now-10m` yields 11 rows — not a training set.
  * JVM metrics are filtered on dt.entity.host, a dimension they do not have,
    so all 7 of them return empty and the failure is never reported.
  * `metric_queries` is keyed by query string, so metrics sharing a selector
    silently overwrite each other.
  * Missing values are written as 0.0, which asserts the resource was idle.
  * The output filename is fixed, so each run destroys the previous one.

Use: netraa backfill --grid coarse
"""

import os

import requests
import pandas as pd

# Dynatrace configuration
BASE_URL = "https://nlq64817.live.dynatrace.com/api/v2/metrics/query"

# B7: the token was hardcoded here in plaintext. It now comes from the
# environment — but the original value was committed to this file, so treat it
# as compromised and rotate it in Dynatrace regardless.
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
    "endpoint_cpm": "builtin:service.keyRequest.count.total",
    "service_instance_cpm": "builtin:service.requestCount.total",
    "database_access_cpm": "builtin:service.dbChildCallCount"
}

# 2. Host & JVM Metrics
host_metrics = {
    "meter_vm_network_receive": "builtin:host.net.nic.bytesRx",
    "meter_vm_network_transmit": "builtin:host.net.nic.bytesTx",
    "meter_vm_cpu_total_percentage": "builtin:host.cpu.usage",
    "meter_vm_cpu_load1": "builtin:host.cpu.load1",
    "meter_vm_cpu_load5": "builtin:host.cpu.load5",
    "meter_vm_cpu_load15": "builtin:host.cpu.load15",
    "instance_jvm_thread_live_count": "builtin:tech.jvm.threads.count",
    "instance_jvm_thread_runnable_state": 'builtin:tech.jvm.threads.count:filter(eq(state,"RUNNABLE"))',
    "meter_instance_jvm_process_cpu_utilization": "builtin:tech.jvm.processCpuUsage",
    "meter_instance_jvm_total_cpu_utilization": "builtin:tech.jvm.cpuTime",
    "instance_jvm_gc_count": "builtin:tech.jvm.memory.pool.collectionCount",
    "instance_jvm_gc_time": "builtin:tech.jvm.memory.gc.collectionTime"
}

# Map final query strings to your custom column names
metric_queries = {}

for custom_name, m in service_metrics.items():
    query = f'{m}:filter(eq("dt.entity.service","{SERVICE_ID}"))'
    metric_queries[query] = custom_name

for custom_name, m in host_metrics.items():
    query = f'{m}:filter(eq("dt.entity.host","{HOST_ID}"))'
    metric_queries[query] = custom_name

all_data = []

print("Fetching CPU-influencing metrics from Dynatrace...")

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
                timestamps = series.get('timestamps', [])
                values = series.get('values', [])
                
                for t, v in zip(timestamps, values):
                    ts_str = pd.to_datetime(t, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
                    all_data.append({
                        "timestamp": ts_str,
                        "metric": custom_name,
                        "value": v if v is not None else 0
                    })
    else:
        print(f"Failed to fetch {custom_name} | Error: {response.status_code}")

df = pd.DataFrame(all_data)

if not df.empty:
    pivot_df = df.pivot_table(index="timestamp", columns="metric", values="value", aggfunc="mean")
    output_file = "cpu_influencers_last_10m.csv"
    pivot_df.to_csv(output_file)
    print(f"\nSuccess! Saved to {output_file}")
else:
    print("\nNo data retrieved.")
