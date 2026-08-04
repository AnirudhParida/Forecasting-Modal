"""SUPERSEDED — kept for reference only. See fetch_cpu_influencers.py and README.md.

This file produced the negative memory values (-941.98 for builtin:host.mem.usage,
a percentage) recorded in memory_influencers_last_10m.csv, because the selector
carries no explicit aggregation or dimension split and Dynatrace auto-merges.

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
    "instance_traffic": "builtin:service.requestCount.server"
}

# 2. Host & JVM Metrics
host_metrics = {
    "instance_jvm_thread_live_count": "builtin:tech.jvm.threads.count",
    "instance_jvm_thread_peak_count": "builtin:tech.jvm.threads.peakCount",
    "instance_jvm_memory_heap": "builtin:tech.jvm.memory.runtime.used",
    "instance_jvm_memory_heap_max": "builtin:tech.jvm.memory.runtime.max",
    "instance_jvm_memory_pools": "builtin:tech.jvm.memory.pool.used",
    "meter_vm_memory_used": "builtin:host.mem.usage",
    "meter_vm_memory_available": "builtin:host.mem.available",
    "meter_vm_memory_total": "builtin:host.mem.total",
    "meter_vm_memory_buff_cache": "builtin:host.mem.buffersAndCache",
    "meter_vm_network_receive": "builtin:host.net.nic.bytesRx",
    "meter_vm_network_transmit": "builtin:host.net.nic.bytesTx"
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

print("Fetching memory-influencing metrics from Dynatrace...")

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
                
                # If metric is dimension-split (like JVM memory pools), append the dimension to the column name
                dimension_suffix = ""
                if 'dimensions' in series and len(series['dimensions']) > 0:
                     dimension_suffix = f"_{series['dimensions'][0]}"
                     
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
    output_file = "memory_influencers_last_10m.csv"
    pivot_df.to_csv(output_file)
    print(f"\nSuccess! Saved to {output_file}")
else:
    print("\nNo data retrieved.")
