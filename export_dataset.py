"""
export_dataset.py — Export Training Dataset to Human-Readable CSVs
===================================================================
Reads the panel parquet files and exports them into a clean, organised
folder structure under  dataset_export/  so you can open them in Excel,
pandas, or any BI tool.

Usage
-----
    python export_dataset.py

Output structure
----------------
dataset_export/
├── README.md                    ← legend: what each file contains
├── 00_summary.csv               ← one row per metric: coverage, min, max, mean
├── by_resource/
│   ├── cpu.csv                  ← date × cpu metrics
│   ├── memory.csv               ← date × memory metrics
│   ├── disk.csv                 ← date × disk metrics (all disks)
│   ├── jvm.csv                  ← date × jvm metrics
│   ├── network.csv              ← date × network metrics
│   └── service.csv              ← date × service metrics
├── targets_only.csv             ← only the 30 metrics the model forecasts
├── all_metrics.csv              ← full 72-metric panel (wide format)
└── daily_coverage.csv           ← for each day: how many metrics had data
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np

ROOT      = Path(__file__).parent
PANEL_DIR = ROOT / "data" / "panel"
OUT_DIR   = ROOT / "dataset_export"

# ── Friendly column names ─────────────────────────────────────────────────────
SUFFIX_LABELS = {
    "DISK-06AEE4ED601B8525": "Disk-A",
    "DISK-1C013B34259F92AC":  "Disk-B",
    "DISK-57DE2A1151778A10":  "Disk-C",
    "DISK-AD574AD979302D79":  "Disk-D",
    "DISK-AEAFE33C63A4A3E9":  "Disk-E",
    "PROCESS_GROUP_INSTANCE-1AEF64B598C08A5B": "JVM-1",
    "PROCESS_GROUP_INSTANCE-3264EBB02CB46816": "JVM-2",
    "PROCESS_GROUP_INSTANCE-FC97E4E3BF378F88": "JVM-3",
    "HOST-D9739223FC540A23":   "Host",
}

METRIC_LABELS = {
    "host_cpu_usage":          "CPU_Usage_Pct",
    "host_cpu_load1":          "CPU_Load_1m",
    "host_mem_usage":          "Memory_Usage_Pct",
    "host_mem_available":      "Memory_Available_Bytes",
    "host_mem_buff_cache":     "Memory_BufferCache_Bytes",
    "host_mem_total":          "Memory_Total_Bytes",
    "disk_busy_time":          "Disk_Busy_Pct",
    "disk_queue_length":       "Disk_Queue_Length",
    "disk_read_iops":          "Disk_Read_IOPS",
    "disk_read_throughput":    "Disk_Read_Throughput_Bps",
    "disk_read_latency":       "Disk_Read_Latency_ms",
    "disk_write_iops":         "Disk_Write_IOPS",
    "disk_write_throughput":   "Disk_Write_Throughput_Bps",
    "disk_write_latency":      "Disk_Write_Latency_ms",
    "host_net_rx":             "Net_Receive_Bps",
    "host_net_tx":             "Net_Transmit_Bps",
    "host_net_rx_packets":     "Net_Rx_Packets",
    "host_net_tx_packets":     "Net_Tx_Packets",
    "jvm_memory_heap_used":    "JVM_Heap_Used_Bytes",
    "jvm_memory_heap_max":     "JVM_Heap_Max_Bytes",
    "jvm_memory_pool_used":    "JVM_MemPool_Used_Bytes",
    "jvm_gc_collection_count": "JVM_GC_Count",
    "jvm_gc_collection_time":  "JVM_GC_Time_ms",
    "jvm_process_cpu":         "JVM_CPU_Pct",
    "jvm_thread_live_count":   "JVM_Thread_Count",
    "service_cpm":             "Service_Calls_Per_Min",
    "instance_traffic":        "Instance_Traffic",
    "database_access_cpm":     "DB_Access_Per_Min",
}


def friendly_name(node_id: str) -> str:
    """Convert raw node_id to a readable column name."""
    if "|" not in node_id:
        return METRIC_LABELS.get(node_id, node_id)
    base, entity = node_id.split("|", 1)
    label   = METRIC_LABELS.get(base, base)
    suffix  = SUFFIX_LABELS.get(entity, entity[-8:])
    return f"{label}_{suffix}"


def load_panel():
    vals  = pd.read_parquet(PANEL_DIR / "coarse_values.parquet")
    mask_ = pd.read_parquet(PANEL_DIR / "coarse_mask.parquet")
    nodes = pd.read_parquet(PANEL_DIR / "coarse_nodes.parquet")

    # Localise index to IST for display
    vals.index  = pd.to_datetime(vals.index).tz_convert("Asia/Kolkata")
    mask_.index = pd.to_datetime(mask_.index).tz_convert("Asia/Kolkata")
    vals.index.name  = "date"
    mask_.index.name = "date"

    return vals, mask_, nodes


def rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: friendly_name(c) for c in df.columns})


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "by_resource").mkdir(exist_ok=True)

    print(f"Loading panel from {PANEL_DIR} …")
    vals, mask_, nodes = load_panel()

    n_days    = len(vals)
    n_metrics = len(vals.columns)
    date_min  = vals.index[0].strftime("%Y-%m-%d")
    date_max  = vals.index[-1].strftime("%Y-%m-%d")
    print(f"  {n_days} daily observations × {n_metrics} metrics")
    print(f"  Date range: {date_min}  →  {date_max}  (IST)")

    # ── 1. Summary CSV ────────────────────────────────────────────────────────
    rows = []
    for node_id in vals.columns:
        s        = vals[node_id]
        node_row = nodes[nodes["node_id"] == node_id]
        resource = node_row["resource"].iloc[0] if len(node_row) else "?"
        role     = node_row["role"].iloc[0]     if len(node_row) else "?"
        unit     = node_row["unit"].iloc[0]     if len(node_row) else "?"
        cov      = node_row["coverage"].iloc[0] if len(node_row) else float("nan")
        rows.append({
            "node_id":       node_id,
            "friendly_name": friendly_name(node_id),
            "resource":      resource,
            "role":          role,
            "unit":          unit,
            "coverage_pct":  round(float(cov) * 100, 1),
            "days_with_data":int(s.notna().sum()),
            "days_total":    n_days,
            "min":           round(float(s.min()), 4) if s.notna().any() else None,
            "max":           round(float(s.max()), 4) if s.notna().any() else None,
            "mean":          round(float(s.mean()), 4) if s.notna().any() else None,
            "median":        round(float(s.median()), 4) if s.notna().any() else None,
            "std":           round(float(s.std()), 4) if s.notna().any() else None,
        })

    summary = pd.DataFrame(rows).sort_values(["resource", "node_id"])
    summary.to_csv(OUT_DIR / "00_summary.csv", index=False)
    print(f"  ✓  00_summary.csv  ({len(summary)} metrics)")

    # ── 2. All metrics (wide) ──────────────────────────────────────────────────
    all_wide = rename_cols(vals)
    all_wide.to_csv(OUT_DIR / "all_metrics.csv")
    print(f"  ✓  all_metrics.csv  ({all_wide.shape[0]} rows × {all_wide.shape[1]} cols)")

    # ── 3. Targets only ────────────────────────────────────────────────────────
    target_ids = nodes[nodes["role"] == "target"]["node_id"].tolist()
    targets_df = rename_cols(vals[target_ids])
    targets_df.to_csv(OUT_DIR / "targets_only.csv")
    print(f"  ✓  targets_only.csv  ({len(target_ids)} target metrics)")

    # ── 4. By resource ─────────────────────────────────────────────────────────
    for resource, grp in nodes.groupby("resource"):
        cols = [c for c in grp["node_id"].tolist() if c in vals.columns]
        if not cols:
            continue
        df = rename_cols(vals[cols])
        fname = f"by_resource/{resource}.csv"
        df.to_csv(OUT_DIR / fname)
        print(f"  ✓  {fname}  ({len(cols)} metrics)")

    # ── 5. Daily coverage ──────────────────────────────────────────────────────
    coverage_df = pd.DataFrame({
        "date":              vals.index,
        "metrics_total":     n_metrics,
        "metrics_with_data": vals.notna().sum(axis=1).values,
        "coverage_pct":      (vals.notna().sum(axis=1) / n_metrics * 100).round(1).values,
        "cpu_usage":         vals.get("host_cpu_usage", pd.Series(dtype=float)).values
                             if "host_cpu_usage" in vals.columns else None,
        "memory_usage":      vals.get("host_mem_usage", pd.Series(dtype=float)).values
                             if "host_mem_usage" in vals.columns else None,
    }).set_index("date")
    coverage_df.to_csv(OUT_DIR / "daily_coverage.csv")
    print(f"  ✓  daily_coverage.csv  ({len(coverage_df)} rows)")

    # ── 6. README ──────────────────────────────────────────────────────────────
    readme = f"""# Netraa — Training Dataset Export
Generated: {pd.Timestamp.now(tz='Asia/Kolkata').strftime('%Y-%m-%d %H:%M IST')}

## Dataset Overview
- **Source**: Dynatrace (https://nlq64817.live.dynatrace.com)
- **Grid**: Coarse (1-day resolution)
- **Date range**: {date_min} → {date_max}
- **Total days**: {n_days}
- **Total metrics**: {n_metrics} (columns)
- **Overall coverage**: {round(vals.notna().mean().mean()*100, 1)}%

## File Guide

| File | Description |
|------|-------------|
| `00_summary.csv` | One row per metric: resource group, role (target/intermediate), unit, coverage %, min/max/mean/median/std |
| `all_metrics.csv` | Full dataset — {n_days} rows × {n_metrics} columns. Use this for custom analysis. |
| `targets_only.csv` | Only the **{len(target_ids)} metrics the model forecasts** (CPU, Memory, Disk, JVM Heap) |
| `by_resource/cpu.csv` | CPU metrics only (2 metrics) |
| `by_resource/memory.csv` | Memory metrics (13 metrics) |
| `by_resource/disk.csv` | Disk metrics — IOPS, throughput, latency, busy time for 5 disks (34 metrics) |
| `by_resource/jvm.csv` | JVM metrics — Heap, GC, threads, CPU (12 metrics) |
| `by_resource/network.csv` | Network Rx/Tx bytes and packets (8 metrics) |
| `by_resource/service.csv` | Service call counts and DB access (3 metrics) |
| `daily_coverage.csv` | For each day: how many metrics had data, plus CPU and Memory values |

## Column Name Convention
Columns use friendly names: `{{MetricType}}_{{DiskLabel}}` or `{{MetricType}}_{{JVMLabel}}`

Disk labels: Disk-A through Disk-E (mapped from entity IDs)
JVM labels: JVM-1, JVM-2, JVM-3 (different JVM process groups)

## Roles
- **target**: Metrics the model directly forecasts (30 metrics)
- **intermediate**: Supporting metrics used as features/context (40 metrics)
- **driver**: High-level driver metrics (2 metrics)

## NaN Values
NaN means no data was available from Dynatrace for that day.
Some disk metrics have ~41% coverage (they are on standby disks).
Core metrics (CPU, Memory, Disk-B, Disk-E) have >98% coverage.
"""
    (OUT_DIR / "README.md").write_text(readme)
    print(f"  ✓  README.md")

    # ── Print summary table ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DATASET EXPORT COMPLETE → {OUT_DIR}/")
    print(f"{'='*60}")
    print(f"\n  {'Resource':<12} {'Metrics':>8} {'Avg Coverage':>14}")
    print("  " + "─"*38)
    for resource, grp in nodes.groupby("resource"):
        cols = [c for c in grp["node_id"].tolist() if c in vals.columns]
        if not cols:
            continue
        cov = round(vals[cols].notna().mean().mean() * 100, 1)
        print(f"  {resource:<12} {len(cols):>8} {cov:>13.1f}%")
    print()
    print(f"  Total   : {n_metrics} metrics  |  {n_days} daily observations")
    print(f"  Targets : {len(target_ids)} metrics the model forecasts")
    print(f"  Period  : {date_min} → {date_max}")
    print()


if __name__ == "__main__":
    main()
