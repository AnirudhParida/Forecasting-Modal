# Netraa — Training Guide
## How to Train the Model on Dynatrace Data

> **Step-by-step guide** for running the full training pipeline against a live Dynatrace tenant.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [One-Time Setup](#2-one-time-setup)
3. [Offline Smoke Test (No API Token Needed)](#3-offline-smoke-test-no-api-token-needed)
4. [Live Training Pipeline — Step by Step](#4-live-training-pipeline--step-by-step)
   - Step 1: Probe Tenant Retention
   - Step 2: Resolve Entity Topology
   - Step 3: Validate Metrics (STOP HERE if failures)
   - Step 4: Backfill Historical Data
   - Step 5: Build Panels
   - Step 6: Build Statistical Dependency Map
   - Step 7: Train the ST-GNN
5. [Keeping Data Current (Incremental Collection)](#5-keeping-data-current-incremental-collection)
6. [Useful Variations and Debug Commands](#6-useful-variations-and-debug-commands)
7. [Understanding the Output Files](#7-understanding-the-output-files)
8. [Troubleshooting Common Errors](#8-troubleshooting-common-errors)
9. [Re-Training After New Data](#9-re-training-after-new-data)

---

## 1. Prerequisites

**Software**:
- Python 3.10+
- `uv` (fast pip/venv alternative) or standard `pip`
- Internet access to your Dynatrace tenant

**Dynatrace requirements**:
- A Dynatrace API token with the following scopes:
  - `metrics.read`
  - `entities.read`
- Your Dynatrace tenant base URL (e.g., `https://abc12345.live.dynatrace.com`)
- The entity IDs for the HOST and SERVICE you want to forecast:
  - `HOST-xxxxxxxxxxxxxxxx` (from Settings → Infrastructure in Dynatrace UI)
  - `SERVICE-xxxxxxxxxxxxxxxx` (from Services in Dynatrace UI)

---

## 2. One-Time Setup

### 2a. Navigate to the project

```bash
cd /home/anirudh.parida@apmosys.mahape/Documents/Forecasting-Modal
```

### 2b. Create a virtual environment and install dependencies

```bash
# Using uv (recommended)
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -r requirements.txt

# --- OR using standard pip ---
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2c. Configure your secrets

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and fill in your values
nano .env
```

Your `.env` should contain:
```bash
DYNATRACE_API_TOKEN=dt0c01.XXXXXXXXXX...
DYNATRACE_BASE_URL=https://abc12345.live.dynatrace.com

# Optional overrides (otherwise taken from configs/v1.yaml)
NETRAA_HOST_ID=HOST-D9739223FC540A23
NETRAA_SERVICE_ID=SERVICE-C16E984A4D0792A9
```

> **IMPORTANT**: Never commit your `.env` file. It is already in `.gitignore`.

### 2d. Update `configs/v1.yaml` with your entity IDs

```yaml
# configs/v1.yaml
base_url: https://abc12345.live.dynatrace.com

scope:
  host_id: HOST-D9739223FC540A23       # ← change to your HOST
  service_id: SERVICE-C16E984A4D0792A9 # ← change to your SERVICE
```

### 2e. (Optional) Set a convenience alias

```bash
alias netraa='.venv/bin/python -m netraa.cli'
```

From here, all commands below can use either:
- `.venv/bin/python -m netraa.cli <command>` (always works)
- `netraa <command>` (if the alias is set)

---

## 3. Offline Smoke Test (No API Token Needed)

Before connecting to the live API, verify all code paths connect end-to-end:

```bash
# Run 13 unit tests (offline, ~seconds)
.venv/bin/python tests/test_pipeline.py

# Run the full 6-stage pipeline on the 11-row legacy CSVs (offline, ~seconds)
.venv/bin/python -m netraa.cli smoke
```

**Expected output for `smoke`**:
```
======================================================================
SMOKE TEST — legacy CSVs, ~11 rows each. Numbers below are meaningless.
======================================================================

[1] legacy import: 62 rows, 8 distinct nodes
[2] panel
    grid=smoke freq=1min
    shape: 11 timesteps x 6 nodes
[3] transforms: negative values 6 -> 0 (B5 abs applied)
[4] dependency discovery: 3 edges
[5] dataset: ...
[6] backtest: ...

======================================================================
SMOKE TEST PASSED — every stage ran. Now run a real backfill.
======================================================================
```

If this passes, all code paths are wired correctly.

---

## 4. Live Training Pipeline — Step by Step

### STEP 1 — Probe Tenant Retention

**What it does**: Queries `builtin:host.cpu.usage` across a grid of (resolution, lookback) combinations to empirically determine what history your tenant actually retains. This tells you what to set in `configs/v1.yaml`.

```bash
.venv/bin/python -m netraa.cli probe
```

**Expected output**:
```
resolution  lookback   points  observed  requested  status
1m              1d       1440     60s        60s     ok
1m              7d          0       -        60s     no data (beyond retention)
5m              1d        288    300s       300s     ok
5m             14d       4032    300s       300s     ok
5m             30d          0       -       300s     no data (beyond retention)
1h             90d       2160   3600s      3600s     ok
1h            180d       4320   3600s      3600s     ok
1d            400d        400  86400s     86400s     ok

Recommended grids:
  fine     {'resolution': '5m', 'lookback_days': 14}
  coarse   {'resolution': '1d', 'lookback_days': 400}
```

**Action**: Update `configs/v1.yaml` with the recommended values:
```yaml
grids:
  fine:
    resolution: 5m        # ← from probe recommendation
    pandas_freq: 5min
    lookback_days: 14     # ← from probe recommendation
  coarse:
    resolution: 1d        # ← from probe recommendation
    pandas_freq: 1D
    lookback_days: 400    # ← from probe recommendation
```

Output file: `data/retention_probe.json`

---

### STEP 2 — Resolve Entity Topology

**What it does**: Walks the Dynatrace entity relationship graph from your HOST and SERVICE to discover all related entity IDs: Process Group Instances (for JVM metrics), Disks, and Service Methods (for key requests). These IDs are required to correctly filter metrics in subsequent steps.

```bash
.venv/bin/python -m netraa.cli topology
```

**Expected output**:
```
host:    HOST-D9739223FC540A23
service: SERVICE-C16E984A4D0792A9
  DISK                       3
      DISK-A1B2C3D4E5F6G7H8
      DISK-B2C3D4E5F6G7H8I9
      DISK-C3D4E5F6G7H8I9J0
  PROCESS_GROUP_INSTANCE     1
      PROCESS_GROUP_INSTANCE-E5F6G7H8I9J0K1L2
  SERVICE_METHOD             4
      SERVICE_METHOD-F6G7H8I9J0K1L2M3
      ... 3 more

saved -> data/topology.json
```

> **If you see warnings** like "no PROCESS_GROUP_INSTANCE entities resolved", check that the service is actually running on the specified host.

Output file: `data/topology.json`

---

### STEP 3 — Validate All Metrics

**What it does**: Runs pre-flight checks on every metric in `metrics_registry.yaml` against the live Dynatrace API. Catches problems (wrong dimension, metric doesn't exist, no data) before spending time on a backfill.

```bash
.venv/bin/python -m netraa.cli validate
```

**Expected output**:
```
metric key                 status              series  points  null%  detail
─────────────────────────────────────────────────────────────────────────────
host_cpu_usage             OK                       1      24   0.0%
host_mem_usage             OK                       1      24   0.0%
disk_read_iops             OK                       3      72   0.0%
disk_write_iops            OK                       3      72   0.0%
jvm_memory_heap_used       OK                       1      24   0.0%
jvm_thread_live_count      OK                       1      24   0.0%
service_cpm                OK                       1      24   0.0%
...

22 OK, 0 unit warnings, 0 failed
```

**STOP AND FIX if you see failures like**:

| Status | What to do |
|--------|-----------|
| `MISSING_METRIC` | The metric ID doesn't exist on your tenant. Remove or disable it in `metrics_registry.yaml` (set `enabled: false`) |
| `DIMENSION_MISMATCH` | The `entity_type` in the registry is wrong for this metric. Check the Dynatrace metric descriptor in the UI |
| `UNRESOLVED` | Topology didn't find entity IDs of this type. Re-check Step 2 output and your `host_id`/`service_id` |
| `NO_DATA` | Query succeeded but returned zero points. The host may not have this metric. Disable it |
| `UNIT_VIOLATION` | Values contradict the declared unit (e.g., negative percent). Check `transform` field |

```bash
# Re-run with full traceback if you need to debug a specific failure
.venv/bin/python -m netraa.cli validate -v
```

Output file: `data/validation_report.json`

**Do not proceed to Step 4 until validate reports 0 failed rows.**

---

### STEP 4 — Backfill Historical Data

**What it does**: Pulls historical time-series data for all enabled metrics in the registry, for both the fine grid (5-min, 14 days) and the coarse grid (daily, 400 days). Data is written to date-partitioned Parquet files.

Both grids must be backfilled — the fine grid feeds dependency discovery (Step 6) and the coarse grid feeds the forecaster (Step 7).

```bash
# Backfill the fine grid (5-minute data, 14 days of history)
# Approximate time: 5–20 minutes depending on tenant size
.venv/bin/python -m netraa.cli backfill --grid fine

# Backfill the coarse grid (daily data, 400 days of history)
# Approximate time: 2–10 minutes
.venv/bin/python -m netraa.cli backfill --grid coarse
```

**Expected output per metric**:
```
metric key                 status       rows  series  detail
─────────────────────────────────────────────────────────────
host_cpu_usage             OK           4032       1
host_mem_usage             OK           4032       1
disk_read_iops             OK          12096       3
disk_write_iops            OK          12096       3
jvm_memory_heap_used       OK           4032       1
service_cpm                OK           4032       1
host_net_rx                OK           8064       2
...

grid=fine: 22/22 metrics returned data, 287,430 rows written
```

**Understanding status columns**:

| Status | Meaning |
|--------|---------|
| `OK` | Data returned and written successfully |
| `PARTIAL` | Some chunks succeeded, some failed (API error on that chunk) |
| `ERROR` | All chunks failed — check API token / network |
| `NO_DATA` | Query succeeded but returned zero points |
| `UNRESOLVED` | No entity IDs resolved — fix topology first |

**Check what was stored**:
```bash
.venv/bin/python -m netraa.cli status
```

Output:
```
=== grid=fine ===
  nodes: 28   rows: 287,430
  span:  2024-01-15 → 2024-01-29
  node_id                          points   first          last     null_fraction
  host_cpu_usage                     4032   2024-01-15...  2024-01-29...  0.000
  ...

=== grid=coarse ===
  nodes: 28   rows: 11,200
  span:  2023-01-01 → 2024-01-29
```

Output directory: `data/raw/grid=fine/`, `data/raw/grid=coarse/`

---

### STEP 5 — Build Panels

**What it does**: Transforms the raw long-format Parquet store into wide (T × N) panels — one panel per grid. Each panel has a `values` matrix (timesteps × nodes) and a parallel `mask` matrix (1.0 = observed, 0.0 = missing).

```bash
# Build the fine panel (for dependency discovery)
.venv/bin/python -m netraa.cli panel --grid fine

# Build the coarse panel (for forecasting)
.venv/bin/python -m netraa.cli panel --grid coarse
```

**Expected output**:
```
grid=fine freq=5min
shape: 4032 timesteps x 28 nodes
span: 2024-01-15 00:00:00+00:00 -> 2024-01-29 00:00:00+00:00
targets: 8, drivers: 4
mean coverage: 94.3%

lowest-coverage nodes:
  endpoint_cpm|SERVICE_METHOD-F6G7...     72.1%
  jvm_gc_collection_count|PGI-E5F6...     88.4%
  ...

saved -> data/panel/fine_*.parquet
```

> **If a node is dropped** (below 20% coverage), it is logged with its coverage percentage. It was not observed frequently enough to be modelled.

Output files: `data/panel/fine_values.parquet`, `data/panel/fine_mask.parquet`, `data/panel/fine_nodes.parquet`, `data/panel/fine_meta.json`
(And same for `coarse_`)

---

### STEP 6 — Build Statistical Dependency Map

**What it does**: Engine (a) — no training, no GPU. Discovers causal dependencies between all metric pairs using:
- Lagged cross-correlation (lags 0 to 24 × 5min = 2 hours)
- Mutual information
- Granger causality tests

Outputs an edge list describing which metrics influence which other metrics, with lag and strength information. This map is also used as a regularisation prior for the neural model in Step 7.

```bash
# Run on the fine grid (default — minute-scale lags matter here)
# Approximate time: 5–30 minutes (Granger tests are expensive)
.venv/bin/python -m netraa.cli graph
```

**Expected output**:
```
source                           target                          strength  dir     lag    MI  granger p  evidence
────────────────────────────────────────────────────────────────────────────────────────────────────────────────
service_cpm                      host_cpu_usage                     0.712    +    5min  0.31    0.0021    xcorr+mi+granger
jvm_memory_heap_used|PGI-...     host_mem_usage                     0.681    +   10min  0.28    0.0089    xcorr+mi+granger
service_cpm                      disk_write_iops|DISK-A...          0.523    +   15min  0.22    0.0134    xcorr+granger
database_access_cpm              disk_read_iops|DISK-A...           0.491    +   10min  0.19    0.0251    xcorr+mi+granger+structural
host_net_rx|eth0                 host_cpu_usage                     0.388    +    5min  0.14    0.0432    xcorr+granger
jvm_gc_collection_time|PGI-...   host_cpu_usage                     0.342    +    0min  0.12    0.0389    xcorr+granger+contemporaneous
...

47 edges -> data/graph/dependency_map_fine.json
```

**Faster alternative (skip Granger, show more edges)**:
```bash
# Skip the slow Granger pass (faster but less evidence for precedence)
.venv/bin/python -m netraa.cli graph --no-granger

# Show top 60 edges instead of top 25
.venv/bin/python -m netraa.cli graph --top 60

# Run dependency discovery on the coarse (daily) grid
.venv/bin/python -m netraa.cli graph --grid coarse
```

Output file: `data/graph/dependency_map_fine.json`

---

### STEP 7 — Train the ST-GNN (and Score It)

**What it does**: Engine (b). Trains the Spatio-Temporal GNN forecaster on the coarse panel, comparing:
1. `stgnn_graph` — full model with learned adjacency regularised toward the statistical prior from Step 6
2. `stgnn_nograph` — same architecture but graph convolutions disabled (ablation)
3. Four classical baselines: persistence, seasonal naive, drift, climatology

All six are evaluated on the same held-out test windows. The output tells you whether the graph earns its place.

```bash
# Train and backtest (uses coarse grid by default)
# Approximate time: 15 minutes to 2 hours depending on hardware
.venv/bin/python -m netraa.cli backtest
```

**Expected output**:
```
windows: 273 (train 191 / val 41 / test 41)
nodes: 28, targets: 8
input_steps: 90, horizons: [7, 30, 60, 90]
channels: 6 [value, mask, calendar x 4]

prior: 47 edges from dependency_map_fine.json

training stgnn_graph ...
training stgnn_nograph ...

model                MAE         RMSE      sMAPE %      pinball   P10-90 cov %
────────────────────────────────────────────────────────────────────────────────
stgnn_graph          0.0312      0.0419       4.21       0.0187        81.3
stgnn_nograph        0.0341      0.0459       4.63       0.0204        78.9
drift                0.0394      0.0521       5.12       -             -
persistence          0.0441      0.0583       5.81       -             -
seasonal_naive       0.0489      0.0634       6.23       -             -
climatology          0.0612      0.0789       7.94       -             -

Per-horizon MAE:
model                    h=7       h=30       h=60       h=90
────────────────────────────────────────────────────────────
stgnn_graph           0.0195     0.0284     0.0341     0.0428
stgnn_nograph         0.0198     0.0298     0.0371     0.0497
drift                 0.0221     0.0341     0.0412     0.0603

Learned graph vs statistical prior:
  learned edges       52
  prior edges         47
  overlap             38
  precision v prior   73.08%
  recall v prior      80.85%

Best model: stgnn_graph (MAE 0.0312)
The graph earns its place: 8.5% lower MAE than the identical graph-free model.
Beats the best classical baseline (0.0394).

saved -> artifacts/backtest_coarse.json
```

**Key flag — shorter input window if data is thin**:
```bash
# If your backfill is thin (< ~180 days), reduce the context window
.venv/bin/python -m netraa.cli backtest --input-steps 60
```

**Key flag — skip the ablation (faster, but you lose the "does the graph help?" answer)**:
```bash
.venv/bin/python -m netraa.cli backtest --no-ablation
```

**Output files**:

| File | Content |
|------|---------|
| `artifacts/stgnn.pt` | Model weights (best epoch on validation) |
| `artifacts/stgnn_scaler.json` | RobustScaler parameters for inference |
| `artifacts/stgnn_meta.json` | Node IDs, target IDs, horizons, quantiles, architecture |
| `artifacts/stgnn_history.json` | Epoch-by-epoch train and val loss |
| `artifacts/stgnn_learned_graph.json` | Learned adjacency edge list + agreement with statistical prior |
| `artifacts/backtest_coarse.json` | All scores, per-target breakdown, graph agreement, meta |

---

## 5. Keeping Data Current (Incremental Collection)

After the initial backfill, keep data fresh with incremental collection:

```bash
# Fetch only new data since the last stored timestamp (cron-safe, idempotent)
.venv/bin/python -m netraa.cli backfill --grid coarse --incremental
.venv/bin/python -m netraa.cli backfill --grid fine --incremental
```

**Recommended cron job** (run daily at 02:00):
```cron
0 2 * * * cd /path/to/Forecasting-Modal && .venv/bin/python -m netraa.cli backfill --grid coarse --incremental >> logs/backfill.log 2>&1
```

**After accumulating more data, retrain**:
```bash
# Rebuild panels
.venv/bin/python -m netraa.cli panel --grid fine
.venv/bin/python -m netraa.cli panel --grid coarse

# Redo dependency map (optional — graph evolves slowly)
.venv/bin/python -m netraa.cli graph

# Retrain the model
.venv/bin/python -m netraa.cli backtest
```

**Check current coverage**:
```bash
.venv/bin/python -m netraa.cli status
.venv/bin/python -m netraa.cli status --top 30  # show top 30 nodes
```

---

## 6. Useful Variations and Debug Commands

### Debug a specific metric selector

```bash
# Backfill only two specific metrics (for debugging)
.venv/bin/python -m netraa.cli backfill --grid fine --only host_cpu_usage,jvm_thread_live_count
```

### Full traceback on any error

```bash
.venv/bin/python -m netraa.cli <command> -v
```

### Use a different config file

```bash
.venv/bin/python -m netraa.cli backfill --grid fine -c configs/my_other_config.yaml
```

### Dependency map: show 60 edges, skip Granger

```bash
.venv/bin/python -m netraa.cli graph --top 60 --no-granger
```

### Dependency map on the daily grid (instead of fine)

```bash
.venv/bin/python -m netraa.cli graph --grid coarse
```

### Train with a shorter input window (fewer context days)

```bash
# Useful when backfill is thin (< ~200 days on the coarse grid)
.venv/bin/python -m netraa.cli backtest --input-steps 60
```

### Skip the graph ablation (saves ~50% training time)

```bash
.venv/bin/python -m netraa.cli backtest --no-ablation
```

### Run with pytest (if installed)

```bash
uv pip install --python .venv/bin/python pytest
.venv/bin/python -m pytest tests/ -v
```

---

## 7. Understanding the Output Files

### `data/retention_probe.json`
From `netraa probe`. Records how much history is actually available at each resolution and what grids are recommended. Use this to set `grids.fine.lookback_days` and `grids.coarse.lookback_days` in `configs/v1.yaml`.

### `data/topology.json`
From `netraa topology`. Maps your HOST and SERVICE to all related entity IDs:
```json
{
  "host_id": "HOST-D97...",
  "service_id": "SERVICE-C16...",
  "entities": {
    "HOST": ["HOST-D97..."],
    "SERVICE": ["SERVICE-C16..."],
    "DISK": ["DISK-A1B...", "DISK-B2C...", "DISK-C3D..."],
    "PROCESS_GROUP_INSTANCE": ["PROCESS_GROUP_INSTANCE-E5F..."],
    "SERVICE_METHOD": ["SERVICE_METHOD-F6G...", ...]
  }
}
```

### `data/validation_report.json`
From `netraa validate`. Per-metric status, series count, null fraction, observed min/max.

### `data/raw/grid=*/date=*/data.parquet`
From `netraa backfill`. Long-format time series: one row per `(timestamp, node_id)`.

### `data/panel/{grid}_values.parquet`
From `netraa panel`. Wide matrix: rows = timesteps, columns = node IDs. NaN = unobserved.

### `data/graph/dependency_map_fine.json`
From `netraa graph`. **Deliverable (a)**: the statistical dependency map.
```json
{
  "meta": { "n_nodes": 28, "n_edges": 47, "span_start": "...", ... },
  "nodes": [{ "node_id": "host_cpu_usage", "role": "target", ... }],
  "edges": [
    {
      "source": "service_cpm",
      "target": "host_cpu_usage",
      "strength": 0.712,
      "direction": "+",
      "lag_steps": 1,
      "lag_seconds": 300,
      "mutual_information": 0.31,
      "granger_p": 0.0021,
      "structural": true,
      "evidence": "xcorr+mi+granger+structural"
    },
    ...
  ]
}
```

### `artifacts/stgnn.pt`
PyTorch model weights (best validation epoch). Load with:
```python
import torch
from netraa.models.stgnn import STGNN
import json

meta = json.load(open("artifacts/stgnn_meta.json"))
model = STGNN(
    n_nodes=len(meta["node_ids"]),
    n_targets=len(meta["target_ids"]),
    in_channels=meta["n_channels"],
    input_steps=meta["input_steps"],
    n_horizons=len(meta["horizons"]),
    n_quantiles=len(meta["quantiles"]),
    target_idx=[meta["node_ids"].index(t) for t in meta["target_ids"]],
)
model.load_state_dict(torch.load("artifacts/stgnn.pt"))
model.eval()
```

### `artifacts/stgnn_learned_graph.json`
From `netraa backtest`. The learned adjacency edge list and agreement with the statistical prior.
```json
{
  "agreement_with_prior": {
    "learned_edges": 52,
    "prior_edges": 47,
    "overlap": 38,
    "precision_vs_prior": 0.7308,
    "recall_vs_prior": 0.8085
  },
  "edges": [
    { "source": "service_cpm", "target": "host_cpu_usage", 
      "weight": 0.234, "prior_weight": 0.712, "in_prior": true },
    ...
  ]
}
```

### `artifacts/backtest_coarse.json`
From `netraa backtest`. Complete evaluation results including per-model, per-horizon, per-target scores and the plain-English verdict.

---

## 8. Troubleshooting Common Errors

### Error: `DYNATRACE_API_TOKEN is not set`

```
RuntimeError: DYNATRACE_API_TOKEN is not set. Copy .env.example to .env and fill it in.
```

**Fix**: Create `.env` from `.env.example` and fill in your API token:
```bash
cp .env.example .env
# Edit .env: set DYNATRACE_API_TOKEN=dt0c01.XXXX...
```

---

### Error: `data/topology.json not found. Run netraa topology first.`

```
SystemExit: data/topology.json not found. Run `netraa topology` first.
```

**Fix**: Run `netraa topology` before `validate`, `backfill`, `panel`, `graph`, or `backtest`.

---

### Error: `no panel found for grid=coarse`

```
SystemExit: no panel found for grid=coarse. Run: python -m netraa.cli panel --grid coarse
```

**Fix**: Run `netraa panel --grid coarse` before `netraa backtest`.

---

### Error: `input_steps=90 is too short for blocks=2, kernel_size=3 (needs > 6)`

This means your panel has fewer timesteps than the minimum required. With `blocks=2` and `kernel_size=3`, the minimum is:
```
receptive = (3-1)×2^0 + (3-1)×2^1 = 2 + 4 = 6 steps minimum
```
But in practice you need `input_steps + max_horizon + 3 windows`, so at least 90+90+3 = 183 days.

**Fix**: Either collect more history (run backfill with a longer period) or reduce input steps:
```bash
.venv/bin/python -m netraa.cli backtest --input-steps 30
```

---

### Error: `panel has X steps; a window needs Y and at least 3 windows are required`

The coarse panel doesn't have enough data to create at least 3 non-overlapping train/val/test windows.

**Fix**: 
1. Run `netraa status` to see how much data you have
2. If insufficient, either run `backfill` for a longer period or use `--input-steps` to reduce the window size:
```bash
# Minimum viable: 60 days input + 90 days max horizon + 3 windows
# = 153 + 3 = 156 days minimum
.venv/bin/python -m netraa.cli backtest --input-steps 30
```

---

### Error: All metrics show `NO_DATA` after backfill

**Possible causes**:
1. Wrong `host_id` or `service_id` — the entity IDs don't match your Dynatrace tenant
2. The specified host has no data in the selected time range
3. API token missing `metrics.read` scope

**Debug**:
```bash
# Verify topology resolved correctly
cat data/topology.json

# Run with verbose flag
.venv/bin/python -m netraa.cli validate -v

# Test a single metric manually
.venv/bin/python -m netraa.cli backfill --grid fine --only host_cpu_usage -v
```

---

### Error: `DIMENSION_MISMATCH` in validate for JVM metrics

```
jvm_memory_heap_used  DIMENSION_MISMATCH  registry filters on dt.entity.host but...
```

This means `topology.py` did not find any `PROCESS_GROUP_INSTANCE` entities.

**Fix**: Check whether your Dynatrace service is monitored by a Java OneAgent. If not, disable JVM metrics in `metrics_registry.yaml`:
```yaml
- key: jvm_memory_heap_used
  ...
  enabled: false  # ← add this
```

---

### Granger tests take too long

The full Granger pass runs O(N²) tests which can take 10–30 minutes on a 28-node panel.

**Fix**: Skip Granger for faster iteration (still uses xcorr + MI + structural):
```bash
.venv/bin/python -m netraa.cli graph --no-granger
```

---

## 9. Re-Training After New Data

Re-training is safe and idempotent. The recommended cadence:

**Daily** (cron):
```bash
# Fetch new data (incremental — safe to run daily)
.venv/bin/python -m netraa.cli backfill --grid coarse --incremental
.venv/bin/python -m netraa.cli backfill --grid fine --incremental
```

**Weekly** (or after significant new data):
```bash
# Rebuild panels and retrain
.venv/bin/python -m netraa.cli panel --grid fine
.venv/bin/python -m netraa.cli panel --grid coarse
.venv/bin/python -m netraa.cli graph              # redo dependency map
.venv/bin/python -m netraa.cli backtest           # retrain + re-score
```

**Note on artifacts**: `netraa backtest` overwrites all artifacts in `artifacts/`. Previous weights are not automatically versioned — back up `artifacts/stgnn.pt` if you want to preserve a specific model version before retraining.

---

## Quick Reference — Complete Pipeline

```bash
# 0. Setup (one-time)
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env   # then edit: set DYNATRACE_API_TOKEN

# 0b. Smoke test (offline, no token needed)
.venv/bin/python -m netraa.cli smoke
.venv/bin/python tests/test_pipeline.py

# 1. Measure what your tenant retains
.venv/bin/python -m netraa.cli probe
# → update configs/v1.yaml with recommended grids

# 2. Resolve entity IDs
.venv/bin/python -m netraa.cli topology

# 3. Validate every metric (STOP HERE if failures)
.venv/bin/python -m netraa.cli validate

# 4. Pull history into the store
.venv/bin/python -m netraa.cli backfill --grid fine
.venv/bin/python -m netraa.cli backfill --grid coarse

# 4b. Check what was stored
.venv/bin/python -m netraa.cli status

# 5. Build the T×N panels
.venv/bin/python -m netraa.cli panel --grid fine
.venv/bin/python -m netraa.cli panel --grid coarse

# 6. (a) Statistical dependency map — no training, no GPU
.venv/bin/python -m netraa.cli graph

# 7. (b) Train ST-GNN + ablation + 4 baselines → score all
.venv/bin/python -m netraa.cli backtest

# Ongoing: keep data current
.venv/bin/python -m netraa.cli backfill --grid coarse --incremental
```

**Output locations summary**:

| File | Stage |
|------|-------|
| `data/retention_probe.json` | Step 1 — probe |
| `data/topology.json` | Step 2 — topology |
| `data/validation_report.json` | Step 3 — validate |
| `data/raw/grid=*/date=*/data.parquet` | Step 4 — backfill |
| `data/panel/{grid}_*.parquet` | Step 5 — panel |
| `data/graph/dependency_map_fine.json` | Step 6 — graph ← Deliverable (a) |
| `artifacts/stgnn.pt` | Step 7 — backtest ← Deliverable (b) |
| `artifacts/stgnn_learned_graph.json` | Step 7 — backtest |
| `artifacts/backtest_coarse.json` | Step 7 — backtest (scores) |

---

*End of Training Guide*
