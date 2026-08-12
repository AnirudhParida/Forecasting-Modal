# Netraa — Technical Documentation
## Spatio-Temporal GNN Infrastructure Forecasting System

> **Version**: v1 (POC) | **Scope**: One Service on One Host | **Horizon**: Next quarter (7/30/60/90 days)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Repository Layout](#3-repository-layout)
4. [File-by-File Deep Dive](#4-file-by-file-deep-dive)
5. [Data Ingestion Deep Dive](#5-data-ingestion-deep-dive)
6. [Inter-Metric Dependency Discovery](#6-inter-metric-dependency-discovery)
7. [Forecasting: CPU, Memory, Disk](#7-forecasting-cpu-memory-disk)
8. [End-to-End Data Flow](#8-end-to-end-data-flow)
9. [Key Design Decisions and Bug Fixes](#9-key-design-decisions-and-bug-fixes)
10. [Configuration Reference](#10-configuration-reference)
11. [Metrics Registry Reference](#11-metrics-registry-reference)

---

## 1. Project Overview

**Netraa** is a capacity forecasting system for infrastructure metrics. It ingests time-series data from **Dynatrace** (an observability platform), discovers causal relationships between metrics, and trains a **Spatio-Temporal Graph Neural Network (ST-GNN)** to forecast future values of CPU usage, memory utilization, and disk I/O at horizons up to one quarter (90 days) ahead.

The system has two distinct engines:

| Engine | What it does | Output |
|--------|-------------|--------|
| **(a) Statistical** | Discovers causal dependencies between metrics using lagged cross-correlation, mutual information, and Granger causality — no training, no GPU | `dependency_map_fine.json` |
| **(b) Neural** | Trains an ST-GNN forecaster whose graph adjacency is jointly learned and regularised toward the statistical prior | `stgnn.pt`, `stgnn_learned_graph.json` |

---

## 2. High-Level Architecture

Two grids are used simultaneously because a quarter-ahead forecast from a 5-minute series would be 129,600 steps ahead — unreasonable to model directly:

| Grid | Resolution | Lookback | Purpose |
|------|-----------|---------|---------|
| `fine` | 5 minutes | 14 days | Dependency discovery (minute-scale lags matter) |
| `coarse` | 1 day | 400 days | Quarter-ahead forecasting |

```
[Dynatrace API v2]
        |
        | DynatraceClient (HTTP, retries, pagination)
        v
[INGEST LAYER]
  probe.py       → measures actual API retention
  topology.py    → walks entity graph: HOST → PGI, DISK, SM
  validate.py    → sanity-checks every registry metric
  backfill.py    → pulls history, chunked, into Parquet store
  store.py       → append-only, date-partitioned Parquet store
  legacy_csv.py  → import the 3 existing CSVs (smoke tests)
        |
        | long-format Parquet rows
        v
[FEATURES LAYER]
  panel.py       → long → wide (T × N) panel + observation mask
  transforms.py  → abs/log1p/diff, RobustScaler, windowing
        |
   _____|_____________
  |                   |
  | fine panel        | coarse panel
  v                   v
[GRAPH LAYER]     [MODELS LAYER]
  statistical.py    dataset.py   → sliding windows
  learned.py        stgnn.py     → ST-GNN model
                    train.py     → training loop
        |           baselines.py → 4 comparators
        | A_prior        |
        |________________|
                    v
              [EVAL LAYER]
                backtest.py → test scores
                metrics.py  → MAE/RMSE/sMAPE
```

---

## 3. Repository Layout

```
Forecasting-Modal/
├── configs/
│   └── v1.yaml                   # Grid, model, graph, forecast config
├── netraa/
│   ├── cli.py                    # Command-line interface (8 sub-commands)
│   ├── config.py                 # Config dataclasses + YAML/env loader
│   ├── ingest/
│   │   ├── dynatrace_client.py   # HTTP client: retries, pagination, auth
│   │   ├── registry.py           # MetricSpec, Registry — metric catalogue
│   │   ├── topology.py           # Entity graph walker (HOST→PGI/DISK/SM)
│   │   ├── backfill.py           # Historical + incremental fetch
│   │   ├── store.py              # Append-only Parquet store
│   │   ├── probe.py              # Retention probing
│   │   ├── validate.py           # Pre-backfill metric validation
│   │   └── legacy_csv.py         # Import existing 3 CSVs
│   ├── features/
│   │   ├── panel.py              # Long → wide panel builder
│   │   └── transforms.py         # Transforms, RobustScaler, windowing
│   ├── graph/
│   │   ├── statistical.py        # Statistical dependency discovery
│   │   └── learned.py            # Learnable graph adjacency (nn.Module)
│   ├── models/
│   │   ├── dataset.py            # Panel → supervised Dataset
│   │   ├── stgnn.py              # ST-GNN architecture
│   │   ├── train.py              # Training loop + artifact saver
│   │   └── baselines.py          # Persistence/seasonal/drift/climatology
│   └── eval/
│       ├── backtest.py           # Chronological backtest runner
│       └── metrics.py            # MAE, RMSE, sMAPE, pinball, coverage
├── metrics_registry.yaml         # Single source of truth for all metrics
├── tests/
│   └── test_pipeline.py          # 13 correctness tests (offline)
├── .env.example                  # Template for secrets
└── commands.txt                  # Quick-reference for operators
```

---

## 4. File-by-File Deep Dive

### 4.1 Configuration Layer

#### `netraa/config.py`

**Purpose**: Loads all configuration from YAML and environment variables. Secrets (`DYNATRACE_API_TOKEN`) come from the environment, never from YAML.

**Key dataclasses**:

| Class | Key Fields | Role |
|-------|-----------|------|
| `GridConfig` | `resolution`, `pandas_freq`, `lookback_days`, `purpose` | Defines one sampling grid |
| `ForecastConfig` | `grid`, `input_steps`, `horizons`, `quantiles` | Forecaster parameters |
| `GraphConfig` | `max_lag_steps`, `xcorr_threshold`, `granger_alpha`, `mi_threshold`, `top_k` | Graph discovery thresholds |
| `ModelConfig` | `hidden`, `blocks`, `kernel_size`, `dropout`, `lr`, `epochs`, `patience` | ST-GNN hyperparameters |
| `Config` | Aggregates all above + `host_id`, `service_id`, `api_token` | Master config |

**Loading chain**:
1. Reads `.env` via a minimal inline parser (no `python-dotenv` dependency)
2. Loads `configs/v1.yaml` with PyYAML
3. Environment variables override YAML for `host_id`, `service_id`, `base_url`, `api_token`

**Path helpers**: `cfg.raw_dir` → `data/raw/`, `cfg.panel_dir` → `data/panel/`, `cfg.graph_dir` → `data/graph/`, `cfg.topology_path` → `data/topology.json`

---

### 4.2 Ingest Layer

#### `netraa/ingest/dynatrace_client.py`

**Purpose**: The sole HTTP gateway to Dynatrace API v2. All previous bare `requests.get` calls were replaced by this client.

**What it solves**:
- Token is read from environment, never hardcoded
- Retry-with-backoff (5 retries, 1.5x backoff) honouring `Retry-After` on HTTP 429
- Automatic pagination via `nextPageKey` — old scripts read only the first page and silently lost data
- Failed queries raise explicit errors instead of printing and moving on

**Core dataclasses**:

```python
@dataclass
class Series:
    dimensions: list[str]        # raw dimension values
    dimension_map: dict[str,str] # named dimension → value
    timestamps: list[int]        # ms-epoch
    values: list[float | None]   # None = null bucket
    null_fraction: float         # computed property

@dataclass
class QueryResult:
    metric_id: str
    series: list[Series]
    warnings: list[str]
    is_empty: bool      # True if no timestamps at all
    point_count: int
```

**Key methods**:

| Method | What it does |
|--------|-------------|
| `query(selector, from, to, resolution)` | Runs a metric query, follows all pagination pages |
| `metric_descriptor(metric_id)` | Fetches metric metadata (unit, available dimensions) |
| `entity(entity_id)` | Fetches entity with all relationships |
| `entities(entity_selector)` | Paginated list of entities |

**`chunk_range(start_ms, end_ms, chunk_days)`**: Generator that splits a long time window into API-sized chunks. Dynatrace caps points per request; chunking ensures no data is silently truncated.

---

#### `netraa/ingest/registry.py`

**Purpose**: Defines `MetricSpec` (one metric's complete specification) and `Registry` (the full catalogue loaded from `metrics_registry.yaml`). This is the single source of truth for metric identity.

**`MetricSpec` fields**:

| Field | Meaning |
|-------|---------|
| `key` | Canonical node name (e.g., `host_cpu_usage`) |
| `selector` | Dynatrace metric ID (e.g., `builtin:host.cpu.usage`) |
| `entity_type` | `HOST`, `SERVICE`, `PROCESS_GROUP_INSTANCE`, `SERVICE_METHOD`, or `DISK` |
| `split_by` | Which dimension to keep as separate series (e.g., `dt.entity.disk`, `nic`) |
| `agg` | `avg`, `sum`, `max`, `min`, `count` |
| `role` | `driver` (input feature), `intermediate`, or `target` (what to forecast) |
| `resource` | `cpu`, `memory`, `disk`, `network`, `jvm`, `service` |
| `transform` | `none`, `abs`, `log1p`, or `diff` |
| `unit` | `percent`, `byte`, `count`, `millisecond`, etc. |
| `aliases` | Legacy column names mapping to this canonical key |

**`ENTITY_DIMENSION` mapping** (critical for correct Dynatrace filtering):

```python
{
  "HOST":                    "dt.entity.host",
  "SERVICE":                 "dt.entity.service",
  "PROCESS_GROUP_INSTANCE":  "dt.entity.process_group_instance",
  "SERVICE_METHOD":          "dt.entity.service_method",
  "DISK":                    "dt.entity.disk",
}
```

> **Critical fix**: JVM metrics (`builtin:tech.jvm.*`) use dimension `dt.entity.process_group_instance`. Old scripts filtered them on `dt.entity.host` — matched nothing, every JVM column was silently empty.

**`MetricSpec.build_selector(entity_ids)`**: Composes the full Dynatrace selector:
```
builtin:host.cpu.usage:filter(eq("dt.entity.host","HOST-D9..."))
  :splitBy(""):avg
```

**`node_id(metric_key, dimension)`**: Ensures one name per physical signal:
- `node_id("host_net_rx", "eth0")` → `"host_net_rx|eth0"`
- `node_id("host_cpu_usage", "")` → `"host_cpu_usage"`

---

#### `netraa/ingest/topology.py`

**Purpose**: Walks the Dynatrace entity relationship graph from `host_id` and `service_id` to discover all related entity IDs needed to correctly filter metrics.

**`resolve(client, host_id, service_id)` logic**:
1. Fetch entity payloads for the host and service via `client.entity()`
2. Parse `toRelationships` and `fromRelationships`, categorise each related entity ID by type prefix (`HOST-`, `SERVICE-`, `DISK-`, etc.)
3. **Process Group Instances**: Prefer PGIs related to both host AND service (intersection). Falls back to host-only PGIs if intersection is empty
4. **Disks**: Fallback to querying `type(DISK),fromRelationships.isDiskOf(entityId(...))`
5. **Service Methods**: Fallback to querying `type(SERVICE_METHOD),fromRelationships.isServiceMethodOf(entityId(...))`

Result is saved to `data/topology.json` for all subsequent commands.

---

#### `netraa/ingest/store.py`

**Purpose**: Append-only, date-partitioned Parquet store for all raw time series data.

**Long-format schema** (one row per observation):

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | `datetime64[ns, UTC]` | UTC timestamp |
| `node_id` | `str` | Canonical node name |
| `metric_key` | `str` | Canonical metric key |
| `dimension` | `str` | Split dimension value ("" when unsplit) |
| `entity_type` | `str` | `HOST`, `SERVICE`, etc. |
| `value` | `float64` | Observed value (NaN for null buckets) |

**Partition layout**:
```
data/raw/
  grid=fine/
    date=2024-01-01/data.parquet
    date=2024-01-02/data.parquet
  grid=coarse/
    date=2024-01-01/data.parquet
```

**Idempotent writes** (`write_rows`): Merges into daily chunks. If partition exists, concatenates and deduplicates on `(timestamp, node_id)` keeping latest value — re-runs are safe.

---

#### `netraa/ingest/backfill.py`

**Purpose**: Orchestrates fetching historical time series from Dynatrace and writing them to the store.

**`backfill(client, registry, topology, grid, raw_dir, ...)` per metric**:
1. Resolve entity IDs from topology: `topology.ids_for(spec.entity_type)`
2. If no IDs → record `UNRESOLVED` status and skip
3. Build the full Dynatrace selector: `spec.build_selector(entity_ids)`
4. Split time range into chunks: `chunk_range(start_ms, end_ms, chunk_days)`
5. For each chunk: query → convert to long rows via `series_to_rows()` → write via `store.write_rows()`
6. Classify outcome: `OK`, `PARTIAL`, `ERROR`, or `NO_DATA`

**`collect_incremental`**: Finds latest stored timestamp, re-fetches last 2 steps (overlap for Dynatrace bucket revisions), calls `backfill()` for the new window only.

---

#### `netraa/ingest/probe.py`

**Purpose**: Empirically measures what data the Dynatrace tenant actually retains at each resolution and lookback combination.

**How it works**: Queries `host_cpu_usage` across a grid of `(resolution, lookback)` combinations — e.g., `(1m, 1d)`, `(5m, 14d)`, `(1h, 90d)`.

Checks per combination:
- `points`: How many data points were returned?
- `observed_spacing_s`: Actual median gap between timestamps
- `downgraded`: Is observed spacing >1.5× requested? (Dynatrace serves coarser buckets when fine history expired)

**`recommend(rows)`**: Picks fine (finest resolution covering ≥7 days without downgrade) and coarse (deepest lookback) grids. Recommendation goes into `configs/v1.yaml`.

---

#### `netraa/ingest/validate.py`

**Purpose**: Pre-flight validation of every registry entry before spending a backfill.

**Validation checks per metric**:

| Step | Check | Failure status |
|------|-------|---------------|
| 1 | Does the metric ID exist? | `MISSING_METRIC` |
| 2 | Is the filter dimension one the metric actually has? | `DIMENSION_MISMATCH` |
| 3 | Does topology have entity IDs of the required type? | `UNRESOLVED` |
| 4 | Does a 24h probe query return data? | `NO_DATA` |
| 5 | Are all values null? | `ALL_NULL` |
| 6 | Do values contradict the declared unit? (negative percent) | `UNIT_VIOLATION` |

> Do not run `backfill` until this reports no FAIL rows.

---

#### `netraa/ingest/legacy_csv.py`

**Purpose**: Imports the three original CSVs into the long-format store, mapping inconsistent column names to canonical node IDs.

**Key problem**: Three files used three naming conventions for the same signal:
- CPU file: `meter_vm_network_receive`
- Memory file: `meter_vm_network_receive_HOST-D9739223FC540A23`

Both collapse onto canonical node `host_net_rx` via the `aliases` field in the registry.

Used **only for smoke testing** (11 rows each) — never for real model quality assessment.

---

### 4.3 Features Layer

#### `netraa/features/panel.py`

**Purpose**: Transforms the long-format Parquet store into a wide **(T × N) panel** where rows are timesteps and columns are nodes.

**`Panel` dataclass**:

| Field | Type | Description |
|-------|------|-------------|
| `values` | `pd.DataFrame` (T×N) | Metric values; NaN where unobserved |
| `mask` | `pd.DataFrame` (T×N) | 1.0 = observed, 0.0 = missing |
| `nodes` | `pd.DataFrame` | Per-node metadata (role, resource, unit, etc.) |
| `grid` | str | Which grid this panel was built from |
| `freq` | str | Pandas frequency alias |

> **Critical design**: Missingness is carried in a parallel `mask` DataFrame, NOT filled with 0.0. A zero for "CPU was not reported" claims the CPU was idle — it poisons both correlation scores and forecast training.

**`build_panel()` pipeline**:
```
read_grid() → raw long DataFrame
    → pivot_table() → wide irregular DataFrame
    → resample(freq).agg(per-node aggregation method)
    → Rebuild mask from raw counts (not from resampled NaN)
    → Reindex to full regular DatetimeIndex (make gaps explicit)
    → Drop nodes below min_coverage (20%) — logged, never silent
    → ffill(limit=2) — bounded forward-fill for short gaps
    → Panel(values, mask, nodes, grid, freq)
```

**Per-node aggregation**: Each metric uses its declared `agg` method — `sum` for counters, `avg` for gauges.

---

#### `netraa/features/transforms.py`

**Purpose**: Per-node value transforms, robust scaling, calendar feature generation, and sliding window construction.

**`apply_node_transforms(panel)`**: Applies each node's declared `transform`:
- `abs` → absolute value (fix for negative memory percentages — blocker B5)
- `log1p` → `log(1 + x)` for right-skewed counters (network bytes, disk throughput)
- `diff` → first difference for non-stationary series
- `none` → no change

**`clip_outliers(panel, lower_q=0.001, upper_q=0.999)`**: Winsorises each column. Dynatrace counters emit single absurd spikes on agent restart; one spike dominates a cross-correlation over a short panel.

**`RobustScaler`**: Median/IQR scaling per node.
- `fit(df)`: Computes median and IQR on **training slice only** — no future leakage
- `transform(df)`: `(x − median) / IQR`
- `save/load`: Persists to JSON for inference

**`calendar_features(index, freq_seconds)`**: Cyclical (sine/cosine) encodings:
- Day-of-week: `sin/cos(2π × dow / 7)` — always included
- Day-of-month: `sin/cos(2π × (dom−1) / 31)` — always included
- Time-of-day: `sin/cos(2π × seconds / 86400)` — only on sub-daily grids

Cyclical encodings avoid the artificial discontinuity between Monday=0 and Sunday=6.

**`make_windows(values, mask, calendar, input_steps, horizons, target_idx)`**: Sliding window construction:
- `X` shape: `(S, N, T_in, C)` where `C = [value, mask, *calendar]`
- `Y` shape: `(S, N_target, H)` — target values at each horizon
- `Y_mask` shape: `(S, N_target, H)` — whether each target observation is present

**`chronological_split`**: Contiguous, ordered train/val/test split. Never shuffles — time series must not be shuffled.

---

### 4.4 Graph Layer

#### `netraa/graph/statistical.py`

**Purpose**: Discovers causal dependencies between all pairs of metric nodes. Engine (a) — no training, no GPU.

**Three signals combined**:

| Signal | What it detects | Algorithm |
|--------|----------------|-----------|
| Lagged cross-correlation | Linear coupling, sign, and delay | Vectorised NaN-aware correlation at lags 0...max_lag |
| Mutual information | Non-linear coupling correlation misses | 2D histogram, normalised MI |
| Granger causality | Does source's past improve prediction of target? | `statsmodels.grangercausalitytests`, applied on first-differenced series |

**`STRUCTURAL_FLOW`** (hard-coded causal flows from topology knowledge):
```python
[("service", "jvm"), ("service", "memory"), ("service", "cpu"),
 ("service", "disk"), ("service", "network"),
 ("jvm", "cpu"), ("jvm", "memory"),
 ("network", "cpu"), ("disk", "cpu")]
```

**`discover(panel, ...)` algorithm**:
```
Step 1: lagged_correlation_scan → best_corr (N×N), best_lag (N×N), overlap (N×N)
Step 2: structural_pairs → set of (src, dst) from topology knowledge
Step 3: For each (i,j): filter by |corr| >= threshold OR structural
Step 4: Compute MI for each candidate pair
Step 5: Granger causality on candidates only (expensive)
Step 6: Accept edge if:
    relevance: |corr| >= 0.30 OR MI >= 0.05
    AND (precedence: Granger p <= 0.05 OR lag > 0) OR structural
Step 7: Drop lag-0 mirror artifacts
Step 8: Prune to top_k=8 strongest incoming edges per target
```

**`Edge` fields**: `source`, `target`, `strength`, `direction`, `lag_steps`, `lag_seconds`, `mutual_information`, `granger_p`, `structural`, `evidence`, `n_overlap`

**`to_adjacency(edges, node_ids)`**: Converts edges to row-normalised N×N adjacency matrix `A_prior`.

---

#### `netraa/graph/learned.py`

**Purpose**: Learnable adjacency matrix as `nn.Module`. Learned jointly with the forecaster — an edge exists because it reduces prediction error.

**Formula**: `A = ReLU(tanh(alpha × (E1 @ E2.T − E2 @ E1.T)))`

- `E1`, `E2` are `(N, embed_dim)` learnable embeddings
- Antisymmetric inner product makes the graph directed (i→j ≠ j→i)
- `ReLU(tanh(...))` produces values in `[0, 1)`

**`forward()`**: Top-k sparsification (keep top_k strongest outgoing edges per node) then row-normalise.

**Regularisation losses**:
- `prior_loss()`: `L1(A_learned − A_prior)` — pulls toward statistical map
- `sparsity_loss()`: `L1(A_dense)` — prevents fully-connected graph

---

### 4.5 Models Layer

#### `netraa/models/dataset.py`

**Purpose**: Converts a Panel into a Dataset — supervised sliding windows with leak-free scaling.

**`prepare()` steps**:
1. Apply node transforms
2. Clip outliers
3. Identify target nodes (`role == "target"`)
4. Compute `n_windows = T − input_steps − max_horizon + 1`
5. Split chronologically: 70% train / 15% val / 15% test
6. Fit `RobustScaler` only on training timesteps
7. Apply scaler to full panel
8. Compute calendar features
9. Slide windows via `make_windows()`

**`Dataset` key shapes**:

| Field | Shape | Content |
|-------|-------|---------|
| `X` | `(S, N, T_in, C)` | Input windows: value, mask, calendar |
| `Y` | `(S, N_target, H)` | Target values at each horizon |
| `Y_mask` | `(S, N_target, H)` | Observation presence |

---

#### `netraa/models/stgnn.py`

**Purpose**: The Spatio-Temporal GNN forecaster (MTGNN/Graph-WaveNet family).

**Architecture**:
```
Input: (B, N, T_in, C)
  → start Conv2d(C → hidden)  [1×1 channel projection]
  → STBlock × blocks (dilation doubles each block)
      ├─ GatedTemporalConv  [dilated causal conv along time]
      └─ MixHopPropagation  [graph conv, 2 hop orders]
  → head: ReLU → Conv2d → ReLU → Dropout → Conv2d
Output: (B, N_target, H, Q)  [H horizons × Q quantiles]
```

**`GatedTemporalConv`**: Two parallel dilated causal convolutions. Output: `tanh(filter(x)) × sigmoid(gate(x))`. Gating controls which features pass through.

**`MixHopPropagation`**: Propagates features over adjacency for `order` hops. Concatenates all hop orders to avoid over-smoothing.

**`STBlock`**: Temporal → Graph → Residual + BatchNorm. Returns `(out, skip)`.

**`masked_quantile_loss`** (Pinball loss):
```
error = target - pred
loss = max(q × error, (q−1) × error) × mask
```
The mask ensures missing observations contribute zero gradient — model never learns "unobserved = idle."

**`graph_regularisation`**: `prior_weight × prior_loss + sparsity_weight × sparsity_loss`

`use_graph=False` disables all graph convolutions — this is the ablation model.

---

#### `netraa/models/train.py`

**Purpose**: Training loop. One run produces the forecaster and the learned adjacency.

**Training loop per epoch**:
```
1. Forward pass → pred (B, N_target, H, Q)
2. loss = masked_quantile_loss(pred, y, mask, quantiles)
3. total = loss + graph_regularisation(prior_weight, sparsity_weight)
4. total.backward()
5. Gradient clip (max norm 5.0)
6. Adam step
7. Validation pass (no grad)
8. ReduceLROnPlateau scheduler step
9. Save best state on val improvement; early stop after patience epochs
```

**`save_artifacts()`** saves: `stgnn.pt`, `stgnn_scaler.json`, `stgnn_meta.json`, `stgnn_history.json`, `stgnn_learned_graph.json`.

---

#### `netraa/models/baselines.py`

**Purpose**: Four classical baselines the ST-GNN must beat before being considered worth deploying.

| Baseline | Algorithm | Note |
|----------|-----------|------|
| `persistence` | Last observed value held flat | Strong for short horizons |
| `seasonal_naive` | Value from one season ago | Strong for weekly patterns |
| `drift` | Linear extrapolation via Theil-Sen slope | **Hard to beat at quarter horizon** |
| `climatology` | Training-period median | Ignores all recent signal |

If any baseline wins, that is the finding — it belongs in the report, not quietly dropped.

---

### 4.6 Evaluation Layer

#### `netraa/eval/metrics.py`

All metrics are masked — missing actuals never count as a hit or miss.

| Metric | Formula | Notes |
|--------|---------|-------|
| `mae` | `mean(|y_true − y_pred|)` | Primary ranking metric |
| `rmse` | `sqrt(mean((y_true − y_pred)²))` | Penalises large errors |
| `smape` | `mean(|y-ŷ| / ((|y|+|ŷ|)/2)) × 100%` | Symmetric — avoids division by ~0 |
| `pinball` | `mean(max(q×e, (q−1)×e))` | Quantile loss |
| `interval_coverage` | Fraction inside P10–P90 band | Well-calibrated ≈ 80% |

---

#### `netraa/eval/backtest.py`

**Purpose**: Chronological backtest runner comparing all models on the test split.

**`run()` steps**:
1. Run 4 classical baselines on `test_idx`
2. Train `stgnn_graph` (full model)
3. Train `stgnn_nograph` (ablation — same architecture, no graph convolutions)
4. Evaluate all models on same test windows
5. Compute graph agreement statistics

**`BacktestResult.verdict()`** produces a plain-English verdict:
- Does `stgnn_graph` beat `stgnn_nograph`? (Is the graph earning its place?)
- Does ST-GNN beat the best classical baseline?

---

### 4.7 CLI Entry Point

#### `netraa/cli.py`

8 sub-commands with descriptive error messages and next-step hints:

```
netraa probe      → measures tenant retention per resolution
netraa topology   → resolves HOST/SERVICE → PGI/DISK/SM entity IDs
netraa validate   → sanity-checks every registry metric vs live API
netraa backfill   → pulls history into the Parquet store
netraa status     → shows per-node coverage summary
netraa panel      → builds the T×N panel + mask
netraa graph      → (a) statistical dependency map — no training
netraa backtest   → (b) train ST-GNN + ablation + baselines, score all
netraa smoke      → offline end-to-end run on the 11-row legacy CSVs
```

---

## 5. Data Ingestion Deep Dive

### 5.1 Complete Ingestion Flow

```
User runs: netraa backfill --grid fine
    |
    v
load_config("configs/v1.yaml")
  → DYNATRACE_API_TOKEN from .env
  → host_id, service_id, base_url
  → GridConfig(fine): resolution=5m, lookback_days=14
    |
    v
cmd_backfill()
  → Registry.load("metrics_registry.yaml")  [validates 27 MetricSpecs]
  → Topology.load("data/topology.json")     [resolved entity IDs]
  → DynatraceClient(base_url, api_token)    [retry adapter: 5 retries]
    |
    v
backfill(client, registry, topology, grid, raw_dir)

For each MetricSpec (27 metrics):
  → entity_ids = topology.ids_for(spec.entity_type)
      e.g., DISK metrics: ["DISK-A1B2", "DISK-C3D4"]
      e.g., JVM metrics:  ["PROCESS_GROUP_INSTANCE-E5F6"]
  
  → selector = spec.build_selector(entity_ids)
      e.g.: "builtin:host.disk.reads
             :filter(in("dt.entity.disk",entityId("DISK-A1B2","DISK-C3D4")))
             :splitBy("dt.entity.disk"):sum"
  
  → For each time chunk (~5000 points):
      result = client.query(selector, start_ms, end_ms, "5m")
              [follows nextPageKey pagination]
      
      rows = series_to_rows(spec, result)
             [long format: one row per (timestamp, node_id)]
             [node_id = "disk_read_iops|DISK-A1B2"]
      
      store.write_rows(rows, raw_dir, "fine")
             [partitioned by date, idempotent upsert on (timestamp, node_id)]
```

### 5.2 Entity Type → Dimension Mapping (Critical)

```
Metric family           Dynatrace filter dimension       Entity IDs from
──────────────────────  ───────────────────────────────  ─────────────────────
host.cpu.usage          dt.entity.host                   topology["HOST"]
host.mem.*              dt.entity.host                   topology["HOST"]
host.net.nic.*          dt.entity.host (split by nic)    topology["HOST"]
host.disk.*             dt.entity.disk                   topology["DISK"]
service.requestCount    dt.entity.service                topology["SERVICE"]
service.keyRequest.*    dt.entity.service_method         topology["SERVICE_METHOD"]
tech.jvm.*              dt.entity.process_group_instance topology["PROCESS_GROUP_INSTANCE"]
```

> Old scripts used `dt.entity.host` for JVM metrics — matched nothing, silently returned empty.

---

## 6. Inter-Metric Dependency Discovery

### 6.1 Three-Stage Discovery Pipeline

```
Panel (fine grid, 5-minute, T×N)
    |
    v Stage 1: Lagged Cross-Correlation Scan
    |
    |  For each lag l in {0, 1, ..., 24}:
    |    Vectorised NaN-aware correlation: source[t] vs target[t+l]
    |    Track best lag and strongest correlation per (i,j) pair
    |
    |  Output: best_corr[N×N], best_lag[N×N], overlap[N×N]
    |
    v Stage 2: Structural Filtering + Mutual Information
    |
    |  Candidates: |corr| >= 0.30 OR in STRUCTURAL_FLOW
    |  Compute MI for each candidate pair
    |  Filter: (|corr| >= 0.30 OR MI >= 0.05)
    |
    v Stage 3: Granger Causality (on candidates only)
    |
    |  H0: "source does not Granger-cause target"
    |  Both series are first-differenced (avoids spurious causality from trends)
    |  statsmodels.grangercausalitytests → p-value
    |
    v Acceptance Gate
    |
    |  Accept edge if:
    |    relevance: |corr| >= 0.30 OR MI >= 0.05
    |    AND (Granger p <= 0.05 OR lag > 0) OR structural
    |
    v Artifact Resolution (drop lag-0 mirrors)
    v Top-k Pruning (keep 8 strongest incoming edges per target)
    v
dependency_map_fine.json
```

### 6.2 Expected CPU/Memory/Disk Dependency Chains

Based on `STRUCTURAL_FLOW` and registry `resource` fields:

```
service_cpm          → cpu (traffic → CPU rises)
                     → memory (heap allocation grows)
                     → disk (DB writes increase)

jvm_thread_live_count
jvm_memory_heap_used → cpu (GC pauses → CPU spikes)
jvm_gc_collection    → memory (heap pressure → OS memory)

database_access_cpm  → disk (DB calls → disk I/O)

host_net_rx/tx       → cpu (NIC interrupt overhead)

disk_read/write_iops → cpu (I/O wait → CPU utilisation)
```

### 6.3 Adjacency Matrix as GNN Input

The edge list is converted to a row-normalised N×N adjacency matrix:
```python
A_prior[i, j] = strength of edge i→j  # normalised so each row sums to 1
```

This matrix:
1. Is saved to `dependency_map_fine.json` — deliverable for phase (a)
2. Passed to `AdaptiveAdjacency` as a regularisation prior for the learned graph
3. Used to compute agreement statistics post-training

---

## 7. Forecasting: CPU, Memory, Disk

### 7.1 Target Metrics (what the model forecasts)

| Target node | Dynatrace ID | Entity | Unit |
|-------------|-------------|--------|------|
| `host_cpu_usage` | `builtin:host.cpu.usage` | HOST | percent |
| `host_mem_usage` | `builtin:host.mem.usage` | HOST | percent |
| `disk_read_iops` | `builtin:host.disk.reads` | DISK | count |
| `disk_write_iops` | `builtin:host.disk.writes` | DISK | count |
| `disk_read_throughput` | `builtin:host.disk.bytesRead` | DISK | byte |
| `disk_write_throughput` | `builtin:host.disk.bytesWritten` | DISK | byte |
| `disk_busy_time` | `builtin:host.disk.activeTime` | DISK | percent |
| `jvm_memory_heap_used` | `builtin:tech.jvm.memory.runtime.used` | PGI | byte |

### 7.2 Input Features per Node

```
C channels per node:
  Channel 0: value (RobustScaled, 0.0 for missing)
  Channel 1: observation mask (1.0=observed, 0.0=missing)
  Channel 2: sin(2π×seconds/86400)   [sub-daily grids only]
  Channel 3: cos(2π×seconds/86400)   [sub-daily grids only]
  Channel 4: sin(2π×dow/7)
  Channel 5: cos(2π×dow/7)
  Channel 6: sin(2π×(dom-1)/31)
  Channel 7: cos(2π×(dom-1)/31)
```

### 7.3 Forecast Output

```
Output shape: (B, N_target, H=4, Q=3)

H = [7 days, 30 days, 60 days, 90 days]
Q = [P10 (pessimistic), P50 (median), P90 (optimistic)]
```

The P10–P90 interval is the capacity planning band. Well-calibrated = actual falls inside ~80% of days.

### 7.4 How the Graph Helps

Without graph convolutions (`stgnn_nograph`):
- Each target node attends only to its own past values
- Cannot see that `service_cpm` rose 2 weeks ago and CPU is about to respond

With graph convolutions (`stgnn_graph`):
- `service_cpm → host_cpu_usage` information flows through learned adjacency
- `jvm_memory_heap_used → host_mem_usage` information flows
- 2-hop MixHop propagation: `service_cpm → jvm → cpu` captured in one block

### 7.5 Capacity Prediction Interpretation

```
host_cpu_usage at 90-day horizon:
  P10 = 42.3%  (optimistic — CPU stays manageable)
  P50 = 61.7%  (median — plan around this)
  P90 = 79.1%  (pessimistic — risk band)

Decision rules:
  P90 at 90 days > 80% → initiate scaling now
  P50 at 30 days > 70% → plan for upgrade next quarter
```

---

## 8. End-to-End Data Flow

```
[Dynatrace API]
      |
      | backfill (5-min and daily)
      v
data/raw/grid=fine/date=*/data.parquet       (long format, per node-day)
data/raw/grid=coarse/date=*/data.parquet
      |
      | panel build (pivot + resample + mask)
      v
data/panel/fine_values.parquet     (T×N, 5min, 14 days)
data/panel/fine_mask.parquet
data/panel/coarse_values.parquet   (T×N, daily, 400 days)
data/panel/coarse_mask.parquet
      |
   ___|_______________________________________________
  |                                                   |
  | fine panel                                        | coarse panel
  v                                                   v
[graph/statistical.py]                        [models/dataset.py]
  Cross-corr scan (N×N, lags 0..24)            RobustScaler (train split only)
  Mutual information                            Sliding windows
  Granger causality                             X(S,N,T,C), Y(S,NT,H), Y_mask
  STRUCTURAL_FLOW prior                               |
      |                                               v
      | A_prior (N×N adjacency)               [models/stgnn.py + train.py]
      |_______________________________________________>|
                                                AdaptiveAdjacency (regularised)
                                                STBlock × 2 (temporal + graph)
                                                Pinball loss + graph regularisation
                                                      |
                                                      v
                                              artifacts/stgnn.pt
                                              artifacts/stgnn_learned_graph.json
                                                      |
                                              [eval/backtest.py]
                                                stgnn_graph vs stgnn_nograph
                                                vs persistence vs drift
                                                vs seasonal_naive vs climatology
                                                      |
                                                      v
                                              artifacts/backtest_coarse.json
                                              [verdict: graph earns its place / not]
```

---

## 9. Key Design Decisions and Bug Fixes

| Blocker | Problem | Fix |
|---------|---------|-----|
| **B1** | Old scripts hardcoded `from=now-10m` → only 11 rows, not a training set | `backfill.py` pulls configurable lookback (up to 400 days), chunked |
| **B2** | Failed/empty queries silently swallowed | Every metric outcome recorded in `FetchReport`; empty results raise |
| **B3** | Three names for same selector — dict dropped two of them | Registry keyed by unique `key`, duplicates declared as `aliases` |
| **B4** | Same signal appeared with different names across files | All producers use `registry.node_id()` — one name everywhere |
| **B5** | Negative memory percent (−941.98%) due to wrong selector | `transform: abs` declared in registry; `validate` reports unit violations |
| **B6** | Each run overwrote `*_last_10m.csv` — no history accumulation | Append-only, date-partitioned Parquet store with idempotent upserts |
| **B7** | API token hardcoded in scripts | Token only from environment (`DYNATRACE_API_TOKEN`); never in YAML |

---

## 10. Configuration Reference

### `configs/v1.yaml` — All Parameters

```yaml
base_url: https://<tenant>.live.dynatrace.com
registry: metrics_registry.yaml
data_dir: data
artifacts_dir: artifacts

scope:
  host_id: HOST-<ID>       # Override: NETRAA_HOST_ID env var
  service_id: SERVICE-<ID> # Override: NETRAA_SERVICE_ID env var

grids:
  fine:
    resolution: 5m           # Dynatrace resolution token
    pandas_freq: 5min        # Pandas offset alias
    lookback_days: 14        # How far back to backfill
    purpose: dependency_map
  coarse:
    resolution: 1d
    pandas_freq: 1D
    lookback_days: 400
    purpose: forecast

min_coverage: 0.20   # Drop panel nodes below this observed fraction

season_steps:
  coarse: 7    # 7 daily steps = one week
  fine: 288    # 288 × 5min = one day

forecast:
  grid: coarse
  input_steps: 90              # Days of context fed to the model
  horizons: [7, 30, 60, 90]   # Forecast horizons in days
  quantiles: [0.1, 0.5, 0.9]  # P10, P50, P90

graph:
  grid: fine
  max_lag_steps: 24      # 24 × 5min = 2 hours max lag
  xcorr_threshold: 0.30  # Min |correlation| to accept an edge
  granger_alpha: 0.05    # Significance level for Granger test
  mi_threshold: 0.05     # Min normalised MI
  top_k: 8               # Max incoming edges per node
  min_overlap: 60        # Min jointly-observed samples to score a pair

model:
  hidden: 32               # Channels per ST block
  blocks: 2                # Number of ST blocks (dilation doubles each)
  kernel_size: 3           # Temporal convolution kernel width
  dropout: 0.2
  node_embed_dim: 16       # Embedding dimension for AdaptiveAdjacency
  graph_prior_weight: 0.1  # lambda_1: ||A_learned − A_prior||_1
  graph_sparsity_weight: 0.01  # lambda_2: ||A_learned||_1
  lr: 0.001
  weight_decay: 0.0001
  epochs: 200
  batch_size: 16
  patience: 25
  seed: 42
```

---

## 11. Metrics Registry Reference

The registry (`metrics_registry.yaml`) defines 27 metrics across 5 categories.

### SERVICE (4 metrics)
| Key | Dynatrace ID | Role | Split |
|-----|-------------|------|-------|
| `service_cpm` | `builtin:service.requestCount.total` | driver | none |
| `endpoint_cpm` | `builtin:service.keyRequest.count.total` | driver | per service method |
| `instance_traffic` | `builtin:service.requestCount.server` | driver | none |
| `database_access_cpm` | `builtin:service.dbChildCallCount` | intermediate | none |

### HOST CPU (2 metrics)
| Key | Dynatrace ID | Role |
|-----|-------------|------|
| `host_cpu_usage` | `builtin:host.cpu.usage` | **target** |
| `host_cpu_load1` | `builtin:host.cpu.load` | intermediate |

### HOST MEMORY (4 metrics)
| Key | Dynatrace ID | Role | Transform |
|-----|-------------|------|-----------|
| `host_mem_usage` | `builtin:host.mem.usage` | **target** | `abs` |
| `host_mem_available` | `builtin:host.mem.available` | intermediate | `abs` |
| `host_mem_total` | `builtin:host.mem.total` | intermediate | none |
| `host_mem_buff_cache` | `builtin:host.mem.buffersAndCache` | intermediate | `abs` |

### HOST NETWORK (2 metrics, per NIC)
| Key | Dynatrace ID | Role | Transform |
|-----|-------------|------|-----------|
| `host_net_rx` | `builtin:host.net.nic.bytesRx` | intermediate | `log1p` |
| `host_net_tx` | `builtin:host.net.nic.bytesTx` | intermediate | `log1p` |

### HOST DISK (8 metrics, per DISK entity)
| Key | Dynatrace ID | Role | Transform |
|-----|-------------|------|-----------|
| `disk_read_iops` | `builtin:host.disk.reads` | **target** | none |
| `disk_write_iops` | `builtin:host.disk.writes` | **target** | none |
| `disk_read_throughput` | `builtin:host.disk.bytesRead` | **target** | `log1p` |
| `disk_write_throughput` | `builtin:host.disk.bytesWritten` | **target** | `log1p` |
| `disk_read_latency` | `builtin:host.disk.readTime` | intermediate | none |
| `disk_write_latency` | `builtin:host.disk.writeTime` | intermediate | none |
| `disk_queue_length` | `builtin:host.disk.queueLength` | intermediate | none |
| `disk_busy_time` | `builtin:host.disk.activeTime` | **target** | none |

### JVM / Process Group Instance (8 metrics)
| Key | Dynatrace ID | Role |
|-----|-------------|------|
| `jvm_thread_live_count` | `builtin:tech.jvm.threads.count` | intermediate |
| `jvm_thread_peak_count` | `builtin:tech.jvm.threads.peakCount` | intermediate |
| `jvm_process_cpu` | `builtin:tech.jvm.processCpuUsage` | intermediate |
| `jvm_memory_heap_used` | `builtin:tech.jvm.memory.runtime.used` | **target** |
| `jvm_memory_heap_max` | `builtin:tech.jvm.memory.runtime.max` | intermediate |
| `jvm_memory_pool_used` | `builtin:tech.jvm.memory.pool.used` | intermediate |
| `jvm_gc_collection_count` | `builtin:tech.jvm.memory.pool.collectionCount` | intermediate |
| `jvm_gc_collection_time` | `builtin:tech.jvm.memory.gc.collectionTime` | intermediate |

> All JVM metrics use entity type `PROCESS_GROUP_INSTANCE`, NOT `HOST`. This is the single most important distinction from the old scripts.

---

*End of Technical Documentation*
