# Netraa — Metric Dependency Graph + Temporal Forecasting

**Status:** approved and implemented. See [README.md](README.md) for the runbook.

**Goal:** (1) learn how metrics depend on each other (e.g. how `endpoint_cpm` drives
`database_access_cpm`), (2) use that dependency structure to forecast CPU, Disk and
Memory utilisation.

**Approved decisions**

| Question | Decision | Consequence |
|----------|----------|-------------|
| Retention | Whatever suits the POC | `netraa probe` measures it empirically instead of assuming a policy |
| Scope | One service on one host | Entity IDs resolved by topology walk; ~60–100 nodes; dense adjacency, no PyG |
| Horizon | Next quarter | **Forces the dual-grid design** — see §1 |
| Negative memory | Treat as positive for now | `transform: abs` on `host_mem_usage`, declared in the registry and still reported by `validate` |
| Legacy CSVs | Keep for smoke tests | `netraa smoke` runs the whole chain offline against them |

---

## 0. What the original data looked like — and where each blocker was fixed

Findings from `cpu_influencers_last_10m.csv`, `disk_io_influencers_last_10m.csv` and
`memory_influencers_last_10m.csv`. All seven are now closed.

| # | Issue | Evidence | Fixed by |
|---|-------|----------|----------|
| B1 | **11 rows of history per file** | `from=now-10m`, `resolution=1m` | `ingest/backfill.py` — configurable history, chunked so no request exceeds the API point cap |
| B2 | **Metrics silently dropped** | CPU script requests 15 metrics, CSV has 5 columns. All JVM metrics, `cpu.load1/5/15` and `endpoint_cpm` absent. | `ingest/validate.py` — every metric lands in a named status bucket; `metrics_registry.yaml` — JVM metrics re-pointed at `dt.entity.process_group_instance`, which is the dimension they actually carry |
| B3 | **Dict-key collision drops metrics** | `metric_queries[query] = custom_name` keyed by the *query string*; `service_cpm` / `service_instance_cpm` / `transaction_volume` all resolve to `builtin:service.requestCount.total`, so only the last survived | `ingest/registry.py` — keyed by unique `key`, genuine duplicates declared as `aliases` |
| B4 | **Inconsistent column naming** | `meter_vm_network_receive` (CPU file) vs `meter_vm_network_receive_HOST-D97…` (memory file) | `registry.node_id()` — one canonical name, used by every producer |
| B5 | **Negative memory values** | `meter_vm_memory_used = -941.98` from a percentage metric | Explicit `:splitBy():agg` in every selector; `transform: abs` as the approved interim measure; `validate` reports the observed range so the root cause stays visible |
| B6 | **Every run overwrites the CSV** | fixed filename `*_last_10m.csv` | `ingest/store.py` — append-only Parquet partitioned by date, idempotent on `(timestamp, node_id)` |
| B7 | **API token hardcoded in 3 files** | plaintext `dt0c01…` | Read from `DYNATRACE_API_TOKEN`; the three legacy scripts now refuse to run without it. **The original token is still compromised — rotate it.** |

**Root cause behind B2/B5:** the selectors carry no explicit aggregation or split.
`builtin:host.disk.reads:filter(...)` with no `:splitBy("dt.entity.disk")` and no
`:avg`/`:sum` lets Dynatrace auto-merge dimensions, which returns empty or nonsensical
series for multi-dimensional metrics. Correct form:

```
builtin:host.disk.reads:filter(eq("dt.entity.host","HOST-…")):splitBy("dt.entity.disk"):avg
```

---

## 1. Two grids — a consequence of the quarter-ahead horizon

A 90-day forecast on a 1-minute grid is 129,600 steps ahead. That is not a forecast, so
the horizon decision splits the pipeline into two sampling grids with different jobs:

| Grid | Resolution | Window | Serves |
|------|-----------|--------|--------|
| `fine` | 5 min | ~14 days | **Dependency map** — minute-scale lags (`endpoint_cpm → database_access_cpm` at 1 min) only exist at this resolution |
| `coarse` | 1 day | ~400 days | **Quarter-ahead forecaster** — 90 daily steps out |

Both are set from measurement, not assumption: `netraa probe` sweeps
(resolution × lookback) against a reference metric and reports where the server starts
serving coarser buckets than requested. Its recommendation goes straight into
`configs/v1.yaml`.

**Sample-size reality.** At the daily grid, 400 days with `input_steps=90` and a 90-day
horizon yields ~220 windows, split ~154 / 33 / 34. That is a small training set for a
neural model, and it is the reason the backtest ships with four classical baselines
rather than treating them as a formality — at this horizon and this sample size, a trend
line is a serious competitor. Losing to `drift` is a finding to report, not a bug to hide.

The backfill is the long pole: it is wall-clock-bound, not effort-bound. Start it first.

---

## 2. Target architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ 1. INGESTION                                                     │
│    dynatrace_client.py  +  metrics_registry.yaml                 │
│    · one client, one metric registry (union of the 3 scripts)    │
│    · backfill mode (chunked from/to) + incremental mode          │
│    · append-only Parquet, partitioned by date                    │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ 2. PANEL BUILDER                                                 │
│    long → wide matrix X ∈ R^{T×N}, one column per NODE           │
│    node_id = {entity_type}:{entity_id}:{metric_key}:{dimension}  │
│    + missingness mask M ∈ {0,1}^{T×N}                            │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ 3. FEATURE LAYER                                                 │
│    grid alignment · gap policy · outlier clip · robust scaling    │
│    counter differencing · calendar encodings                     │
└──────────┬─────────────────────────────────┬─────────────────────┘
           ▼                                 ▼
┌────────────────────────┐      ┌────────────────────────────────┐
│ 4a. STATISTICAL GRAPH  │      │ 4b. LEARNED GRAPH              │
│ (no training required) │─────▶│ adaptive adjacency, trained    │
│ lagged xcorr · MI ·    │ prior│ jointly with the forecaster    │
│ Granger → A_prior      │      │ → A_learned                    │
└───────────┬────────────┘      └───────────────┬────────────────┘
            │                                   │
            │        ┌──────────────────────────┘
            ▼        ▼
┌──────────────────────────────────────────────────────────────────┐
│ 5. SPATIO-TEMPORAL FORECASTER (MTGNN / Graph-WaveNet family)     │
│    dilated TCN ⇄ graph convolution, stacked                      │
│    multi-horizon quantile output (P10/P50/P90)                   │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ 6. EVALUATION · EXPLAINABILITY · SERVING                         │
│    rolling-origin backtest vs baselines · ablation without graph │
│    dependency map export (source,target,strength,direction,lag)  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. The node definition (the key modelling decision)

A node is **not** a metric name. It is an `(entity, metric, dimension)` triple:

```
HOST:HOST-D97…:host.cpu.usage:—
HOST:HOST-D97…:host.disk.reads:/dev/sda
HOST:HOST-D97…:host.disk.reads:/dev/sdb
SERVICE:SERVICE-C16…:service.dbChildCallCount:—
PROCESS:PROCESS-…:tech.jvm.memory.pool.used:G1 Old Gen
```

Deduplicating the three scripts gives ~25 unique selectors; after dimension splitting
(per-disk, per-NIC, per-GC-pool, per-memory-pool) this expands to roughly **60–100 nodes**.

That N is small, and it drives two engineering simplifications:

- Dense `N×N` adjacency is cheap — **no PyTorch Geometric, no neighbour sampling.**
  Plain PyTorch with dense matmuls. One less fragile dependency.
- The whole graph fits in one batch. Full-batch training throughout.

---

## 4. On "train a GNN to get dependencies, then train a temporal model"

Worth being direct about this, because it changes the design.

A GNN trained in isolation has no supervision signal — there is no label for "is there an
edge from `endpoint_cpm` to `database_access_cpm`". Feeding it metrics with no objective
gives it nothing to learn from. The two formulations that actually work:

**(a) Statistical / causal discovery — unsupervised, no neural net.**
Lagged cross-correlation, mutual information and Granger causality over the panel.
Produces exactly the deliverable you described: an edge list with strength, direction and
lag. Runs in minutes, is fully explainable, and is available as soon as the backfill lands.

**(b) Learned adjacency, trained jointly with the forecaster.**
The graph becomes a model parameter, supervised by forecast error — an edge exists
because it demonstrably improves prediction.

```
E₁, E₂ ∈ R^{N×d}                              # learned node embeddings
A = ReLU(tanh(α · (E₁E₂ᵀ − E₂E₁ᵀ)))           # asymmetric → directed
A = top_k(A, k=8)                             # sparsify
```

The antisymmetric term makes the graph directed, so `endpoint_cpm → database_access_cpm`
is distinguishable from the reverse.

**Recommendation: build both.** (a) is the explainable dependency map for the dashboard
and is your day-1 artefact. (b) is what the forecaster actually consumes, regularised
toward (a) with a `λ‖A_learned − A_prior‖₁` penalty so the learned graph stays
physically plausible instead of drifting into spurious correlations.

This is one training run, not two sequential ones — but the dependency map is still a
first-class, separately inspectable output.

---

## 5. Dependency discovery detail (component 4a)

For every ordered node pair `(i, j)` and lag `ℓ ∈ [0, L]` with `L = 30` min:

| Signal | Captures | Output |
|--------|----------|--------|
| Lagged cross-correlation | linear coupling + delay | `strength`, `direction` (sign), `lag = argmax` |
| Mutual information | non-linear coupling | `mi_score` |
| Granger causality (F-test) | temporal precedence | `p_value` |
| Structural prior | topology (service runs on host, JVM in process on host) | hard edge |

An edge is accepted when `|xcorr| ≥ τ₁` **and** (`granger_p ≤ 0.05` **or** structural
prior). Requiring both predictive relevance and temporal precedence is what keeps
"these two rise together at 9am" out of the graph.

Output: `graph/dependency_map.json` — edge list ready for a dashboard graph view.

---

## 6. Forecaster detail (component 5)

```
Input   X ∈ R^{B × N × T_in × C}
        C = [value, mask, tod_sin, tod_cos, dow_sin, dow_cos]
        T_in = 60 steps lookback

Block × 3:
        dilated causal conv (dilations 1,2,4,8)   ← temporal
        mix-hop graph propagation over A          ← cross-metric
        residual + skip connection

Output  Ŷ ∈ R^{B × N_target × H × Q}
        H = [1, 3, 6, 12] steps ahead
        Q = [0.1, 0.5, 0.9]
```

- **Loss:** masked pinball (quantile) loss — masked so gaps do not contribute gradient,
  quantile so you get prediction intervals rather than a bare point estimate.
- **Targets:** host CPU usage %, host memory used, disk read/write IOPS, disk
  read/write throughput, disk busy time, JVM heap used.
- **Why one model for all targets:** CPU, memory and disk share drivers (request volume,
  thread count, network I/O). Separate models per resource would relearn the same
  structure three times and lose the cross-resource coupling.

---

## 7. Evaluation — what must be true before this ships

Rolling-origin backtest, chronological split (70 / 15 / 15), **never shuffled**.

Baselines the model must beat, in order:

1. **Persistence** (`ŷ_{t+h} = y_t`) — the bar most forecasting models quietly fail.
2. **Seasonal naive** — the value one season back (weekly on the daily grid).
3. **Drift** — Theil–Sen slope extrapolated, so a couple of spikes cannot set the trend.
4. **Climatology** — the training-period median.
5. **Ablation: the same model with every graph convolution disabled.**

Item 5 is the one that matters, and it is why `use_graph` is a constructor flag rather
than a separate model: `stgnn_graph` and `stgnn_nograph` share architecture, data and
seed and differ in nothing else. `BacktestResult.verdict()` states the outcome in words —
if the graph does not win, it says so and recommends the simpler model.

Reported per horizon: MAE, RMSE, sMAPE, pinball loss, and empirical P10–P90 coverage.
Coverage is the calibration check: a well-behaved P10–P90 band contains ~80% of actuals.
Much tighter and the intervals cannot be planned against; much wider and they are useless.

**Leakage control.** The scaler is fitted only on timesteps strictly before the first
validation window, splits are contiguous and ordered, and nothing is shuffled. Fitting
the scaler on the full panel is the easiest way to make these numbers look better than
the model is.

---

## 8. Layout as built

```
netraa/
├── cli.py                        # probe / topology / validate / backfill /
│                                 #   status / panel / graph / backtest / smoke
├── config.py                     # YAML + env; secrets never in YAML
├── ingest/
│   ├── dynatrace_client.py       # retry, Retry-After, pagination, typed errors
│   ├── registry.py               # MetricSpec, selector builder, node_id  [B3, B4]
│   ├── topology.py               # host/service → PGI, disk, service-method  [B2]
│   ├── validate.py               # per-metric status buckets  [B2, B5]
│   ├── probe.py                  # measured retention → grid recommendation
│   ├── store.py                  # append-only Parquet, idempotent  [B6]
│   ├── backfill.py               # chunked history + incremental  [B1]
│   └── legacy_csv.py             # old CSVs → canonical nodes (smoke fixtures)
├── features/
│   ├── panel.py                  # long → (T×N) values + mask
│   └── transforms.py             # abs/log1p/diff, robust scaling, calendar, windows
├── graph/
│   ├── statistical.py            # (a) xcorr / MI / Granger → dependency_map.json
│   └── learned.py                # (b) adaptive adjacency, regularised toward (a)
├── models/
│   ├── stgnn.py                  # gated dilated TCN ⇄ mix-hop graph conv
│   ├── baselines.py              # persistence / seasonal / drift / climatology
│   ├── dataset.py                # leak-free windowing and splits
│   └── train.py                  # training loop, early stopping, artefacts
└── eval/
    ├── metrics.py                # masked MAE / RMSE / sMAPE / pinball / coverage
    └── backtest.py               # ablation + baselines + verdict

metrics_registry.yaml             # single source of truth
configs/v1.yaml                   # grids, horizons, thresholds, hyperparameters
tests/test_pipeline.py            # 13 correctness tests
data/  artifacts/                 # generated; gitignored
```

Stack: PyTorch (CPU), pandas + pyarrow, scikit-learn, statsmodels (Granger), PyYAML.
**No PyTorch Geometric** — at 60–100 nodes the adjacency is a dense matrix and mix-hop
propagation is a single `einsum`, so PyG would add a fragile dependency for nothing.

---

## 9. Phasing

| Phase | Work | Depends on | Output |
|-------|------|------------|--------|
| **0. Data foundation** | Fix B1–B7. Unified client, metric registry, explicit `splitBy`/aggregation, Parquet store, env-var token. Kick off backfill. | — | Trustworthy collector; backfill running |
| **1. Panel + statistical graph** | Panel builder, feature transforms, xcorr/MI/Granger. | ≥30 days landed | `panel.parquet`, `dependency_map.json` — **the dependency deliverable, no NN required** |
| **2. Forecaster** | ST-GNN with adaptive adjacency, quantile loss, training loop. | Phase 1 | Trained model + learned graph |
| **3. Evaluation** | Backtest vs all 5 baselines, ablation, calibration. | Phase 2 | Go / no-go evidence |
| **4. Serving** | Inference entrypoint, dependency map + forecast export. | Phase 3 | Dashboard-ready artefacts |

Phase 0 is blocking and its backfill is wall-clock-bound, not effort-bound. Phase 1
delivers the dependency map on its own — that answers "how does `endpoint_cpm` depend on
`database_access_cpm`" without waiting for Phase 2.

**All code for phases 0–3 is written and tested.** What remains is wall-clock: run
`probe`, `topology` and `validate` against the live tenant, then let the backfill
accumulate history. Phase 4 (serving) is not built.

---

## 10. Verification status

`tests/test_pipeline.py` — 13 tests, all passing. The ones that matter:

| Test | Proves |
|------|--------|
| `test_dependency_discovery_recovers_known_structure` | On synthetic data with planted dependencies, discovery recovers the correct **source, sign and lag** (lag 3 and lag 5 as constructed) and attaches no edge to a pure-noise node |
| `test_lagged_correlation_is_directional` | `A[i,j]` means *i leads j* — the direction is not accidentally reversed |
| `test_pairwise_correlation_handles_gaps` | Masked positions contribute nothing; a 100-step gap does not destroy a real signal |
| `test_window_alignment` | `Y[i,:,j]` really is the value at `start + input_steps + horizon − 1` — an off-by-one here silently makes every score wrong |
| `test_masked_quantile_loss_ignores_gaps` | Poisoning masked entries with `1e6` does not move the loss |
| `test_store_write_is_idempotent` | Re-running a backfill neither duplicates nor clobbers; revisions overwrite in place |
| `test_registry_has_no_selector_collision` | B3 cannot recur, and no `builtin:tech.jvm.*` metric is filtered on the host dimension |

`netraa smoke` additionally runs all six stages against the legacy CSVs offline. It
confirms the canonicalisation: `service_instance_cpm`, `service_cpm_SERVICE-…` and
`transaction_volume_SERVICE-…` collapse onto one `service_cpm` node (17 observations from
three legacy columns), and `meter_vm_network_receive` merges with
`meter_vm_network_receive_HOST-…` — B3 and B4 demonstrated on the real files.

---

## 11. Remaining open questions

1. **`host_mem_usage` root cause.** `transform: abs` is an approved interim measure, not a
   diagnosis. A percentage metric returning −941.98 means the selector or the tenant's
   metric definition is wrong; `netraa validate` will report the observed range against
   the descriptor unit once it runs against the live tenant.
2. **Does the graph earn its place?** Unanswerable until real history exists. The ablation
   is built and will answer it; the honest possible outcome is that at a 90-day horizon
   with ~220 windows, `drift` wins and the ST-GNN is not justified for problem 2.
3. **Serving (phase 4).** Not built. Needs an inference entrypoint and an export format
   agreed with whoever builds the dashboard.
