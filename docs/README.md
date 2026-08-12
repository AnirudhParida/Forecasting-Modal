# Netraa — Metric Dependency Discovery + Capacity Forecasting

Two deliverables on one data foundation:

**(a) Dependency map** — how each metric depends on the others, with strength,
direction and lag. Statistical, explainable, needs no training.

**(b) Quarter-ahead forecaster** — an ST-GNN whose adjacency matrix is learned
jointly with the forecast objective and regularised toward (a).

One training run produces both. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design.

---

## Setup

```bash
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -r requirements.txt

cp .env.example .env      # then fill in DYNATRACE_API_TOKEN
```

> The Dynatrace token was previously hardcoded in `fetch_*.py`. It is exposed in
> plaintext in this workspace — **rotate it in Dynatrace before going further.**

Verify the install without touching the API:

```bash
.venv/bin/python tests/test_pipeline.py       # 13 correctness tests
.venv/bin/python -m netraa.cli smoke          # full pipeline on the legacy CSVs
```

---

## Runbook

Run in order. Each step assumes the previous one succeeded.

| # | Command | What it does |
|---|---------|--------------|
| 1 | `netraa probe` | Measures what your tenant actually retains at each resolution. Writes a recommendation for `grids:` in `configs/v1.yaml`. |
| 2 | `netraa topology` | Resolves the host and service into the process-group-instance, disk and service-method IDs the metrics are really dimensioned by. |
| 3 | `netraa validate` | Checks every registry metric: does it exist, is the filter dimension right, does it return data, do the values match the declared unit. **Exits non-zero on failure — do not skip.** |
| 4 | `netraa backfill --grid coarse` | Pulls history into the append-only Parquet store. This is the long pole. |
| 5 | `netraa panel --grid coarse` | Builds the (T × N) panel plus the observation mask. |
| 6 | `netraa graph` | **(a)** The dependency map → `data/graph/dependency_map_*.json`. |
| 7 | `netraa backtest` | **(b)** Trains the ST-GNN + graph ablation, scores both against four classical baselines. |

Invoke as `.venv/bin/python -m netraa.cli <command>`.

Afterwards, `netraa backfill --grid coarse --incremental` keeps the store current
(idempotent — safe to run on a cron).

Useful anytime: `netraa status` shows per-node coverage and how much history you
actually have.

---

## Reading the output

**`netraa validate`** — every metric lands in exactly one bucket. `OK`,
`MISSING_METRIC`, `DIMENSION_MISMATCH`, `UNRESOLVED`, `NO_DATA`, `ALL_NULL`,
`UNIT_VIOLATION`. The original scripts requested 15 metrics, received 5, and
reported nothing about the other 10; this is what replaces that silence.

**`netraa graph`** — one row per edge:

```
source                target                strength  dir   lag     MI  granger p  evidence
endpoint_cpm          database_access_cpm      0.841    +    1m  0.312     0.0021  xcorr+mi+granger
```

`evidence` lists which tests fired. An edge tagged only `xcorr+contemporaneous`
is a correlation at zero lag with no temporal precedence — weaker than one
carrying `granger`.

**`netraa backtest`** — the line that matters is the ablation verdict:

```
The graph earns its place: 8.3% lower MAE than the identical graph-free model.
```

`stgnn_graph` and `stgnn_nograph` are the same architecture, data and seed,
differing only in whether graph convolutions run. If the graph version doesn't
win, the dependency structure isn't helping the forecaster and the report says
so rather than burying it.

The four baselines — persistence, seasonal naive, drift, climatology — are there
to be beaten. At a 90-day horizon a trend line is genuinely competitive; losing
to `drift` is a real result, not a bug.

---

## Layout

```
netraa/
├── ingest/          Dynatrace client, registry, topology, validation, backfill
├── features/        panel construction, transforms, windowing
├── graph/
│   ├── statistical.py   (a) xcorr / MI / Granger  → dependency_map.json
│   └── learned.py       (b) adaptive adjacency, regularised toward (a)
├── models/          ST-GNN, baselines, dataset prep, training
└── eval/            metrics, backtest with graph ablation

metrics_registry.yaml    single source of truth for every metric
configs/v1.yaml          grids, horizons, thresholds, hyperparameters
tests/test_pipeline.py   13 correctness tests
```

`fetch_cpu_influencers.py`, `fetch_disk_io_influencers.py` and
`fetch_memory_influencers.py` are superseded and kept only for reference; each
carries a header documenting the specific defect it exhibited. The three CSVs
they produced are retained as smoke-test fixtures.

---

## Configuration notes

**Two grids, deliberately.** A quarter-ahead forecast on a 1-minute grid would be
129,600 steps ahead. So the `fine` grid (5 min) serves dependency discovery,
where minute-scale lags live, and the `coarse` grid (daily) serves the
forecaster. Run `netraa probe` and set both from what your tenant actually keeps.

**Data volume.** At the daily grid, 400 days of history yields ~220 training
windows at `input_steps=90, horizon=90`. That is a small training set, and the
classical baselines exist precisely so you can tell whether the neural model is
adding anything over a trend line at that sample size.

**`host_mem_usage` uses `transform: abs`.** This is an approved interim measure
for the negative percentages in the original data, not a diagnosis.
`netraa validate` still reports the raw observed range so the underlying problem
stays visible.
