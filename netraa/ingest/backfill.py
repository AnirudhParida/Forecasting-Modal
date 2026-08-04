"""Historical backfill and incremental collection.

Blocker B1: the old scripts hardcoded `from=now-10m`, producing 11 rows. That
is not a training set. This module pulls a configurable history, chunked so no
single request exceeds the API's point cap, and appends to the Parquet store.

Every metric's outcome is recorded and returned. A metric that yields nothing
is reported, not silently skipped (B2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..config import GridConfig
from .dynatrace_client import DynatraceClient, DynatraceError, QueryResult, chunk_range
from .registry import MetricSpec, Registry, node_id
from .store import empty_frame, last_timestamp, write_rows
from .topology import Topology

log = logging.getLogger(__name__)

TARGET_POINTS_PER_REQUEST = 5000


@dataclass
class FetchReport:
    key: str
    status: str
    rows: int = 0
    series: int = 0
    chunks: int = 0
    detail: str = ""


@dataclass
class BackfillSummary:
    grid: str
    reports: list[FetchReport] = field(default_factory=list)

    @property
    def total_rows(self) -> int:
        return sum(r.rows for r in self.reports)

    def format(self) -> str:
        hdr = f"{'metric key':<26} {'status':<12} {'rows':>9} {'series':>7}  detail"
        lines = [hdr, "-" * len(hdr)]
        for r in sorted(self.reports, key=lambda r: (r.status != "OK", r.key)):
            lines.append(
                f"{r.key:<26} {r.status:<12} {r.rows:>9} {r.series:>7}  {r.detail[:60]}"
            )
        ok = sum(r.status == "OK" for r in self.reports)
        lines += [
            "",
            f"grid={self.grid}: {ok}/{len(self.reports)} metrics returned data, "
            f"{self.total_rows:,} rows written",
        ]
        return "\n".join(lines)


def chunk_days_for(grid: GridConfig) -> int:
    return max(1, int(TARGET_POINTS_PER_REQUEST * grid.seconds / 86_400))


def series_to_rows(spec: MetricSpec, result: QueryResult) -> pd.DataFrame:
    """Flatten a query result into long-format rows with canonical node IDs."""
    records: list[dict] = []

    for series in result.series:
        dimension = ""
        if spec.split_by:
            dimension = series.dimension_map.get(spec.split_by, "")
            if not dimension and series.dimensions:
                dimension = str(series.dimensions[0])

        nid = node_id(spec.key, dimension)
        for ts, value in zip(series.timestamps, series.values):
            records.append(
                {
                    "timestamp": ts,
                    "node_id": nid,
                    "metric_key": spec.key,
                    "dimension": dimension,
                    "entity_type": spec.entity_type,
                    "value": value,
                }
            )

    if not records:
        return empty_frame()

    df = pd.DataFrame.from_records(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def backfill(
    client: DynatraceClient,
    registry: Registry,
    topology: Topology,
    grid: GridConfig,
    raw_dir: Path,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    only_keys: list[str] | None = None,
) -> BackfillSummary:
    end = end or pd.Timestamp.now(tz="UTC").floor(grid.pandas_freq)
    start = start or (end - pd.Timedelta(days=grid.lookback_days))

    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    chunk_days = chunk_days_for(grid)

    summary = BackfillSummary(grid=grid.name)
    specs = registry.enabled
    if only_keys:
        specs = [s for s in specs if s.key in only_keys]

    for spec in specs:
        entity_ids = topology.ids_for(spec.entity_type)
        if not entity_ids:
            summary.reports.append(
                FetchReport(
                    key=spec.key,
                    status="UNRESOLVED",
                    detail=f"no {spec.entity_type} entities in topology",
                )
            )
            continue

        selector = spec.build_selector(entity_ids)
        total_rows = 0
        series_seen: set[str] = set()
        chunks = 0
        errors: list[str] = []

        for c_start, c_end in chunk_range(start_ms, end_ms, chunk_days):
            chunks += 1
            try:
                result = client.query(selector, c_start, c_end, grid.resolution)
            except DynatraceError as exc:
                errors.append(str(exc).splitlines()[0][:100])
                continue

            rows = series_to_rows(spec, result)
            if rows.empty:
                continue
            series_seen.update(rows["node_id"].unique())
            total_rows += write_rows(rows, raw_dir, grid.name)

        if errors:
            status = "PARTIAL" if total_rows else "ERROR"
            detail = errors[0]
        elif total_rows == 0:
            status, detail = "NO_DATA", "query succeeded, zero points returned"
        else:
            status, detail = "OK", ""

        summary.reports.append(
            FetchReport(
                key=spec.key,
                status=status,
                rows=total_rows,
                series=len(series_seen),
                chunks=chunks,
                detail=detail,
            )
        )
        log.info("%s: %s (%d rows, %d series)", spec.key, status, total_rows, len(series_seen))

    return summary


def collect_incremental(
    client: DynatraceClient,
    registry: Registry,
    topology: Topology,
    grid: GridConfig,
    raw_dir: Path,
    overlap_steps: int = 2,
) -> BackfillSummary:
    """Fetch only what is newer than the store, with a small overlap.

    The overlap re-fetches the most recent points because Dynatrace can revise
    the newest buckets after they are first served. Writes are idempotent, so
    the overlap corrects those values rather than duplicating them.
    """
    last = last_timestamp(raw_dir, grid.name)
    end = pd.Timestamp.now(tz="UTC").floor(grid.pandas_freq)

    if last is None:
        log.info("empty store for grid=%s — running a full backfill", grid.name)
        return backfill(client, registry, topology, grid, raw_dir)

    start = last - pd.Timedelta(seconds=grid.seconds * overlap_steps)
    if start >= end:
        return BackfillSummary(grid=grid.name)

    return backfill(client, registry, topology, grid, raw_dir, start=start, end=end)
