"""Append-only Parquet store.

Blocker B6: every run of the old scripts wrote to a fixed filename
(`*_last_10m.csv`), so each run destroyed the previous one and history could
never accumulate. Data is now written in long format, partitioned by date, and
writes are idempotent — re-running a backfill over a window already collected
updates those rows instead of duplicating or clobbering them.

Long schema (one row per observation):
    timestamp    datetime64[ns, UTC]
    node_id      canonical node name (registry.node_id)
    metric_key   canonical metric key
    dimension    split dimension value ("" when unsplit)
    entity_type  HOST | SERVICE | PROCESS_GROUP_INSTANCE | SERVICE_METHOD
    value        float64
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

COLUMNS = ["timestamp", "node_id", "metric_key", "dimension", "entity_type", "value"]
KEY = ["timestamp", "node_id"]


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.Series([], dtype="datetime64[ns, UTC]"),
            "node_id": pd.Series([], dtype="object"),
            "metric_key": pd.Series([], dtype="object"),
            "dimension": pd.Series([], dtype="object"),
            "entity_type": pd.Series([], dtype="object"),
            "value": pd.Series([], dtype="float64"),
        }
    )


def _partition_path(raw_dir: Path, grid: str, day: pd.Timestamp) -> Path:
    return raw_dir / f"grid={grid}" / f"date={day:%Y-%m-%d}" / "data.parquet"


def write_rows(df: pd.DataFrame, raw_dir: Path, grid: str) -> int:
    """Merge rows into the store. Returns the number of new-or-updated rows."""
    if df.empty:
        return 0

    df = df.loc[:, COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    written = 0
    for day, chunk in df.groupby(df["timestamp"].dt.floor("D")):
        path = _partition_path(raw_dir, grid, day)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            existing = pd.read_parquet(path)
            existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
            combined = pd.concat([existing, chunk], ignore_index=True)
        else:
            combined = chunk

        before = len(combined)
        combined = (
            combined.drop_duplicates(subset=KEY, keep="last")
            .sort_values(KEY)
            .reset_index(drop=True)
        )
        combined.to_parquet(path, index=False, compression="snappy")
        written += len(combined) - (before - len(chunk))

    return written


def read_grid(
    raw_dir: Path,
    grid: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Read the whole store for one grid, optionally time-bounded."""
    root = raw_dir / f"grid={grid}"
    if not root.exists():
        return empty_frame()

    parts = sorted(root.glob("date=*/data.parquet"))
    if not parts:
        return empty_frame()

    frames = []
    for path in parts:
        day = pd.Timestamp(path.parent.name.split("=", 1)[1], tz="UTC")
        if start is not None and day < start.floor("D"):
            continue
        if end is not None and day > end.ceil("D"):
            continue
        frames.append(pd.read_parquet(path))

    if not frames:
        return empty_frame()

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    if start is not None:
        df = df[df["timestamp"] >= start]
    if end is not None:
        df = df[df["timestamp"] <= end]

    return df.sort_values(KEY).reset_index(drop=True)


def last_timestamp(raw_dir: Path, grid: str) -> pd.Timestamp | None:
    """Latest stored timestamp for a grid — the resume point for `collect`."""
    root = raw_dir / f"grid={grid}"
    if not root.exists():
        return None
    parts = sorted(root.glob("date=*/data.parquet"))
    if not parts:
        return None
    tail = pd.read_parquet(parts[-1], columns=["timestamp"])
    if tail.empty:
        return None
    return pd.to_datetime(tail["timestamp"], utc=True).max()


def coverage(raw_dir: Path, grid: str) -> pd.DataFrame:
    """Per-node coverage summary — how much history each node actually has."""
    df = read_grid(raw_dir, grid)
    if df.empty:
        return pd.DataFrame(
            columns=["node_id", "points", "first", "last", "null_fraction"]
        )

    grouped = df.groupby("node_id").agg(
        points=("value", "size"),
        non_null=("value", "count"),
        first=("timestamp", "min"),
        last=("timestamp", "max"),
    )
    grouped["null_fraction"] = 1.0 - grouped["non_null"] / grouped["points"]
    return (
        grouped.drop(columns=["non_null"])
        .reset_index()
        .sort_values("points", ascending=False)
    )
