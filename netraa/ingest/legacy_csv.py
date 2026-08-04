"""Import the original cpu/disk/memory CSVs into the store.

Kept deliberately (approved) so the pipeline has something to smoke-test
against without hitting the API. Importing them also demonstrates the B4 fix:
the three files use three different naming conventions for the same signals —

    cpu file     meter_vm_network_receive
    memory file  meter_vm_network_receive_HOST-D9739223FC540A23

— and both collapse onto the single canonical node `host_net_rx` here.

These files hold 11 rows each. They are enough to exercise the code paths and
nothing else; do not read model results off them.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from .registry import Registry, node_id
from .store import empty_frame

log = logging.getLogger(__name__)

LEGACY_FILES = [
    "cpu_influencers_last_10m.csv",
    "disk_io_influencers_last_10m.csv",
    "memory_influencers_last_10m.csv",
]

# The old scripts appended series['dimensions'][0] to the column name. For these
# metrics that first dimension was the HOST or SERVICE entity itself, not a real
# split — which is why the same signal ended up with two different names across
# the three files.
ENTITY_SUFFIX = re.compile(
    r"_(HOST|SERVICE|SERVICE_METHOD|PROCESS_GROUP_INSTANCE|DISK)-[0-9A-F]+$"
)

# Suffixes naming the scope entity carry no information when the scope is one
# host and one service, so they are discarded and the columns collapse onto a
# single node. A DISK or PGI suffix is a genuine split and is kept.
SCOPE_ENTITY_TYPES = {"HOST", "SERVICE"}


def parse_legacy_column(column: str, alias_map: dict[str, str]) -> tuple[str, str] | None:
    """Legacy CSV header -> (canonical metric key, dimension)."""
    dimension = ""
    match = ENTITY_SUFFIX.search(column)
    base = column
    if match:
        entity_type = match.group(1)
        base = column[: match.start()]
        if entity_type not in SCOPE_ENTITY_TYPES:
            dimension = match.group(0).lstrip("_")

    canonical = alias_map.get(base)
    if canonical is None:
        return None
    return canonical, dimension


def load_legacy_csvs(
    project_root: Path, registry: Registry, files: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Returns (long-format rows, list of unmapped columns)."""
    alias_map = registry.alias_map()
    frames: list[pd.DataFrame] = []
    unmapped: list[str] = []

    for name in files or LEGACY_FILES:
        path = project_root / name
        if not path.exists():
            log.warning("legacy CSV not found: %s", path)
            continue

        df = pd.read_csv(path)
        if "timestamp" not in df.columns:
            log.warning("%s has no timestamp column, skipping", name)
            continue

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        for column in df.columns:
            if column == "timestamp":
                continue
            parsed = parse_legacy_column(column, alias_map)
            if parsed is None:
                unmapped.append(f"{name}:{column}")
                continue

            metric_key, dimension = parsed
            spec = registry.by_key(metric_key)
            sub = pd.DataFrame(
                {
                    "timestamp": df["timestamp"],
                    "node_id": node_id(metric_key, dimension),
                    "metric_key": metric_key,
                    "dimension": dimension,
                    "entity_type": spec.entity_type,
                    "value": pd.to_numeric(df[column], errors="coerce"),
                }
            )
            frames.append(sub.dropna(subset=["value"]))

    if not frames:
        return empty_frame(), unmapped

    out = pd.concat(frames, ignore_index=True)
    # The same node can appear in more than one legacy file at the same
    # timestamp; average rather than letting one file win arbitrarily.
    out = (
        out.groupby(["timestamp", "node_id", "metric_key", "dimension", "entity_type"], as_index=False)["value"]
        .mean()
    )
    return out, sorted(set(unmapped))
