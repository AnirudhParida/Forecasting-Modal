"""Ingest UPI metrics (CSV/XLSX format) for one or more UPI hosts into the Parquet store.

Parses units (like %, B, kB, MB, GB, kiB, MiB, GiB, k, m, ms, µs),
parses comparison-prefixed values (e.g. '< 0.001'),
parses timestamps, and builds a long-format frame.

Supports multi-host ingestion: when a list of host IDs is supplied each host's
column is written as its own node (metric_key__host_id) so a single model can
cover all hosts sharing the same metric schema.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import pandas as pd

from .registry import Registry, node_id
from .store import empty_frame, write_rows

log = logging.getLogger(__name__)


def parse_val(v, unit_hint: str = "") -> float:
    """Parse a raw metric value (possibly with unit suffix) into a plain float.

    Supported unit suffixes (case-insensitive):
      Byte multiples : B, kB, MB, GB, kiB, MiB, GiB  → values in bytes
      SI multiples   : k (×1 000), m (×1 000 000)
      Time           : ms → milliseconds (stored as-is), µs/us → converted to ms
      Percent        : % → stripped, value kept as-is
    """
    if pd.isna(v):
        return float("nan")
    s = str(v).replace(",", "").strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return float("nan")

    # Strip leading comparison operators: "< 0.001", "> 100" etc.
    # Keep the numeric value as-is (the < means display truncation, not a real bound)
    if s.startswith(("<", ">")):
        s = s[1:].strip()

    # Strip percent sign – value already represents 0-100 scale
    if s.endswith("%"):
        try:
            return float(s[:-1].strip())
        except ValueError:
            return float("nan")

    # Match numeric part + optional unit suffix
    # Accept µ (U+00B5) and Œº (mis-encoded µ) as microsecond prefix
    match = re.match(
        r"^([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*(.*?)$", s
    )
    if not match:
        try:
            return float(s)
        except ValueError:
            return float("nan")

    num = float(match.group(1))
    raw_unit = match.group(2).strip()

    # Normalise unit: collapse mis-encoded µ characters (Œº → µ)
    norm = raw_unit.replace("Œº", "µ").replace("\u00b5", "µ").lower()

    # ── byte multiples ────────────────────────────────────────────────────────
    if norm in ("b", "byte", "bytes"):
        pass  # already in bytes
    elif norm in ("kb", "kib"):
        num *= 1024.0
    elif norm in ("mb", "mib"):
        num *= 1024.0 ** 2
    elif norm in ("gb", "gib"):
        num *= 1024.0 ** 3
    # ── SI multiples ─────────────────────────────────────────────────────────
    elif norm == "k":
        num *= 1_000.0
    elif norm == "m":
        num *= 1_000_000.0
    # ── time ─────────────────────────────────────────────────────────────────
    elif norm == "ms":
        pass  # keep in milliseconds
    elif norm in ("µs", "us"):
        num /= 1_000.0   # convert microseconds → milliseconds
    # ── percent (already handled above, but catch trailing "%" here) ─────────
    elif norm == "%":
        pass
    # unknown / empty → keep raw numeric value
    return num


def find_file_for_spec(spec, files: list[str]) -> str | None:
    candidates = [spec.key] + spec.aliases
    for cand in candidates:
        cand_lower = cand.lower().strip()
        for f in files:
            # Skip duplicate files marked with (1), (2), (3) … suffixes
            if re.search(r"\(\d+\)\.xlsx?$", f, re.IGNORECASE):
                continue
            f_lower = f.lower()
            if f_lower.startswith(cand_lower) or cand_lower in f_lower:
                return f
    return None


def _host_node_id(metric_key: str, host_slug: str) -> str:
    """Build a per-host node ID using the canonical pipe separator.

    Example: host_cpu_usage|HYDUPINTAPP16
    This is consistent with split_node_id() which partitions on '|'.
    """
    from .registry import node_id
    return node_id(metric_key, host_slug)


def ingest_upi_directory(
    data_dir: Path,
    registry: Registry,
    raw_dir: Path,
    host: "str | list[str]",
    grid: str = "coarse"
) -> int:
    """Ingest CSV/XLSX files from data_dir for one or multiple hosts.

    Parameters
    ----------
    host : str | list[str]
        A single host identifier string (e.g. "hydupiapp001") OR a list of
        host identifier strings when training a shared multi-host model
        (e.g. ["10.50.98.26", "10.78.33.83"]).  Each host's column is written
        to a separate node whose node_id is  ``<metric_key>__<host_slug>``  so
        that all hosts can coexist in the same panel without collision.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Source directory {data_dir} does not exist.")

    # Normalise to a list
    hosts: list[str] = [host] if isinstance(host, str) else list(host)
    multi_host = len(hosts) > 1

    files = sorted(os.listdir(data_dir))
    total_written = 0

    for spec in registry.enabled:
        matching_file = find_file_for_spec(spec, files)
        if not matching_file:
            log.warning("No matching file found for metric key: %s (aliases: %s)", spec.key, spec.aliases)
            continue

        filepath = data_dir / matching_file
        ext = filepath.suffix.lower()
        log.info("Processing metric %s using file %s", spec.key, matching_file)

        try:
            if ext == ".csv":
                df = pd.read_csv(filepath)
            elif ext in (".xlsx", ".xls"):
                df = pd.read_excel(filepath)
            else:
                log.warning("Unsupported file extension %s for file %s", ext, matching_file)
                continue

            if df.empty:
                log.warning("File %s is empty", matching_file)
                continue

            # Identify timestamp column
            time_col = None
            for col in df.columns:
                if "date" in str(col).lower() or "time" in str(col).lower():
                    time_col = col
                    break
            if time_col is None:
                time_col = df.columns[0]

            # Parse timestamps
            if ext == ".csv":
                timestamps = pd.to_datetime(df[time_col], format="%d/%m/%y %H:%M", errors="coerce")
                if timestamps.isna().all():
                    timestamps = pd.to_datetime(df[time_col], errors="coerce")
            else:
                timestamps = pd.to_datetime(df[time_col], errors="coerce")

            # Make timezone-aware UTC
            if timestamps.dt.tz is None:
                timestamps = timestamps.dt.tz_localize("UTC")
            else:
                timestamps = timestamps.dt.tz_convert("UTC")

            # ── per-host ingestion ────────────────────────────────────────────
            for h in hosts:
                # Find the column for this host
                host_col = None
                for col in df.columns:
                    if h.lower() in str(col).lower():
                        host_col = col
                        break

                if host_col is None:
                    log.warning(
                        "Host %r not found in columns of %s. Columns: %s",
                        h, matching_file, list(df.columns)
                    )
                    continue

                # Parse numeric values
                parsed_values = df[host_col].apply(parse_val)

                # Ratio → percentage scaling
                if spec.unit == "percent" and parsed_values.max() <= 1.0:
                    parsed_values = parsed_values * 100.0

                # Build node_id: single-host uses plain metric_key,
                # multi-host uses the canonical pipe-separated format
                # metric_key|HOST_SLUG so that split_node_id() correctly
                # extracts the metric_key and panel.py can look up role/spec.
                if multi_host:
                    # Derive a clean slug from the host string
                    if " - " in h:
                        # "10.50.98.26 - HYDUPINTAPP16" -> "HYDUPINTAPP16"
                        slug = h.split(" - ", 1)[1].strip()
                    else:
                        # "10.50.98.26" -> "10_50_98_26"
                        slug = h.replace(".", "_")
                    nid = _host_node_id(spec.key, slug)
                else:
                    from .registry import node_id as _node_id
                    nid = _node_id(spec.key, "")

                sub_df = pd.DataFrame(
                    {
                        "timestamp": timestamps,
                        "node_id": nid,
                        "metric_key": spec.key,
                        "dimension": h if multi_host else "",
                        "entity_type": spec.entity_type,
                        "value": parsed_values,
                    }
                ).dropna(subset=["timestamp", "value"])

                if not sub_df.empty:
                    written = write_rows(sub_df, raw_dir, grid)
                    total_written += written
                    log.info(
                        "Ingested %d rows for metric %s (host=%s, node=%s)",
                        written, spec.key, h, nid
                    )
                else:
                    log.warning("No valid rows after parsing %s for host %s", matching_file, h)

        except Exception as e:
            log.error("Failed to process file %s: %s", matching_file, e, exc_info=True)

    return total_written
