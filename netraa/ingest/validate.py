"""Pre-flight validation of every registry entry.

This is the direct fix for blocker B2. The old scripts requested 15 metrics,
received 5, and printed nothing about the other 10. Here every metric is checked
against the metric-descriptor endpoint and a short probe query, and each one
lands in exactly one status bucket. A backfill should not be started until this
reports no FAIL rows.

Statuses
  OK                  metric exists, dimension matches, data returned
  MISSING_METRIC      the metric ID does not exist on this tenant
  DIMENSION_MISMATCH  the registry filters on a dimension the metric lacks
                      (this is why every builtin:tech.jvm.* column was empty)
  UNRESOLVED          no entity IDs of the required type in the topology
  NO_DATA             query succeeded but returned zero points
  ALL_NULL            points returned, every value null
  UNIT_VIOLATION      values contradict the declared unit (negative percent —
                      the -941.98 in memory_influencers_last_10m.csv, B5)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

from .dynatrace_client import DynatraceClient, DynatraceError
from .registry import MetricSpec, Registry
from .topology import Topology

log = logging.getLogger(__name__)

FAIL_STATUSES = {
    "MISSING_METRIC",
    "DIMENSION_MISMATCH",
    "UNRESOLVED",
    "NO_DATA",
    "ALL_NULL",
}


@dataclass
class ValidationRow:
    key: str
    selector: str
    status: str
    detail: str = ""
    descriptor_unit: str = ""
    descriptor_dimensions: str = ""
    series_count: int = 0
    point_count: int = 0
    null_fraction: float = 0.0
    observed_min: float | None = None
    observed_max: float | None = None
    resolved_selector: str = ""

    @property
    def failed(self) -> bool:
        return self.status in FAIL_STATUSES


def _check_unit(spec: MetricSpec, lo: float | None, hi: float | None) -> str | None:
    if lo is None or hi is None:
        return None
    if spec.unit == "percent":
        if lo < 0:
            return f"declared percent but minimum is {lo:.2f}"
        if hi > 100.0001:
            return f"declared percent but maximum is {hi:.2f}"
    if spec.unit in {"byte", "count", "millisecond"} and lo < 0:
        return f"declared {spec.unit} but minimum is {lo:.2f}"
    return None


def validate_registry(
    client: DynatraceClient,
    registry: Registry,
    topology: Topology,
    probe_from: str = "now-24h",
    probe_to: str = "now",
    probe_resolution: str = "1h",
) -> list[ValidationRow]:
    rows: list[ValidationRow] = []

    for spec in registry.enabled:
        row = ValidationRow(key=spec.key, selector=spec.selector, status="OK")

        # 1. Does the metric exist, and what does it actually carry?
        try:
            desc = client.metric_descriptor(spec.selector)
        except DynatraceError as exc:
            row.status = "MISSING_METRIC"
            row.detail = str(exc).splitlines()[0]
            rows.append(row)
            continue

        dims = [d.get("key", "") for d in desc.get("dimensionDefinitions", []) or []]
        row.descriptor_unit = desc.get("unit", "")
        row.descriptor_dimensions = ",".join(dims)

        # 2. Is the dimension we filter on one this metric actually has?
        if dims and spec.entity_dimension not in dims:
            row.status = "DIMENSION_MISMATCH"
            row.detail = (
                f"registry filters on {spec.entity_dimension} but metric exposes "
                f"[{row.descriptor_dimensions}]"
            )
            rows.append(row)
            continue

        # 3. Do we have entity IDs of the required type?
        entity_ids = topology.ids_for(spec.entity_type)
        if not entity_ids:
            row.status = "UNRESOLVED"
            row.detail = f"topology has no {spec.entity_type} entities"
            rows.append(row)
            continue

        # 4. Does a real query return anything?
        resolved = spec.build_selector(entity_ids)
        row.resolved_selector = resolved
        try:
            result = client.query(resolved, probe_from, probe_to, probe_resolution)
        except DynatraceError as exc:
            row.status = "NO_DATA"
            row.detail = str(exc).splitlines()[0]
            rows.append(row)
            continue

        row.series_count = len(result.series)
        row.point_count = result.point_count
        if result.warnings:
            row.detail = "; ".join(result.warnings)[:200]

        if result.is_empty:
            row.status = "NO_DATA"
            rows.append(row)
            continue

        values = [v for s in result.series for v in s.values if v is not None]
        total = sum(len(s.values) for s in result.series)
        row.null_fraction = 1.0 - (len(values) / total) if total else 1.0

        if not values:
            row.status = "ALL_NULL"
            rows.append(row)
            continue

        row.observed_min = float(min(values))
        row.observed_max = float(max(values))

        # 5. Do the values agree with the declared unit?
        violation = _check_unit(spec, row.observed_min, row.observed_max)
        if violation:
            row.status = "UNIT_VIOLATION"
            row.detail = (
                f"{violation}. transform={spec.transform} will be applied at "
                "panel-build time; investigate the selector before trusting it."
            )

        rows.append(row)

    return rows


def format_report(rows: list[ValidationRow]) -> str:
    hdr = f"{'metric key':<26} {'status':<19} {'series':>6} {'points':>7} {'null%':>6}  detail"
    lines = [hdr, "-" * len(hdr)]
    for r in sorted(rows, key=lambda r: (not r.failed, r.status, r.key)):
        lines.append(
            f"{r.key:<26} {r.status:<19} {r.series_count:>6} {r.point_count:>7} "
            f"{r.null_fraction * 100:>5.1f}%  {r.detail[:70]}"
        )

    failed = [r for r in rows if r.failed]
    warned = [r for r in rows if r.status == "UNIT_VIOLATION"]
    lines.append("")
    lines.append(
        f"{len(rows) - len(failed) - len(warned)} OK, {len(warned)} unit warnings, "
        f"{len(failed)} failed"
    )
    if failed:
        lines.append("Backfill will skip failed metrics. Fix the registry first.")
    return "\n".join(lines)


def save_report(rows: list[ValidationRow], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([asdict(r) for r in rows], indent=2))
