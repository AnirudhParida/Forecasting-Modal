"""Measure what this tenant actually retains, instead of assuming.

Dynatrace keeps fine-grained points for a shorter window than coarse ones and
silently serves a coarser bucket when you ask for more history than the
requested resolution covers. Rather than hardcoding a retention policy, this
probes a reference metric across a grid of (resolution, lookback) pairs and
reports what actually came back — requested spacing vs observed spacing.

The recommendation drives `grids:` in configs/v1.yaml:
  fine grid   -> finest resolution that still covers a usable dependency window
  coarse grid -> resolution that reaches furthest back, for the quarter-ahead model
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import json

import pandas as pd

from .dynatrace_client import DynatraceClient, DynatraceError

log = logging.getLogger(__name__)

RESOLUTIONS = ["1m", "5m", "1h", "1d"]
LOOKBACK_DAYS = [1, 7, 14, 30, 90, 180, 365, 400]

RES_SECONDS = {"1m": 60, "5m": 300, "1h": 3600, "1d": 86400}


@dataclass
class ProbeRow:
    resolution: str
    lookback_days: int
    points: int
    observed_spacing_s: float | None
    requested_spacing_s: int
    downgraded: bool
    earliest: str | None
    error: str = ""

    @property
    def usable(self) -> bool:
        return self.points > 0 and not self.error


def probe_retention(
    client: DynatraceClient,
    reference_selector: str,
    resolutions: list[str] | None = None,
    lookbacks: list[int] | None = None,
) -> list[ProbeRow]:
    rows: list[ProbeRow] = []
    resolutions = resolutions or RESOLUTIONS
    lookbacks = lookbacks or LOOKBACK_DAYS

    for res in resolutions:
        for days in lookbacks:
            row = ProbeRow(
                resolution=res,
                lookback_days=days,
                points=0,
                observed_spacing_s=None,
                requested_spacing_s=RES_SECONDS[res],
                downgraded=False,
                earliest=None,
            )
            try:
                result = client.query(
                    reference_selector, f"now-{days}d", "now", res
                )
            except DynatraceError as exc:
                row.error = str(exc).splitlines()[0][:120]
                rows.append(row)
                continue

            stamps = sorted({t for s in result.series for t in s.timestamps})
            row.points = len(stamps)
            if len(stamps) >= 2:
                diffs = pd.Series(stamps).diff().dropna() / 1000.0
                row.observed_spacing_s = float(diffs.median())
                row.downgraded = row.observed_spacing_s > row.requested_spacing_s * 1.5
            if stamps:
                row.earliest = pd.to_datetime(
                    stamps[0], unit="ms", utc=True
                ).isoformat()

            rows.append(row)

    return rows


def recommend(rows: list[ProbeRow], min_fine_days: int = 7) -> dict:
    """Pick the two grids from measured behaviour."""
    ok = [r for r in rows if r.usable and not r.downgraded]

    fine = None
    for res in RESOLUTIONS:                      # finest first
        candidates = [r for r in ok if r.resolution == res and r.lookback_days >= min_fine_days]
        if candidates:
            best = max(candidates, key=lambda r: r.lookback_days)
            fine = {"resolution": res, "lookback_days": best.lookback_days}
            break

    coarse = None
    if ok:
        deepest = max(ok, key=lambda r: (r.lookback_days, -RES_SECONDS[r.resolution]))
        coarse = {
            "resolution": deepest.resolution,
            "lookback_days": deepest.lookback_days,
        }

    return {
        "fine": fine,
        "coarse": coarse,
        "note": (
            "fine grid feeds dependency discovery; coarse grid feeds the "
            "quarter-ahead forecaster. Copy these into configs/v1.yaml."
        ),
    }


def format_report(rows: list[ProbeRow], rec: dict) -> str:
    hdr = (
        f"{'resolution':<11}{'lookback':>9}{'points':>8}{'observed':>10}"
        f"{'requested':>11}  status"
    )
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        if r.error:
            status = f"ERROR {r.error[:40]}"
        elif r.points == 0:
            status = "no data (beyond retention)"
        elif r.downgraded:
            status = "DOWNGRADED — server served coarser buckets"
        else:
            status = "ok"
        obs = f"{r.observed_spacing_s:.0f}s" if r.observed_spacing_s else "-"
        lines.append(
            f"{r.resolution:<11}{r.lookback_days:>7}d{r.points:>8}{obs:>10}"
            f"{r.requested_spacing_s:>10}s  {status}"
        )

    lines += ["", "Recommended grids:"]
    for name in ("fine", "coarse"):
        val = rec.get(name)
        lines.append(
            f"  {name:<8} {val}" if val else f"  {name:<8} could not be determined"
        )
    return "\n".join(lines)


def save_report(rows: list[ProbeRow], rec: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"probes": [asdict(r) for r in rows], "recommendation": rec}, indent=2)
    )
