"""Long store -> wide (T x N) panel plus an observation mask.

The panel is the single object both engines consume: the statistical dependency
graph scores its columns pairwise, and the forecaster slides windows over it.

Two things the old CSVs got wrong and this fixes:

  B4  One node, one name. Column names come from registry.node_id, so
      `meter_vm_network_receive` and `meter_vm_network_receive_HOST-D97…`
      resolve to the same column instead of two half-empty ones.

  Gaps are represented, not invented. Missingness is carried in a parallel mask
  rather than being filled with 0.0 as the old scripts did — a zero for
  "CPU was not reported" is a data point claiming the CPU was idle, and it
  poisons both correlation scores and forecast training.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..ingest.registry import Registry, split_node_id
from ..ingest.store import read_grid

log = logging.getLogger(__name__)

AGG_TO_PANDAS = {"avg": "mean", "sum": "sum", "max": "max", "min": "min", "count": "sum"}


@dataclass
class Panel:
    values: pd.DataFrame     # T x N, NaN where unobserved
    mask: pd.DataFrame       # T x N, 1.0 observed / 0.0 missing
    nodes: pd.DataFrame      # per-node metadata
    grid: str
    freq: str

    @property
    def node_ids(self) -> list[str]:
        return list(self.values.columns)

    @property
    def n_nodes(self) -> int:
        return self.values.shape[1]

    @property
    def n_steps(self) -> int:
        return self.values.shape[0]

    def targets(self) -> list[str]:
        return self.nodes.loc[self.nodes["role"] == "target", "node_id"].tolist()

    def drivers(self) -> list[str]:
        return self.nodes.loc[self.nodes["role"] == "driver", "node_id"].tolist()

    def describe(self) -> str:
        cov = self.mask.mean().sort_values()
        lines = [
            f"grid={self.grid} freq={self.freq}",
            f"shape: {self.n_steps} timesteps x {self.n_nodes} nodes",
            f"span: {self.values.index.min()} -> {self.values.index.max()}",
            f"targets: {len(self.targets())}, drivers: {len(self.drivers())}",
            f"mean coverage: {self.mask.values.mean() * 100:.1f}%",
            "",
            "lowest-coverage nodes:",
        ]
        for nid, c in cov.head(5).items():
            lines.append(f"  {nid:<40} {c * 100:5.1f}%")
        return "\n".join(lines)

    # --------------------------------------------------------------- storage
    def save(self, panel_dir: Path) -> None:
        panel_dir.mkdir(parents=True, exist_ok=True)
        self.values.to_parquet(panel_dir / f"{self.grid}_values.parquet")
        self.mask.to_parquet(panel_dir / f"{self.grid}_mask.parquet")
        self.nodes.to_parquet(panel_dir / f"{self.grid}_nodes.parquet", index=False)
        (panel_dir / f"{self.grid}_meta.json").write_text(
            json.dumps(
                {
                    "grid": self.grid,
                    "freq": self.freq,
                    "n_steps": self.n_steps,
                    "n_nodes": self.n_nodes,
                    "start": str(self.values.index.min()),
                    "end": str(self.values.index.max()),
                    "coverage": float(self.mask.values.mean()),
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, panel_dir: Path, grid: str) -> "Panel":
        meta = json.loads((panel_dir / f"{grid}_meta.json").read_text())
        return cls(
            values=pd.read_parquet(panel_dir / f"{grid}_values.parquet"),
            mask=pd.read_parquet(panel_dir / f"{grid}_mask.parquet"),
            nodes=pd.read_parquet(panel_dir / f"{grid}_nodes.parquet"),
            grid=meta["grid"],
            freq=meta["freq"],
        )


def build_panel(
    raw_dir: Path,
    grid: str,
    freq: str,
    registry: Registry,
    source_grid: str | None = None,
    min_coverage: float = 0.20,
    ffill_limit: int = 2,
    long_df: pd.DataFrame | None = None,
    min_tail_coverage: float = 0.60,
    min_observed_steps: int = 90,
) -> Panel:
    """Build a panel for `grid`.

    `source_grid` lets the coarse (daily) panel be aggregated down from the
    fine store rather than refetched — each node aggregated by its own declared
    method, so counters sum and gauges average.
    """
    df = long_df if long_df is not None else read_grid(raw_dir, source_grid or grid)
    if df.empty:
        raise ValueError(
            f"no data in the store for grid={source_grid or grid}. "
            "Run `netraa backfill` first (or `netraa smoke` to use the legacy CSVs)."
        )

    wide = df.pivot_table(
        index="timestamp", columns="node_id", values="value", aggfunc="mean"
    ).sort_index()

    # Per-node aggregation onto the target grid.
    agg_map: dict[str, str] = {}
    for nid in wide.columns:
        metric_key, _ = split_node_id(nid)
        try:
            spec = registry.by_key(metric_key)
            agg_map[nid] = AGG_TO_PANDAS[spec.agg]
        except KeyError:
            agg_map[nid] = "mean"

    resampled = wide.resample(freq).agg(agg_map)

    # A resample of an all-NaN bucket yields 0.0 for sum but NaN for mean.
    # Re-derive observation from the raw counts so summed counters are not
    # credited with a fabricated zero.
    observed = wide.notna().resample(freq).sum() > 0
    observed = observed.reindex(columns=resampled.columns, fill_value=False)
    resampled = resampled.where(observed)

    # Regular grid across the full span — missing buckets become explicit rows.
    full_index = pd.date_range(
        resampled.index.min(), resampled.index.max(), freq=freq, tz="UTC"
    )
    resampled = resampled.reindex(full_index)
    resampled.index.name = "timestamp"

    mask = resampled.notna().astype("float64")

    # Drop nodes too sparse to model. Reported, never silent.
    #
    # Overall coverage alone cannot tell a gappy node from a short-history one:
    # a metric onboarded halfway through a 400-day span scores ~50% while being
    # solid ever since. Those are kept when their coverage *since first
    # observation* clears min_tail_coverage with at least min_observed_steps
    # points — the mask channel carries the missing early span.
    coverage = mask.mean()
    observed_steps = mask.sum()
    T = len(mask)
    first_obs = mask.to_numpy().argmax(axis=0)  # index of first observed bucket
    tail_coverage = pd.Series(
        np.where(
            observed_steps.to_numpy() > 0,
            observed_steps.to_numpy() / np.maximum(T - first_obs, 1),
            0.0,
        ),
        index=mask.columns,
    )

    keep, rescued = [], []
    for nid in resampled.columns:
        if coverage[nid] >= min_coverage:
            keep.append(nid)
        elif (
            tail_coverage[nid] >= min_tail_coverage
            and observed_steps[nid] >= min_observed_steps
        ):
            keep.append(nid)
            rescued.append(nid)

    dropped = sorted(set(resampled.columns) - set(keep))
    if rescued:
        log.info(
            "keeping %d short-history node(s) on tail coverage: %s",
            len(rescued),
            ", ".join(
                f"{r} ({coverage[r] * 100:.1f}% overall, "
                f"{tail_coverage[r] * 100:.1f}% since start)"
                for r in rescued
            ),
        )
    if dropped:
        log.warning(
            "dropping %d node(s) below %.0f%% coverage: %s",
            len(dropped),
            min_coverage * 100,
            ", ".join(f"{d} ({coverage[d] * 100:.1f}%)" for d in dropped),
        )
    resampled = resampled[keep]
    mask = mask[keep]

    if resampled.empty or not keep:
        raise ValueError(
            f"every node fell below the {min_coverage:.0%} coverage floor for "
            f"grid={grid}. Lower min_coverage or collect more history."
        )

    # Bounded forward fill for short gaps; longer gaps stay NaN and the mask
    # keeps them out of the loss.
    filled = resampled.ffill(limit=ffill_limit) if ffill_limit else resampled

    nodes = _node_metadata(keep, registry, coverage)
    return Panel(values=filled, mask=mask, nodes=nodes, grid=grid, freq=freq)


def _node_metadata(
    node_ids: list[str], registry: Registry, coverage: pd.Series
) -> pd.DataFrame:
    records = []
    for nid in node_ids:
        metric_key, dimension = split_node_id(nid)
        try:
            spec = registry.by_key(metric_key)
            role, resource = spec.role, spec.resource
            unit, transform, entity_type = spec.unit, spec.transform, spec.entity_type
        except KeyError:
            role, resource, unit, transform, entity_type = (
                "intermediate", "unknown", "unspecified", "none", "UNKNOWN",
            )
        records.append(
            {
                "node_id": nid,
                "metric_key": metric_key,
                "dimension": dimension,
                "entity_type": entity_type,
                "role": role,
                "resource": resource,
                "unit": unit,
                "transform": transform,
                "coverage": float(coverage.get(nid, np.nan)),
            }
        )
    return pd.DataFrame.from_records(records)
