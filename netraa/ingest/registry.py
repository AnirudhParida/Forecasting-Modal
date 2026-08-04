"""Metric registry -> concrete Dynatrace selectors.

Blocker B3 lived here in the old code:

    metric_queries[query] = custom_name      # keyed by the QUERY string

Three registry names collapsing onto one selector meant two of them were
discarded before a request was made. Entries are now keyed by `key`, which is
unique by construction, and genuine duplicates are declared as `aliases`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Which dimension key each entity type is addressed by, and which topology
# bucket supplies the IDs. Getting this wrong is what silently killed every
# JVM metric in the old scripts (B2).
ENTITY_DIMENSION = {
    "HOST": "dt.entity.host",
    "SERVICE": "dt.entity.service",
    "PROCESS_GROUP_INSTANCE": "dt.entity.process_group_instance",
    "SERVICE_METHOD": "dt.entity.service_method",
    "DISK": "dt.entity.disk",
}

VALID_AGGS = {"avg", "sum", "max", "min", "count"}
VALID_ROLES = {"driver", "intermediate", "target"}
VALID_TRANSFORMS = {"none", "abs", "log1p", "diff"}


@dataclass
class MetricSpec:
    key: str
    selector: str
    entity_type: str
    split_by: str
    agg: str
    role: str
    resource: str
    transform: str = "none"
    unit: str = "unspecified"
    aliases: list[str] = field(default_factory=list)
    enabled: bool = True
    note: str = ""

    def __post_init__(self) -> None:
        if self.entity_type not in ENTITY_DIMENSION:
            raise ValueError(
                f"{self.key}: unknown entity_type {self.entity_type!r}. "
                f"Expected one of {sorted(ENTITY_DIMENSION)}"
            )
        if self.agg not in VALID_AGGS:
            raise ValueError(f"{self.key}: unknown agg {self.agg!r}")
        if self.role not in VALID_ROLES:
            raise ValueError(f"{self.key}: unknown role {self.role!r}")
        if self.transform not in VALID_TRANSFORMS:
            raise ValueError(f"{self.key}: unknown transform {self.transform!r}")

    @property
    def entity_dimension(self) -> str:
        return ENTITY_DIMENSION[self.entity_type]

    def build_selector(self, entity_ids: list[str]) -> str:
        """Compose the full metric selector.

        Always emits an explicit filter, an explicit splitBy and an explicit
        aggregation. The old scripts emitted only a filter, letting Dynatrace
        auto-merge dimensions — the likely source of the negative memory
        values in memory_influencers_last_10m.csv (B5).
        """
        if not entity_ids:
            raise ValueError(
                f"{self.key}: no {self.entity_type} entity IDs resolved. "
                "Run `netraa topology` first."
            )

        quoted = ",".join(f'"{e}"' for e in entity_ids)
        if len(entity_ids) == 1:
            flt = f'filter(eq("{self.entity_dimension}",{quoted}))'
        else:
            flt = f'filter(in("{self.entity_dimension}",entityId({quoted})))'

        split = f'splitBy("{self.split_by}")' if self.split_by else "splitBy()"
        return f"{self.selector}:{flt}:{split}:{self.agg}"


@dataclass
class Registry:
    metrics: list[MetricSpec]

    @classmethod
    def load(cls, path: str | Path) -> "Registry":
        raw = yaml.safe_load(Path(path).read_text())
        specs = [MetricSpec(**entry) for entry in raw["metrics"]]

        seen: dict[str, str] = {}
        for spec in specs:
            if spec.key in seen:
                raise ValueError(f"duplicate registry key: {spec.key}")
            seen[spec.key] = spec.selector
        return cls(metrics=specs)

    @property
    def enabled(self) -> list[MetricSpec]:
        return [m for m in self.metrics if m.enabled]

    def by_key(self, key: str) -> MetricSpec:
        for m in self.metrics:
            if m.key == key or key in m.aliases:
                return m
        raise KeyError(key)

    def targets(self) -> list[MetricSpec]:
        return [m for m in self.enabled if m.role == "target"]

    def drivers(self) -> list[MetricSpec]:
        return [m for m in self.enabled if m.role == "driver"]

    def alias_map(self) -> dict[str, str]:
        """Legacy column name -> canonical key. Used to read the old CSVs."""
        out: dict[str, str] = {}
        for m in self.metrics:
            out[m.key] = m.key
            for alias in m.aliases:
                out[alias] = m.key
        return out


def node_id(metric_key: str, dimension: str = "") -> str:
    """Canonical node identifier.

    Blocker B4: the same physical signal appeared as `meter_vm_network_receive`
    in one CSV and `meter_vm_network_receive_HOST-D97…` in another, because one
    script appended dimensions and the others did not. Every producer now goes
    through this function, so a node has exactly one name everywhere.
    """
    dimension = (dimension or "").strip()
    if not dimension:
        return metric_key
    safe = dimension.replace("/", "_").replace("\\", "_").replace(":", "-").replace(" ", "_")
    return f"{metric_key}|{safe}"


def split_node_id(nid: str) -> tuple[str, str]:
    metric_key, _, dim = nid.partition("|")
    return metric_key, dim
