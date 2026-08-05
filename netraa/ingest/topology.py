"""Resolve the entity IDs a metric selector needs.

The old scripts assumed every metric could be filtered by dt.entity.host or
dt.entity.service. That is false for two families:

  * builtin:tech.jvm.*             -> dt.entity.process_group_instance
  * builtin:service.keyRequest.*   -> dt.entity.service_method

Filtering those on the host/service dimension matches nothing and returns an
empty result, which the old code swallowed (blocker B2). This module walks the
entity relationship graph from the one host and one service in scope and
collects the real IDs, so the selectors filter on a dimension that exists.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .dynatrace_client import DynatraceClient, DynatraceError

log = logging.getLogger(__name__)

# Dynatrace entity IDs are prefixed by their type, so relationships can be
# classified without knowing the (version-dependent) relationship names.
ID_PREFIXES = {
    "HOST": "HOST-",
    "SERVICE": "SERVICE-",
    "PROCESS_GROUP_INSTANCE": "PROCESS_GROUP_INSTANCE-",
    "PROCESS_GROUP": "PROCESS_GROUP-",
    "SERVICE_METHOD": "SERVICE_METHOD-",
    "DISK": "DISK-",
}


@dataclass
class Topology:
    host_id: str
    service_id: str
    entities: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def ids_for(self, entity_type: str) -> list[str]:
        return self.entities.get(entity_type, [])

    def to_dict(self) -> dict:
        return {
            "host_id": self.host_id,
            "service_id": self.service_id,
            "entities": self.entities,
            "warnings": self.warnings,
        }

    @classmethod
    def load(cls, path: str | Path) -> "Topology":
        data = json.loads(Path(path).read_text())
        return cls(
            host_id=data["host_id"],
            service_id=data["service_id"],
            entities=data["entities"],
            warnings=data.get("warnings", []),
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))


def _collect_related(entity_payload: dict) -> dict[str, set[str]]:
    """Pull every related entity ID out of an entity payload, typed by prefix."""
    found: dict[str, set[str]] = {k: set() for k in ID_PREFIXES}

    for direction in ("toRelationships", "fromRelationships"):
        for _rel_name, entries in (entity_payload.get(direction) or {}).items():
            for entry in entries or []:
                eid = entry.get("id", "")
                for etype, prefix in ID_PREFIXES.items():
                    if eid.startswith(prefix):
                        found[etype].add(eid)
                        break
    return found


def resolve(
    client: DynatraceClient, host_id: str, service_id: str
) -> Topology:
    """Walk relationships from the host and the service in scope."""
    topo = Topology(host_id=host_id, service_id=service_id)
    buckets: dict[str, set[str]] = {k: set() for k in ID_PREFIXES}
    buckets["HOST"].add(host_id)
    buckets["SERVICE"].add(service_id)

    for label, eid in (("host", host_id), ("service", service_id)):
        try:
            payload = client.entity(eid)
        except DynatraceError as exc:
            topo.warnings.append(f"could not read {label} entity {eid}: {exc}")
            log.warning("could not read %s entity %s: %s", label, eid, exc)
            continue
        for etype, ids in _collect_related(payload).items():
            buckets[etype] |= ids

    # Process group instances: use ALL PGIs on the host.
    # The intersection of host PGIs ∩ service PGIs only yields the PGI directly
    # linked to the target service (e.g. nginx), which has no JVM instrumentation.
    # JVM metrics (builtin:tech.jvm.*) live on Tomcat/Java PGIs that are related
    # to the host but not necessarily to the service entity. For the POC scope
    # (one host, one service) using all host PGIs gives the correct JVM coverage.
    host_pgis = set()
    service_pgis = set()
    try:
        host_pgis = _collect_related(client.entity(host_id))["PROCESS_GROUP_INSTANCE"]
        service_pgis = _collect_related(client.entity(service_id))["PROCESS_GROUP_INSTANCE"]
    except DynatraceError:
        pass

    if host_pgis:
        buckets["PROCESS_GROUP_INSTANCE"] = host_pgis
        if not (host_pgis & service_pgis):
            topo.warnings.append(
                "no process group instance is directly related to both the host "
                "and the service; using all PGIs on the host. JVM metrics may "
                "include processes outside the service in scope."
            )

    # Disks: relationship walk is the primary source; entity selector is a
    # fallback for tenants that do not expose isDiskOf on the host payload.
    if not buckets["DISK"]:
        try:
            disks = client.entities(
                f'type(DISK),fromRelationships.isDiskOf(entityId("{host_id}"))'
            )
            buckets["DISK"] = {d["entityId"] for d in disks}
        except DynatraceError as exc:
            topo.warnings.append(f"disk entity lookup failed: {exc}")

    # Service methods (key requests) belonging to the service.
    if not buckets["SERVICE_METHOD"]:
        try:
            methods = client.entities(
                f'type(SERVICE_METHOD),fromRelationships.isServiceMethodOf(entityId("{service_id}"))'
            )
            buckets["SERVICE_METHOD"] = {m["entityId"] for m in methods}
        except DynatraceError as exc:
            topo.warnings.append(f"service method entity lookup failed: {exc}")

    topo.entities = {k: sorted(v) for k, v in buckets.items() if v}

    for etype in ("PROCESS_GROUP_INSTANCE", "DISK", "SERVICE_METHOD"):
        if not topo.entities.get(etype):
            topo.warnings.append(
                f"no {etype} entities resolved — metrics with "
                f"entity_type={etype} will be skipped by the backfill and "
                f"reported as UNRESOLVED by `netraa validate`."
            )

    return topo
