"""Dynatrace API v2 client.

Replaces the bare `requests.get` calls in the three fetch_* scripts. Differences
that matter:

* Token is read from the environment, never hardcoded (blocker B7).
* Retries with backoff, and honours Retry-After on 429 (the old scripts would
  drop a metric permanently on a transient rate limit).
* Pagination via nextPageKey (the old scripts read only the first page).
* A failed or empty response raises / returns an explicit EMPTY result that the
  caller must handle. The old scripts printed to stdout and moved on, which is
  how two thirds of the requested metrics vanished without anyone noticing (B2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)


class DynatraceError(RuntimeError):
    """Raised when the API returns a non-recoverable error."""


@dataclass
class Series:
    """One returned time series, tagged with the dimensions that identify it."""

    dimensions: list[str]
    dimension_map: dict[str, str]
    timestamps: list[int]
    values: list[float | None]

    @property
    def null_fraction(self) -> float:
        if not self.values:
            return 1.0
        return sum(v is None for v in self.values) / len(self.values)


@dataclass
class QueryResult:
    metric_id: str
    series: list[Series] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.series or all(not s.timestamps for s in self.series)

    @property
    def point_count(self) -> int:
        return sum(len(s.timestamps) for s in self.series)


class DynatraceClient:
    def __init__(self, base_url: str, api_token: str, timeout: int = 60):
        if not api_token:
            raise DynatraceError(
                "No API token supplied. Set DYNATRACE_API_TOKEN in .env (blocker B7)."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers.update(
            {
                "accept": "application/json; charset=utf-8",
                "Authorization": f"Api-Token {api_token}",
            }
        )
        retry = Retry(
            total=5,
            backoff_factor=1.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    # ------------------------------------------------------------------ core
    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, params=params or {}, timeout=self.timeout)
        if resp.status_code != 200:
            body = resp.text[:600]
            raise DynatraceError(
                f"GET {path} -> HTTP {resp.status_code}\n"
                f"params={params}\nbody={body}"
            )
        return resp.json()

    # -------------------------------------------------------------- metadata
    def metric_descriptor(self, metric_id: str) -> dict[str, Any]:
        """Metric metadata: unit, default aggregation, available dimensions.

        This is what makes blockers B2 and B5 detectable before a backfill runs
        instead of after: a metric that does not exist 404s here, and a metric
        whose declared unit disagrees with the observed values is visible.
        """
        base_id = metric_id.split(":filter")[0].split(":splitBy")[0]
        return self._get(f"/api/v2/metrics/{base_id}")

    def entity(self, entity_id: str, fields: str = "+toRelationships,+fromRelationships,+properties") -> dict[str, Any]:
        return self._get(f"/api/v2/entities/{entity_id}", {"fields": fields})

    def entities(self, entity_selector: str, fields: str = "") -> list[dict[str, Any]]:
        params: dict[str, Any] = {"entitySelector": entity_selector, "pageSize": 500}
        if fields:
            params["fields"] = fields
        out: list[dict[str, Any]] = []
        while True:
            page = self._get("/api/v2/entities", params)
            out.extend(page.get("entities", []))
            key = page.get("nextPageKey")
            if not key:
                return out
            params = {"nextPageKey": key}

    # ----------------------------------------------------------------- query
    def query(
        self,
        metric_selector: str,
        time_from: str | int,
        time_to: str | int,
        resolution: str,
        entity_selector: str | None = None,
    ) -> QueryResult:
        """Run a metric query, following pagination.

        `time_from` / `time_to` accept ms-epoch ints or Dynatrace relative
        tokens ("now-10m").
        """
        params: dict[str, Any] = {
            "metricSelector": metric_selector,
            "from": time_from,
            "to": time_to,
            "resolution": resolution,
        }
        if entity_selector:
            params["entitySelector"] = entity_selector

        result = QueryResult(metric_id=metric_selector)

        while True:
            payload = self._get("/api/v2/metrics/query", params)
            for warn in payload.get("warnings", []) or []:
                result.warnings.append(str(warn))

            for block in payload.get("result", []):
                for raw_series in block.get("data", []):
                    result.series.append(
                        Series(
                            dimensions=list(raw_series.get("dimensions", [])),
                            dimension_map=dict(raw_series.get("dimensionMap", {})),
                            timestamps=list(raw_series.get("timestamps", [])),
                            values=list(raw_series.get("values", [])),
                        )
                    )

            key = payload.get("nextPageKey")
            if not key:
                break
            params = {"nextPageKey": key}

        return result

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "DynatraceClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def chunk_range(
    start_ms: int, end_ms: int, chunk_days: int
) -> Iterator[tuple[int, int]]:
    """Split a backfill window into API-sized chunks.

    Dynatrace caps the number of points a single query returns; a year of
    5-minute data in one request is silently truncated. Chunking keeps every
    request inside the limit.
    """
    step = chunk_days * 86_400_000
    cursor = start_ms
    while cursor < end_ms:
        yield cursor, min(cursor + step, end_ms)
        cursor += step
