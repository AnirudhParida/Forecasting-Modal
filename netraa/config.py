"""Configuration loading. Secrets come from the environment, never from YAML."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader so the package works without python-dotenv installed."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass
class GridConfig:
    """One sampling grid. The POC uses two — see ARCHITECTURE.md section 2."""

    name: str
    resolution: str          # Dynatrace resolution token, e.g. "5m", "1h", "1d"
    pandas_freq: str         # equivalent pandas offset alias
    lookback_days: int
    purpose: str
    # Set to another grid's name to aggregate this panel down from that grid's
    # stored data instead of fetching it separately.
    source_grid: str | None = None

    @property
    def seconds(self) -> int:
        unit = self.resolution[-1]
        n = int(self.resolution[:-1])
        return n * {"m": 60, "h": 3600, "d": 86400}[unit]


@dataclass
class ForecastConfig:
    grid: str = "coarse"
    input_steps: int = 90
    horizons: list[int] = field(default_factory=lambda: [7, 30, 60, 90])
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])


@dataclass
class GraphConfig:
    grid: str = "fine"
    max_lag_steps: int = 24
    xcorr_threshold: float = 0.30
    granger_alpha: float = 0.05
    mi_threshold: float = 0.05
    top_k: int = 8
    min_overlap: int = 60


@dataclass
class ModelConfig:
    hidden: int = 32
    blocks: int = 2
    kernel_size: int = 3
    dropout: float = 0.2
    node_embed_dim: int = 16
    top_k: int = 12             # max incoming edges kept per node in the learned adjacency
                                # (was hardcoded to 8 in train.py; now configurable)
    graph_prior_weight: float = 0.1
    graph_sparsity_weight: float = 0.01
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    batch_size: int = 16
    patience: int = 25
    seed: int = 42
    # Gaussian noise (in scaled units) added to the value channel during
    # training. Overlapping windows over a short panel repeat almost the same
    # sample hundreds of times; jitter is the cheapest defence against the
    # model memorising the panel by epoch ~5.
    input_noise: float = 0.1


@dataclass
class Config:
    host_id: str
    service_id: str
    base_url: str
    api_token: str
    grids: dict[str, GridConfig]
    forecast: ForecastConfig
    graph: GraphConfig
    model: ModelConfig
    data_dir: Path
    artifacts_dir: Path
    registry_path: Path
    raw: dict[str, Any]

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def panel_dir(self) -> Path:
        return self.data_dir / "panel"

    @property
    def graph_dir(self) -> Path:
        return self.data_dir / "graph"

    @property
    def topology_path(self) -> Path:
        return self.data_dir / "topology.json"

    def require_token(self) -> str:
        if not self.api_token:
            raise RuntimeError(
                "DYNATRACE_API_TOKEN is not set. Copy .env.example to .env and fill it in. "
                "Do not hardcode the token in source files (blocker B7)."
            )
        return self.api_token


def load_config(path: str | Path = "configs/v1.yaml") -> Config:
    _load_dotenv(PROJECT_ROOT / ".env")

    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    raw = yaml.safe_load(cfg_path.read_text())

    grids = {
        name: GridConfig(name=name, **spec) for name, spec in raw["grids"].items()
    }

    data_dir = PROJECT_ROOT / raw.get("data_dir", "data")
    artifacts_dir = PROJECT_ROOT / raw.get("artifacts_dir", "artifacts")

    return Config(
        host_id=os.environ.get("NETRAA_HOST_ID", raw["scope"].get("host_id", "")),
        service_id=os.environ.get("NETRAA_SERVICE_ID", raw["scope"].get("service_id", "")),
        base_url=os.environ.get(
            "DYNATRACE_BASE_URL", raw.get("base_url", "")
        ).rstrip("/"),
        api_token=os.environ.get("DYNATRACE_API_TOKEN", ""),
        grids=grids,
        forecast=ForecastConfig(**raw.get("forecast", {})),
        graph=GraphConfig(**raw.get("graph", {})),
        model=ModelConfig(**raw.get("model", {})),
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        registry_path=PROJECT_ROOT / raw.get("registry", "metrics_registry.yaml"),
        raw=raw,
    )
