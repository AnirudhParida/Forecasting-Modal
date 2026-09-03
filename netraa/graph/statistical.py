"""(a) Statistical dependency discovery — no neural network, no training.

Answers the question directly: how does endpoint_cpm depend on
database_access_cpm, in which direction, with what delay?

Three independent signals are combined:

  lagged cross-correlation   linear coupling, its sign, and the delay
  mutual information         non-linear coupling the correlation misses
  Granger causality          does the source's past improve prediction of the
                             target beyond the target's own past

An edge is accepted only with predictive relevance AND temporal precedence
(or a structural relationship). Requiring both is what keeps "these two both
rise at 09:00" out of the graph — correlation alone would accept it.

The output is this phase's deliverable in its own right, and doubles as the
prior A_prior that regularises the learned adjacency in models/stgnn.py.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.panel import Panel

log = logging.getLogger(__name__)

# Structural relationships that hold by topology, independent of any data.
# Workload lands on the JVM, the JVM consumes host resources.
STRUCTURAL_FLOW = [
    ("service", "jvm"),
    ("service", "memory"),
    ("service", "cpu"),
    ("service", "disk"),
    ("service", "network"),
    ("jvm", "cpu"),
    ("jvm", "memory"),
    ("network", "cpu"),
    ("disk", "cpu"),
]


def _host_of(node_id: str) -> str:
    """Return the host dimension of a pipe-formatted node ID, or '' for single-host.

    Examples
    --------
    _host_of('host_cpu_usage|10_50_98_26') -> '10_50_98_26'
    _host_of('host_cpu_usage')             -> ''
    """
    return node_id.split("|", 1)[1] if "|" in node_id else ""


@dataclass
class Edge:
    source: str
    target: str
    strength: float           # |max lagged correlation|
    direction: str            # "+" or "-"
    lag_steps: int
    lag_seconds: int
    mutual_information: float
    granger_p: float | None
    structural: bool
    evidence: str
    n_overlap: int

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ xcorr
def _pairwise_lagged_corr(
    values: np.ndarray, mask: np.ndarray, lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """Exact pairwise-complete correlation between source[t] and target[t+lag].

    Returns (corr NxN, overlap-count NxN). Fully vectorised: the NaN handling is
    done with mask matmuls rather than a Python loop over N^2 pairs.
    """
    if lag == 0:
        A, B = values, values
        MA, MB = mask, mask
    else:
        A, B = values[:-lag], values[lag:]
        MA, MB = mask[:-lag], mask[lag:]

    Af = np.nan_to_num(A) * MA
    Bf = np.nan_to_num(B) * MB

    n = MA.T @ MB                                  # jointly observed count
    with np.errstate(divide="ignore", invalid="ignore"):
        sum_a = Af.T @ MB
        sum_b = MA.T @ Bf
        sum_aa = (Af * Af).T @ MB
        sum_bb = MA.T @ (Bf * Bf)
        sum_ab = Af.T @ Bf

        mean_a = sum_a / n
        mean_b = sum_b / n
        cov = sum_ab / n - mean_a * mean_b
        var_a = sum_aa / n - mean_a**2
        var_b = sum_bb / n - mean_b**2
        denom = np.sqrt(np.clip(var_a, 0, None) * np.clip(var_b, 0, None))
        corr = np.where(denom > 1e-12, cov / denom, 0.0)

    return np.nan_to_num(corr), n


def lagged_correlation_scan(
    panel: Panel, max_lag: int, min_overlap: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scan lags 0..max_lag. Returns (best_corr, best_lag, overlap) each NxN.

    Entry [i, j] describes source i leading target j.
    """
    values = panel.values.to_numpy(dtype="float64")
    mask = panel.mask.to_numpy(dtype="float64")
    N = values.shape[1]

    best_corr = np.zeros((N, N))
    best_lag = np.zeros((N, N), dtype=int)
    best_overlap = np.zeros((N, N), dtype=int)

    for lag in range(0, max_lag + 1):
        if lag >= values.shape[0] - 2:
            break
        corr, n = _pairwise_lagged_corr(values, mask, lag)
        corr = np.where(n >= min_overlap, corr, 0.0)

        improved = np.abs(corr) > np.abs(best_corr)
        best_corr = np.where(improved, corr, best_corr)
        best_lag = np.where(improved, lag, best_lag)
        best_overlap = np.where(improved, n.astype(int), best_overlap)

    np.fill_diagonal(best_corr, 0.0)
    return best_corr, best_lag, best_overlap


# --------------------------------------------------------- mutual information
def _mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 12) -> float:
    """Histogram-based MI, normalised to [0, 1] by min(H(x), H(y)).

    A histogram estimator rather than sklearn's kNN estimator: it is far cheaper
    across thousands of candidate pairs and this only needs to rank, not measure.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 20:
        return 0.0
    x, y = x[ok], y[ok]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0

    hist, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = hist / hist.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        nz = pxy > 0
        mi = float(np.sum(pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])))
        hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
        hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])))

    norm = min(hx, hy)
    return float(np.clip(mi / norm, 0.0, 1.0)) if norm > 1e-12 else 0.0


# ----------------------------------------------------------------- screening
def _informative_nodes(
    values: np.ndarray, mask: np.ndarray, min_unique: int = 8
) -> np.ndarray:
    """Boolean flag per node: does this series carry enough information to score?

    Near-constant series (host_mem_total, a config value that changed once) have
    almost no entropy, so the normalised MI against them saturates to 1.0 and
    they float to the top of the edge table. They cannot support a dependency
    claim in either direction, so they are excluded from scoring entirely.
    """
    N = values.shape[1]
    ok = np.zeros(N, dtype=bool)
    for j in range(N):
        obs = (mask[:, j] > 0) & np.isfinite(values[:, j])
        col = values[obs, j]
        ok[j] = len(np.unique(col)) >= min_unique and np.std(col) > 1e-12
    return ok


def _linear_detrend(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remove each node's linear trend, fitted on its observed points only.

    Two unrelated metrics that both drift over a 400-day panel correlate near
    1.0 at whatever lag the scan happens to try — that is how a disk counter
    ends up 'causing' JVM heap at 13 days. Correlation and MI are therefore
    scored on detrended series; Granger differences internally already.
    """
    out = values.copy()
    t = np.arange(values.shape[0], dtype="float64")
    for j in range(values.shape[1]):
        obs = (mask[:, j] > 0) & np.isfinite(values[:, j])
        if obs.sum() < 3 or np.std(values[obs, j]) < 1e-12:
            continue
        slope, intercept = np.polyfit(t[obs], values[obs, j], 1)
        out[:, j] = values[:, j] - (slope * t + intercept)
    return out


# ------------------------------------------------------------ Granger causality
def _granger_p(
    source: np.ndarray, target: np.ndarray, max_lag: int
) -> float | None:
    """p-value for 'source does not Granger-cause target'. Low = precedence.

    Both series are differenced first: Granger on a trending series reports
    spurious causality between any two things that drift together.
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        return None

    ok = np.isfinite(source) & np.isfinite(target)
    s, t = source[ok], target[ok]
    lag = max(1, min(max_lag, 8))
    if len(s) < 5 * lag + 20:
        return None

    s, t = np.diff(s), np.diff(t)
    if np.std(s) < 1e-12 or np.std(t) < 1e-12:
        return None

    data = np.column_stack([t, s])          # [target, source] is the required order
    try:
        # statsmodels prints its full test table to stdout regardless of the
        # (deprecated) verbose flag; discard it rather than flooding the console
        # once per candidate pair.
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            res = grangercausalitytests(data, maxlag=[lag])
        return float(res[lag][0]["ssr_ftest"][1])
    except Exception as exc:                 # near-singular systems are common here
        log.debug("granger failed: %s", exc)
        return None


# ------------------------------------------------------------------ structural
def structural_pairs(panel: Panel) -> set[tuple[str, str]]:
    """Edges implied by topology rather than by the data.

    Cross-host pairs are always excluded: structural flow (disk→cpu, etc.)
    only makes sense within the same physical host.
    """
    nodes = panel.nodes.set_index("node_id")
    pairs: set[tuple[str, str]] = set()

    for src in nodes.index:
        for dst in nodes.index:
            if src == dst:
                continue
            # ── host isolation: skip cross-host structural edges ──────────────
            h_src, h_dst = _host_of(src), _host_of(dst)
            if h_src and h_dst and h_src != h_dst:
                continue
            r_src = nodes.loc[src, "resource"]
            r_dst = nodes.loc[dst, "resource"]
            if (r_src, r_dst) in STRUCTURAL_FLOW:
                pairs.add((src, dst))
            # Siblings: same metric split across dimensions (per-disk, per-NIC).
            if (
                nodes.loc[src, "metric_key"] == nodes.loc[dst, "metric_key"]
                and nodes.loc[src, "dimension"] != nodes.loc[dst, "dimension"]
            ):
                pairs.add((src, dst))
    return pairs


# ---------------------------------------------------------------- main entry
def discover(
    panel: Panel,
    max_lag_steps: int = 24,
    xcorr_threshold: float = 0.30,
    granger_alpha: float = 0.05,
    mi_threshold: float = 0.05,
    top_k: int = 8,
    min_overlap: int = 60,
    run_granger: bool = True,
) -> list[Edge]:
    node_ids = panel.node_ids
    N = len(node_ids)
    freq_seconds = int(pd.Timedelta(panel.freq).total_seconds())

    values = panel.values.to_numpy(dtype="float64")
    mask_np = panel.mask.to_numpy(dtype="float64")

    informative = _informative_nodes(values, mask_np)
    if not informative.all():
        excluded = [node_ids[j] for j in range(N) if not informative[j]]
        log.info(
            "excluding %d near-constant node(s) from dependency scoring: %s",
            len(excluded), ", ".join(excluded),
        )

    # Correlation and MI are scored on detrended values; Granger runs on the
    # originals (it differences internally).
    detrended = _linear_detrend(values, mask_np)
    det_panel = Panel(
        values=pd.DataFrame(
            detrended, index=panel.values.index, columns=panel.values.columns
        ),
        mask=panel.mask, nodes=panel.nodes, grid=panel.grid, freq=panel.freq,
    )

    log.info("scanning %d nodes x %d lags", N, max_lag_steps + 1)
    best_corr, best_lag, overlap = lagged_correlation_scan(
        det_panel, max_lag_steps, min_overlap
    )

    structural = structural_pairs(panel)

    candidates: list[Edge] = []
    for i in range(N):
        for j in range(N):
            if i == j or not (informative[i] and informative[j]):
                continue
            r = best_corr[i, j]
            src, dst = node_ids[i], node_ids[j]
            # ── host isolation: skip cross-host statistical edges ─────────────
            h_src, h_dst = _host_of(src), _host_of(dst)
            if h_src and h_dst and h_src != h_dst:
                continue
            is_structural = (src, dst) in structural

            if abs(r) < xcorr_threshold and not is_structural:
                continue
            if overlap[i, j] < min_overlap:
                continue

            lag = int(best_lag[i, j])
            x = detrended[: len(detrended) - lag, i] if lag else detrended[:, i]
            y = detrended[lag:, j] if lag else detrended[:, j]
            mi = _mutual_information(x, y)

            if abs(r) < xcorr_threshold and mi < mi_threshold and not is_structural:
                continue

            candidates.append(
                Edge(
                    source=src,
                    target=dst,
                    strength=float(abs(r)),
                    direction="+" if r >= 0 else "-",
                    lag_steps=lag,
                    lag_seconds=lag * freq_seconds,
                    mutual_information=float(mi),
                    granger_p=None,
                    structural=is_structural,
                    evidence="",
                    n_overlap=int(overlap[i, j]),
                )
            )

    log.info("%d candidate pairs passed screening", len(candidates))

    # Granger only on candidates — it is the expensive test and would be
    # wasteful across all N^2 pairs.
    granger_ran = False
    if run_granger:
        try:
            import statsmodels  # noqa: F401
            granger_ran = True
        except ImportError:
            log.warning("statsmodels not installed — falling back to lag-based precedence")
    if granger_ran:
        idx = {n: k for k, n in enumerate(node_ids)}
        for edge in candidates:
            edge.granger_p = _granger_p(
                values[:, idx[edge.source]],
                values[:, idx[edge.target]],
                max(1, edge.lag_steps),
            )

    accepted: list[Edge] = []
    for edge in candidates:
        granger_pass = edge.granger_p is not None and edge.granger_p <= granger_alpha
        if granger_ran:
            # A lag alone is not precedence: best_lag is the argmax of |corr|
            # over the scan, so any two drifting series peak at *some* lag > 0.
            # When the Granger test ran, the edge must survive it; a pair the
            # test could not fit (near-singular) is rejected, not waved through.
            precedence = granger_pass
        else:
            precedence = granger_pass or edge.lag_steps > 0
        relevance = edge.strength >= xcorr_threshold or edge.mutual_information >= mi_threshold

        if not (relevance and (precedence or edge.structural)):
            continue

        tags = []
        if edge.strength >= xcorr_threshold:
            tags.append("xcorr")
        if edge.mutual_information >= mi_threshold:
            tags.append("mi")
        if edge.granger_p is not None and edge.granger_p <= granger_alpha:
            tags.append("granger")
        if edge.structural:
            tags.append("structural")
        if edge.lag_steps == 0:
            tags.append("contemporaneous")
        edge.evidence = "+".join(tags)
        accepted.append(edge)

    # Resolve two-way pairs. Strongly autocorrelated metrics produce a genuine
    # i->j edge at a real lag AND a mirror j->i edge at lag 0; Granger fires on
    # both, so the acceptance rule alone cannot separate them. When one
    # direction carries temporal precedence and the other does not, the
    # lag-0 mirror is an artifact and is dropped.
    by_pair = {(e.source, e.target): e for e in accepted}
    artifacts: set[tuple[str, str]] = set()
    for (src, dst), edge in by_pair.items():
        reverse = by_pair.get((dst, src))
        if reverse is None:
            continue
        if edge.lag_steps == 0 and reverse.lag_steps > 0 and not edge.structural:
            artifacts.add((src, dst))
    if artifacts:
        log.info("dropped %d lag-0 mirror edge(s)", len(artifacts))
    accepted = [e for e in accepted if (e.source, e.target) not in artifacts]

    # Collapse symmetric contemporaneous pairs. When both directions were
    # accepted at lag 0 (iops <-> throughput on the same disk), the pair carries
    # no directional information — keeping both just spends two of the target's
    # top_k slots on one signal. One canonical direction is kept.
    by_pair = {(e.source, e.target): e for e in accepted}
    collapsed: set[tuple[str, str]] = set()
    for (src, dst), edge in by_pair.items():
        reverse = by_pair.get((dst, src))
        if (
            reverse is not None
            and edge.lag_steps == 0
            and reverse.lag_steps == 0
            and not (edge.structural or reverse.structural)
            and src > dst
        ):
            collapsed.add((src, dst))
    if collapsed:
        log.info("collapsed %d symmetric contemporaneous pair(s)", len(collapsed))
    accepted = [e for e in accepted if (e.source, e.target) not in collapsed]

    # Keep the strongest top_k incoming edges per target: without this the graph
    # saturates and the "dependency map" stops being a map of anything.
    by_target: dict[str, list[Edge]] = {}
    for edge in accepted:
        by_target.setdefault(edge.target, []).append(edge)

    pruned: list[Edge] = []
    for target, edges in by_target.items():
        edges.sort(key=lambda e: (e.strength, e.mutual_information), reverse=True)
        pruned.extend(edges[:top_k])

    pruned.sort(key=lambda e: e.strength, reverse=True)
    log.info("%d edges accepted after pruning", len(pruned))
    return pruned


# ------------------------------------------------------------------- outputs
def to_adjacency(edges: list[Edge], node_ids: list[str]) -> np.ndarray:
    """A_prior[i, j] = strength of edge i -> j, row-normalised."""
    idx = {n: k for k, n in enumerate(node_ids)}
    A = np.zeros((len(node_ids), len(node_ids)), dtype="float32")
    for e in edges:
        if e.source in idx and e.target in idx:
            A[idx[e.source], idx[e.target]] = e.strength

    row_sum = A.sum(axis=1, keepdims=True)
    return np.divide(A, row_sum, out=np.zeros_like(A), where=row_sum > 0)


def save_dependency_map(
    edges: list[Edge], panel: Panel, path: Path, params: dict | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "grid": panel.grid,
            "freq": panel.freq,
            "n_nodes": panel.n_nodes,
            "n_steps": panel.n_steps,
            "n_edges": len(edges),
            "span_start": str(panel.values.index.min()),
            "span_end": str(panel.values.index.max()),
            "params": params or {},
        },
        "nodes": panel.nodes.to_dict(orient="records"),
        "edges": [e.to_dict() for e in edges],
    }
    path.write_text(json.dumps(payload, indent=2, default=str))


def load_dependency_map(path: Path) -> tuple[list[Edge], dict]:
    payload = json.loads(Path(path).read_text())
    edges = [Edge(**e) for e in payload["edges"]]
    return edges, payload["meta"]


def format_report(edges: list[Edge], top: int = 25) -> str:
    if not edges:
        return "No dependencies accepted. Lower graph.xcorr_threshold or collect more history."

    hdr = (
        f"{'source':<32}{'target':<32}{'strength':>9}{'dir':>5}{'lag':>8}"
        f"{'MI':>7}{'granger p':>11}  evidence"
    )
    lines = [hdr, "-" * len(hdr)]
    for e in edges[:top]:
        gp = f"{e.granger_p:.4f}" if e.granger_p is not None else "-"
        lag = f"{e.lag_seconds // 60}m" if e.lag_seconds < 86400 else f"{e.lag_seconds // 86400}d"
        lines.append(
            f"{e.source[:31]:<32}{e.target[:31]:<32}{e.strength:>9.3f}{e.direction:>5}"
            f"{lag:>8}{e.mutual_information:>7.3f}{gp:>11}  {e.evidence}"
        )
    if len(edges) > top:
        lines.append(f"... {len(edges) - top} more edges")
    return "\n".join(lines)
