"""Correctness tests.

The smoke test proves the stages connect. These prove the algorithms recover
structure that is known by construction — most importantly that dependency
discovery finds the right source, the right sign and the right lag on synthetic
data where the answer is not in doubt.

Run:  .venv/bin/python -m pytest tests/ -v
      .venv/bin/python tests/test_pipeline.py      (no pytest needed)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from netraa.features.panel import Panel
from netraa.features.transforms import RobustScaler, make_windows
from netraa.graph import statistical
from netraa.ingest.registry import Registry, node_id, split_node_id
from netraa.ingest.store import read_grid, write_rows

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ fixtures
def synthetic_panel(
    n_steps: int = 1500, lag_a: int = 3, lag_b: int = 5, seed: int = 0
) -> Panel:
    """Panel with dependencies that are true by construction.

        driver         -> dep_positive   at +lag_a, positive
        driver         -> dep_negative   at +lag_b, negative
        independent    -> nothing
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)

    driver = np.sin(t / 24.0) * 10 + np.sin(t / 168.0) * 4 + rng.normal(0, 0.5, n_steps)

    dep_pos = np.zeros(n_steps)
    dep_pos[lag_a:] = 0.9 * driver[:-lag_a]
    dep_pos += rng.normal(0, 0.4, n_steps)

    dep_neg = np.zeros(n_steps)
    dep_neg[lag_b:] = -0.8 * driver[:-lag_b]
    dep_neg += rng.normal(0, 0.4, n_steps)

    independent = rng.normal(0, 1, n_steps)

    index = pd.date_range("2026-01-01", periods=n_steps, freq="5min", tz="UTC")
    values = pd.DataFrame(
        {
            "driver": driver,
            "dep_positive": dep_pos,
            "dep_negative": dep_neg,
            "independent": independent,
        },
        index=index,
    )
    values.index.name = "timestamp"

    nodes = pd.DataFrame(
        [
            {"node_id": "driver", "metric_key": "driver", "dimension": "",
             "entity_type": "SERVICE", "role": "driver", "resource": "service",
             "unit": "count", "transform": "none", "coverage": 1.0},
            {"node_id": "dep_positive", "metric_key": "dep_positive", "dimension": "",
             "entity_type": "HOST", "role": "target", "resource": "cpu",
             "unit": "percent", "transform": "none", "coverage": 1.0},
            {"node_id": "dep_negative", "metric_key": "dep_negative", "dimension": "",
             "entity_type": "HOST", "role": "target", "resource": "memory",
             "unit": "percent", "transform": "none", "coverage": 1.0},
            {"node_id": "independent", "metric_key": "independent", "dimension": "",
             "entity_type": "HOST", "role": "intermediate", "resource": "disk",
             "unit": "count", "transform": "none", "coverage": 1.0},
        ]
    )
    mask = pd.DataFrame(1.0, index=values.index, columns=values.columns)
    return Panel(values=values, mask=mask, nodes=nodes, grid="test", freq="5min")


# --------------------------------------------------------------------- tests
def test_dependency_discovery_recovers_known_structure():
    """The core claim: correct source, correct sign, correct lag."""
    lag_a, lag_b = 3, 5
    panel = synthetic_panel(lag_a=lag_a, lag_b=lag_b)

    edges = statistical.discover(
        panel, max_lag_steps=12, xcorr_threshold=0.4, top_k=4,
        min_overlap=100, run_granger=True,
    )
    found = {(e.source, e.target): e for e in edges}

    pos = found.get(("driver", "dep_positive"))
    assert pos is not None, "failed to find driver -> dep_positive"
    assert pos.lag_steps == lag_a, f"lag {pos.lag_steps}, expected {lag_a}"
    assert pos.direction == "+", f"direction {pos.direction}, expected +"
    assert pos.strength > 0.8, f"strength {pos.strength:.3f} too low"

    neg = found.get(("driver", "dep_negative"))
    assert neg is not None, "failed to find driver -> dep_negative"
    assert neg.lag_steps == lag_b, f"lag {neg.lag_steps}, expected {lag_b}"
    assert neg.direction == "-", f"direction {neg.direction}, expected -"

    # The pure-noise node must not be attached to anything.
    for (src, dst) in found:
        assert "independent" not in (src, dst), (
            f"spurious edge involving the independent node: {src} -> {dst}"
        )

    # The lag-0 mirror of a real lagged edge is an artifact and must be dropped.
    assert ("dep_positive", "driver") not in found, (
        "lag-0 mirror edge survived; direction of causality is ambiguous"
    )
    assert ("dep_negative", "driver") not in found, (
        "lag-0 mirror edge survived; direction of causality is ambiguous"
    )
    print("  recovered lags:", {k: v.lag_steps for k, v in found.items()})


def test_lagged_correlation_is_directional():
    """A[i,j] must mean i leads j, not the reverse."""
    panel = synthetic_panel(lag_a=4)
    corr, lag, _ = statistical.lagged_correlation_scan(panel, 12, 100)
    ids = panel.node_ids
    i, j = ids.index("driver"), ids.index("dep_positive")

    assert lag[i, j] == 4, f"driver->dep lag {lag[i, j]}, expected 4"
    assert abs(corr[i, j]) > abs(corr[j, i]) or lag[j, i] != 4, (
        "reverse direction scored as strongly as the true one"
    )


def test_pairwise_correlation_handles_gaps():
    """Masked positions must not contribute to the correlation."""
    panel = synthetic_panel(n_steps=600)
    values = panel.values.copy()
    mask = panel.mask.copy()
    values.iloc[100:200, 0] = np.nan
    mask.iloc[100:200, 0] = 0.0
    gapped = Panel(values=values, mask=mask, nodes=panel.nodes,
                   grid="test", freq="5min")

    corr, lag, overlap = statistical.lagged_correlation_scan(gapped, 8, 50)
    i, j = gapped.node_ids.index("driver"), gapped.node_ids.index("dep_positive")
    assert np.isfinite(corr[i, j]), "NaN leaked into the correlation"
    assert abs(corr[i, j]) > 0.7, f"gap destroyed a real signal: {corr[i, j]:.3f}"
    assert overlap[i, j] < len(values), "overlap count ignored the mask"


def test_adjacency_orientation_survives_export():
    edges = [
        statistical.Edge("a", "b", 0.9, "+", 2, 600, 0.4, 0.01, False, "xcorr", 100),
        statistical.Edge("b", "c", 0.5, "-", 1, 300, 0.2, 0.04, False, "xcorr", 100),
    ]
    A = statistical.to_adjacency(edges, ["a", "b", "c"])
    assert A[0, 1] > 0 and A[1, 0] == 0, "edge direction flipped in the adjacency"
    assert A[1, 2] > 0 and A[2, 1] == 0
    assert np.allclose(A.sum(axis=1)[[0, 1]], 1.0), "rows not normalised"


def test_registry_has_no_selector_collision():
    """Blocker B3: three names must not silently collapse to one entry."""
    reg = Registry.load(PROJECT_ROOT / "metrics_registry.yaml")
    keys = [m.key for m in reg.metrics]
    assert len(keys) == len(set(keys)), "duplicate registry keys"

    aliases = reg.alias_map()
    for legacy in ("service_instance_cpm", "transaction_volume", "service_cpm"):
        assert aliases[legacy] == "service_cpm", (
            f"{legacy} did not resolve to the canonical node"
        )
    # Every JVM metric must be addressed by the PGI dimension, not the host.
    for m in reg.metrics:
        if m.selector.startswith("builtin:tech.jvm"):
            assert m.entity_type == "PROCESS_GROUP_INSTANCE", (
                f"{m.key} would be filtered on the wrong dimension (blocker B2)"
            )


def test_node_id_round_trip():
    assert node_id("disk_read_iops", "DISK-ABC") == "disk_read_iops|DISK-ABC"
    assert node_id("host_cpu_usage", "") == "host_cpu_usage"
    assert split_node_id("disk_read_iops|DISK-ABC") == ("disk_read_iops", "DISK-ABC")
    assert split_node_id("host_cpu_usage") == ("host_cpu_usage", "")
    # Characters that would break a path or a CSV header.
    assert "/" not in node_id("disk_read_iops", "/dev/sda")


def test_store_write_is_idempotent(tmp_path: Path | None = None):
    """Blocker B6: re-running a backfill must not duplicate or clobber."""
    import tempfile

    tmp = Path(tmp_path or tempfile.mkdtemp())
    index = pd.date_range("2026-01-01", periods=50, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": index,
            "node_id": "host_cpu_usage",
            "metric_key": "host_cpu_usage",
            "dimension": "",
            "entity_type": "HOST",
            "value": np.arange(50, dtype="float64"),
        }
    )

    write_rows(df, tmp, "coarse")
    first = read_grid(tmp, "coarse")
    write_rows(df, tmp, "coarse")
    second = read_grid(tmp, "coarse")

    assert len(first) == len(second) == 50, (
        f"re-write changed row count: {len(first)} -> {len(second)}"
    )

    # A revised value for an existing timestamp must overwrite, not append.
    revised = df.copy()
    revised["value"] = 999.0
    write_rows(revised, tmp, "coarse")
    third = read_grid(tmp, "coarse")
    assert len(third) == 50, "revision appended instead of replacing"
    assert (third["value"] == 999.0).all(), "revision did not take effect"


def test_window_alignment():
    """Y[i, :, j] must be the value at start + input_steps + horizon - 1."""
    T, N = 200, 3
    values = np.arange(T * N, dtype="float32").reshape(T, N)
    mask = np.ones((T, N), dtype="float32")
    calendar = np.zeros((T, 2), dtype="float32")
    horizons = [1, 5, 10]

    X, Y, Ymask, starts = make_windows(
        values, mask, calendar, input_steps=20, horizons=horizons, target_idx=[0, 2]
    )
    assert X.shape == (len(starts), N, 20, 4)
    assert Y.shape == (len(starts), 2, 3)

    for i in (0, 7, len(starts) - 1):
        s = starts[i]
        assert np.allclose(X[i, :, :, 0], values[s : s + 20].T), "input window misaligned"
        for j, h in enumerate(horizons):
            expected = values[s + 20 + h - 1, [0, 2]]
            assert np.allclose(Y[i, :, j], expected), (
                f"target misaligned at window {i}, horizon {h}"
            )


def test_scaler_is_invertible():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {"a": rng.normal(100, 15, 500), "b": rng.exponential(3, 500), "flat": 7.0}
    )
    scaler = RobustScaler.fit(df)
    restored = scaler.inverse_transform_array(
        scaler.transform(df).to_numpy(), list(df.columns)
    )
    assert np.allclose(restored, df.to_numpy(), atol=1e-6), "scaling is not invertible"
    assert scaler.scale["flat"] != 0, "zero-variance column produced a zero scale"


def test_masked_quantile_loss_ignores_gaps():
    import torch

    from netraa.models.stgnn import masked_quantile_loss

    pred = torch.zeros(4, 2, 3, 3)
    target = torch.ones(4, 2, 3)
    mask = torch.ones(4, 2, 3)

    full = masked_quantile_loss(pred, target, mask, [0.1, 0.5, 0.9])

    # Make the masked-out entries wildly wrong; the loss must not move.
    mask_half = mask.clone()
    mask_half[:, :, 1:] = 0.0
    target_poisoned = target.clone()
    target_poisoned[:, :, 1:] = 1e6
    half = masked_quantile_loss(pred, target_poisoned, mask_half, [0.1, 0.5, 0.9])

    assert torch.isclose(full, half, atol=1e-5), (
        f"masked positions leaked into the loss: {float(full)} vs {float(half)}"
    )


def test_stgnn_shapes_and_ablation_parity():
    import torch

    from netraa.models.stgnn import STGNN

    B, N, T, C = 3, 6, 40, 5
    horizons, quantiles = [1, 7, 30], [0.1, 0.5, 0.9]
    x = torch.randn(B, N, T, C)

    for use_graph in (True, False):
        model = STGNN(
            n_nodes=N, n_targets=2, in_channels=C, input_steps=T,
            n_horizons=len(horizons), n_quantiles=len(quantiles),
            target_idx=[0, 3], hidden=16, blocks=2, use_graph=use_graph,
        )
        out = model(x)
        assert out.shape == (B, 2, len(horizons), len(quantiles)), (
            f"bad output shape {tuple(out.shape)} (use_graph={use_graph})"
        )

    # Too-short input must fail loudly rather than silently truncating.
    try:
        STGNN(n_nodes=N, n_targets=1, in_channels=C, input_steps=4,
              n_horizons=1, n_quantiles=1, target_idx=[0], blocks=3, kernel_size=3)
    except ValueError as exc:
        assert "too short" in str(exc)
    else:
        raise AssertionError("accepted an input window shorter than the receptive field")


def test_adaptive_adjacency_is_directed_and_sparse():
    import torch

    from netraa.graph.learned import AdaptiveAdjacency

    torch.manual_seed(0)
    N, k = 10, 3
    prior = np.zeros((N, N), dtype="float32")
    prior[0, 1] = 1.0

    adj_module = AdaptiveAdjacency(N, embed_dim=8, top_k=k, prior=prior)
    A = adj_module()

    assert A.shape == (N, N)
    assert torch.allclose(A.diagonal(), torch.zeros(N), atol=1e-6), "self-loops present"
    assert int((A > 0).sum(dim=1).max()) <= k, "top-k sparsification not applied"

    asym = (A - A.t()).abs().sum().detach().item()
    assert asym > 1e-3, "adjacency is symmetric — direction is not represented"
    assert adj_module.prior_loss().detach().item() > 0, "prior regulariser is inactive"


def test_chronological_split_does_not_shuffle():
    from netraa.features.transforms import chronological_split

    tr, va, te = chronological_split(100, 0.7, 0.15)
    assert tr.max() < va.min() < te.min(), "splits overlap or are out of order"
    assert len(tr) + len(va) + len(te) == 100
    assert np.array_equal(tr, np.sort(tr)), "training indices were shuffled"


# ----------------------------------------------------------------- runner
def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:
            failures.append((fn.__name__, exc))
            print(f"FAIL  {fn.__name__}: {exc}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    for name, exc in failures:
        print(f"  {name}: {type(exc).__name__}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
