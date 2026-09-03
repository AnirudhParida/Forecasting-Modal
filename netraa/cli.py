"""Netraa CLI.

Pipeline order:

    netraa probe       measure what the tenant actually retains
    netraa topology    resolve host/service -> PGI, disk, service-method IDs
    netraa validate    check every registry metric before spending a backfill
    netraa backfill    pull history into the Parquet store
    netraa panel       build the (T x N) panel + mask
    netraa graph       (a) statistical dependency map — no training
    netraa backtest    (b) train ST-GNN + ablation + baselines, score them
    netraa smoke       exercise the whole chain on the legacy CSVs, offline
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import PROJECT_ROOT, load_config


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _client(cfg):
    from .ingest.dynatrace_client import DynatraceClient

    return DynatraceClient(cfg.base_url, cfg.require_token())


def _registry(cfg):
    from .ingest.registry import Registry

    return Registry.load(cfg.registry_path)


def _topology(cfg):
    from .ingest.topology import Topology

    if not cfg.topology_path.exists():
        raise SystemExit(
            f"{cfg.topology_path} not found. Run `netraa topology` first."
        )
    return Topology.load(cfg.topology_path)


def _load_panel(cfg, grid: str):
    """Load a saved panel, with a next-step hint instead of a stack trace."""
    from .features.panel import Panel

    meta = cfg.panel_dir / f"{grid}_meta.json"
    if not meta.exists():
        raise SystemExit(
            f"no panel found for grid={grid} ({meta} missing).\n"
            f"Run: python -m netraa.cli panel --grid {grid}"
        )
    return Panel.load(cfg.panel_dir, grid)


# --------------------------------------------------------------------- probe
def cmd_probe(args, cfg) -> int:
    from .ingest import probe

    registry = _registry(cfg)
    ref = registry.by_key("host_cpu_usage")
    selector = ref.build_selector()  # probe only needs a valid selector; no entity filter

    with _client(cfg) as client:
        rows = probe.probe_retention(client, selector)

    rec = probe.recommend(rows)
    print(probe.format_report(rows, rec))
    probe.save_report(rows, rec, cfg.data_dir / "retention_probe.json")
    print(f"\nsaved -> {cfg.data_dir / 'retention_probe.json'}")
    return 0


# ------------------------------------------------------------------ topology
def cmd_topology(args, cfg) -> int:
    from .ingest import topology as topo_mod

    with _client(cfg) as client:
        topo = topo_mod.resolve(client, cfg.host_id, cfg.service_id)

    topo.save(cfg.topology_path)
    print(f"host:    {topo.host_id}")
    print(f"service: {topo.service_id}")
    for etype, ids in sorted(topo.entities.items()):
        print(f"  {etype:<26} {len(ids)}")
        for eid in ids[:4]:
            print(f"      {eid}")
        if len(ids) > 4:
            print(f"      ... {len(ids) - 4} more")
    for warn in topo.warnings:
        print(f"  WARNING: {warn}")
    print(f"\nsaved -> {cfg.topology_path}")
    return 0


# ------------------------------------------------------------------ validate
def cmd_validate(args, cfg) -> int:
    from .ingest import validate as val_mod

    registry, topo = _registry(cfg), _topology(cfg)
    with _client(cfg) as client:
        rows = val_mod.validate_registry(client, registry, topo)

    print(val_mod.format_report(rows))
    out = cfg.data_dir / "validation_report.json"
    val_mod.save_report(rows, out)
    print(f"\nsaved -> {out}")
    return 1 if any(r.failed for r in rows) else 0


# ------------------------------------------------------------------ backfill
def cmd_backfill(args, cfg) -> int:
    from .ingest.backfill import backfill, collect_incremental

    registry, topo = _registry(cfg), _topology(cfg)
    grid = cfg.grids[args.grid]

    with _client(cfg) as client:
        if args.incremental:
            summary = collect_incremental(client, registry, topo, grid, cfg.raw_dir)
        else:
            summary = backfill(
                client,
                registry,
                topo,
                grid,
                cfg.raw_dir,
                only_keys=args.only.split(",") if args.only else None,
            )

    print(summary.format())
    return 0


# ------------------------------------------------------------------ ingest-upi
def cmd_ingest_upi(args, cfg) -> int:
    from .ingest.upi_csv_ingest import ingest_upi_directory
    from pathlib import Path

    registry = _registry(cfg)
    data_dir = Path(args.data_dir)
    raw_dir = cfg.raw_dir
    grid = args.grid

    # Support comma-separated hosts for multi-host shared model
    hosts_raw = args.host
    hosts = [h.strip() for h in hosts_raw.split(",") if h.strip()]
    host_arg = hosts if len(hosts) > 1 else hosts[0]

    print(f"Ingesting UPI CSV/XLSX metrics from {data_dir}")
    if isinstance(host_arg, list):
        print(f"  Hosts ({len(host_arg)}): {', '.join(host_arg)}")
        print(f"  Mode: MULTI-HOST (each host gets its own node_id: metric__HOSTNAME)")
    else:
        print(f"  Host: {host_arg}")
        print(f"  Mode: SINGLE-HOST")

    written = ingest_upi_directory(data_dir, registry, raw_dir, host_arg, grid)
    print(f"Ingestion complete: {written:,} rows written to store at {raw_dir}")
    return 0


# -------------------------------------------------------------------- status
def cmd_status(args, cfg) -> int:
    from .ingest.store import coverage

    for name in cfg.grids:
        cov = coverage(cfg.raw_dir, name)
        print(f"\n=== grid={name} ===")
        if cov.empty:
            print("  (empty — run `netraa backfill`)")
            continue
        print(f"  nodes: {len(cov)}   rows: {cov['points'].sum():,}")
        print(f"  span:  {cov['first'].min()} -> {cov['last'].max()}")
        print(cov.head(args.top).to_string(index=False))
    return 0


# --------------------------------------------------------------------- panel
def _build_panel(cfg, grid_name: str, long_df=None):
    from .features.panel import build_panel

    grid = cfg.grids[grid_name]
    registry = _registry(cfg)
    source = getattr(grid, "source_grid", None)
    return build_panel(
        raw_dir=cfg.raw_dir,
        grid=grid_name,
        freq=grid.pandas_freq,
        registry=registry,
        source_grid=source,
        min_coverage=cfg.raw.get("min_coverage", 0.20),
        min_tail_coverage=cfg.raw.get("min_tail_coverage", 0.60),
        min_observed_steps=cfg.raw.get("min_observed_steps", 90),
        min_nonzero_fraction=cfg.raw.get("min_nonzero_fraction", 0.0),
        long_df=long_df,
    )


def cmd_panel(args, cfg) -> int:
    panel = _build_panel(cfg, args.grid)
    panel.save(cfg.panel_dir)
    print(panel.describe())
    print(f"\nsaved -> {cfg.panel_dir}/{args.grid}_*.parquet")
    return 0


# --------------------------------------------------------------------- graph
def cmd_graph(args, cfg) -> int:
    from .features.transforms import apply_node_transforms, clip_outliers
    from .graph import statistical

    grid = args.grid or cfg.graph.grid
    panel = clip_outliers(apply_node_transforms(_load_panel(cfg, grid)))

    params = {
        "max_lag_steps": cfg.graph.max_lag_steps,
        "xcorr_threshold": cfg.graph.xcorr_threshold,
        "granger_alpha": cfg.graph.granger_alpha,
        "mi_threshold": cfg.graph.mi_threshold,
        "top_k": cfg.graph.top_k,
        "min_overlap": min(cfg.graph.min_overlap, max(10, panel.n_steps // 3)),
    }
    edges = statistical.discover(panel, run_granger=not args.no_granger, **params)

    print(statistical.format_report(edges, top=args.top))
    out = cfg.graph_dir / f"dependency_map_{grid}.json"
    statistical.save_dependency_map(edges, panel, out, params)
    print(f"\n{len(edges)} edges -> {out}")
    return 0


# ------------------------------------------------------------------ backtest
def cmd_backtest(args, cfg) -> int:
    from .eval import backtest as bt
    from .graph import statistical
    from .models import dataset as ds_mod
    from .models.train import save_artifacts

    grid = args.grid or cfg.forecast.grid
    panel = _load_panel(cfg, grid)

    ds = ds_mod.prepare(
        panel,
        input_steps=args.input_steps or cfg.forecast.input_steps,
        horizons=cfg.forecast.horizons,
    )
    print(ds.summary(), "\n")

    # Prior from the statistical map, aligned to this panel's node order.
    adj_prior = None
    for candidate in (
        cfg.graph_dir / f"dependency_map_{cfg.graph.grid}.json",
        cfg.graph_dir / f"dependency_map_{grid}.json",
    ):
        if candidate.exists():
            edges, _ = statistical.load_dependency_map(candidate)
            adj_prior = statistical.to_adjacency(edges, ds.node_ids)
            applied = int((adj_prior > 0).sum())
            print(f"prior: {applied} edges from {candidate.name}")
            if applied < len(edges) // 2:
                print(
                    f"WARNING: only {applied} of {len(edges)} edges in "
                    f"{candidate.name} reference nodes present in this panel. "
                    f"The map was built from a different (probably stale) panel "
                    f"— rerun `netraa panel` and then `netraa graph` so the "
                    f"prior and the forecaster see the same node set."
                )
            print()
            break
    if adj_prior is None:
        print("prior: none found — run `netraa graph` first for the (a)->(b) link\n")

    season = cfg.raw.get("season_steps", {}).get(grid, 7)
    result, trained_models = bt.run(
        ds,
        cfg.model,
        cfg.forecast.quantiles,
        adj_prior=adj_prior,
        season=season,
        run_ablation=not args.no_ablation,
    )

    print(result.format())
    out = cfg.artifacts_dir / f"backtest_{grid}.json"
    result.save(out)
    for name, tr in trained_models.items():
        if name == "stgnn_graph":
            save_artifacts(tr, ds, cfg.artifacts_dir, cfg.forecast.quantiles, tag="stgnn")
            save_artifacts(tr, ds, cfg.artifacts_dir, cfg.forecast.quantiles, tag="stgnn_graph")
        else:
            save_artifacts(tr, ds, cfg.artifacts_dir, cfg.forecast.quantiles, tag=name)
    print(f"\nsaved -> {out}")
    return 0


# ---------------------------------------------------------------------- chart
def cmd_chart(args, cfg) -> int:
    """Generate PNG charts from a completed backtest run."""
    from .eval.charts import generate_all

    grid = args.grid or cfg.forecast.grid

    # Load the panel if it exists; None means coverage heatmap is skipped.
    panel = None
    meta = cfg.panel_dir / f"{grid}_meta.json"
    if meta.exists():
        panel = _load_panel(cfg, grid)
    else:
        print(f"panel not found for grid={grid} -- coverage heatmap will be skipped")

    backtest_json = cfg.artifacts_dir / f"backtest_{grid}.json"
    generate_all(
        backtest_json=backtest_json,
        panel=panel,
        graph_dir=cfg.graph_dir,
        out_dir=cfg.artifacts_dir / "charts",
        grid=grid,
    )
    return 0


# --------------------------------------------------------------------- smoke
def cmd_smoke(args, cfg) -> int:
    """Offline end-to-end run on the legacy CSVs.

    These files hold 11 rows each. This proves the code paths connect; it says
    nothing about model quality and the printed numbers are not results.
    """
    from .eval import backtest as bt
    from .features.panel import build_panel
    from .features.transforms import apply_node_transforms, clip_outliers
    from .graph import statistical
    from .ingest.legacy_csv import load_legacy_csvs
    from .models import dataset as ds_mod

    registry = _registry(cfg)

    print("=" * 78)
    print("SMOKE TEST — legacy CSVs, ~11 rows each. Numbers below are meaningless.")
    print("=" * 78)

    long_df, unmapped = load_legacy_csvs(PROJECT_ROOT, registry)
    print(f"\n[1] legacy import: {len(long_df):,} rows, "
          f"{long_df['node_id'].nunique()} distinct nodes")
    if unmapped:
        print(f"    unmapped columns: {', '.join(unmapped)}")

    collapsed = (
        long_df.groupby("node_id")["value"].size().sort_values(ascending=False)
    )
    print("    nodes after canonicalisation (B3/B4 fix):")
    for nid, n in collapsed.items():
        print(f"      {nid:<34} {n:>4} obs")

    panel = build_panel(
        raw_dir=cfg.raw_dir,
        grid="smoke",
        freq="1min",
        registry=registry,
        min_coverage=0.10,
        long_df=long_df,
    )
    print(f"\n[2] panel\n{panel.describe()}")

    prepped = clip_outliers(apply_node_transforms(panel))
    neg = (panel.values < 0).sum().sum()
    neg_after = (prepped.values < 0).sum().sum()
    print(f"\n[3] transforms: negative values {neg} -> {neg_after} (B5 abs applied)")

    edges = statistical.discover(
        prepped,
        max_lag_steps=3,
        xcorr_threshold=0.5,
        mi_threshold=0.10,
        top_k=3,
        min_overlap=5,
        run_granger=False,
    )
    print(f"\n[4] dependency discovery: {len(edges)} edges")
    print(statistical.format_report(edges, top=10))

    ds = ds_mod.prepare(panel, input_steps=4, horizons=[1], train_frac=0.6, val_frac=0.2)
    print(f"\n[5] dataset\n{ds.summary()}")

    from .config import ModelConfig

    tiny = ModelConfig(
        hidden=8, blocks=1, kernel_size=2, dropout=0.0,
        node_embed_dim=4, epochs=5, batch_size=4, patience=5,
    )
    adj_prior = statistical.to_adjacency(edges, ds.node_ids)
    result, _ = bt.run(
        ds, tiny, cfg.forecast.quantiles, adj_prior=adj_prior, season=2
    )
    print(f"\n[6] backtest\n{result.format()}")

    print("\n" + "=" * 78)
    print("SMOKE TEST PASSED — every stage ran. Now run a real backfill.")
    print("=" * 78)
    return 0


# ---------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="netraa", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", default="configs/v1.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("probe", help="measure tenant retention per resolution")
    sub.add_parser("topology", help="resolve related entity IDs")
    sub.add_parser("validate", help="check every registry metric")

    b = sub.add_parser("backfill", help="pull history into the store")
    b.add_argument("--grid", default="coarse", choices=["fine", "coarse"])
    b.add_argument("--incremental", action="store_true")
    b.add_argument("--only", default="", help="comma-separated metric keys")

    s = sub.add_parser("status", help="store coverage per node")
    s.add_argument("--top", type=int, default=15)

    pa = sub.add_parser("panel", help="build the T x N panel")
    pa.add_argument("--grid", default="coarse", choices=["fine", "coarse"])

    g = sub.add_parser("graph", help="(a) statistical dependency map")
    g.add_argument("--grid", default=None)
    g.add_argument("--top", type=int, default=25)
    g.add_argument("--no-granger", action="store_true")

    bt_ = sub.add_parser("backtest", help="(b) train + ablation + baselines")
    bt_.add_argument("--grid", default=None)
    bt_.add_argument("--input-steps", type=int, default=None)
    bt_.add_argument("--no-ablation", action="store_true")

    sub.add_parser("smoke", help="offline end-to-end run on the legacy CSVs")

    ch = sub.add_parser("chart", help="generate PNG charts from a backtest run")
    ch.add_argument("--grid", default=None,
                    help="grid to visualise (default: forecast.grid from config)")

    ing = sub.add_parser("ingest-upi", help="ingest local UPI CSV/XLSX metrics")
    ing.add_argument("--data-dir", required=True, help="directory path containing the metrics files")
    ing.add_argument(
        "--host", required=True,
        help="host identifier(s) to filter columns. Single host: '10.51.1.103'. "
             "Multi-host shared model: '10.50.98.26,10.78.33.83' (comma-separated)."
    )
    ing.add_argument("--grid", default="coarse", choices=["fine", "coarse"])

    return p


COMMANDS = {
    "probe":     cmd_probe,
    "topology":  cmd_topology,
    "validate":  cmd_validate,
    "backfill":  cmd_backfill,
    "ingest-upi": cmd_ingest_upi,
    "status":    cmd_status,
    "panel":     cmd_panel,
    "graph":     cmd_graph,
    "backtest":  cmd_backtest,
    "chart":     cmd_chart,
    "smoke":     cmd_smoke,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)

    try:
        cfg = load_config(args.config)
        return COMMANDS[args.command](args, cfg)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except SystemExit as exc:
        # Deliberate, already-explained exits (missing topology, missing panel).
        if exc.code not in (0, None):
            print(f"\n{exc.code}", file=sys.stderr)
            return 1
        raise
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        # Expected operator errors: show the message and the next step, not a
        # 40-line traceback. Re-run with -v for the full trace.
        print(f"\nerror: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        print("(re-run with -v for the full traceback)", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
