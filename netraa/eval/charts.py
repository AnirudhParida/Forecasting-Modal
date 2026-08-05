"""Visualization charts for backtest results and panel data.

Run after backtest:  python -m netraa.cli chart [--grid coarse]

Charts generated
----------------
  model_comparison.png         MAE / sMAPE / pinball / P10-P90 coverage bars
  per_horizon_mae.png          MAE & sMAPE degradation vs horizon, per model
  training_curves.png          epoch-by-epoch train/val loss for STGNN variants
  coverage_heatmap_<grid>.png  null-fraction heatmap across all panel nodes
  dependency_graph.png         network diagram of statistical edges (top 50)
  per_target_comparison.png    per-node STGNN vs climatology MAE improvement
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive; safe on headless servers
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


# ─────────────────────────────────────────────────── Design system ──────────

BG      = "#0d1117"
SURFACE = "#161b22"
BORDER  = "#30363d"
TEXT    = "#c9d1d9"
MUTED   = "#8b949e"

PALETTE = [
    "#58a6ff",   # stgnn_graph    — blue
    "#f78166",   # stgnn_nograph  — coral
    "#3fb950",   # climatology    — green
    "#d2a8ff",   # persistence    — purple
    "#ffa657",   # seasonal_naive — amber
    "#ff7b72",   # drift          — red
]

RESOURCE_COLORS: dict[str, str] = {
    "cpu":     "#58a6ff",
    "memory":  "#3fb950",
    "disk":    "#f78166",
    "network": "#d2a8ff",
    "jvm":     "#ffa657",
    "service": "#ff7b72",
}

MODEL_ORDER = [
    "stgnn_graph", "stgnn_nograph", "climatology",
    "persistence", "seasonal_naive", "drift",
]

MODEL_LABELS: dict[str, str] = {
    "stgnn_graph":    "STGNN + Graph",
    "stgnn_nograph":  "STGNN (no graph)",
    "climatology":    "Climatology",
    "persistence":    "Persistence",
    "seasonal_naive": "Seasonal Naive",
    "drift":          "Drift",
}

_STYLE: dict[str, Any] = {
    "figure.facecolor":   BG,
    "axes.facecolor":     SURFACE,
    "axes.edgecolor":     BORDER,
    "axes.labelcolor":    TEXT,
    "text.color":         TEXT,
    "xtick.color":        MUTED,
    "ytick.color":        MUTED,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "axes.titlesize":     11,
    "axes.labelsize":     9,
    "axes.titlepad":      10,
    "axes.grid":          True,
    "grid.color":         BORDER,
    "grid.alpha":         0.45,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.6,
    "legend.facecolor":   SURFACE,
    "legend.edgecolor":   BORDER,
    "legend.fontsize":    8,
    "figure.titlesize":   13,
    "figure.titleweight": "bold",
}


# ──────────────────────────────────────────── Helpers ───────────────────────

def _ctx():
    return plt.rc_context(_STYLE)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("saved %s", path.name)
    print(f"  checkmark  {path.name}")


def _model_color(name: str) -> str:
    return dict(zip(MODEL_ORDER, PALETTE)).get(name, MUTED)


def _short(nid: str, n: int = 26) -> str:
    """Shorten long entity-dimension node IDs for axis labels."""
    if "|" in nid:
        metric, dim = nid.split("|", 1)
        tail = dim.split("-")[-1][:8]
        nid = f"{metric}|{tail}"
    return nid[:n]


def _resource(nid: str, nodes_meta: list | None = None) -> str:
    """Return the resource family for a node ID."""
    if nodes_meta:
        for n in nodes_meta:
            if n.get("node_id") == nid:
                return str(n.get("resource", "service"))
    name = nid.lower()
    for r in ("cpu", "disk", "network", "jvm", "service"):
        if r in name:
            return r
    if "mem" in name:
        return "memory"
    return "service"


# ─────────────────────────────────────────── Chart functions ────────────────

def chart_model_comparison(data: dict, out_dir: Path) -> None:
    """2x2 bar grid: MAE, sMAPE, pinball loss, P10-P90 interval coverage."""
    scores = data.get("scores", {})
    models = [m for m in MODEL_ORDER if m in scores]
    if not models:
        log.warning("chart_model_comparison: no scores in data")
        return

    specs = [
        ("MAE",                  "mae",              True,  None),
        ("sMAPE (%)",            "smape",            True,  None),
        ("Pinball Loss",         "pinball",          True,  None),
        ("P10-P90 Coverage (%)", "coverage_p10_p90", False, 80.0),
    ]

    with _ctx():
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("Model Performance Comparison", y=1.02)

        for ax, (title, key, lower_better, ref) in zip(axes.flatten(), specs):
            vals, lbls, cols = [], [], []
            for m in models:
                v = scores[m].get(key, float("nan"))
                if np.isfinite(v):
                    vals.append(v)
                    lbls.append(MODEL_LABELS.get(m, m))
                    cols.append(_model_color(m))

            if not vals:
                ax.set_title(title)
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color=MUTED)
                continue

            best = min(vals) if lower_better else max(vals)
            bars = ax.barh(lbls, vals, color=cols, height=0.6, alpha=0.85)

            for bar, v in zip(bars, vals):
                if abs(v - best) < 1e-6 * max(abs(best), 1):
                    bar.set_edgecolor("#ffffff")
                    bar.set_linewidth(1.8)
                offset = max(vals) * 0.006
                ax.text(v + offset, bar.get_y() + bar.get_height() / 2,
                        f"{v:,.1f}", va="center", ha="left",
                        fontsize=7.5, color=TEXT)

            if ref is not None:
                ax.axvline(ref, color="#ffa657", ls="--", lw=1.2, alpha=0.8,
                           label=f"target {ref}%")
                ax.legend(fontsize=7)

            ax.set_title(title)
            ax.set_xlabel(
                "lower is better" if lower_better else "higher is better",
                color=MUTED, fontsize=8)
            ax.invert_yaxis()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.tight_layout()
        _save(fig, out_dir / "model_comparison.png")


def chart_per_horizon(data: dict, out_dir: Path) -> None:
    """Line chart: MAE and sMAPE per forecast horizon, one line per model."""
    scores = data.get("scores", {})
    models = [m for m in MODEL_ORDER if m in scores]
    if not models:
        return

    sample_ph = next(iter(scores.values())).get("per_horizon", {})
    h_vals = sorted(int(h) for h in sample_ph)
    if not h_vals:
        return

    with _ctx():
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Forecast Degradation by Horizon")

        for ax, (key, ylabel) in zip(axes, [("mae", "MAE"), ("smape", "sMAPE (%)")]):
            for m in models:
                ph = scores[m].get("per_horizon", {})
                ys = [ph.get(str(h), {}).get(key, float("nan")) for h in h_vals]
                if all(np.isnan(y) for y in ys):
                    continue
                ax.plot(h_vals, ys,
                        marker="o", ms=5, lw=2.0,
                        color=_model_color(m),
                        label=MODEL_LABELS.get(m, m),
                        alpha=0.9)

            ax.set_xlabel("Horizon (days)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} vs Horizon")
            ax.legend(loc="upper left")
            ax.set_xticks(h_vals)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.tight_layout()
        _save(fig, out_dir / "per_horizon_mae.png")


def chart_training_curves(data: dict, out_dir: Path) -> None:
    """Epoch-by-epoch train/val pinball loss for STGNN variants."""
    training = data.get("training", {})
    variants = [v for v in ("stgnn_graph", "stgnn_nograph") if v in training]
    if not variants:
        log.warning("chart_training_curves: no training data")
        return

    has_history = any("history" in training[v] for v in variants)

    with _ctx():
        ncols = len(variants)
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5), squeeze=False)
        fig.suptitle("Training Loss Curves")

        for ax, v in zip(axes[0], variants):
            info     = training[v]
            best_ep  = info.get("best_epoch", 0)
            best_val = info.get("best_val_loss", float("nan"))
            label    = MODEL_LABELS.get(v, v)

            if has_history and "history" in info:
                history    = info["history"]
                epochs     = [h["epoch"]      for h in history]
                train_loss = [h["train_loss"] for h in history]
                val_loss   = [h["val_loss"]   for h in history]

                ax.plot(epochs, train_loss, color=PALETTE[0],
                        lw=1.8, label="Train", alpha=0.9)
                ax.plot(epochs, val_loss,   color=PALETTE[1],
                        lw=1.8, label="Validation", alpha=0.9)
                ax.fill_between(epochs, train_loss, val_loss,
                                alpha=0.08, color=PALETTE[0])

                if best_ep:
                    ax.axvline(best_ep, color="#ffa657", ls="--",
                               lw=1.2, alpha=0.8, label=f"best (ep {best_ep})")
                    ax.scatter([best_ep], [best_val],
                               color="#ffa657", zorder=5, s=60)

                ax.legend()
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Pinball Loss")
            else:
                # Only summary stats available — minimal bar chart
                ax.bar(["Best Val Loss"], [best_val],
                       color=PALETTE[0], alpha=0.8)
                ax.set_ylabel("Loss")
                ax.text(0.5, 0.92,
                        f"best epoch {best_ep} of {info.get('epochs_run','?')} run",
                        ha="center", transform=ax.transAxes,
                        color=MUTED, fontsize=8)

            ax.set_title(label)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.tight_layout()
        _save(fig, out_dir / "training_curves.png")


def chart_coverage_heatmap(panel: Any, out_dir: Path) -> None:
    """Null-fraction heatmap: rows = nodes sorted by coverage, cols = time.

    ``panel`` must be a ``netraa.features.panel.Panel`` instance.
    """
    mask   = panel.mask    # T x N DataFrame, 1.0 = observed
    values = panel.values  # T x N DataFrame, NaN = missing

    null_frac = (1.0 - mask.mean()).sort_values(ascending=False)
    show      = null_frac.head(35).index.tolist()

    # Downsample time for display
    step     = max(1, len(values) // 250)
    sub_mask = mask[show].iloc[::step]
    sub_vals = values[show].iloc[::step]

    # Per-column 1st-99th percentile normalization for visual contrast
    sub_norm = sub_vals.copy()
    for col in sub_norm.columns:
        c = sub_norm[col].dropna()
        if len(c) > 1:
            lo, hi = c.quantile(0.01), c.quantile(0.99)
            if hi > lo:
                sub_norm[col] = (sub_norm[col] - lo) / (hi - lo)
    # Keep missing cells visually dark
    sub_norm[sub_mask == 0] = np.nan

    cmap = plt.cm.plasma
    cmap.set_bad(color=BG)

    with _ctx():
        fig, (ax_h, ax_b) = plt.subplots(
            1, 2, figsize=(16, max(5, len(show) * 0.34)),
            gridspec_kw={"width_ratios": [5, 1]},
        )
        fig.suptitle(
            f"Node Coverage Heatmap  grid={panel.grid}  "
            f"(top {len(show)} lowest-coverage nodes)"
        )

        im = ax_h.imshow(
            sub_norm.T.values, aspect="auto",
            cmap=cmap, vmin=0, vmax=1,
            interpolation="nearest",
        )
        ax_h.set_yticks(range(len(show)))
        ax_h.set_yticklabels([_short(n) for n in show], fontsize=6.5)

        n_ticks  = min(8, len(sub_norm))
        tick_pos = np.linspace(0, len(sub_norm) - 1, n_ticks, dtype=int)
        tick_lbl = [str(sub_norm.index[i])[:10] for i in tick_pos]
        ax_h.set_xticks(tick_pos)
        ax_h.set_xticklabels(tick_lbl, rotation=30, ha="right", fontsize=7)
        ax_h.set_xlabel("Time")
        ax_h.set_title("Value (plasma scale, black = null)")
        plt.colorbar(im, ax=ax_h, fraction=0.02, pad=0.01, label="norm. value")

        nf      = null_frac[show].values * 100
        bar_col = [
            "#ef4444" if v > 50 else
            "#ffa657" if v > 40 else
            "#3fb950"
            for v in nf
        ]
        ax_b.barh(range(len(show)), nf, color=bar_col, height=0.7)
        ax_b.axvline(20, color=MUTED,    ls="--", lw=0.8, alpha=0.7, label="old 20% floor")
        ax_b.axvline(40, color="#ffa657", ls="--", lw=0.8, alpha=0.7, label="new 40% floor")
        ax_b.set_yticks(range(len(show)))
        ax_b.set_yticklabels([])
        ax_b.set_xlabel("Null %")
        ax_b.set_xlim(0, 100)
        ax_b.set_title("Null %")
        ax_b.legend(fontsize=7)
        ax_b.spines["top"].set_visible(False)
        ax_b.spines["right"].set_visible(False)

        fig.tight_layout()
        _save(fig, out_dir / f"coverage_heatmap_{panel.grid}.png")


def chart_dependency_graph(dep_map_path: Path, out_dir: Path) -> None:
    """Network diagram of top-50 statistical dependency edges."""
    payload    = json.loads(dep_map_path.read_text())
    edges_raw  = payload.get("edges", [])
    nodes_meta = payload.get("nodes", [])
    if not edges_raw:
        log.warning("dependency map has no edges: %s", dep_map_path)
        return

    top = sorted(edges_raw, key=lambda e: e.get("strength", 0), reverse=True)[:50]

    if HAS_NX:
        _dep_graph_nx(top, nodes_meta, dep_map_path.stem, out_dir)
    else:
        _dep_graph_matrix(top, nodes_meta, dep_map_path.stem, out_dir)


def _dep_graph_nx(edges_raw: list, nodes_meta: list, title: str, out_dir: Path) -> None:
    import networkx as nx

    G = nx.DiGraph()
    for e in edges_raw:
        G.add_edge(e["source"], e["target"],
                   weight=float(e.get("strength", 0.5)))

    nodes     = list(G.nodes())
    n_col     = [RESOURCE_COLORS.get(_resource(n, nodes_meta), MUTED) for n in nodes]
    n_size    = [250 + G.degree(n) * 90 for n in nodes]
    e_weights = [G[u][v]["weight"] for u, v in G.edges()]

    with _ctx():
        fig, ax = plt.subplots(figsize=(16, 12))
        fig.suptitle(f"Statistical Dependency Graph  {title}  (top 50 edges)")

        try:
            pos = nx.spring_layout(G, k=2.8, seed=42, iterations=120)
        except Exception:
            pos = nx.circular_layout(G)

        nx.draw_networkx_nodes(G, pos, ax=ax,
                               nodelist=nodes, node_color=n_col,
                               node_size=n_size, alpha=0.88)
        nx.draw_networkx_edges(G, pos, ax=ax,
                               width=[1.0 + 2.5 * w for w in e_weights],
                               alpha=[0.3 + 0.5 * w for w in e_weights],
                               edge_color="#58a6ff", arrows=True,
                               arrowsize=10,
                               connectionstyle="arc3,rad=0.08",
                               min_source_margin=12, min_target_margin=12)
        nx.draw_networkx_labels(G, pos, {n: _short(n, 18) for n in nodes},
                                ax=ax, font_size=6, font_color=TEXT)

        legend = [
            plt.Line2D([0], [0], marker="o", color="none",
                       markerfacecolor=c, markersize=9, label=r.capitalize())
            for r, c in RESOURCE_COLORS.items()
            if any(_resource(n, nodes_meta) == r for n in nodes)
        ]
        ax.legend(handles=legend, loc="upper right", framealpha=0.9)
        ax.axis("off")
        fig.tight_layout()
        _save(fig, out_dir / "dependency_graph.png")


def _dep_graph_matrix(edges_raw: list, nodes_meta: list, title: str, out_dir: Path) -> None:
    """Adjacency heatmap fallback when networkx is unavailable."""
    all_nodes = sorted(
        {e["source"] for e in edges_raw} | {e["target"] for e in edges_raw}
    )
    idx = {n: i for i, n in enumerate(all_nodes)}
    mat = np.zeros((len(all_nodes), len(all_nodes)))
    for e in edges_raw:
        i = idx.get(e["source"], -1)
        j = idx.get(e["target"], -1)
        if i >= 0 and j >= 0:
            mat[i, j] = e.get("strength", 0.5)

    with _ctx():
        fig, ax = plt.subplots(figsize=(14, 12))
        fig.suptitle(f"Dependency Matrix  {title}  (install networkx for graph layout)")
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
        lbls = [_short(n, 22) for n in all_nodes]
        ax.set_xticks(range(len(all_nodes)))
        ax.set_xticklabels(lbls, rotation=90, fontsize=6)
        ax.set_yticks(range(len(all_nodes)))
        ax.set_yticklabels(lbls, fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.02, label="Edge strength")
        ax.set_title("Source (rows) -> Target (cols)")
        fig.tight_layout()
        _save(fig, out_dir / "dependency_graph.png")


def chart_per_target(data: dict, out_dir: Path) -> None:
    """Horizontal bars: per-node MAE improvement of STGNN over climatology."""
    per_target = data.get("per_target", {})
    stgnn_pt   = per_target.get("stgnn_graph", {})
    clim_pt    = per_target.get("climatology",  {})
    if not stgnn_pt or not clim_pt:
        log.warning("chart_per_target: per_target data missing")
        return

    rows = []
    for t in stgnn_pt:
        if t not in clim_pt:
            continue
        sm = stgnn_pt[t].get("mae", float("nan"))
        cm = clim_pt[t].get("mae",  float("nan"))
        if np.isfinite(sm) and np.isfinite(cm) and cm > 1e-9:
            rows.append((t, (cm - sm) / cm * 100))

    if not rows:
        log.warning("chart_per_target: no comparable rows")
        return

    rows.sort(key=lambda x: x[1])
    names    = [_short(r[0]) for r in rows]
    imps     = [r[1] for r in rows]
    cols     = ["#3fb950" if v > 0 else "#ef4444" for v in imps]
    n_better = sum(1 for v in imps if v > 0)

    with _ctx():
        fig, ax = plt.subplots(figsize=(12, max(6, len(rows) * 0.30)))
        fig.suptitle(
            "STGNN vs Climatology  Per-Target MAE Improvement\n"
            "(green = STGNN wins, red = climatology wins)"
        )
        ax.barh(names, imps, color=cols, height=0.7, alpha=0.85)
        ax.axvline(0, color=TEXT, lw=1.0, alpha=0.5)
        ax.set_xlabel("MAE Improvement over Climatology (%)")
        ax.text(0.02, 0.98,
                f"STGNN wins: {n_better} / {len(rows)} nodes",
                transform=ax.transAxes, va="top", ha="left",
                color=TEXT, fontsize=9)
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        _save(fig, out_dir / "per_target_comparison.png")


# ─────────────────────────────────────────────── Entry point ────────────────

def generate_all(
    backtest_json: Path,
    panel: Any,
    graph_dir: Path,
    out_dir: Path,
    grid: str = "coarse",
) -> None:
    """Generate all charts from a completed backtest run.

    Parameters
    ----------
    backtest_json
        JSON file saved by ``netraa backtest``.
    panel
        Loaded ``Panel`` object (from ``Panel.load``), or None to skip heatmap.
    graph_dir
        Directory containing ``dependency_map_*.json`` files.
    out_dir
        Output directory for PNG files (created if missing).
    grid
        Grid name used when looking for a dependency map.
    """
    if not HAS_MPL:
        raise RuntimeError(
            "matplotlib is required for charts.\n"
            "Install:  pip install 'matplotlib>=3.7'\n"
            "Optional: pip install 'networkx>=3.0'  (for graph network layout)"
        )

    if not HAS_NX:
        log.info(
            "networkx not installed — dependency chart uses adjacency matrix fallback. "
            "For the network layout install: pip install 'networkx>=3.0'"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating charts -> {out_dir}/\n")

    data: dict = {}
    if backtest_json.exists():
        data = json.loads(backtest_json.read_text())
    else:
        log.warning(
            "backtest JSON not found: %s  --  run `netraa backtest` first",
            backtest_json,
        )

    if data:
        chart_model_comparison(data, out_dir)
        chart_per_horizon(data, out_dir)
        chart_training_curves(data, out_dir)
        chart_per_target(data, out_dir)

    if panel is not None:
        chart_coverage_heatmap(panel, out_dir)

    candidates = [
        graph_dir / f"dependency_map_{grid}.json",
        graph_dir / "dependency_map_coarse.json",
        graph_dir / "dependency_map_fine.json",
    ]
    for dep_path in candidates:
        if dep_path.exists():
            chart_dependency_graph(dep_path, out_dir)
            break
    else:
        log.warning("no dependency map found in %s", graph_dir)

    print(f"\nAll charts saved -> {out_dir}/")
