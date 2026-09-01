"""
impact_analysis.py  —  What-If Impact Analyzer
================================================
Given a metric that changes by X%, propagate the effect across the entire
dependency graph and report which metrics are impacted, by how much, and
through which path.

Usage
-----
  Interactive (prompts you):
      python impact_analysis.py

  Single-shot CLI:
      python impact_analysis.py --metric disk_write_iops --change 20
      python impact_analysis.py --metric host_cpu_usage  --change -15
      python impact_analysis.py --metric service_cpm     --change 50 --layers 3 --chart

  Options:
      --metric    Partial or full metric name (fuzzy matched)
      --change    % change (positive = increase, negative = decrease)
      --layers    Propagation depth  [default: 3]
      --chart     Save a bar chart PNG to artifacts/
      --graph     Path to graph JSON  [default: artifacts/stgnn_learned_graph.json]
      --threshold Minimum impact% to show in output [default: 0.5]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT      = Path(__file__).parent
GRAPH_F   = ROOT / "artifacts" / "stgnn_learned_graph.json"
SCALER_F  = ROOT / "artifacts" / "stgnn_scaler.json"
ART_DIR   = ROOT / "artifacts"

# ── colour helpers (ANSI, gracefully degraded on Windows) ─────────────────────
try:
    import os; _colours = os.get_terminal_size().columns > 0
except Exception: _colours = False

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _colours else text

RED    = lambda t: _c("91", t)
YELLOW = lambda t: _c("93", t)
GREEN  = lambda t: _c("92", t)
CYAN   = lambda t: _c("96", t)
BOLD   = lambda t: _c("1",  t)
DIM    = lambda t: _c("2",  t)
RESET  = lambda t: t

# ── human-readable short names ────────────────────────────────────────────────
SHORT: dict[str, str] = {
    "host_cpu_usage":                   "CPU Usage%",
    "host_cpu_load1":                   "CPU Load-1m",
    "host_mem_usage":                   "Memory Usage%",
    "host_mem_available":               "Memory Free",
    "host_mem_buff_cache":              "Buffer/Cache",
    "host_mem_total":                   "Memory Total",
    "service_cpm":                      "Service CPM",
    "instance_traffic":                 "Instance Traffic",
    "database_access_cpm":              "DB Access CPM",
    "jvm_gc_collection_count":          "GC Count",
    "jvm_gc_collection_time":           "GC Time",
    "jvm_memory_heap_used":             "Heap Used",
    "jvm_memory_heap_max":              "Heap Max",
    "jvm_memory_pool_used":             "Mem Pool Used",
    "jvm_process_cpu":                  "JVM CPU%",
    "jvm_thread_live_count":            "Thread Count",
    "host_cpu_iowait":                  "CPU IO-Wait%",
    "host_net_rx_bytes":                "NIC Rx Bytes",
    "host_net_tx_bytes":                "NIC Tx Bytes",
    "host_net_rx_packets":              "NIC Rx Packets",
    "host_net_tx_packets":              "NIC Tx Packets",
    "host_net_rx_errors":               "NIC Rx Errors",
    "host_net_tx_errors":               "NIC Tx Errors",
    "host_net_rx_dropped":              "NIC Rx Dropped",
    "host_net_tx_dropped":              "NIC Tx Dropped",
    "host_sessions_new":                "Sessions New",
    "host_sessions_reset":              "Sessions Reset",
    "host_availability":                "Host Availability",
}

def _short(node_id: str) -> str:
    base = node_id.split("|")[0]
    suffix = ""
    if "|" in node_id:
        raw = node_id.split("|")[1]
        # Shorten entity IDs: keep last 6 chars
        suffix = f"[…{raw[-6:]}]"
    if base in SHORT:
        return SHORT[base] + suffix
    return base.replace("_", " ").title() + suffix


# ══════════════════════════════════════════════════════════════════════════════
# Graph loading
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Graph:
    node_ids: list[str]
    A: np.ndarray        # (N, N) row-normalised adjacency; A[i,j] = i→j weight
    A_raw: np.ndarray    # (N, N) un-normalised (raw weights, for display)
    in_prior: np.ndarray # (N, N) bool — edge confirmed by statistical test too
    scaler_center: dict[str, float] = field(default_factory=dict)
    scaler_scale:  dict[str, float] = field(default_factory=dict)

    @property
    def N(self) -> int:
        return len(self.node_ids)

    def idx(self, node_id: str) -> int:
        return self.node_ids.index(node_id)


def load_graph(path: Path, scaler_path: Path | None = None) -> Graph:
    data   = json.loads(path.read_text())
    edges  = data["edges"]

    # Collect all unique node names
    nodes: list[str] = []
    seen:  set[str]  = set()
    for e in edges:
        for k in ("source", "target"):
            if e[k] not in seen:
                nodes.append(e[k])
                seen.add(e[k])
    nodes.sort()
    idx = {n: i for i, n in enumerate(nodes)}
    N   = len(nodes)

    A_raw    = np.zeros((N, N), dtype=float)
    in_prior = np.zeros((N, N), dtype=bool)
    for e in edges:
        i, j = idx[e["source"]], idx[e["target"]]
        A_raw[i, j]    = e["weight"]
        in_prior[i, j] = e.get("in_prior", False)

    # Row-normalise for propagation
    row_sum = A_raw.sum(axis=1, keepdims=True)
    A = np.divide(A_raw, row_sum, out=np.zeros_like(A_raw), where=row_sum > 0)

    # Load scaler if available
    center, scale = {}, {}
    if scaler_path and scaler_path.exists():
        sc = json.loads(scaler_path.read_text())
        center = sc.get("center", {})
        scale  = sc.get("scale",  {})

    return Graph(node_ids=nodes, A=A, A_raw=A_raw, in_prior=in_prior,
                 scaler_center=center, scaler_scale=scale)


# ══════════════════════════════════════════════════════════════════════════════
# Fuzzy metric matching
# ══════════════════════════════════════════════════════════════════════════════

def fuzzy_match(query: str, node_ids: list[str], top: int = 8) -> list[str]:
    """Return node_ids that contain the query string (case-insensitive)."""
    q = query.lower().replace("-", "_").replace(" ", "_")
    exact   = [n for n in node_ids if n.lower() == q]
    if exact:
        return exact
    prefix  = [n for n in node_ids if n.lower().startswith(q)]
    contain = [n for n in node_ids if q in n.lower() and n not in prefix]
    return (prefix + contain)[:top]


def resolve_metric(query: str, graph: Graph) -> str | None:
    """Resolve a user query to a single node_id. Returns None if ambiguous."""
    matches = fuzzy_match(query, graph.node_ids)
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    print(f"\n  Found {len(matches)} metrics matching {CYAN(repr(query))}:\n")
    for i, m in enumerate(matches, 1):
        prior_marker = " ✓" if any(graph.in_prior[graph.idx(m), :]) or any(graph.in_prior[:, graph.idx(m)]) else ""
        print(f"  {BOLD(str(i)):>4}. {_short(m):<28} {DIM(m)}{prior_marker}")
    print()
    while True:
        raw = input("  Enter number to select: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(matches):
            return matches[int(raw) - 1]
        print("  Please enter a valid number.")


# ══════════════════════════════════════════════════════════════════════════════
# Propagation engine
# ══════════════════════════════════════════════════════════════════════════════

DAMPENING = [1.0, 0.5, 0.25]   # weight multiplier per hop

@dataclass
class ImpactResult:
    source:       str
    delta_pct:    float
    impacts:      list[dict]    # sorted by |total_impact|, descending
    upstream:     list[dict]    # metrics that directly drive the source


def propagate(graph: Graph, source_id: str, delta_pct: float, n_layers: int = 3) -> ImpactResult:
    s   = graph.idx(source_id)
    N   = graph.N
    A   = graph.A

    # ── downstream: what does source affect? ─────────────────────────────────
    hop_contributions = np.zeros((N, n_layers), dtype=float)
    Ak = A.copy()
    for layer in range(n_layers):
        raw_signal          = Ak[s, :] * delta_pct * DAMPENING[min(layer, 2)]
        hop_contributions[:, layer] = raw_signal
        Ak = Ak @ A

    total_signal = hop_contributions.sum(axis=1)

    impacts = []
    for j in range(N):
        if j == s:
            continue
        total = float(total_signal[j])
        if abs(total) < 1e-6:
            continue

        hops = []
        for layer in range(n_layers):
            v = float(hop_contributions[j, layer])
            if abs(v) > 1e-6:
                hops.append({"layer": layer + 1, "value": round(v, 4)})

        direct_weight = float(graph.A_raw[s, j])
        confirmed = bool(graph.in_prior[s, j])

        # Absolute scale estimate (IQR-based)
        iqr   = graph.scaler_scale.get(graph.node_ids[j], None)
        med   = graph.scaler_center.get(graph.node_ids[j], None)
        abs_delta = None
        if iqr and med:
            abs_delta = round(total / 100 * iqr, 4)  # proportional signal × IQR

        impacts.append({
            "metric":         graph.node_ids[j],
            "short":          _short(graph.node_ids[j]),
            "total_pct":      round(total, 4),
            "direct_weight":  round(direct_weight, 4),
            "confirmed":      confirmed,
            "hops":           hops,
            "abs_delta":      abs_delta,
        })

    impacts.sort(key=lambda x: abs(x["total_pct"]), reverse=True)

    # ── upstream: what drives source? ────────────────────────────────────────
    upstream = []
    for i in range(N):
        if i == s:
            continue
        w = float(graph.A_raw[i, s])
        if w > 1e-4:
            upstream.append({
                "metric":    graph.node_ids[i],
                "short":     _short(graph.node_ids[i]),
                "weight":    round(w, 4),
                "confirmed": bool(graph.in_prior[i, s]),
            })
    upstream.sort(key=lambda x: x["weight"], reverse=True)

    return ImpactResult(
        source    = source_id,
        delta_pct = delta_pct,
        impacts   = impacts,
        upstream  = upstream,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Output formatting
# ══════════════════════════════════════════════════════════════════════════════

def _bar(value: float, width: int = 20) -> str:
    """ASCII progress bar, scaled to ±width characters."""
    filled = int(min(abs(value) / 30 * width, width))
    bar    = "█" * filled + "░" * (width - filled)
    return bar


def _impact_colour(pct: float) -> str:
    a = abs(pct)
    if a >= 10: return RED(f"{pct:+.2f}%")
    if a >=  5: return YELLOW(f"{pct:+.2f}%")
    if a >=  1: return GREEN(f"{pct:+.2f}%")
    return DIM(f"{pct:+.2f}%")


def print_results(result: ImpactResult, threshold: float = 0.5) -> None:
    delta_str = f"{result.delta_pct:+.1f}%"
    colour_fn = RED if result.delta_pct > 0 else GREEN
    src_short = _short(result.source)

    print()
    print("═" * 72)
    print(f"  WHAT-IF IMPACT ANALYSIS")
    print(f"  Source  : {BOLD(src_short)}")
    print(f"  Change  : {colour_fn(BOLD(delta_str))}")
    print(f"  Metric  : {DIM(result.source)}")
    print("═" * 72)

    # ── upstream context ──────────────────────────────────────────────────────
    if result.upstream:
        print(f"\n  {CYAN('▲ UPSTREAM — What drives')} {CYAN(BOLD(src_short))} {CYAN('(context only)')}")
        print(f"  {'Influencer':<30} {'Weight':>8}  {'Confirmed?':>10}")
        print("  " + "─" * 54)
        for u in result.upstream[:6]:
            ck = GREEN("✓ stat+model") if u["confirmed"] else DIM("model only")
            print(f"  {u['short']:<30} {u['weight']:>8.3f}  {ck}")
        if len(result.upstream) > 6:
            print(f"  {DIM(f'  ... {len(result.upstream)-6} more upstream influencers')}")

    # ── downstream impacts ────────────────────────────────────────────────────
    visible = [x for x in result.impacts if abs(x["total_pct"]) >= threshold]

    print(f"\n  {CYAN('▼ DOWNSTREAM IMPACTS')}  ({len(visible)} metrics affected above {threshold}% signal threshold)")
    print()
    print(f"  {'Metric':<30} {'Signal Δ':>10}  {'Layer':>6}  {'Evidence':<14}  Bar")
    print("  " + "─" * 72)

    for row in visible:
        pct   = row["total_pct"]
        layer = row["hops"][0]["layer"] if row["hops"] else "?"
        layer_label = {1: "Direct", 2: "2-hop", 3: "3-hop"}.get(layer, f"{layer}-hop")
        ck    = GREEN("stat+model") if row["confirmed"] else DIM("model")
        bar   = _bar(pct)
        print(f"  {row['short']:<30} {_impact_colour(pct):>10}  {layer_label:>6}  {ck:<14}  {bar}")

    # ── hop breakdown for top 5 ───────────────────────────────────────────────
    top5 = visible[:5]
    if any(len(r["hops"]) > 1 for r in top5):
        print(f"\n  {CYAN('Hop breakdown for top impacts:')}")
        for row in top5:
            if len(row["hops"]) <= 1:
                continue
            breakdown = "  +  ".join(
                f"Layer{h['layer']} {h['value']:+.2f}%" for h in row["hops"]
            )
            print(f"  {row['short']:<28}  {breakdown}")

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    print("═" * 72)
    up = [x for x in visible if x["total_pct"] > 0]
    dn = [x for x in visible if x["total_pct"] < 0]
    print(f"  Summary: {BOLD(str(len(up)))} metrics {RED('increase')}  |  "
          f"{BOLD(str(len(dn)))} metrics {GREEN('decrease')}  |  "
          f"{DIM(str(len(result.impacts) - len(visible)))} below threshold")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Chart output
# ══════════════════════════════════════════════════════════════════════════════

def make_chart(result: ImpactResult, art_dir: Path, threshold: float = 0.5) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [chart skipped — matplotlib not installed]")
        return None

    visible = [x for x in result.impacts if abs(x["total_pct"]) >= threshold][:20]
    if not visible:
        return None

    labels = [x["short"] for x in visible]
    values = [x["total_pct"] for x in visible]
    colors = ["#f85149" if v > 0 else "#3fb950" for v in values]
    confirmed = [x["confirmed"] for x in visible]

    BG, SURFACE, TEXT, MUTED, BORDER = "#0d1117","#161b22","#c9d1d9","#8b949e","#30363d"

    fig, ax = plt.subplots(figsize=(12, max(5, len(labels) * 0.45 + 2)), facecolor=BG)
    ax.set_facecolor(SURFACE)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.tick_params(colors=TEXT)

    bars = ax.barh(labels, values, color=colors, alpha=0.85, height=0.65)

    # Hatch confirmed edges
    for bar, conf in zip(bars, confirmed):
        if conf:
            bar.set_hatch("///")
            bar.set_edgecolor("#ffa657")
            bar.set_linewidth(0.8)

    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Propagated signal change (%)", color=MUTED, fontsize=9)

    src_short = _short(result.source)
    delta_str = f"{result.delta_pct:+.1f}%"
    ax.set_title(
        f"Impact Propagation  |  {src_short} {delta_str}  →  downstream metrics\n"
        f"{DIM('Hatched = also confirmed by statistical test')}",
        color=TEXT, fontsize=11, loc="left", pad=10
    )

    for bar, val in zip(bars, values):
        ax.text(
            val + (0.3 if val >= 0 else -0.3), bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f}%", va="center",
            ha="left" if val >= 0 else "right",
            color=TEXT, fontsize=7,
        )

    plt.yticks(fontsize=8, color=TEXT)
    plt.xticks(fontsize=8, color=MUTED)
    plt.tight_layout()

    art_dir.mkdir(parents=True, exist_ok=True)
    safe_name = result.source.replace("|", "_").replace("/", "_")[:40]
    out = art_dir / f"impact_{safe_name}_{result.delta_pct:+.0f}pct.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# JSON export
# ══════════════════════════════════════════════════════════════════════════════

def save_json(result: ImpactResult, art_dir: Path, threshold: float = 0.5) -> Path:
    payload = {
        "source_metric": result.source,
        "source_short":  _short(result.source),
        "delta_pct":     result.delta_pct,
        "interpretation": (
            f"A {result.delta_pct:+.1f}% change in {_short(result.source)} "
            f"propagates the following signal changes through the dependency graph."
        ),
        "upstream_drivers": result.upstream[:8],
        "downstream_impacts": [
            x for x in result.impacts if abs(x["total_pct"]) >= threshold
        ],
    }
    art_dir.mkdir(parents=True, exist_ok=True)
    safe = result.source.replace("|", "_").replace("/", "_")[:40]
    out  = art_dir / f"impact_{safe}_{result.delta_pct:+.0f}pct.json"
    out.write_text(json.dumps(payload, indent=2))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def run_interactive(graph: Graph, art_dir: Path, args) -> None:
    """Loop: ask for metric + change, show results, repeat."""
    print(BOLD("\n  Netraa — What-If Impact Analyzer"))
    print(DIM("  Type a metric name (partial OK), a % change, and see the ripple effect.\n"))
    print(DIM(f"  {graph.N} metrics loaded from graph.  Type 'quit' to exit.\n"))

    while True:
        try:
            # ── metric input ──────────────────────────────────────────────────
            if args.metric:
                query = args.metric
                args.metric = None   # only use once in loop
            else:
                query = input(CYAN("  Metric name › ")).strip()
            if not query or query.lower() in ("quit", "exit", "q"):
                print(DIM("\n  Goodbye.\n"))
                break

            source_id = resolve_metric(query, graph)
            if source_id is None:
                print(RED(f"  No metric found matching '{query}'. Try a shorter term.\n"))
                continue

            # ── change input ──────────────────────────────────────────────────
            if args.change is not None:
                delta_pct = float(args.change)
                args.change = None
            else:
                raw = input(CYAN(f"  % change for '{_short(source_id)}' › ")).strip()
                delta_pct = float(raw.replace("%", "").replace("+", ""))

            # ── propagate ─────────────────────────────────────────────────────
            result = propagate(graph, source_id, delta_pct, n_layers=args.layers)
            print_results(result, threshold=args.threshold)

            # ── export ────────────────────────────────────────────────────────
            json_path = save_json(result, art_dir, threshold=args.threshold)
            print(f"  JSON saved → {json_path}")

            if args.chart:
                chart_path = make_chart(result, art_dir, threshold=args.threshold)
                if chart_path:
                    print(f"  Chart saved → {chart_path}")
            print()

        except (KeyboardInterrupt, EOFError):
            print(DIM("\n  Goodbye.\n"))
            break
        except ValueError as e:
            print(RED(f"  Invalid input: {e}\n"))


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    from netraa.config import load_config

    p = argparse.ArgumentParser(
        description="What-If Impact Analyzer — propagate a metric change across the dependency graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("-c", "--config", default="configs/v1.yaml", help="Path to config file (default: configs/v1.yaml)")
    p.add_argument("--metric",    default=None,  help="Source metric name (partial OK)")
    p.add_argument("--change",    default=None,  type=float, help="Percent change (e.g. 20 or -15)")
    p.add_argument("--layers",    default=3,     type=int,   help="Propagation depth [default 3]")
    p.add_argument("--threshold", default=0.5,   type=float, help="Min |impact%%| to display [default 0.5]")
    p.add_argument("--chart",     action="store_true",       help="Save bar chart PNG to artifacts/")
    p.add_argument("--graph",     default=None,  help="Path to graph JSON (default: derived from config)")
    args = p.parse_args()

    cfg = load_config(args.config)
    art_dir = cfg.artifacts_dir

    graph_path = Path(args.graph) if args.graph else (art_dir / "stgnn_learned_graph.json")
    scaler_path = art_dir / "stgnn_scaler.json"

    if not graph_path.exists():
        sys.exit(f"Graph not found: {graph_path}\nRun `netraa backtest` first.")

    print(DIM(f"\n  Loading graph from {graph_path} …"))
    graph = load_graph(graph_path, scaler_path=scaler_path)
    print(DIM(f"  {graph.N} metrics  |  {int((graph.A_raw > 0).sum())} edges loaded\n"))

    run_interactive(graph, art_dir, args)


if __name__ == "__main__":
    main()
