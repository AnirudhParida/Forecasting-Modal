"""
forecast_next_30d.py
====================
Run the trained STGNN model on the *most recent* 56-day context window
and produce 30-day-ahead forecasts (P10 / P50 / P90) for every target metric.

Usage
-----
    python forecast_next_30d.py

Outputs
-------
    artifacts/forecast_30d.json           -- machine-readable forecast table
    artifacts/forecast_charts/<metric>.png -- one chart per target metric
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
PANEL_DIR  = ROOT / "data" / "panel"
ART_DIR    = ROOT / "artifacts"
CHART_DIR  = ART_DIR / "forecast_charts"
META_FILE  = ART_DIR / "stgnn_meta.json"
WEIGHTS    = ART_DIR / "stgnn.pt"
SCALER_F   = ART_DIR / "stgnn_scaler.json"

# ── imports ───────────────────────────────────────────────────────────────────
sys.path.insert(0, str(ROOT))
from netraa.features.panel import Panel
from netraa.features.transforms import (
    RobustScaler,
    apply_node_transforms,
    calendar_features,
    clip_outliers,
)
from netraa.models.stgnn import STGNN

try:
    import torch
except ImportError:
    sys.exit("PyTorch not installed. Run: pip install torch")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not installed -- skipping charts (pip install matplotlib)")


# ── human-friendly labels ─────────────────────────────────────────────────────
FRIENDLY: dict[str, str] = {
    "host_cpu_usage":                                                         "CPU Utilization (%)",
    "host_mem_usage":                                                         "Memory Utilization (%)",
    "disk_busy_time|DISK-1C013B34259F92AC":                                   "Disk Busy Time - Disk 1 (%)",
    "disk_busy_time|DISK-AEAFE33C63A4A3E9":                                   "Disk Busy Time - Disk 2 (%)",
    "disk_read_iops|DISK-1C013B34259F92AC":                                   "Disk Read IOPS - Disk 1",
    "disk_read_iops|DISK-AEAFE33C63A4A3E9":                                   "Disk Read IOPS - Disk 2",
    "disk_read_throughput|DISK-1C013B34259F92AC":                             "Disk Read Throughput - Disk 1 (B/s)",
    "disk_read_throughput|DISK-AEAFE33C63A4A3E9":                             "Disk Read Throughput - Disk 2 (B/s)",
    "disk_write_iops|DISK-1C013B34259F92AC":                                  "Disk Write IOPS - Disk 1",
    "disk_write_iops|DISK-AEAFE33C63A4A3E9":                                  "Disk Write IOPS - Disk 2",
    "disk_write_throughput|DISK-1C013B34259F92AC":                            "Disk Write Throughput - Disk 1 (B/s)",
    "disk_write_throughput|DISK-AEAFE33C63A4A3E9":                            "Disk Write Throughput - Disk 2 (B/s)",
    "jvm_memory_heap_used|PROCESS_GROUP_INSTANCE-1AEF64B598C08A5B":           "JVM Heap Used (bytes)",
    "host_cpu_iowait":                                                        "CPU I/O Wait (%)",
    "host_net_rx_bytes":                                                      "NIC Bytes Received",
    "host_net_tx_bytes":                                                      "NIC Bytes Sent",
    "host_sessions_new":                                                      "New Sessions Received",
    "host_sessions_reset":                                                    "Sessions Reset Received",
}

RESOURCE_COLOR: dict[str, str] = {
    "host_cpu":  "#58a6ff",
    "host_mem":  "#3fb950",
    "disk_busy": "#f78166",
    "disk_read": "#ffa657",
    "disk_write":"#ff7b72",
    "jvm":       "#d2a8ff",
    "host_net":  "#79c0ff",
    "host_sess": "#d2a8ff",
}

def _color(node_id: str) -> str:
    for k, v in RESOURCE_COLOR.items():
        if k.replace("_", "") in node_id.replace("_", "").lower():
            return v
    return "#c9d1d9"

def _safe_filename(node_id: str) -> str:
    return node_id.replace("|", "_").replace("/", "_")

def _get_label(node_id: str) -> str:
    if node_id in FRIENDLY:
        return FRIENDLY[node_id]
    base = node_id.split("|")[0]
    return base.replace("_", " ").title()


# ── load panel ─────────────────────────────────────────────────────────────────
def load_panel(panel_dir: Path, grid: str = "coarse") -> Panel:
    panel = Panel.load(panel_dir, grid)
    panel = apply_node_transforms(panel)
    panel = clip_outliers(panel)
    return panel


# ── load model ─────────────────────────────────────────────────────────────────
def load_model(meta: dict, n_nodes: int, device: str, cfg, weights_path: Path) -> STGNN:
    model = STGNN(
        n_nodes        = n_nodes,
        n_targets      = len(meta["target_ids"]),
        in_channels    = meta["n_channels"],
        input_steps    = meta["input_steps"],
        n_horizons     = len(meta["horizons"]),
        n_quantiles    = len(meta["quantiles"]),
        target_idx     = [meta["node_ids"].index(t) for t in meta["target_ids"]],
        hidden         = cfg.model.hidden,
        blocks         = cfg.model.blocks,
        kernel_size    = cfg.model.kernel_size,
        dropout        = 0.0,
        node_embed_dim = cfg.model.node_embed_dim,
        top_k          = cfg.model.top_k,
        use_graph      = meta.get("use_graph", True),
    )
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


# ── build input window ────────────────────────────────────────────────────────
def build_input(panel: Panel, meta: dict, scaler: RobustScaler) -> np.ndarray:
    input_steps = meta["input_steps"]
    node_ids    = meta["node_ids"]

    vals  = panel.values.reindex(columns=node_ids, fill_value=np.nan)
    mask_ = panel.mask.reindex(columns=node_ids, fill_value=0.0)

    vals_  = vals.iloc[-input_steps:]
    mask_w = mask_.iloc[-input_steps:]

    scaled = scaler.transform(vals_).to_numpy(dtype="float32")
    m_np   = mask_w.to_numpy(dtype="float32")

    freq_seconds = int(pd.Timedelta(panel.freq).total_seconds())
    cal = calendar_features(vals_.index, freq_seconds)

    n_cal = cal.shape[1]
    cal_broadcast = np.broadcast_to(
        cal[:, None, :], (input_steps, len(node_ids), n_cal)
    ).copy()

    channels = np.concatenate(
        [scaled[:, :, None], m_np[:, :, None], cal_broadcast],
        axis=2,
    )  # (T, N, C)
    return channels.transpose(1, 0, 2)[None]  # (1, N, T, C)


# ── run inference ──────────────────────────────────────────────────────────────
@torch.no_grad()
def run_forecast(model: STGNN, x_np: np.ndarray, device: str) -> np.ndarray:
    x   = torch.from_numpy(x_np).to(device)
    out = model(x)  # (1, N_target, H, Q)
    return out.cpu().numpy()[0]  # (N_target, H, Q)


# ── inverse scale ──────────────────────────────────────────────────────────────
def inverse_scale_forecast(
    pred: np.ndarray,
    target_ids: list[str],
    scaler: RobustScaler,
) -> np.ndarray:
    centers = np.array([scaler.center.get(t, 0.0) for t in target_ids], dtype="float32")
    scales  = np.array([scaler.scale.get(t, 1.0)  for t in target_ids], dtype="float32")
    return pred * scales[:, None, None] + centers[:, None, None]


# ── chart ──────────────────────────────────────────────────────────────────────
BG      = "#0d1117"
SURFACE = "#161b22"
BORDER  = "#30363d"
TEXT    = "#c9d1d9"
MUTED   = "#8b949e"

def _make_chart(
    node_id: str,
    horizon_days: int,
    history_dates: pd.DatetimeIndex,
    history_vals:  np.ndarray,
    forecast_dates: pd.DatetimeIndex,
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4), facecolor=BG)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.tick_params(colors=TEXT, which="both")

    color = _color(node_id)
    label = _get_label(node_id)

    tail = min(90, len(history_dates))
    ax.plot(
        history_dates[-tail:], history_vals[-tail:],
        color=color, linewidth=1.6, label="History (90 days)", alpha=0.9,
    )

    ax.fill_between(forecast_dates, p10, p90, color=color, alpha=0.18, label="P10-P90 confidence band")
    ax.plot(forecast_dates, p50, color=color, linewidth=2.2, linestyle="--", label=f"P50 Forecast ({horizon_days}d median)")
    ax.plot(forecast_dates, p10, color=color, linewidth=0.8, linestyle=":", alpha=0.55)
    ax.plot(forecast_dates, p90, color=color, linewidth=0.8, linestyle=":", alpha=0.55)

    ax.axvline(forecast_dates[0] - pd.Timedelta(days=1), color=MUTED, linewidth=1.0, linestyle="--", alpha=0.6)

    ax.set_title(f"{horizon_days}-Day Forecast  |  {label}", color=TEXT, fontsize=11, pad=10, loc="left", fontweight="bold")
    ax.set_xlabel("Date", color=MUTED, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    interval = 1 if horizon_days <= 30 else (2 if horizon_days <= 60 else 3)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=interval))
    plt.xticks(rotation=30, ha="right", fontsize=8, color=MUTED)
    plt.yticks(fontsize=8, color=MUTED)

    legend = ax.legend(
        loc="upper left", framealpha=0.3,
        facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=8,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    import argparse
    from netraa.config import load_config

    parser = argparse.ArgumentParser(description="Netraa Multi-Horizon Capacity Forecast")
    parser.add_argument("-c", "--config", default="configs/v1.yaml", help="Path to config file (default: configs/v1.yaml)")
    parser.add_argument("-d", "--days", type=int, default=30, help="Forecast horizon days (default: 30)")
    parser.add_argument("--all-charts", action="store_true", help="Generate all milestone charts (15d, 30d, 45d, 60d, 90d) instead of just requested horizon chart")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("=" * 80)
    print(f"  Netraa -- Capacity Forecast ({args.days}-Day Forecast)")
    print(f"  Config : {args.config}")
    print("=" * 80)

    art_dir      = cfg.artifacts_dir
    panel_dir    = cfg.panel_dir
    chart_dir    = art_dir / "forecast_charts"
    meta_file    = art_dir / "stgnn_meta.json"
    weights_path = art_dir / "stgnn.pt"
    scaler_file  = art_dir / "stgnn_scaler.json"
    trend_file   = art_dir / "stgnn_trend.json"

    if not meta_file.exists():
        sys.exit(f"\nModel meta not found: {meta_file}\nRun `netraa backtest` first.")
    if not weights_path.exists():
        sys.exit(f"\nModel weights not found: {weights_path}\nRun `netraa backtest` first.")

    meta = json.loads(meta_file.read_text())
    print(f"\nModel   : {meta['tag']}  (best epoch {meta['best_epoch']}, val_loss {meta['best_val_loss']:.4f})")
    print(f"Targets : {len(meta['target_ids'])} metrics")
    print(f"Horizons: {meta['horizons']} days  |  Quantiles: {meta['quantiles']}")

    scaler = RobustScaler.load(scaler_file)

    print(f"\nLoading panel from {panel_dir} ...")
    panel = load_panel(panel_dir, grid=cfg.forecast.grid)
    last_date = panel.values.index[-1]
    print(f"Panel   : {panel.n_steps} daily steps, last observed: {last_date.date()}")

    print(f"Building input window (last {meta['input_steps']} days) ...")
    x_np = build_input(panel, meta, scaler)
    print(f"Input   : shape {x_np.shape}  (batch=1, nodes={x_np.shape[1]}, time={x_np.shape[2]}, channels={x_np.shape[3]})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nRunning model on {device} ...")
    model    = load_model(meta, len(meta["node_ids"]), device, cfg, weights_path)
    raw_pred = run_forecast(model, x_np, device)   # (N_target, H, Q)
    pred     = inverse_scale_forecast(raw_pred, meta["target_ids"], scaler)

    # ── re-add trend ──────────────────────────────────────────────────────────
    from netraa.features.transforms import LinearDetrend
    if trend_file.exists():
        print("Re-adding linear trend slopes ...")
        detrend = LinearDetrend.load(trend_file)
        panel_end_step = len(panel.values) - 1
        h_days = np.array(meta["horizons"])
        for i, target in enumerate(meta["target_ids"]):
            s = detrend.slopes.get(target, 0.0)
            b = detrend.intercepts.get(target, 0.0)
            if abs(s) > 1e-12 or abs(b) > 1e-12:
                for h_idx, h in enumerate(h_days):
                    trend_val = s * (panel_end_step + h) + b
                    pred[i, h_idx, :] += trend_val

    # ── dynamic quantile mapping ──────────────────────────────────────────────
    quantiles = meta["quantiles"]
    q_lower = quantiles[0]
    q_mid   = quantiles[1]
    q_upper = quantiles[2]

    h_days  = np.array(meta["horizons"])
    q_idx   = {q: idx for idx, q in enumerate(meta["quantiles"])}

    # Milestone horizons to report
    all_milestones = [15, 30, 45, 60, 90]
    target_milestones = [m for m in all_milestones if m <= args.days]
    if not target_milestones:
        target_milestones = [args.days]

    max_days = max(args.days, max(target_milestones))
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=max_days, freq="D"
    )

    chart_dir.mkdir(parents=True, exist_ok=True)
    results = []

    print("\n" + "=" * 80)
    print("  FORECAST SUMMARY BY MILESTONE HORIZON")
    print("=" * 80)

    for i, target in enumerate(meta["target_ids"]):
        label = _get_label(target)
        print(f"\nMetric: {label} ({target})")
        print("─" * 80)
        print(f"  {'Horizon':<12}  {'Date':<12}  {f'P{int(q_lower*100)}':>10}  {f'P{int(q_mid*100)} (Median)':>15}  {f'P{int(q_upper*100)}':>10}")
        print("─" * 80)

        p10_sparse = pred[i, :, q_idx[q_lower]]
        p50_sparse = pred[i, :, q_idx[q_mid]]
        p90_sparse = pred[i, :, q_idx[q_upper]]

        days_out  = np.arange(1, max_days + 1)
        p10_daily = np.interp(days_out, h_days, p10_sparse)
        p50_daily = np.interp(days_out, h_days, p50_sparse)
        p90_daily = np.interp(days_out, h_days, p90_sparse)

        milestones_data = {}
        for m in target_milestones:
            idx_m = m - 1
            v_p10 = float(p10_daily[idx_m])
            v_p50 = float(p50_daily[idx_m])
            v_p90 = float(p90_daily[idx_m])
            m_date = str((last_date + pd.Timedelta(days=m)).date())

            print(f"  {f'{m} Days':<12}  {m_date:<12}  {v_p10:>10.2f}  {v_p50:>15.2f}  {v_p90:>10.2f}")

            milestones_data[f"{m}d"] = {
                "day": m,
                "date": m_date,
                "p10": round(v_p10, 4),
                "p50": round(v_p50, 4),
                "p90": round(v_p90, 4),
                "chart": f"{_safe_filename(target)}_{m}d.png"
            }

        # Targeted day index
        req_idx = args.days - 1 if max_days >= args.days else max_days - 1
        results.append({
            "metric":        target,
            "label":         label,
            "as_of_date":    str(last_date.date()),
            "forecast_days": args.days,
            "forecast_date": str((last_date + pd.Timedelta(days=args.days)).date()),
            "p10":           round(float(p10_daily[req_idx]), 4),
            "p50":           round(float(p50_daily[req_idx]), 4),
            "p90":           round(float(p90_daily[req_idx]), 4),
            "milestones":    milestones_data,
            "daily_forecast": {
                "dates": [str(d.date()) for d in forecast_dates[:args.days]],
                "p10":   [round(float(v), 4) for v in p10_daily[:args.days]],
                "p50":   [round(float(v), 4) for v in p50_daily[:args.days]],
                "p90":   [round(float(v), 4) for v in p90_daily[:args.days]],
            },
        })

        if HAS_MPL:
            hist_vals  = panel.values[target].values if target in panel.values.columns else np.full(len(panel.values), np.nan)
            hist_dates = panel.values.index

            if args.all_charts:
                # Generate individual charts for all milestone horizons up to max_days
                for m in target_milestones:
                    if m <= len(forecast_dates):
                        f_dates = forecast_dates[:m]
                        f_p10   = p10_daily[:m]
                        f_p50   = p50_daily[:m]
                        f_p90   = p90_daily[:m]
                        chart_path = chart_dir / f"{_safe_filename(target)}_{m}d.png"
                        try:
                            _make_chart(target, m, hist_dates, hist_vals, f_dates,
                                        f_p10, f_p50, f_p90, chart_path)
                        except Exception as e:
                            print(f"  [chart error for {target} ({m}d): {e}]")
            else:
                # Generate ONLY the chart for the requested forecast horizon (e.g. 30d, 60d)
                m = args.days
                if m <= len(forecast_dates):
                    f_dates = forecast_dates[:m]
                    f_p10   = p10_daily[:m]
                    f_p50   = p50_daily[:m]
                    f_p90   = p90_daily[:m]
                    chart_path = chart_dir / f"{_safe_filename(target)}_{m}d.png"
                    try:
                        _make_chart(target, m, hist_dates, hist_vals, f_dates,
                                    f_p10, f_p50, f_p90, chart_path)
                    except Exception as e:
                        print(f"  [chart error for {target} ({m}d): {e}]")

    print("\n" + "=" * 80)

    out_json = art_dir / "forecast_30d.json"
    payload  = {
        "as_of_date":    str(last_date.date()),
        "forecast_days": args.days,
        "model":         meta["tag"],
        "best_val_loss": meta["best_val_loss"],
        "horizons_available": meta["horizons"],
        "milestone_horizons": target_milestones,
        "forecasts":     results,
    }
    out_json.write_text(json.dumps(payload, indent=2))

    print(f"\n  Forecast JSON saved : {out_json}")
    if HAS_MPL:
        charts = list(chart_dir.glob("*.png"))
        print(f"  Charts saved ({len(charts)}) : {chart_dir}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
