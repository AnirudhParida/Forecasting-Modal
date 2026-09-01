"""validate_forecast.py / validate_august_forecast.py
===============================================
Dynamic Forecast Validation script for ST-GNN models.

Supports flexible Start Date (--st) and End Date (--et) parameters.

Usage Examples:
---------------
1. Default (August 1 to August 28, 2026):
   python validate_august_forecast.py -c configs/v4_multihost_v1.yaml

2. Custom Date Range (e.g. August 10 to August 25, 2026):
   python validate_august_forecast.py -c configs/v4_multihost_v1.yaml --st 2026-08-10 --et 2026-08-25

3. Single Host Config (e.g. hydupiapp001):
   python validate_august_forecast.py -c configs/v3_hydupiapp001.yaml --st 2026-08-01 --et 2026-08-28
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from netraa.config import load_config
from netraa.features.panel import Panel
from netraa.features.transforms import (
    LinearDetrend,
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


def _get_clean_label(node_id: str) -> tuple[str, str]:
    """Returns (metric_label, host_label)."""
    key, dim = node_id.split("|") if "|" in node_id else (node_id, "")
    metric = key.replace("host_", "").replace("_", " ").title()
    if "10_50_98_26" in dim or "HYDUPINTAPP16" in dim:
        host = "HYDUPINTAPP16 (10.50.98.26)"
    elif "10_78_33_83" in dim or "JPRUPIWEBCRP02" in dim:
        host = "JPRUPIWEBCRP02 (10.78.33.83)"
    elif dim:
        host = dim
    else:
        host = "Default Host"
    return f"{metric} - {host}", host


def run_validation(config_path: str, st_str: str, et_str: str) -> dict:
    cfg = load_config(config_path)
    art_dir      = cfg.artifacts_dir
    panel_dir    = cfg.panel_dir
    chart_dir    = art_dir / "forecast_charts"
    meta_file    = art_dir / "stgnn_meta.json"
    weights_path = art_dir / "stgnn.pt"
    scaler_file  = art_dir / "stgnn_scaler.json"
    trend_file   = art_dir / "stgnn_trend.json"

    if not meta_file.exists() or not weights_path.exists():
        raise FileNotFoundError(f"Trained model files missing in {art_dir}. Run `netraa backtest` first.")

    meta   = json.loads(meta_file.read_text())
    scaler = RobustScaler.load(scaler_file)
    panel  = Panel.load(panel_dir, grid=cfg.forecast.grid)
    panel  = apply_node_transforms(panel)
    panel  = clip_outliers(panel)

    st_dt = pd.Timestamp(st_str, tz="UTC")
    et_dt = pd.Timestamp(et_str, tz="UTC")

    if st_dt >= et_dt:
        raise ValueError(f"Start date (--st {st_str}) must be strictly earlier than end date (--et {et_str})")

    # The cutoff date for historical context window is 1 day before --st
    cutoff_dt = st_dt - pd.Timedelta(days=1)
    
    panel_min = panel.values.index.min()
    panel_max = panel.values.index.max()

    if cutoff_dt < panel_min:
        raise ValueError(f"Cutoff date ({cutoff_dt.date()}) is earlier than available panel data start ({panel_min.date()})")

    # Locate cutoff index in panel
    cutoff_idx = panel.values.index.get_indexer([cutoff_dt], method="pad")[0]
    
    input_steps = meta["input_steps"]
    node_ids    = meta["node_ids"]
    target_ids  = meta["target_ids"]

    # History slice for input context (last N input_steps days up to cutoff_dt)
    slice_start = max(0, cutoff_idx - input_steps + 1)
    slice_end   = cutoff_idx + 1

    actual_input_len = slice_end - slice_start
    if actual_input_len < input_steps:
        raise ValueError(
            f"Not enough history before {st_str} for input window. "
            f"Need {input_steps} days, only found {actual_input_len} days."
        )

    context_values = panel.values.iloc[slice_start:slice_end].reindex(columns=node_ids, fill_value=np.nan)
    context_mask   = panel.mask.iloc[slice_start:slice_end].reindex(columns=node_ids, fill_value=0.0)

    # Scaled input tensor
    scaled = scaler.transform(context_values).to_numpy(dtype="float32")
    m_np   = context_mask.to_numpy(dtype="float32")
    freq_seconds = int(pd.Timedelta(panel.freq).total_seconds())
    cal = calendar_features(context_values.index, freq_seconds)

    n_cal = cal.shape[1]
    cal_broadcast = np.broadcast_to(
        cal[:, None, :], (input_steps, len(node_ids), n_cal)
    ).copy()

    channels = np.concatenate(
        [scaled[:, :, None], m_np[:, :, None], cal_broadcast],
        axis=2,
    )  # (T, N, C)
    x_np = channels.transpose(1, 0, 2)[None]  # (1, N, T, C)

    # Load Model & Run Inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = STGNN(
        n_nodes        = len(node_ids),
        n_targets      = len(target_ids),
        in_channels    = meta["n_channels"],
        input_steps    = meta["input_steps"],
        n_horizons     = len(meta["horizons"]),
        n_quantiles    = len(meta["quantiles"]),
        target_idx     = [node_ids.index(t) for t in target_ids],
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
    model.to(device)

    with torch.no_grad():
        x_tensor = torch.from_numpy(x_np).to(device)
        raw_pred = model(x_tensor).cpu().numpy()[0]  # (N_target, H, Q)

    # Inverse Scale
    centers = np.array([scaler.center.get(t, 0.0) for t in target_ids], dtype="float32")
    scales  = np.array([scaler.scale.get(t, 1.0)  for t in target_ids], dtype="float32")
    pred    = raw_pred * scales[:, None, None] + centers[:, None, None]

    # Re-add linear trend if present
    if trend_file.exists():
        detrend = LinearDetrend.load(trend_file)
        h_days  = np.array(meta["horizons"])
        for i, target in enumerate(target_ids):
            s = detrend.slopes.get(target, 0.0)
            b = detrend.intercepts.get(target, 0.0)
            if abs(s) > 1e-12 or abs(b) > 1e-12:
                for h_idx, h in enumerate(h_days):
                    trend_val = s * (cutoff_idx + h) + b
                    pred[i, h_idx, :] += trend_val

    # Forecast dates range [st, et]
    forecast_dates = pd.date_range(start=st_dt, end=et_dt, freq="D")
    n_forecast_days = len(forecast_dates)

    quantiles = meta["quantiles"]
    q_idx = {q: idx for idx, q in enumerate(quantiles)}
    q_lower, q_mid, q_upper = quantiles[0], quantiles[1], quantiles[2]

    h_days = np.array(meta["horizons"])
    days_out = np.arange(1, n_forecast_days + 1)

    chart_dir.mkdir(parents=True, exist_ok=True)
    validation_results = {}

    print("=" * 95)
    print(f"  FORECAST VALIDATION REPORT")
    print(f"  Config     : {config_path}")
    print(f"  Start Date : {st_str}")
    print(f"  End Date   : {et_str} ({n_forecast_days} Days)")
    print(f"  Cutoff Date: {cutoff_dt.date()} (History Window: {context_values.index.min().date()} -> {cutoff_dt.date()})")
    print("=" * 95)

    for i, target in enumerate(target_ids):
        clean_label, host_label = _get_clean_label(target)
        
        # Interpolate predictions to daily steps for the requested horizon
        p10_sparse = pred[i, :, q_idx[q_lower]]
        p50_sparse = pred[i, :, q_idx[q_mid]]
        p90_sparse = pred[i, :, q_idx[q_upper]]

        p10_daily = np.interp(days_out, h_days, p10_sparse)
        p50_daily = np.interp(days_out, h_days, p50_sparse)
        p90_daily = np.interp(days_out, h_days, p90_sparse)

        # Extract actual values from panel if present within date range
        actual_in_span = panel.values.reindex(forecast_dates)[target].to_numpy()

        valid_mask  = ~np.isnan(actual_in_span)
        has_actuals = np.any(valid_mask)

        if has_actuals:
            act_clean = actual_in_span[valid_mask]
            p50_clean = p50_daily[valid_mask]
            p10_clean = p10_daily[valid_mask]
            p90_clean = p90_daily[valid_mask]

            mae  = float(np.mean(np.abs(act_clean - p50_clean)))
            rmse = float(np.sqrt(np.mean((act_clean - p50_clean) ** 2)))
            coverage = float(np.mean((act_clean >= p10_clean) & (act_clean <= p90_clean)) * 100.0)
            summary_str = f"MAE = {mae:.2f}% | RMSE = {rmse:.2f}% | P5-P95 Band Coverage = {coverage:.1f}%"
        else:
            mae, rmse, coverage = float("nan"), float("nan"), float("nan")
            summary_str = "No ground-truth actual data available in panel for this range (Future Forecast)"

        print(f"\nHost: {host_label}")
        print(f"Target Node: {target}")
        print(f"Validation Summary: {summary_str}")
        print("─" * 95)
        print(f"  {'Date':<12}  {'Actual CPU %':>15}  {'P50 Forecast':>15}  {'P5 (Lower)':>12}  {'P95 (Upper)':>12}  {'Abs Error':>10}")
        print("─" * 95)

        daily_rows = []
        for d_idx, dt in enumerate(forecast_dates):
            act_v = actual_in_span[d_idx]
            p50_v = float(p50_daily[d_idx])
            p10_v = float(p10_daily[d_idx])
            p90_v = float(p90_daily[d_idx])
            
            act_str = f"{act_v:>14.2f}%" if not np.isnan(act_v) else f"{'N/A (Future)':>15}"
            err_str = f"{abs(act_v - p50_v):>10.2f}%" if not np.isnan(act_v) else f"{'N/A':>10}"

            dt_str = str(dt.date())
            print(f"  {dt_str:<12}  {act_str}  {p50_v:>15.2f}%  {p10_v:>12.2f}%  {p90_v:>12.2f}%  {err_str}")

            daily_rows.append({
                "date": dt_str,
                "actual": round(float(act_v), 2) if not np.isnan(act_v) else None,
                "forecast_p50": round(p50_v, 2),
                "forecast_p5": round(p10_v, 2),
                "forecast_p95": round(p90_v, 2),
                "abs_error": round(float(abs(act_v - p50_v)), 2) if not np.isnan(act_v) else None,
            })

        validation_results[target] = {
            "host": host_label,
            "target_node": target,
            "start_date": st_str,
            "end_date": et_str,
            "mae": round(mae, 4) if not np.isnan(mae) else None,
            "rmse": round(rmse, 4) if not np.isnan(rmse) else None,
            "coverage_pct": round(coverage, 2) if not np.isnan(coverage) else None,
            "daily_comparison": daily_rows,
        }

        # Plot Validation Chart
        if HAS_MPL:
            fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0d1117")
            ax.set_facecolor("#161b22")
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")
            ax.tick_params(colors="#c9d1d9", which="both")

            # Context history (last 30 days of context before cutoff)
            hist_sub = context_values[target].dropna().tail(30)
            if not hist_sub.empty:
                ax.plot(hist_sub.index, hist_sub.values, color="#8b949e", linewidth=1.5, label=f"Context History (30d before {st_str})", alpha=0.8)

            # Actual vs Forecast
            if has_actuals:
                ax.plot(forecast_dates, actual_in_span, color="#3fb950", linewidth=2.2, label=f"Actual CPU Usage (%)", marker="o", markersize=4)

            ax.plot(forecast_dates, p50_daily, color="#58a6ff", linewidth=2.2, linestyle="--", label=f"ST-GNN P50 Forecast ({st_str} to {et_str})", marker="s", markersize=4)
            ax.fill_between(forecast_dates, p10_daily, p90_daily, color="#58a6ff", alpha=0.20, label="P5-P95 Confidence Interval")

            ax.axvline(cutoff_dt, color="#f78166", linestyle=":", linewidth=1.5, label=f"Forecast Cutoff ({cutoff_dt.date()})")

            ax.set_title(f"Forecast Validation ({st_str} to {et_str})  |  {host_label}", color="#c9d1d9", fontsize=11, pad=12, loc="left", fontweight="bold")
            ax.set_xlabel("Date", color="#8b949e", fontsize=9)
            ax.set_ylabel("CPU Usage (%)", color="#8b949e", fontsize=9)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            interval = 1 if n_forecast_days <= 15 else (2 if n_forecast_days <= 45 else 5)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
            plt.xticks(rotation=30, ha="right", fontsize=8, color="#8b949e")
            plt.yticks(fontsize=8, color="#8b949e")

            ax.legend(loc="upper right", framealpha=0.35, facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)
            plt.tight_layout()

            safe_t = target.replace("|", "_")
            chart_file = chart_dir / f"validation_{st_str}_to_{et_str}_{safe_t}.png"
            fig.savefig(chart_file, dpi=130, bbox_inches="tight", facecolor="#0d1117")
            plt.close(fig)

    # Save JSON report
    val_json = art_dir / f"validation_{st_str}_to_{et_str}.json"
    val_json.write_text(json.dumps(validation_results, indent=2))
    print("\n" + "=" * 95)
    print(f"  Validation JSON saved : {val_json}")
    print(f"  Validation Charts saved: {chart_dir}/validation_{st_str}_to_{et_str}_*.png")
    print("=" * 95)

    return validation_results


def main():
    parser = argparse.ArgumentParser(description="ST-GNN Dynamic Forecast & Validation CLI")
    parser.add_argument("-c", "--config", default="configs/v4_multihost_v1.yaml", help="Path to model config YAML")
    parser.add_argument("--st", default="2026-08-01", help="Forecast start date YYYY-MM-DD (default: 2026-08-01)")
    parser.add_argument("--et", default="2026-08-28", help="Forecast end date YYYY-MM-DD (default: 2026-08-28)")
    args = parser.parse_args()

    run_validation(args.config, args.st, args.et)


if __name__ == "__main__":
    main()
