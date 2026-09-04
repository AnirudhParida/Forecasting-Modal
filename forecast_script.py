"""forecast_script.py
==================
ST-GNN Dynamic Multi-Host Forecast & Validation Script

Supports:
  1. Date Selection  : --st YYYY-MM-DD  --et YYYY-MM-DD
  2. Model Selection : --model graph | nograph | both
     - graph   : ST-GNN model trained WITH statistical dependency graph prior
     - nograph : ST-GNN model evaluated WITHOUT dependency graph convolutions
     - both    : Runs both models side-by-side, compares predictions, and plots comparison charts!

Usage Examples
--------------
1. Forecast / Validate with ST-GNN Graph Model (Default August 1-28, 2026):
   python forecast_script.py -c configs/v4_multihost_v1.yaml --model graph

2. Forecast / Validate with ST-GNN No-Graph Model:
   python forecast_script.py -c configs/v4_multihost_v1.yaml --model nograph

3. Compare BOTH Models (Graph vs No-Graph vs Actuals):
   python forecast_script.py -c configs/v4_multihost_v1.yaml --model both --st 2026-08-01 --et 2026-08-28

4. Custom Date Range Validation:
   python forecast_script.py -c configs/v4_multihost_v1.yaml --st 2026-08-05 --et 2026-08-20 --model both
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
    # pyrefly: ignore [missing-import]
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


def _get_clean_label(node_id: str) -> tuple[str, str, str]:
    """Returns (metric_name, metric_label, host_label)."""
    key, dim = node_id.split("|") if "|" in node_id else (node_id, "")
    if key == "host_cpu_usage":
        metric_name = "CPU Usage (%)"
    elif key == "host_disk_avail_pct":
        metric_name = "Disk Available (%)"
    elif key == "host_mem_avail_pct":
        metric_name = "Memory Available (%)"
    elif key == "host_disk_read_ops_sec":
        metric_name = "Disk Read Ops/sec"
    elif key == "host_disk_write_bytes_sec":
        metric_name = "Disk Write Bytes/sec"
    else:
        metric_name = key.replace("host_", "").replace("_", " ").title()

    if "10_50_98_26" in dim or "HYDUPINTAPP16" in dim:
        host = "HYDUPINTAPP16 (10.50.98.26)"
    elif "10_78_33_83" in dim or "JPRUPIWEBCRP02" in dim:
        host = "JPRUPIWEBCRP02 (10.78.33.83)"
    elif dim:
        host = dim
    else:
        host = "Default Host"
    return metric_name, f"{metric_name} - {host}", host


def _get_clean_driver_name(key: str) -> str:
    name_map = {
        "host_disk_read_bytes_sec": "Disk Read Bytes/sec",
        "host_disk_read_ops_sec": "Disk Read Ops/sec",
        "host_disk_write_bytes_sec": "Disk Write Bytes/sec",
        "host_disk_write_iops": "Disk Write IOPS",
        "host_disk_read_time": "Disk Read Time",
        "host_disk_write_time": "Disk Write Time",
        "host_disk_used": "Disk Used",
        "host_disk_avail_pct": "Disk Available (%)",
        "host_mem_available": "Memory Available",
        "host_mem_avail_pct": "Memory Available (%)",
        "host_mem_swap_used": "Swap Used",
        "host_net_rx_bytes": "Network RX Bytes",
        "host_net_tx_bytes": "Network TX Bytes",
        "host_net_rx_packets": "Network RX Packets",
        "host_net_tx_packets": "Network TX Packets",
        "host_sessions_reset": "Session Resets",
        "host_sessions_timeout": "Session Timeouts",
        "host_sessions_new": "New Sessions",
        "host_cpu_iowait": "CPU I/O Wait",
        "host_cpu_system": "CPU System",
        "host_cpu_user": "CPU User",
        "host_cpu_idle": "CPU Idle",
        "host_cpu_usage": "CPU Usage (%)",
        "host_inodes_total": "Inodes Total",
    }
    base_key = key.split("|")[0]
    return name_map.get(base_key, base_key.replace("host_", "").replace("_", " ").title())


def _build_explainability(
    target_node: str,
    cfg,
    panel: Panel,
    cutoff_idx: int,
    interpolated_preds: dict,
    model_tag: str = "stgnn_graph",
    eval_summary: dict | None = None,
) -> dict:
    """Builds explainability dictionary & justification for target node forecast."""
    metric_name, clean_label, host_label = _get_clean_label(target_node)
    
    # Load dependency graph edges for target
    graph_file = cfg.graph_dir / "dependency_map_coarse.json"
    if not graph_file.exists():
        graph_file = cfg.graph_dir / f"dependency_map_{cfg.forecast.grid}.json"
    
    incoming = []
    if graph_file.exists():
        try:
            gdata = json.loads(graph_file.read_text())
            t_dim = target_node.split("|")[1] if "|" in target_node else ""
            for e in gdata.get("edges", []):
                if e["target"] == target_node:
                    s_dim = e["source"].split("|")[1] if "|" in e["source"] else ""
                    if t_dim == "" or s_dim == "" or t_dim == s_dim:
                        incoming.append(e)
        except Exception:
            pass

    incoming.sort(key=lambda x: x.get("strength", 0.0), reverse=True)
    top_incoming = incoming[:5]
    tot_strength = sum(e.get("strength", 0.0) for e in top_incoming) or 1.0

    drivers = []
    for e in top_incoming:
        str_val = e.get("strength", 0.0)
        direction = e.get("direction", "+")
        pct_contrib = round((str_val / tot_strength) * 100)
        lag_s = e.get("lag_seconds", 0)
        lag_str = f"{lag_s // 86400}d" if lag_s >= 86400 else (f"{lag_s // 60}m" if lag_s >= 60 else "0m")

        drivers.append({
            "metric": _get_clean_driver_name(e["source"]),
            "source_node": e["source"],
            "impact": f"{direction}{pct_contrib}%",
            "direction": direction,
            "strength": round(float(str_val), 3),
            "lag": lag_str,
        })

    pred_data = interpolated_preds.get(model_tag) or next(iter(interpolated_preds.values()))
    p10_final = float(pred_data["p10"][-1])
    p50_final = float(pred_data["p50"][-1])
    p90_final = float(pred_data["p90"][-1])

    hist = panel.values.iloc[max(0, cutoff_idx - 30):cutoff_idx + 1][target_node].dropna()
    base_avg = float(hist.mean()) if not hist.empty else p50_final
    diff = p50_final - base_avg
    diff_str = f"{diff:+.2f}%" if abs(diff) >= 0.01 else "stable"

    spread = abs(p90_final - p10_final)
    conf_score = round(max(60.0, min(98.0, 100.0 - spread * 2.5)), 1)

    key = target_node.split("|")[0]
    if "cpu" in key:
        risk = "HIGH" if p50_final > 75.0 else ("MEDIUM" if p50_final > 45.0 else "LOW")
    elif "avail" in key:
        risk = "HIGH" if p50_final < 15.0 else ("MEDIUM" if p50_final < 30.0 else "LOW")
    else:
        risk = "HIGH" if p50_final > 80.0 else ("MEDIUM" if p50_final > 50.0 else "LOW")

    summary_text = (
        f"{metric_name} is predicted to average {p50_final:.2f}% (P50) over the forecast horizon "
        f"({diff_str} vs 30d baseline avg {base_avg:.2f}%)."
    )
    if drivers:
        top_driver_names = ", ".join(d["metric"] for d in drivers[:3])
        summary_text += f" Primary interdependency drivers: {top_driver_names}."

    # Extract accuracy metrics if evaluation against actuals was run
    accuracy_info = {}
    if eval_summary and model_tag in eval_summary:
        mae_val = eval_summary[model_tag].get("mae", 0.0)
        cov_val = eval_summary[model_tag].get("coverage_pct", 0.0)
        acc_pct = max(0.0, round(100.0 - mae_val, 2))
        accuracy_info = {
            "forecast_accuracy_pct": acc_pct,
            "mae": mae_val,
            "coverage_pct": cov_val,
        }

    return {
        "model_version": "ST-GNN (Graph Prior)" if "graph" in model_tag and "nograph" not in model_tag else "ST-GNN (No-Graph Ablation)",
        "forecast": {
            "p05_or_p10": round(p10_final, 2),
            "p50_median": round(p50_final, 2),
            "p95_or_p90": round(p90_final, 2),
        },
        "accuracy": accuracy_info,
        "baseline_30d_avg": round(base_avg, 2),
        "predicted_shift": round(diff, 2),
        "confidence_score": conf_score,
        "risk_level": risk,
        "summary": summary_text,
        "top_contributing_drivers": drivers,
    }


def _predict_single_model(
    model_tag: str,
    cfg,
    panel: Panel,
    cutoff_idx: int,
) -> tuple[np.ndarray, dict, RobustScaler]:
    """Loads weights and scaler for model_tag, runs inference on panel input window."""
    art_dir = cfg.artifacts_dir
    
    is_nograph = "nograph" in model_tag or "no-graph" in model_tag

    # Dedicated nograph file or fallback to base stgnn weights
    if is_nograph and (art_dir / "stgnn_nograph.pt").exists():
        prefix = "stgnn_nograph"
        use_graph_arch = False
    elif not is_nograph and (art_dir / "stgnn_graph.pt").exists():
        prefix = "stgnn_graph"
        use_graph_arch = True
    else:
        prefix = "stgnn"
        use_graph_arch = True

    meta_file = art_dir / f"{prefix}_meta.json"
    if not meta_file.exists():
        meta_file = art_dir / "stgnn_meta.json"

    weights_path = art_dir / f"{prefix}.pt"
    if not weights_path.exists():
        weights_path = art_dir / "stgnn.pt"

    scaler_file = art_dir / f"{prefix}_scaler.json"
    if not scaler_file.exists():
        scaler_file = art_dir / "stgnn_scaler.json"

    trend_file = art_dir / f"{prefix}_trend.json"
    if not trend_file.exists():
        trend_file = art_dir / "stgnn_trend.json"

    if not meta_file.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights for '{model_tag}' missing in {art_dir}. "
            f"Expected {weights_path.name}. Run `netraa backtest` first."
        )

    meta   = json.loads(meta_file.read_text())
    scaler = RobustScaler.load(scaler_file)

    input_steps = meta["input_steps"]
    node_ids    = meta["node_ids"]
    target_ids  = meta["target_ids"]

    # Slice history input context window
    slice_start = max(0, cutoff_idx - input_steps + 1)
    slice_end   = cutoff_idx + 1

    context_values = panel.values.iloc[slice_start:slice_end].reindex(columns=node_ids, fill_value=np.nan)
    context_mask   = panel.mask.iloc[slice_start:slice_end].reindex(columns=node_ids, fill_value=0.0)

    # Detrend input context window before scaling (matches training pipeline in dataset.py)
    if trend_file.exists():
        detrend = LinearDetrend.load(trend_file)
        context_values = detrend.transform(context_values)

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
        use_graph      = use_graph_arch,
    )
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    # If evaluating nograph mode using base stgnn.pt weights, disable graph propagation
    if is_nograph and use_graph_arch:
        for block in model.blocks:
            if hasattr(block, "graph") and block.graph is not None:
                # Override forward graph convolution to return zero propagation
                block.graph.use_graph = False

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

    # ── Fix 1: Domain Clipping ──────────────────────────────────────────────────
    # Percentage-bounded targets (CPU %, Disk Avail %, Memory Avail %) are physically
    # bounded [0, 100]. Clip only those targets; raw byte/ops metrics must NOT be clipped
    # to 100 as their natural range can be orders of magnitude larger.
    PCT_TARGET_KEYS = {"host_cpu_usage", "host_disk_avail_pct", "host_mem_avail_pct"}
    for i, target in enumerate(target_ids):
        base_key = target.split("|")[0]
        if base_key in PCT_TARGET_KEYS:
            pred[i] = np.clip(pred[i], 0.0, 100.0)
        else:
            # Non-percentage metrics: only clip at 0 (no negatives physically possible)
            pred[i] = np.clip(pred[i], 0.0, None)

    # ── Fix 2: Enforce Quantile Monotonicity ────────────────────────────────────
    # After domain clipping, guarantee P05 <= P50 <= P95 ordering is preserved.
    # Clipping can invert the band ordering when the lower tail is near zero.
    pred[:, :, 0] = np.minimum(pred[:, :, 0], pred[:, :, 1])  # P05 <= P50
    pred[:, :, 2] = np.maximum(pred[:, :, 2], pred[:, :, 1])  # P95 >= P50

    return pred, meta, scaler


def run_validation(config_path: str, st_str: str, et_str: str, model_choice: str = "graph") -> dict:
    cfg = load_config(config_path)
    art_dir   = cfg.artifacts_dir
    panel_dir = cfg.panel_dir
    chart_dir = art_dir / "forecast_charts"

    panel = Panel.load(panel_dir, grid=cfg.forecast.grid)
    panel = apply_node_transforms(panel)
    panel = clip_outliers(panel)

    st_dt = pd.Timestamp(st_str, tz="UTC")
    et_dt = pd.Timestamp(et_str, tz="UTC")

    if st_dt >= et_dt:
        raise ValueError(f"Start date (--st {st_str}) must be strictly earlier than end date (--et {et_str})")

    cutoff_dt  = st_dt - pd.Timedelta(days=1)
    panel_min  = panel.values.index.min()

    if cutoff_dt < panel_min:
        raise ValueError(f"Cutoff date ({cutoff_dt.date()}) is earlier than available panel data start ({panel_min.date()})")

    cutoff_idx = panel.values.index.get_indexer([cutoff_dt], method="pad")[0]
    
    # Determine models to evaluate
    if model_choice in ("both", "compare", "all"):
        models_to_run = ["stgnn_graph", "stgnn_nograph"]
    elif model_choice in ("nograph", "stgnn_nograph", "no-graph"):
        models_to_run = ["stgnn_nograph"]
    else:
        models_to_run = ["stgnn_graph"]

    preds_dict = {}
    meta_dict  = {}
    for mtag in models_to_run:
        p, m, _ = _predict_single_model(mtag, cfg, panel, cutoff_idx)
        preds_dict[mtag] = p
        meta_dict[mtag]  = m

    ref_meta   = meta_dict[models_to_run[0]]
    target_ids = ref_meta["target_ids"]
    horizons   = ref_meta["horizons"]
    quantiles  = ref_meta["quantiles"]
    q_idx      = {q: idx for idx, q in enumerate(quantiles)}
    q_lower, q_mid, q_upper = quantiles[0], quantiles[1], quantiles[2]

    forecast_dates  = pd.date_range(start=st_dt, end=et_dt, freq="D")
    n_forecast_days = len(forecast_dates)
    h_days          = np.array(horizons)
    days_out        = np.arange(1, n_forecast_days + 1)

    chart_dir.mkdir(parents=True, exist_ok=True)
    validation_results = {}

    print("=" * 105)
    print(f"  NETRAA FORECAST & VALIDATION REPORT (forecast_script.py)")
    print(f"  Config     : {config_path}")
    print(f"  Model(s)   : {', '.join(models_to_run)}")
    print(f"  Start Date : {st_str}")
    print(f"  End Date   : {et_str} ({n_forecast_days} Days)")
    print(f"  Cutoff Date: {cutoff_dt.date()}")
    print("=" * 105)

    for i, target in enumerate(target_ids):
        metric_name, clean_label, host_label = _get_clean_label(target)
        actual_in_span = panel.values.reindex(forecast_dates)[target].to_numpy()
        valid_mask     = ~np.isnan(actual_in_span)
        has_actuals    = np.any(valid_mask)

        print(f"\nHost: {host_label} | Metric: {metric_name}")
        print(f"Target Node: {target}")

        model_eval_summary = {}
        interpolated_preds = {}

        for mtag in models_to_run:
            pred = preds_dict[mtag]
            p10_s = pred[i, :, q_idx[q_lower]]
            p50_s = pred[i, :, q_idx[q_mid]]
            p90_s = pred[i, :, q_idx[q_upper]]

            p10_d = np.interp(days_out, h_days, p10_s)
            p50_d = np.interp(days_out, h_days, p50_s)
            p90_d = np.interp(days_out, h_days, p90_s)

            interpolated_preds[mtag] = {"p10": p10_d, "p50": p50_d, "p90": p90_d}

            if has_actuals:
                act_clean = actual_in_span[valid_mask]
                p50_clean = p50_d[valid_mask]
                p10_clean = p10_d[valid_mask]
                p90_clean = p90_d[valid_mask]

                mae  = float(np.mean(np.abs(act_clean - p50_clean)))
                rmse = float(np.sqrt(np.mean((act_clean - p50_clean) ** 2)))
                cov  = float(np.mean((act_clean >= p10_clean) & (act_clean <= p90_clean)) * 100.0)
                m_name = "ST-GNN (Graph Prior)" if "graph" in mtag and "nograph" not in mtag else "ST-GNN (No-Graph Ablation)"
                print(f"  [{m_name}] MAE = {mae:.2f}% | RMSE = {rmse:.2f}% | P5-P95 Band Coverage = {cov:.1f}%")
                model_eval_summary[mtag] = {"mae": round(mae, 4), "rmse": round(rmse, 4), "coverage_pct": round(cov, 2)}
            else:
                m_name = "ST-GNN (Graph Prior)" if "graph" in mtag and "nograph" not in mtag else "ST-GNN (No-Graph Ablation)"
                print(f"  [{m_name}] (Future Forecast - No ground truth actuals)")

        print("─" * 105)
        if len(models_to_run) > 1:
            print(f"  {'Date':<12}  {f'Actual ({metric_name})':>16}  {'Graph P50':>12}  {'Graph Err':>10}  {'NoGraph P50':>13}  {'NoGraph Err':>12}  {'Best Model':>12}")
        else:
            print(f"  {'Date':<12}  {f'Actual ({metric_name})':>18}  {'P50 Forecast':>15}  {'P5 (Lower)':>12}  {'P95 (Upper)':>12}  {'Abs Error':>10}")
        print("─" * 105)

        daily_rows = []
        for d_idx, dt in enumerate(forecast_dates):
            act_v  = actual_in_span[d_idx]
            dt_str = str(dt.date())

            if len(models_to_run) > 1:
                g_p50 = interpolated_preds["stgnn_graph"]["p50"][d_idx]
                ng_p50 = interpolated_preds["stgnn_nograph"]["p50"][d_idx]
                
                if not np.isnan(act_v):
                    g_err  = abs(act_v - g_p50)
                    ng_err = abs(act_v - ng_p50)
                    winner = "Graph" if g_err <= ng_err else "No-Graph"
                    print(f"  {dt_str:<12}  {act_v:>11.2f}%  {g_p50:>12.2f}%  {g_err:>9.2f}%  {ng_p50:>13.2f}%  {ng_err:>11.2f}%  {winner:>12}")
                else:
                    g_err, ng_err, winner = None, None, "N/A"
                    print(f"  {dt_str:<12}  {'N/A':>12}  {g_p50:>12.2f}%  {'N/A':>10}  {ng_p50:>13.2f}%  {'N/A':>12}  {'N/A':>12}")

                daily_rows.append({
                    "date": dt_str,
                    "actual": round(float(act_v), 2) if not np.isnan(act_v) else None,
                    "stgnn_graph_p50": round(float(g_p50), 2),
                    "stgnn_graph_err": round(float(g_err), 2) if g_err is not None else None,
                    "stgnn_nograph_p50": round(float(ng_p50), 2),
                    "stgnn_nograph_err": round(float(ng_err), 2) if ng_err is not None else None,
                    "winner": winner,
                })
            else:
                mtag = models_to_run[0]
                p50_v = interpolated_preds[mtag]["p50"][d_idx]
                p10_v = interpolated_preds[mtag]["p10"][d_idx]
                p90_v = interpolated_preds[mtag]["p90"][d_idx]
                act_str = f"{act_v:>14.2f}%" if not np.isnan(act_v) else f"{'N/A (Future)':>15}"
                err_str = f"{abs(act_v - p50_v):>10.2f}%" if not np.isnan(act_v) else f"{'N/A':>10}"
                print(f"  {dt_str:<12}  {act_str}  {p50_v:>15.2f}%  {p10_v:>12.2f}%  {p90_v:>12.2f}%  {err_str}")

                daily_rows.append({
                    "date": dt_str,
                    "actual": round(float(act_v), 2) if not np.isnan(act_v) else None,
                    "forecast_p50": round(float(p50_v), 2),
                    "forecast_p10": round(float(p10_v), 2),
                    "forecast_p90": round(float(p90_v), 2),
                    "abs_error": round(float(abs(act_v - p50_v)), 2) if not np.isnan(act_v) else None,
                })

        explainability_data = _build_explainability(
            target, cfg, panel, cutoff_idx, interpolated_preds, models_to_run[0], eval_summary=model_eval_summary
        )

        print("\n" + "─" * 105)
        print("  EXPLAINABILITY & DRIVER INTERDEPENDENCY ANALYSIS")
        print(f"  Target Metric : {metric_name} | {host_label}")
        print(f"  Forecast P50  : {explainability_data['forecast']['p50_median']:.2f}% (P5: {explainability_data['forecast']['p05_or_p10']:.2f}% | P95: {explainability_data['forecast']['p95_or_p90']:.2f}%)")
        if explainability_data.get("accuracy"):
            acc = explainability_data["accuracy"]
            print(f"  Model Accuracy: {acc['forecast_accuracy_pct']:.2f}%  (MAE: {acc['mae']:.2f}%, Band Coverage: {acc['coverage_pct']:.1f}%)")
        print(f"  Baseline Shift: {explainability_data['predicted_shift']:+.2f}% vs 30-day historical average ({explainability_data['baseline_30d_avg']:.2f}%)")
        print(f"  Confidence    : {explainability_data['confidence_score']:.1f}% | Risk Level: {explainability_data['risk_level']}")
        print(f"  Summary       : {explainability_data['summary']}")
        
        if explainability_data["top_contributing_drivers"]:
            print("\n  Top Interdependency Drivers:")
            for d_idx_item, drv in enumerate(explainability_data["top_contributing_drivers"], 1):
                print(f"    {d_idx_item}. {drv['metric']:<30} {drv['impact']:>6}  (Strength: {drv['strength']:.3f}, Lag: {drv['lag']})")
        print("─" * 105)

        validation_results[target] = {
            "host": host_label,
            "target_node": target,
            "models_evaluated": models_to_run,
            "metrics": model_eval_summary,
            "explainability": explainability_data,
            "daily_comparison": daily_rows,
        }

        # Plot Comparison / Validation Chart
        if HAS_MPL:
            fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0d1117")
            ax.set_facecolor("#161b22")
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")
            ax.tick_params(colors="#c9d1d9", which="both")

            # Context history (dynamically reads input_steps from model metadata / config)
            context_len  = ref_meta.get("input_steps", cfg.forecast.input_steps)
            context_vals = panel.values.iloc[max(0, cutoff_idx - context_len):cutoff_idx + 1][target].dropna()
            if not context_vals.empty:
                ax.plot(context_vals.index, context_vals.values, color="#8b949e", linewidth=1.5, label=f"Context History ({context_len}d before {st_str})", alpha=0.7)

            # Actuals
            if has_actuals:
                ax.plot(forecast_dates, actual_in_span, color="#3fb950", linewidth=2.5, label=f"Actual {metric_name}", marker="o", markersize=4, zorder=5)

            # ST-GNN Graph Model Plot (Primary Model)
            if "stgnn_graph" in interpolated_preds:
                g_p50 = interpolated_preds["stgnn_graph"]["p50"]
                g_p10 = interpolated_preds["stgnn_graph"]["p10"]
                g_p90 = interpolated_preds["stgnn_graph"]["p90"]

                # Extrapolate P100 (Max Envelope) and P0 (Min Envelope)
                g_p100 = np.clip(g_p90 + 1.28 * (g_p90 - g_p50), 0.0, 100.0)
                g_p0   = np.clip(g_p10 - 1.28 * (g_p50 - g_p10), 0.0, 100.0)
                # Capacity Required (+25% over P100)
                g_cap  = np.clip(g_p100 * 1.25, 0.0, 100.0)

                # Plot P50 and P5-P95 Band
                ax.plot(forecast_dates, g_p50, color="#58a6ff", linewidth=2.2, linestyle="--", label="ST-GNN (Graph Prior) P50", marker="s", markersize=4, zorder=6)
                ax.fill_between(forecast_dates, g_p10, g_p90, color="#58a6ff", alpha=0.18, label="ST-GNN Graph P5-P95 Band")

                # Plot P100 (Max Envelope) and P0 (Min Envelope) dotted lines
                ax.plot(forecast_dates, g_p100, color="#d2a8ff", linewidth=1.4, linestyle=":", label="P100 (Max Envelope)", zorder=5)
                ax.plot(forecast_dates, g_p0, color="#79c0ff", linewidth=1.4, linestyle=":", label="P0 (Min Envelope)", zorder=5)

                # Plot Capacity Required (+25% over P100) line
                ax.plot(forecast_dates, g_cap, color="#f85149", linewidth=2.0, linestyle="-.", label="Capacity Required (+25% over P100)", zorder=7)

                # Annotations at final forecast date
                last_dt   = forecast_dates[-1]
                last_p50  = g_p50[-1]
                last_p100 = g_p100[-1]
                last_cap  = g_cap[-1]

                ax.annotate(
                    f"P50: {last_p50:.2f}%",
                    xy=(last_dt, last_p50),
                    xytext=(12, -4),
                    textcoords="offset points",
                    color="#58a6ff",
                    fontsize=8.0,
                    fontweight="bold",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#0d1117", edgecolor="#58a6ff", alpha=0.95),
                    zorder=10
                )
                ax.annotate(
                    f"P100: {last_p100:.2f}%",
                    xy=(last_dt, last_p100),
                    xytext=(12, 8),
                    textcoords="offset points",
                    color="#d2a8ff",
                    fontsize=8.0,
                    fontweight="bold",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#0d1117", edgecolor="#d2a8ff", alpha=0.95),
                    zorder=10
                )
                ax.annotate(
                    f"Cap Req: {last_cap:.2f}%",
                    xy=(last_dt, last_cap),
                    xytext=(12, 20),
                    textcoords="offset points",
                    color="#f85149",
                    fontsize=8.0,
                    fontweight="bold",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#0d1117", edgecolor="#f85149", alpha=0.95),
                    zorder=10
                )

            # Annotate final Actual value on chart if ground truth is present
            if has_actuals:
                last_dt = forecast_dates[-1]
                last_act = actual_in_span[-1]
                if not np.isnan(last_act):
                    ax.annotate(
                        f"Actual: {last_act:.2f}%",
                        xy=(last_dt, last_act),
                        xytext=(12, -16),
                        textcoords="offset points",
                        color="#3fb950",
                        fontsize=8.0,
                        fontweight="bold",
                        va="center",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d1117", edgecolor="#3fb950", alpha=0.95),
                        zorder=10
                    )

            ax.axvline(cutoff_dt, color="#f78166", linestyle=":", linewidth=1.5, label=f"Cutoff ({cutoff_dt.date()})")

            # Increase title pad to accommodate top horizontal legend bar cleanly
            ax.set_title(f"Forecast ({st_str} to {et_str}) | {metric_name} | {host_label}", color="#c9d1d9", fontsize=11, pad=38, loc="left", fontweight="bold")
            ax.set_xlabel("Date", color="#8b949e", fontsize=9)
            ax.set_ylabel(metric_name, color="#8b949e", fontsize=9)

            # Calculate total plotted span (context history + forecast horizon) to prevent X-axis tick congestion
            total_span_days = context_len + n_forecast_days
            if total_span_days <= 30:
                tick_interval = 2
            elif total_span_days <= 60:
                tick_interval = 5
            elif total_span_days <= 120:
                tick_interval = 10
            else:
                tick_interval = 15

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=tick_interval))
            plt.xticks(rotation=30, ha="right", fontsize=8, color="#8b949e")
            plt.yticks(fontsize=8, color="#8b949e")

            # Extend right margin slightly for callout badges
            ax.set_xlim(right=forecast_dates[-1] + pd.Timedelta(days=5))

            # Lock Y-axis lower bound to 0 and top bound above Capacity Required line
            current_bottom, current_top = ax.get_ylim()
            max_val = max(
                np.nanmax(actual_in_span) if has_actuals else 0.0,
                np.nanmax(context_vals.values) if not context_vals.empty else 0.0,
                np.nanmax(g_cap) if "stgnn_graph" in interpolated_preds else 0.0,
            )
            ax.set_ylim(bottom=0.0, top=max(current_top, max_val * 1.15))

            # Place legend horizontally above the plot area to eliminate overlap with data lines
            ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=4,
                frameon=True,
                facecolor="#161b22",
                edgecolor="#30363d",
                labelcolor="#c9d1d9",
                fontsize=8,
                handletextpad=0.5,
                columnspacing=1.2,
            )
            plt.tight_layout()

            safe_t = target.replace("|", "_")
            model_tag_slug = "compare" if len(models_to_run) > 1 else models_to_run[0]
            chart_file = chart_dir / f"forecast_{model_tag_slug}_{st_str}_to_{et_str}_{safe_t}.png"
            fig.savefig(chart_file, dpi=130, bbox_inches="tight", facecolor="#0d1117")
            plt.close(fig)

    # Save JSON report
    val_json = art_dir / f"forecast_report_{st_str}_to_{et_str}.json"
    val_json.write_text(json.dumps(validation_results, indent=2))
    print("\n" + "=" * 105)
    print(f"  Forecast Report JSON saved : {val_json}")
    print(f"  Forecast Charts saved      : {chart_dir}/forecast_*.png")
    print("=" * 105)

    return validation_results


def main():
    parser = argparse.ArgumentParser(description="ST-GNN Dynamic Multi-Host Forecast & Validation Script (forecast_script.py)")
    parser.add_argument("-c", "--config", default="configs/v4_multihost_v1.yaml", help="Path to model config YAML")
    parser.add_argument("--st", default="2026-08-01", help="Forecast start date YYYY-MM-DD (default: 2026-08-01)")
    parser.add_argument("--et", default="2026-08-28", help="Forecast end date YYYY-MM-DD (default: 2026-08-28)")
    parser.add_argument(
        "-m", "--model", default="graph",
        choices=["graph", "stgnn_graph", "nograph", "stgnn_nograph", "both", "compare"],
        help="Model choice: 'graph' (ST-GNN with Graph Prior), 'nograph' (ST-GNN Ablation), or 'both' (Compare Both Models Side-by-Side)"
    )
    args = parser.parse_args()

    run_validation(args.config, args.st, args.et, args.model)


if __name__ == "__main__":
    main()
