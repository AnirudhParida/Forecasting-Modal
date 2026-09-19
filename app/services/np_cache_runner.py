"""
app/services/np_cache_runner.py
================================
Offline NeuralProphet Forecast Cache Runner.

Run ONCE (or on new data) to pre-compute NeuralProphet forecasts for all
hosts and all horizons (15d, 30d, 60d, 90d). The FastAPI endpoints read
the cached JSON files — no live model training at request time.

Cache layout (relative to Forecasting-Modal root):
  artifacts_np/
    HYDUPINTAPP16_forecast_15d.json
    HYDUPINTAPP16_forecast_30d.json
    HYDUPINTAPP16_forecast_60d.json
    HYDUPINTAPP16_forecast_90d.json
    JPRUPIWEBCRP02_forecast_15d.json
    ...

Each JSON file structure:
  {
    "host": "HYDUPINTAPP16",
    "horizon_days": 30,
    "as_of_date": "2026-09-04",
    "generated_at": "2026-09-05T01:15:00",
    "metrics": {
      "cpu_pct": {
        "label": "CPU Usage (%)",
        "unit": "%",
        "evaluation": {"mae": 2.54, "rmse": 3.27, "mape": 142.0},
        "forecast": [
          {"date": "2026-09-05", "yhat": 12.3, "yhat_lower": 9.1, "yhat_upper": 15.8},
          ...
        ]
      }
    }
  }

Usage:
  cd /home/anirudh.parida@apmosys.mahape/Documents/Forecasting-Modal
  python -m app.services.np_cache_runner
  python -m app.services.np_cache_runner --host HYDUPINTAPP16
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — make the NeuralProphet pipeline importable
# ---------------------------------------------------------------------------
NP_PIPELINE_ROOT = Path("/home/anirudh.parida@apmosys.mahape/Documents/Phrophet_forecast")
if str(NP_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(NP_PIPELINE_ROOT))

# pyrefly: ignore [missing-import]
from pipeline.forecaster import ServerMetricsForecaster 
# pyrefly: ignore [missing-import]
 # noqa: E402
from pipeline.evaluation import compute_metrics      
# pyrefly: ignore [missing-import]
     # noqa: E402
from pipeline.visualization import extract_forecast_rows 
# pyrefly: ignore [missing-import]
 # noqa: E402
from config.settings import HOSTS, METRIC_LABELS          # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts_np"
HORIZONS = [15, 30, 60, 90]      # pre-compute all four
MAX_HORIZON = max(HORIZONS)       # run forecaster once for 90d, then slice

METRIC_UNITS: dict[str, str] = {
    "cpu_pct":          "%",
    "memory_pct":       "%",
    "disk_pct":         "%",
    "disk_read_bytes":  "KB/s",
    "disk_read_ops":    "ops/s",
    "disk_write_bytes": "KB/s",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("np_cache_runner")


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_for_host(host_alias: str) -> None:
    """Fit NeuralProphet for one host and write all horizon JSON cache files."""
    logger.info("=" * 60)
    logger.info("Host: %s  |  Max horizon: %dd", host_alias, MAX_HORIZON)
    logger.info("=" * 60)

    forecaster = ServerMetricsForecaster(n_forecasts=MAX_HORIZON)

    all_data = forecaster.load_and_preprocess()
    if host_alias not in all_data:
        logger.error(
            "Host '%s' not found. Available: %s", host_alias, list(all_data.keys())
        )
        return

    logger.info("Fitting model for %s …", host_alias)
    forecaster.fit(host_alias)

    logger.info("Predicting %d days ahead …", MAX_HORIZON)
    raw_forecasts = forecaster.predict(host_alias, save_charts=False)

    full_df = all_data[host_alias]
    as_of_date = full_df["ds"].max().date()
    generated_at = datetime.now().isoformat(timespec="seconds")

    # -----------------------------------------------------------------------
    # Build full 90-day per-metric payload
    # -----------------------------------------------------------------------
    metric_data: dict[str, dict] = {}
    for metric, forecast_df in raw_forecasts.items():
        rows = extract_forecast_rows(forecast_df, metric=metric, n_steps=MAX_HORIZON)
        if rows is None or rows.empty:
            logger.warning("  [%s] No forecast rows extracted — skipping.", metric)
            continue

        # Attempt evaluation on last MAX_HORIZON rows of history
        eval_metrics: dict[str, object] = {"mae": None, "rmse": None, "mape": None}
        try:
            hist_tail = full_df.tail(MAX_HORIZON)
            actuals = hist_tail[metric].values
            if "yhat1" in forecast_df.columns:
                hist_mask = forecast_df["y"].notna()
                hist_yhat = forecast_df[hist_mask]["yhat1"].values[-len(actuals):]
                if len(hist_yhat) == len(actuals):
                    em = compute_metrics(actuals, hist_yhat, metric)
                    eval_metrics = {
                        "mae":  round(float(em.get("mae", 0.0)), 4),
                        "rmse": round(float(em.get("rmse", 0.0)), 4),
                        "mape": round(float(em.get("mape", 0.0)), 4),
                    }
        except Exception as exc:
            logger.debug("  [%s] Evaluation skipped: %s", metric, exc)

        forecast_list = []
        for _, row in rows.iterrows():
            date_str = (
                str(row["ds"].date())
                if hasattr(row["ds"], "date")
                else str(row["ds"])[:10]
            )
            forecast_list.append({
                "date":       date_str,
                "yhat":       round(float(row["yhat"]), 4),
                "yhat_lower": round(float(row["yhat_lower"]), 4)
                              if "yhat_lower" in rows.columns else None,
                "yhat_upper": round(float(row["yhat_upper"]), 4)
                              if "yhat_upper" in rows.columns else None,
            })

        metric_data[metric] = {
            "label":      METRIC_LABELS.get(metric, metric),
            "unit":       METRIC_UNITS.get(metric, ""),
            "evaluation": eval_metrics,
            "forecast":   forecast_list,
        }

    logger.info("Extracted forecasts for metrics: %s", list(metric_data.keys()))

    # -----------------------------------------------------------------------
    # Slice and write one file per horizon
    # -----------------------------------------------------------------------
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for horizon in HORIZONS:
        sliced: dict[str, dict] = {}
        for metric, mdata in metric_data.items():
            sliced[metric] = {**mdata, "forecast": mdata["forecast"][:horizon]}

        payload = {
            "host":         host_alias,
            "horizon_days": horizon,
            "as_of_date":   str(as_of_date),
            "generated_at": generated_at,
            "metrics":      sliced,
        }

        out_path = CACHE_DIR / f"{host_alias}_forecast_{horizon}d.json"
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("  Written → %s", out_path.name)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NeuralProphet offline forecast cache runner."
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host alias (e.g. HYDUPINTAPP16). Omit to run ALL hosts.",
    )
    args = parser.parse_args()

    hosts_to_run = [args.host] if args.host else list(HOSTS.keys())
    logger.info("Hosts to process: %s", hosts_to_run)

    for host in hosts_to_run:
        try:
            run_for_host(host)
        except Exception as exc:
            logger.error("Failed for host %s: %s", host, exc, exc_info=True)

    logger.info("Cache runner complete. Artifacts at: %s", CACHE_DIR)


if __name__ == "__main__":
    main()
