"""Chronological backtest: ST-GNN vs graph ablation vs classical baselines.

The ablation is the point of this module. `stgnn_graph` and `stgnn_nograph` are
the same architecture, the same data and the same seed, differing only in
whether the graph convolutions run. If the graph version does not win, the
dependency structure is not earning its place in the forecaster and the report
should say so rather than bury it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import ModelConfig
from ..models import baselines
from ..models.dataset import Dataset, inverse_scale
from ..models.train import TrainResult, predict, train
from . import metrics

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    scores: dict[str, dict] = field(default_factory=dict)
    per_target: dict[str, dict] = field(default_factory=dict)
    graph_agreement: dict = field(default_factory=dict)
    training: dict[str, dict] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @staticmethod
    def _rank_metric(score: dict) -> tuple[str, float]:
        """MASE when available, MAE otherwise. MASE weights every target
        equally; aggregate MAE is decided by whichever target has the largest
        units."""
        m = score.get("mase", float("nan"))
        if np.isfinite(m):
            return "MASE", m
        return "MAE", score.get("mae", float("nan"))

    def ranking(self) -> list[tuple[str, float]]:
        return sorted(
            ((k, self._rank_metric(v)[1]) for k, v in self.scores.items()),
            key=lambda kv: (np.isnan(kv[1]), kv[1]),
        )

    def verdict(self) -> str:
        rank = self.ranking()
        if not rank:
            return "no models evaluated"
        winner, best_score = rank[0]
        metric_name = self._rank_metric(self.scores[winner])[0]

        g = self._rank_metric(self.scores.get("stgnn_graph", {}))[1]
        ng = self._rank_metric(self.scores.get("stgnn_nograph", {}))[1]
        best_baseline = min(
            (
                self._rank_metric(v)[1]
                for k, v in self.scores.items()
                if not k.startswith("stgnn")
            ),
            default=float("nan"),
        )

        lines = [f"Best model: {winner} ({metric_name} {best_score:.4f})"]

        if np.isfinite(g) and np.isfinite(ng):
            delta = (ng - g) / ng * 100 if ng else 0.0
            if delta > 1.0:
                lines.append(
                    f"The graph earns its place: {delta:.1f}% lower {metric_name} "
                    f"than the identical graph-free model."
                )
            else:
                lines.append(
                    f"The graph does NOT earn its place on point error: "
                    f"{delta:+.1f}% vs the graph-free model. Ship the simpler "
                    f"model, or revisit the dependency map before trusting it "
                    f"in the forecaster."
                )

        if np.isfinite(best_baseline) and np.isfinite(g):
            if g < best_baseline:
                lines.append(
                    f"Beats the best classical baseline ({best_baseline:.4f})."
                )
            else:
                lines.append(
                    f"Does NOT beat the best classical baseline ({best_baseline:.4f}). "
                    f"At a quarter horizon a trend line is a strong competitor — "
                    f"treat this as the headline result, not a footnote."
                )

        # Point error is not the whole product: capacity planning consumes the
        # P10-P90 band, and the band is judged by pinball loss and coverage.
        pin = {
            k: v.get("pinball", float("nan"))
            for k, v in self.scores.items()
            # Point-only baselines report median-only pinball (= MAE/2), which
            # is not comparable to a multi-quantile average — only rank models
            # that actually emit a band (finite interval coverage).
            if np.isfinite(v.get("pinball", float("nan")))
            and np.isfinite(v.get("coverage_p10_p90", float("nan")))
        }
        if pin:
            pin_winner = min(pin, key=pin.get)
            cov = self.scores[pin_winner].get("coverage_p10_p90", float("nan"))
            cov_s = f", P10-90 coverage {cov:.1f}% (nominal 80%)" if np.isfinite(cov) else ""
            lines.append(
                f"Best probabilistic model: {pin_winner} "
                f"(pinball {pin[pin_winner]:.4f}{cov_s}). If the deliverable is "
                f"the capacity band rather than a point line, judge on this."
            )
        return "\n".join(lines)

    def format(self) -> str:
        hdr = (
            f"{'model':<20}{'MASE':>9}{'MAE':>11}{'RMSE':>11}{'sMAPE %':>11}"
            f"{'pinball':>11}{'P10-90 cov %':>14}"
        )
        lines = [hdr, "-" * len(hdr)]
        for name, _ in self.ranking():
            s = self.scores[name]
            m = s.get("mase", float("nan"))
            m_s = f"{m:.3f}" if np.isfinite(m) else "-"
            cov = s.get("coverage_p10_p90", float("nan"))
            cov_s = f"{cov:.1f}" if np.isfinite(cov) else "-"
            lines.append(
                f"{name:<20}{m_s:>9}{s['mae']:>11.4f}{s['rmse']:>11.4f}"
                f"{s['smape']:>11.2f}{s['pinball']:>11.4f}{cov_s:>14}"
            )

        lines += ["", "Per-horizon MAE:"]
        horizons = list(next(iter(self.scores.values()))["per_horizon"].keys())
        lines.append(f"{'model':<20}" + "".join(f"{'h=' + h:>11}" for h in horizons))
        for name, _ in self.ranking():
            row = self.scores[name]["per_horizon"]
            lines.append(
                f"{name:<20}" + "".join(f"{row[h]['mae']:>11.4f}" for h in horizons)
            )

        # Per-target breakdown for the winning model: the aggregate row above
        # hides which series the model actually fails on.
        best_name = self.ranking()[0][0]
        pt = self.per_target.get(best_name, {})
        if pt:
            lines += ["", f"Per-target ({best_name}, median forecast), worst sMAPE first:"]
            lines.append(f"  {'target':<44}{'MAE':>12}{'MASE':>9}{'sMAPE %':>10}")
            rows = sorted(
                pt.items(),
                key=lambda kv: -(kv[1].get("smape") or 0.0),
            )
            for tid, s in rows[:8]:
                m = s.get("mase", float("nan"))
                m_s = f"{m:.3f}" if np.isfinite(m) else "-"
                lines.append(
                    f"  {tid[:43]:<44}{s['mae']:>12.4f}{m_s:>9}{s['smape']:>10.2f}"
                )
            if len(rows) > 8:
                lines.append(f"  ... {len(rows) - 8} more targets in the saved JSON")

        if self.graph_agreement:
            a = self.graph_agreement
            lines += [
                "",
                "Learned graph vs statistical prior:",
                f"  learned edges     {a['learned_edges']}",
                f"  prior edges       {a['prior_edges']}",
                f"  overlap           {a['overlap']}",
                f"  precision v prior {a['precision_vs_prior']:.2%}",
                f"  recall v prior    {a['recall_vs_prior']:.2%}",
            ]

        lines += ["", self.verdict()]
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "scores": self.scores,
                    "per_target": self.per_target,
                    "graph_agreement": self.graph_agreement,
                    "training": self.training,
                    "meta": self.meta,
                },
                indent=2,
                default=float,
            )
        )


def _mase_scale(ds: Dataset, season: int) -> np.ndarray:
    """Per-target MASE denominator: in-sample seasonal-naive MAE, original units.

    Computed on the training slice only — the same cutoff the climatology
    baseline uses — so the scale itself cannot leak test information.
    """
    values = ds.scaler.inverse_transform_array(ds.values_scaled, ds.node_ids)
    train_end = (
        int(ds.starts[ds.train_idx[-1]] + ds.input_steps)
        if len(ds.train_idx)
        else values.shape[0]
    )
    v = values[:train_end][:, ds.target_idx]
    m = ds.mask[:train_end][:, ds.target_idx]

    scale = np.ones(len(ds.target_idx), dtype="float64")
    for lag in (season, 1):  # fall back to lag-1 when a season never overlaps
        both = m[lag:] * m[:-lag]
        diff = np.abs(np.nan_to_num(v[lag:]) - np.nan_to_num(v[:-lag])) * both
        n = both.sum(axis=0)
        ok = (n > 0) & (diff.sum(axis=0) / np.maximum(n, 1) > 1e-9)
        if lag == season:
            filled = ok.copy()
            scale = np.where(ok, diff.sum(axis=0) / np.maximum(n, 1), scale)
        else:
            use = ~filled & ok
            scale = np.where(use, diff.sum(axis=0) / np.maximum(n, 1), scale)
    return scale


def run(
    ds: Dataset,
    cfg: ModelConfig,
    quantiles: list[float],
    adj_prior: np.ndarray | None,
    season: int,
    device: str = "cpu",
    run_ablation: bool = True,
) -> tuple[BacktestResult, TrainResult]:
    result = BacktestResult()
    test_idx = ds.test_idx
    if len(test_idx) == 0:
        raise ValueError("empty test split — the panel is too short to backtest")

    # Actuals in original units.
    y_true = inverse_scale(ds.Y[test_idx], ds.target_ids, ds.scaler)
    y_mask = ds.Y_mask[test_idx]
    mase_scale = _mase_scale(ds, season)

    # ------------------------------------------------------------- baselines
    base_preds = baselines.run_all(
        values=ds.values_scaled,
        mask=ds.mask,
        starts=ds.starts[test_idx],
        input_steps=ds.input_steps,
        horizons=ds.horizons,
        target_idx=ds.target_idx,
        season=season,
        train_values=ds.values_scaled[: ds.starts[ds.train_idx[-1]] + ds.input_steps]
        if len(ds.train_idx)
        else ds.values_scaled,
    )
    for name, pred in base_preds.items():
        pred = inverse_scale(pred, ds.target_ids, ds.scaler)
        result.scores[name] = metrics.evaluate(
            y_true, pred, y_mask, quantiles, ds.horizons, mase_scale=mase_scale
        )
        result.per_target[name] = metrics.per_target(
            y_true, pred, y_mask, ds.target_ids, mase_scale=mase_scale
        )

    # ------------------------------------------------------------ ST-GNN runs
    variants = [("stgnn_graph", True)]
    if run_ablation:
        variants.append(("stgnn_nograph", False))

    trained: TrainResult | None = None
    for name, use_graph in variants:
        log.info("training %s", name)
        tr = train(
            ds, cfg, quantiles, adj_prior=adj_prior, use_graph=use_graph, device=device
        )
        result.training[name] = {
            "best_epoch": tr.best_epoch,
            "best_val_loss": tr.best_val_loss,
            "epochs_run": len(tr.history),
            "history": tr.history,   # full epoch log for training-curve charts
        }

        raw = predict(tr.model, ds, test_idx, device=device)
        pred = inverse_scale(raw, ds.target_ids, ds.scaler)
        result.scores[name] = metrics.evaluate(
            y_true, pred, y_mask, quantiles, ds.horizons, mase_scale=mase_scale
        )
        qi = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
        result.per_target[name] = metrics.per_target(
            y_true, pred[..., qi], y_mask, ds.target_ids, mase_scale=mase_scale
        )

        if use_graph:
            trained = tr
            if tr.model.adjacency is not None:
                result.graph_agreement = tr.model.adjacency.agreement()

    result.meta = {
        "n_test_windows": int(len(test_idx)),
        "test_start": str(ds.window_time(int(test_idx[0]))),
        "test_end": str(ds.window_time(int(test_idx[-1]))),
        "n_nodes": len(ds.node_ids),
        "n_targets": len(ds.target_ids),
        "horizons": ds.horizons,
        "input_steps": ds.input_steps,
    }
    return result, trained
