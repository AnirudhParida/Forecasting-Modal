"""Training loop for the ST-GNN.

One run produces both halves of the design: the forecaster, and the learned
adjacency that was supervised by its error and regularised toward the
statistical prior.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader, TensorDataset

from ..config import ModelConfig
from .dataset import Dataset
from .stgnn import STGNN, masked_quantile_loss

log = logging.getLogger(__name__)


@dataclass
class TrainResult:
    model: STGNN
    history: list[dict] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    use_graph: bool = True

    def format(self) -> str:
        tag = "with graph" if self.use_graph else "graph ablation"
        return (
            f"{tag}: best val loss {self.best_val_loss:.5f} at epoch "
            f"{self.best_epoch} ({len(self.history)} epochs run)"
        )


def _loaders(
    ds: Dataset, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    def make(idx: np.ndarray, shuffle: bool) -> DataLoader:
        tensors = TensorDataset(
            torch.from_numpy(ds.X[idx]),
            torch.from_numpy(ds.Y[idx]),
            torch.from_numpy(ds.Y_mask[idx]),
        )
        return DataLoader(
            tensors,
            batch_size=min(batch_size, max(1, len(idx))),
            shuffle=shuffle,
            drop_last=False,
        )

    return make(ds.train_idx, True), make(ds.val_idx, False)


def train(
    ds: Dataset,
    cfg: ModelConfig,
    quantiles: list[float],
    adj_prior: np.ndarray | None = None,
    use_graph: bool = True,
    device: str = "cpu",
    verbose: bool = True,
) -> TrainResult:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ── Host-isolation mask ───────────────────────────────────────────────────
    # Build an N×N boolean mask where mask[i, j] = 1 only when node i and
    # node j belong to the same host.  For single-host models all entries are 1
    # (no restriction). The mask is registered as a non-trainable buffer inside
    # AdaptiveAdjacency and permanently zeroes cross-host entries before the
    # top-k sparsification step, so gradients can never promote cross-host edges.
    node_ids = ds.node_ids
    N = len(node_ids)
    def _host_of(nid: str) -> str:
        return nid.split("|", 1)[1] if "|" in nid else ""
    hosts = [_host_of(nid) for nid in node_ids]
    same_host_mask: np.ndarray | None = None
    if any(h != "" for h in hosts):            # multi-host panel
        mask_arr = np.zeros((N, N), dtype="float32")
        for ii in range(N):
            for jj in range(N):
                if hosts[ii] == hosts[jj]:
                    mask_arr[ii, jj] = 1.0
        same_host_mask = mask_arr
        n_cross = int((mask_arr == 0).sum()) - N   # exclude diagonal
        log.info(
            "host-isolation mask: %d same-host entries kept, "
            "%d cross-host entries permanently zeroed",
            int(mask_arr.sum()) - N,
            n_cross,
        )

    model = STGNN(
        n_nodes=len(ds.node_ids),
        n_targets=len(ds.target_ids),
        in_channels=ds.n_channels,
        input_steps=ds.input_steps,
        n_horizons=len(ds.horizons),
        n_quantiles=len(quantiles),
        target_idx=ds.target_idx,
        hidden=cfg.hidden,
        blocks=cfg.blocks,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout,
        node_embed_dim=cfg.node_embed_dim,
        top_k=cfg.top_k,
        use_graph=use_graph,
        adj_prior=adj_prior if use_graph else None,
        same_host_mask=same_host_mask if use_graph else None,
    ).to(device)

    # All targets weighted equally. CPU/Memory weights (1.5× previously) caused
    # best_epoch=3 because those 2 targets converge fast and stall val-loss improvement
    # before disk/JVM metrics learn. CPU already wins by 46% without extra pressure.
    target_weights = torch.ones(len(ds.target_ids), dtype=torch.float32, device=device)

    optimiser = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=cfg.epochs, eta_min=1e-6
    )

    train_loader, val_loader = _loaders(ds, cfg.batch_size)
    result = TrainResult(model=model, use_graph=use_graph)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    stale = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss, n_batches = 0.0, 0
        for xb, yb, mb in train_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            if cfg.input_noise > 0:
                # Jitter only the value channel, only where observed — noise on
                # a masked-out zero would contradict the mask channel.
                noise = torch.randn_like(xb[..., 0]) * cfg.input_noise
                xb = xb.clone()
                xb[..., 0] += noise * xb[..., 1]
            optimiser.zero_grad()
            pred = model(xb)
            loss = masked_quantile_loss(pred, yb, mb, quantiles, target_weights)
            total = loss + model.graph_regularisation(
                cfg.graph_prior_weight, cfg.graph_sparsity_weight
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            train_loss += loss.detach().item()
            n_batches += 1
        train_loss /= max(1, n_batches)

        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                val_loss += float(masked_quantile_loss(model(xb), yb, mb, quantiles, target_weights))
                n_val += 1
        val_loss = val_loss / n_val if n_val else float("nan")

        scheduler.step()
        result.history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )

        if val_loss < result.best_val_loss - 1e-6:
            result.best_val_loss = val_loss
            result.best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            log.info(
                "epoch %3d  train %.5f  val %.5f  (best %.5f @ %d)",
                epoch, train_loss, val_loss, result.best_val_loss, result.best_epoch,
            )

        if stale >= cfg.patience:
            log.info("early stop at epoch %d (no val improvement for %d)", epoch, stale)
            break

    model.load_state_dict(best_state)
    return result


@torch.no_grad()
def predict(
    model: STGNN, ds: Dataset, idx: np.ndarray, device: str = "cpu", batch_size: int = 32
) -> np.ndarray:
    """Predictions for the given window indices -> (S, N_target, H, Q)."""
    model.eval()
    out = []
    for start in range(0, len(idx), batch_size):
        chunk = idx[start : start + batch_size]
        xb = torch.from_numpy(ds.X[chunk]).to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0,))


def save_artifacts(
    result: TrainResult,
    ds: Dataset,
    artifacts_dir: Path,
    quantiles: list[float],
    tag: str = "stgnn",
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), artifacts_dir / f"{tag}.pt")
    ds.scaler.save(artifacts_dir / f"{tag}_scaler.json")
    # Save linear detrend slopes so forecast_next_30d.py can re-add the trend
    # to predicted values at inference time (otherwise predictions are of the
    # de-trended residuals, not the original-scale metric).
    ds.detrend.save(artifacts_dir / f"{tag}_trend.json")

    meta = {
        "tag": tag,
        "use_graph": result.use_graph,
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "node_ids": ds.node_ids,
        "target_ids": ds.target_ids,
        "horizons": ds.horizons,
        "quantiles": quantiles,
        "input_steps": ds.input_steps,
        "n_channels": ds.n_channels,
    }
    (artifacts_dir / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2))
    (artifacts_dir / f"{tag}_history.json").write_text(
        json.dumps(result.history, indent=2)
    )

    if result.use_graph and result.model.adjacency is not None:
        (artifacts_dir / f"{tag}_learned_graph.json").write_text(
            json.dumps(
                {
                    "agreement_with_prior": result.model.adjacency.agreement(),
                    "edges": result.model.adjacency.export(ds.node_ids),
                },
                indent=2,
            )
        )
