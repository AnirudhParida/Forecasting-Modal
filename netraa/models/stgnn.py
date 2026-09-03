"""Spatio-temporal GNN forecaster (MTGNN / Graph-WaveNet family).

Dilated causal convolutions carry information along time; graph convolutions
carry it between metrics. Stacking them lets "database_access_cpm rose 15
minutes ago" reach "host_cpu_usage next quarter" through both axes.

`use_graph=False` disables every graph convolution while leaving the temporal
stack untouched. That is the ablation in eval/backtest.py: if the full model
cannot beat this, the graph is decoration and we should say so rather than ship
it.
"""

from __future__ import annotations

import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.nn.functional as F

from ..graph.learned import AdaptiveAdjacency


class MixHopPropagation(nn.Module):
    """Propagate node features over the graph, keeping every hop order.

    Concatenating hops instead of only using the deepest one avoids
    over-smoothing, which on a 60-100 node graph would otherwise collapse every
    metric onto the same representation within two layers.
    """

    def __init__(self, c_in: int, c_out: int, order: int = 2, dropout: float = 0.0):
        super().__init__()
        self.order = order
        self.dropout = dropout
        self.mlp = nn.Conv2d((order + 1) * c_in, c_out, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N, T);  adj[i, j] = influence of node i on node j
        out = [x]
        h = x
        for _ in range(self.order):
            h = torch.einsum("bcnt,nm->bcmt", h, adj)
            out.append(h)
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return F.dropout(h, self.dropout, training=self.training)


class GatedTemporalConv(nn.Module):
    """Gated dilated causal convolution along the time axis."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.filter = nn.Conv2d(
            channels, channels, kernel_size=(1, kernel_size), dilation=(1, dilation)
        )
        self.gate = nn.Conv2d(
            channels, channels, kernel_size=(1, kernel_size), dilation=(1, dilation)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.filter(x)) * torch.sigmoid(self.gate(x))


class STBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        use_graph: bool,
    ):
        super().__init__()
        self.use_graph = use_graph
        self.temporal = GatedTemporalConv(channels, kernel_size, dilation)
        self.graph = (
            MixHopPropagation(channels, channels, order=2, dropout=dropout)
            if use_graph
            else None
        )
        self.norm = nn.BatchNorm2d(channels)
        self.skip = nn.Conv2d(channels, channels, kernel_size=(1, 1))

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.temporal(x)
        if self.use_graph and adj is not None:
            h = self.graph(h, adj)
        residual = x[..., -h.size(3):]          # align time after the dilation
        out = self.norm(h + residual)
        return out, self.skip(h)


class STGNN(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        n_targets: int,
        in_channels: int,
        input_steps: int,
        n_horizons: int,
        n_quantiles: int,
        target_idx: list[int],
        hidden: int = 32,
        blocks: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.2,
        node_embed_dim: int = 16,
        top_k: int = 8,
        use_graph: bool = True,
        adj_prior: np.ndarray | None = None,
        same_host_mask: np.ndarray | None = None,
    ):
        super().__init__()
        self.use_graph = use_graph
        self.n_horizons = n_horizons
        self.n_quantiles = n_quantiles
        self.register_buffer("target_idx", torch.tensor(target_idx, dtype=torch.long))

        receptive = sum((kernel_size - 1) * (2**b) for b in range(blocks))
        if input_steps <= receptive:
            raise ValueError(
                f"input_steps={input_steps} is too short for blocks={blocks}, "
                f"kernel_size={kernel_size} (needs > {receptive}). "
                "Increase forecast.input_steps or reduce model.blocks."
            )
        self.remaining_steps = input_steps - receptive

        self.start = nn.Conv2d(in_channels, hidden, kernel_size=(1, 1))
        self.blocks = nn.ModuleList(
            [
                STBlock(hidden, kernel_size, 2**b, dropout, use_graph)
                for b in range(blocks)
            ]
        )
        self.adjacency = (
            AdaptiveAdjacency(
                n_nodes, node_embed_dim, top_k,
                prior=adj_prior,
                same_host_mask=same_host_mask,
            )
            if use_graph
            else None
        )

        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden, hidden * 2, kernel_size=(1, self.remaining_steps)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden * 2, n_horizons * n_quantiles, kernel_size=(1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, T, C) -> (B, N_target, H, Q)."""
        h = self.start(x.permute(0, 3, 1, 2))          # (B, C, N, T)
        adj = self.adjacency() if self.use_graph else None

        skip_total = None
        for block in self.blocks:
            h, skip = block(h, adj)
            skip = skip[..., -self.remaining_steps:]
            skip_total = skip if skip_total is None else skip_total + skip

        out = self.head(skip_total)                    # (B, H*Q, N, 1)
        B, _, N, _ = out.shape
        out = out.squeeze(-1).permute(0, 2, 1)         # (B, N, H*Q)
        out = out.reshape(B, N, self.n_horizons, self.n_quantiles)
        return out.index_select(1, self.target_idx)

    # ------------------------------------------------------------------ losses
    def graph_regularisation(
        self, prior_weight: float, sparsity_weight: float
    ) -> torch.Tensor:
        if not self.use_graph or self.adjacency is None:
            return torch.zeros((), device=next(self.parameters()).device)
        return (
            prior_weight * self.adjacency.prior_loss()
            + sparsity_weight * self.adjacency.sparsity_loss()
        )


def masked_quantile_loss(
    pred: torch.Tensor,      # (B, N_target, H, Q)
    target: torch.Tensor,    # (B, N_target, H)
    mask: torch.Tensor,      # (B, N_target, H)
    quantiles: list[float],
    target_weights: torch.Tensor | None = None,  # (N_target,) per-target weight multipliers
) -> torch.Tensor:
    """Pinball loss, masked so gaps contribute no gradient.

    Filling gaps with zero and taking plain MAE would teach the model that
    unobserved means idle. The mask is the reason the panel keeps missingness
    explicit instead of imputing it away.

    ``target_weights`` upweights operationally critical targets (CPU, memory)
    so the model spends proportionally more gradient budget on them. A weight of
    3.0 on CPU means a CPU prediction error contributes 3× more to the loss than
    a disk prediction error of the same magnitude.  All other targets default to
    1.0. The normalisation by ``mask.sum()`` already accounts for missing data;
    the weights are applied *before* summation so the denominator is unaffected.
    """
    q = torch.tensor(quantiles, device=pred.device).view(1, 1, 1, -1)
    target = target.unsqueeze(-1)
    mask = mask.unsqueeze(-1)

    error = target - pred
    loss = torch.maximum(q * error, (q - 1) * error) * mask  # (B, N_target, H, Q)

    if target_weights is not None:
        # Broadcast: (N_target,) → (1, N_target, 1, 1)
        w = target_weights.view(1, -1, 1, 1)
        loss = loss * w

    denom = mask.sum() * len(quantiles)
    return loss.sum() / denom if denom > 0 else loss.sum() * 0.0
