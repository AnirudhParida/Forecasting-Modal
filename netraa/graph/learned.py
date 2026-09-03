"""(b) Learned adjacency, supervised by forecast error.

A graph trained in isolation has no supervision signal — there is no label for
"is there an edge from endpoint_cpm to database_access_cpm". So the adjacency is
made a model parameter and learned jointly with the forecaster: an edge exists
because it demonstrably reduces prediction error.

    A = ReLU(tanh(alpha * (E1 E2^T - E2 E1^T)))

The antisymmetric inner term makes the graph directed, so i->j and j->i are
distinguishable. Left unconstrained this drifts toward whatever spurious
correlation helps the training loss, so it is pulled toward the statistical
prior A_prior from graph/statistical.py by an L1 penalty. That is the
"regularised toward (a)" half of the design — one training run, two graphs
that have to agree with each other.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveAdjacency(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        embed_dim: int = 16,
        top_k: int = 8,
        alpha: float = 3.0,
        prior: np.ndarray | None = None,
        same_host_mask: np.ndarray | None = None,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.top_k = min(top_k, n_nodes - 1) if n_nodes > 1 else 1
        self.alpha = alpha

        self.e1 = nn.Parameter(torch.randn(n_nodes, embed_dim) * 0.1)
        self.e2 = nn.Parameter(torch.randn(n_nodes, embed_dim) * 0.1)

        if prior is not None:
            self.register_buffer("prior", torch.tensor(prior, dtype=torch.float32))
        else:
            self.register_buffer("prior", torch.zeros(n_nodes, n_nodes))

        # Host-isolation mask: 1 where src and dst share the same host, 0 otherwise.
        # Registered as a non-trainable buffer so it moves with the model to any device
        # and is saved/loaded with the state_dict.
        if same_host_mask is not None:
            self.register_buffer(
                "same_host_mask",
                torch.tensor(same_host_mask, dtype=torch.float32),
            )
        else:
            # Default: all edges allowed (single-host or unconstrained mode)
            self.register_buffer(
                "same_host_mask",
                torch.ones(n_nodes, n_nodes, dtype=torch.float32),
            )

    def dense(self) -> torch.Tensor:
        """Unsparsified adjacency in [0, 1)."""
        m = self.e1 @ self.e2.t() - self.e2 @ self.e1.t()
        return F.relu(torch.tanh(self.alpha * m))

    def forward(self) -> torch.Tensor:
        """Sparsified, row-normalised adjacency. A[i, j] = influence of i on j.

        The same_host_mask is applied before sparsification so that cross-host
        entries are zeroed and can never receive top-k selection or gradient.
        """
        adj = self.dense()

        # ── host isolation: zero out all cross-host entries ──────────────────
        adj = adj * self.same_host_mask

        # Keep the top_k strongest outgoing edges per node. Gradients still flow
        # to the retained entries; dropped ones simply receive none this step.
        if self.top_k < self.n_nodes:
            mask = torch.zeros_like(adj)
            _, idx = adj.topk(self.top_k, dim=1)
            mask.scatter_(1, idx, 1.0)
            adj = adj * mask

        adj = adj * (1.0 - torch.eye(self.n_nodes, device=adj.device))
        return adj / (adj.sum(dim=1, keepdim=True) + 1e-8)

    # ------------------------------------------------------------ regularisers
    def prior_loss(self) -> torch.Tensor:
        """Pull the learned graph toward the statistically discovered one."""
        if float(self.prior.abs().sum()) == 0.0:
            return torch.zeros((), device=self.e1.device)
        return (self.forward() - self.prior).abs().mean()

    def sparsity_loss(self) -> torch.Tensor:
        return self.dense().abs().mean()

    @torch.no_grad()
    def export(self, node_ids: list[str], threshold: float = 1e-4) -> list[dict]:
        """Learned graph as an edge list, comparable with the statistical map."""
        adj = self.forward().cpu().numpy()
        prior = self.prior.cpu().numpy()
        edges = []
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j and adj[i, j] > threshold:
                    edges.append(
                        {
                            "source": node_ids[i],
                            "target": node_ids[j],
                            "weight": float(adj[i, j]),
                            "prior_weight": float(prior[i, j]),
                            "in_prior": bool(prior[i, j] > 0),
                        }
                    )
        edges.sort(key=lambda e: e["weight"], reverse=True)
        return edges

    @torch.no_grad()
    def agreement(self) -> dict:
        """How much of the learned graph the statistical prior supports.

        Low overlap is a signal worth reading, not a bug: it means the two
        methods disagree about the structure and neither should be trusted
        without inspection.
        """
        adj = (self.forward() > 1e-4).cpu().numpy()
        prior = (self.prior > 0).cpu().numpy()
        learned_n, prior_n = int(adj.sum()), int(prior.sum())
        overlap = int((adj & prior).sum())
        return {
            "learned_edges": learned_n,
            "prior_edges": prior_n,
            "overlap": overlap,
            "precision_vs_prior": overlap / learned_n if learned_n else 0.0,
            "recall_vs_prior": overlap / prior_n if prior_n else 0.0,
        }
