"""
CumProd-weighted cross-entropy loss for tree-flash training.

Loss formulation
----------------
For each tree node i in the batch:

    weight[b, i] = cumprod_weights[b, i]
                 = ∏_{j on path root→i} p_target(tree_token[b, j] | prefix_j)

    Pre-computed offline in stage-2; loaded directly from the dataset.

Draft loss:
    L_draft = Σ_{b,i} weight[b,i] · CE(draft_logits[b,i], tree_tokens[b,i])
              ───────────────────────────────────────────────────────────────
                          Σ_{b,i} weight[b,i]

AR-head loss (same targets, same weights, scaled by ar_loss_weight):
    L_ar    = Σ_{b,i} weight[b,i] · CE(ar_logits[b,i], tree_tokens[b,i])
              ──────────────────────────────────────────────────────────────
                          Σ_{b,i} weight[b,i]

Total loss:
    L = L_draft + ar_loss_weight * L_ar

Both terms are normalised by the sum of weights (not by B*tree_size) so that
nodes on high-probability paths contribute proportionally more.

torch.compile notes
-------------------
* The function is fully vectorised — no Python loops over batch or tree dims.
* All shapes are static; no dynamic branching on tensor values.
* Suitable for @torch.compile(fullgraph=True).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_loss(
    draft_logits: Tensor,       # [B, tree_size, V]
    ar_logits: Tensor,          # [B, tree_size, V]
    tree_tokens: Tensor,        # [B, tree_size]    int64
    cumprod_weights: Tensor,    # [B, tree_size]    float32
    ar_loss_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute the total training loss.

    Parameters
    ----------
    draft_logits    : [B, tree_size, V]  logits from the draft diffusion model
    ar_logits       : [B, tree_size, V]  logits from the AR pruning head
    tree_tokens     : [B, tree_size]     ground-truth token ids
    cumprod_weights : [B, tree_size]     pre-computed CumProd weights ∈ [0,1]
    ar_loss_weight  : scalar λ

    Returns
    -------
    total_loss  : scalar Tensor (differentiable)
    draft_loss  : scalar Tensor (for logging)
    ar_loss     : scalar Tensor (for logging)
    """
    B, T, V = draft_logits.shape

    # Flatten spatial dims for F.cross_entropy
    flat_targets = tree_tokens.reshape(B * T)           # [B * tree_size]
    flat_weights = cumprod_weights.reshape(B * T)       # [B * tree_size]

    # Normalisation constant (sum of weights, not count)
    weight_sum = flat_weights.sum().clamp(min=1e-8)     # scalar

    def _weighted_ce(logits: Tensor) -> Tensor:
        """Per-position CE, multiplied by CumProd weights, then normalised."""
        # per_token: [B * tree_size]
        per_token = F.cross_entropy(
            logits.reshape(B * T, V),
            flat_targets,
            reduction="none",
        )
        return (per_token * flat_weights).sum() / weight_sum

    draft_loss = _weighted_ce(draft_logits)             # scalar
    ar_loss = _weighted_ce(ar_logits)                   # scalar
    total_loss = draft_loss + ar_loss_weight * ar_loss  # scalar

    return total_loss, draft_loss, ar_loss
