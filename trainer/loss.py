"""
CumProd-weighted cross-entropy loss for tree-flash training.

Loss formulation
----------------
For each tree node i in the batch:

    weight[b, i] = cumprod_weight[b, i]
                 = ∏_{j on path root→i (inclusive)} p_target(tree_token[b, j] | prefix_j)

The cumprod is computed on-the-fly from the individual AR probabilities stored in
the stage-2 dataset:

    log_cumprod[b, i] = Σ_{j: ancestor_matrix[j, i]} log(tree_probs[b, j])
    cumprod_weight[b, i] = exp(log_cumprod[b, i])

This requires tree_spec.ancestor_matrix on the same device as tree_probs.

Nodes with IGNORE_IDX target (tree_tokens == -1) are excluded automatically:
  * their tree_probs entry is 0.0 → log_prob = -inf → cumprod propagates to 0
  * F.cross_entropy is called with ignore_index=-1

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

Both terms are normalised by the sum of weights so that nodes on high-probability
paths contribute proportionally more.

torch.compile notes
-------------------
* Fully vectorised — no Python loops over batch or tree dims.
* The ancestor_matrix matmul is [B, T] × [T, T]; all shapes are static.
* Suitable for @torch.compile(fullgraph=True).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_loss(
    draft_logits: Tensor,       # [B, tree_size, V]
    ar_logits: Tensor,          # [B, tree_size, V]
    tree_tokens: Tensor,        # [B, tree_size]    int64   (-1 = ignore)
    tree_probs: Tensor,         # [B, tree_size]    float32 (individual AR probs; 0.0 = ignore)
    ar_loss_weight: float,
    ancestor_matrix: Tensor,    # [tree_size, tree_size] bool or float32 (device-resident)
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute the total training loss.

    Parameters
    ----------
    draft_logits    : [B, tree_size, V]  logits from the draft diffusion model
    ar_logits       : [B, tree_size, V]  logits from the AR pruning head
    tree_tokens     : [B, tree_size]     ground-truth token ids  (-1 = ignore)
    tree_probs      : [B, tree_size]     individual AR probabilities ∈ [0, 1]
    ar_loss_weight  : scalar λ
    ancestor_matrix : [tree_size, tree_size] bool/float — ancestor_matrix[k, q] = True
                      iff k is ancestor-or-self of q (from TreeSpec)

    Returns
    -------
    total_loss  : scalar Tensor (differentiable)
    draft_loss  : scalar Tensor (for logging)
    ar_loss     : scalar Tensor (for logging)
    """
    B, T, V = draft_logits.shape

    # ── Compute cumprod weights from individual probs ─────────────────────────
    # log_probs[b, k] = log p_target(tree_token[b, k] | prefix_k)
    # 0.0 probs (IGNORE_IDX nodes) → log = -inf → cumprod of descendants = 0
    log_probs = torch.log(tree_probs.clamp(min=1e-9))          # [B, T]

    # cumprod[b, q] = exp(Σ_{k: ancestor of q} log_probs[b, k])
    anc = ancestor_matrix.to(dtype=torch.float32, device=log_probs.device)  # [T, T]
    cumprod_weights = torch.exp(log_probs @ anc)                # [B, T]

    # ── Flatten for F.cross_entropy ───────────────────────────────────────────
    flat_targets = tree_tokens.reshape(B * T)                   # [B*T]
    flat_weights = cumprod_weights.reshape(B * T)               # [B*T]

    # Normalisation: sum of weights over valid (non-ignored) positions
    # IGNORE_IDX nodes have weight ≈ 0 so they don't inflate the denominator
    weight_sum = flat_weights.sum().clamp(min=1e-8)             # scalar

    def _weighted_ce(logits: Tensor) -> Tensor:
        """Per-token CE (ignoring IGNORE_IDX), multiplied by cumprod weights."""
        per_token = F.cross_entropy(
            logits.reshape(B * T, V),
            flat_targets,
            reduction="none",
            ignore_index=-1,     # skip IGNORE_IDX positions
        )                                                        # [B*T]
        return (per_token * flat_weights).sum() / weight_sum

    draft_loss = _weighted_ce(draft_logits)                     # scalar
    ar_loss    = _weighted_ce(ar_logits)                        # scalar
    total_loss = draft_loss + ar_loss_weight * ar_loss          # scalar

    return total_loss, draft_loss, ar_loss
