"""
Auto-regressive pruning head for tree-structured speculative decoding.

The AR head is the ONLY tree-topology-aware component in the draft model.
It takes the draft backbone's hidden state at each node together with the
embedding of that node's parent token, and predicts a token distribution.
This distribution is used as a q-value during pruning (Exp 2+).

During training the AR head is trained jointly with the draft diffusion model
using the same CumProd-weighted cross-entropy loss (scaled by ar_loss_weight).

Architecture
------------
  input  : cat([backbone_hs, parent_embed])  [B, tree_size, 2 * H]
  proj   : Linear(2H → H, bias=False)        [B, tree_size, H]
  norm   : RMSNorm(H)                        [B, tree_size, H]
  output : lm_head(norm_out)                 [B, tree_size, V]

lm_head is shared with the target model (injected after construction).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ARHead(nn.Module):
    """
    Auto-regressive pruning head.

    Parameters
    ----------
    hidden_size : H — hidden dimension of the draft backbone (= target model hidden_size)
    rms_eps     : epsilon for RMSNorm

    Attributes
    ----------
    lm_head : nn.Linear [H → V]
        Must be set externally before the first forward call.
        Shared with the frozen target model to keep the output space aligned.
    """

    def __init__(self, hidden_size: int, rms_eps: float = 1e-6) -> None:
        super().__init__()
        # Project concatenated [backbone_hs ‖ parent_embed] → H
        self.proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.norm = nn.RMSNorm(hidden_size, eps=rms_eps)
        # Injected post-construction; typing hint only — not an nn.Parameter
        self.lm_head: nn.Linear | None = None

    def forward(
        self,
        backbone_hs: Tensor,    # [B, tree_size, H]
        parent_embeds: Tensor,  # [B, tree_size, H]
    ) -> Tensor:                # [B, tree_size, V]
        """
        Parameters
        ----------
        backbone_hs   : [B, tree_size, H]
            Final normed hidden states from the draft diffusion model.
        parent_embeds : [B, tree_size, H]
            Token embedding of each node's parent (anchor embedding for root).

        Returns
        -------
        logits : [B, tree_size, V]
        """
        assert self.lm_head is not None, "ARHead.lm_head must be set before forward()"

        # [B, tree_size, 2H] → [B, tree_size, H]
        x = torch.cat([backbone_hs, parent_embeds], dim=-1)
        x = self.norm(self.proj(x))  # [B, tree_size, H]

        # [B, tree_size, H] → [B, tree_size, V]
        return self.lm_head(x)
