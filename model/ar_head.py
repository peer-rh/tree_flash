"""
Auto-regressive pruning head for tree-structured speculative decoding.

Architecture (shared backbone design)
--------------------------------------
The AR head receives shared hidden states from the draft backbone (all-but-last
layers) and injects parent-token information before running its own dedicated
transformer block.

  parent_proj : Linear(H → H, bias=False)
  ar_layer    : Qwen3DFlashDecoderLayer  (layer_idx = num_draft_layers)
  norm        : RMSNorm(H)
  lm_head     : shared Linear [H → V]  (injected after construction)

Forward:
  ar_input  = shared_hs + parent_proj(parent_embeds)   [B, tree_size, H]
  ar_hs     = ar_layer(ar_input, target_hidden_proj, position_embeddings)
  logits    = lm_head(norm(ar_hs))                     [B, tree_size, V]

lm_head is shared with the target model (injected after construction).
"""

from __future__ import annotations
from typing import Optional

import torch.nn as nn
from torch import Tensor
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3Config

from dflash.model.dflash import Qwen3DFlashDecoderLayer


class ARHead(nn.Module):
    """
    Auto-regressive pruning head.

    Parameters
    ----------
    config          : Qwen3Config — target/draft model config
    num_draft_layers: number of draft layers (used to pick a safe cache layer_idx
                      that doesn't collide with the diffusion path's layer indices)

    Attributes
    ----------
    lm_head : nn.Linear [H → V]
        Must be set externally before the first forward call.
        Shared with the frozen target model to keep the output space aligned.
    """

    def __init__(self, config: Qwen3Config, num_draft_layers: int) -> None:
        super().__init__()
        H = config.hidden_size
        # layer_idx = num_draft_layers keeps the AR layer's KV cache slot
        # separate from all diffusion layers (0 .. num_draft_layers-1)
        self.parent_proj = nn.Linear(H, H, bias=False)
        self.ar_layer = Qwen3DFlashDecoderLayer(config, layer_idx=num_draft_layers)
        self.norm = Qwen3RMSNorm(H, eps=config.rms_norm_eps)
        # Injected post-construction; typing hint only — not an nn.Parameter
        self.lm_head: nn.Linear | None = None

    def forward(
        self,
        shared_hs: Tensor,                          # [B, tree_size, H]
        parent_embeds: Tensor,                      # [B, tree_size, H]
        target_hidden_proj: Tensor,                 # [B, ctx_len, H]
        position_embeddings: tuple[Tensor, Tensor], # (cos, sin) from rotary_emb
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:                                    # [B, tree_size, V]
        """
        Parameters
        ----------
        shared_hs           : [B, tree_size, H]  backbone output (layers[:-1])
        parent_embeds       : [B, tree_size, H]  embedding of each node's parent token
        target_hidden_proj  : [B, ctx_len, H]    projected target features (from shared_forward)
        position_embeddings : (cos, sin)          from rotary_emb (shared with draft)
        attention_mask      : optional causal/tree mask

        Returns
        -------
        logits : [B, tree_size, V]
        """
        assert self.lm_head is not None, "ARHead.lm_head must be set before forward()"

        # Inject parent information into shared hidden states
        ar_input = shared_hs + self.parent_proj(parent_embeds)  # [B, tree_size, H]

        # Dedicated AR transformer block
        ar_hs = self.ar_layer(
            hidden_states=ar_input,
            target_hidden=target_hidden_proj,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )  # [B, tree_size, H]

        return self.lm_head(self.norm(ar_hs))  # [B, tree_size, V]
