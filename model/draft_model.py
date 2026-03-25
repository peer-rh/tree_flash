"""
TreeDraftModel: DFlashDraftModel with split forward for shared backbone + heads.

Splits the N draft layers into:
  - Shared backbone  : layers[0 .. N-2]  (all-but-last)
  - Diffusion head   : layers[N-1] + norm

Both the diffusion head and the AR head operate on the shared backbone's output
hidden states, allowing them to share computation while using independent final
transformer blocks to specialise their respective tasks.
"""

from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor
from transformers.cache_utils import Cache

from dflash.model.dflash import DFlashDraftModel


class TreeDraftModel(DFlashDraftModel):
    """
    Extends DFlashDraftModel to expose shared_forward / diffusion_head.

    The parent class forward() is left intact; the split methods are additive.
    """

    def shared_forward(
        self,
        position_ids: Tensor,                        # [B, T] or [B, ctx+T]
        noise_embedding: Tensor,                     # [B, tree_size, H]
        target_hidden: Tensor,                       # [B, ctx_len, n_layers*H]  raw
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        attention_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        """
        Run fc + hidden_norm + rotary_emb + layers[:-1].

        Returns
        -------
        shared_hs          : [B, tree_size, H]  pre-norm hidden states
        target_hidden_proj : [B, ctx_len, H]    projected + normed target features
        position_embeddings: (cos, sin) tuple from rotary_emb
        """
        target_hidden_proj = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(noise_embedding, position_ids)

        hidden_states = noise_embedding
        for layer in self.layers[:-1]:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden_proj,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return hidden_states, target_hidden_proj, position_embeddings

    def diffusion_head(
        self,
        shared_hs: Tensor,                           # [B, tree_size, H]
        target_hidden_proj: Tensor,                  # [B, ctx_len, H]
        position_embeddings: tuple[Tensor, Tensor],  # (cos, sin)
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        attention_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Run layers[-1] + norm.

        Returns
        -------
        backbone_hs : [B, tree_size, H]  normed hidden states (ready for lm_head)
        """
        hidden_states = self.layers[-1](
            hidden_states=shared_hs,
            target_hidden=target_hidden_proj,
            attention_mask=attention_mask,
            past_key_value=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        return self.norm(hidden_states)
