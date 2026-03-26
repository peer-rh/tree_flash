"""TreeDFlashDraftModel — DFlashDraftModel subclass with per-layer score_mod routing."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
from transformers.cache_utils import Cache

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "dflash"))

from model.dflash import DFlashDraftModel  # noqa: E402


class TreeDFlashDraftModel(DFlashDraftModel):
    """DFlashDraftModel with per-layer score_mod support for tree position embeddings.

    Usage:
        draft = DFlashDraftModel.from_pretrained(...)
        draft.__class__ = TreeDFlashDraftModel
    """

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        score_mods: Optional[list] = None,
        **kwargs,
    ):
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, layer in enumerate(self.layers):
            layer_kwargs = dict(kwargs)
            if score_mods is not None:
                layer_kwargs["score_mod"] = score_mods[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **layer_kwargs,
            )
        return self.norm(hidden_states)
