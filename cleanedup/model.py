"""Cleaned-up Tree Flash model definitions.

This file is the local, reduced copy of the original DFlash model. It keeps
only the pieces used by the cleaned-up trainer and inference path:

- target-context feature extraction
- relation-aware tree attention bias
- the main drafter forward path
- q-head support
- Hugging Face checkpoint loading via ``from_pretrained``

Removed on purpose:

- all AR-head modules and helpers
- AR-specific forward arguments and returns
- AR-specific config handling

The goal is to keep the cleaned-up directory self-contained without carrying the
full feature surface of the original project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Tuple, Unpack

from src.trees.prunable import PrunableTreeProcessor


RANK_RELATION_CAP = 8

REL_SELF = 0
REL_PARENT = 1
REL_CHILD = 2
REL_SIBLING = 3
REL_ANCESTOR = 4
REL_DESCENDANT = 5
REL_OTHER = 6

REL_PARENT_RANK_BASE = 7
REL_CHILD_RANK_BASE = REL_PARENT_RANK_BASE + RANK_RELATION_CAP
REL_SIBLING_RANK_BASE = REL_CHILD_RANK_BASE + RANK_RELATION_CAP
RELATION_VOCAB_SIZE = REL_SIBLING_RANK_BASE + (RANK_RELATION_CAP * RANK_RELATION_CAP)


def clamp_relation_rank(rank: int) -> int:
    """Clamp child-rank ids into the finite relation vocabulary."""
    rank = int(rank)
    if rank <= 0:
        return 0
    return min(rank, RANK_RELATION_CAP)


def relation_id_for_parent_rank(rank: int) -> int:
    bucket = clamp_relation_rank(rank)
    if bucket == 0:
        return REL_PARENT
    return REL_PARENT_RANK_BASE + bucket - 1


def relation_id_for_child_rank(rank: int) -> int:
    bucket = clamp_relation_rank(rank)
    if bucket == 0:
        return REL_CHILD
    return REL_CHILD_RANK_BASE + bucket - 1


def relation_id_for_sibling_ranks(rank_i: int, rank_j: int) -> int:
    bucket_i = clamp_relation_rank(rank_i)
    bucket_j = clamp_relation_rank(rank_j)
    if bucket_i == 0 or bucket_j == 0:
        return REL_SIBLING
    return REL_SIBLING_RANK_BASE + ((bucket_i - 1) * RANK_RELATION_CAP) + (bucket_j - 1)


@dataclass
class TreeInfo:
    """Dynamic tree metadata consumed by the drafter.

    Shapes:
    - `tree_mask`: `(batch_size, num_anchors, tree_size, tree_size)`
    - `parent_idx`: `(batch_size, num_anchors, tree_size)`
    - `depth`: `(batch_size, num_anchors, tree_size)`
    - `relation_map`: `(batch_size, num_anchors, tree_size, tree_size)`
    - `tree_position_ids`: `(batch_size, num_anchors, tree_size)`
    - `non_root_mask`: `(batch_size, num_anchors, tree_size)`
    - `primary_path_mask`: `(batch_size, num_anchors, tree_size)`
    - `primary_path_indices`: `(batch_size, num_anchors, tree_size)` with `-1`
      padding after valid indices.
    """

    tree_mask: torch.Tensor
    parent_idx: torch.Tensor
    depth: torch.Tensor
    relation_map: torch.Tensor
    tree_position_ids: torch.Tensor
    non_root_mask: torch.Tensor
    primary_path_mask: torch.Tensor
    primary_path_indices: torch.Tensor
    block_size: int
    num_blocks: int


def _build_single_tree_relations(
    *,
    parent_idx: torch.Tensor,
    node_ranks: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ancestor visibility and relation ids for one padded tree."""
    block_size = int(parent_idx.numel())
    device = parent_idx.device
    tree_mask = torch.zeros((block_size, block_size), dtype=torch.bool, device=device)

    for node_idx in range(block_size):
        if not bool(valid_mask[node_idx]):
            continue
        cur = node_idx
        while cur >= 0:
            if not bool(valid_mask[cur]):
                break
            tree_mask[node_idx, cur] = True
            cur = int(parent_idx[cur].item())

    relation_map = torch.full((block_size, block_size), REL_OTHER, dtype=torch.long, device=device)
    for q_idx in range(block_size):
        for k_idx in range(block_size):
            if not bool(valid_mask[q_idx]) or not bool(valid_mask[k_idx]):
                continue
            if q_idx == k_idx:
                relation_map[q_idx, k_idx] = REL_SELF
            elif int(parent_idx[q_idx].item()) == k_idx:
                relation_map[q_idx, k_idx] = relation_id_for_parent_rank(int(node_ranks[q_idx].item()))
            elif int(parent_idx[k_idx].item()) == q_idx:
                relation_map[q_idx, k_idx] = relation_id_for_child_rank(int(node_ranks[k_idx].item()))
            elif (
                int(parent_idx[q_idx].item()) >= 0
                and int(parent_idx[q_idx].item()) == int(parent_idx[k_idx].item())
            ):
                relation_map[q_idx, k_idx] = relation_id_for_sibling_ranks(
                    int(node_ranks[q_idx].item()),
                    int(node_ranks[k_idx].item()),
                )
            elif bool(tree_mask[q_idx, k_idx]):
                relation_map[q_idx, k_idx] = REL_ANCESTOR
            elif bool(tree_mask[k_idx, q_idx]):
                relation_map[q_idx, k_idx] = REL_DESCENDANT
    return tree_mask, relation_map


def _build_primary_path_indices(primary_path_mask: torch.Tensor) -> torch.Tensor:
    """Pack primary-path indices into a padded tensor with `-1` tail padding."""
    batch_size, num_blocks, block_size = primary_path_mask.shape
    primary_indices = torch.full(
        (batch_size, num_blocks, block_size),
        -1,
        dtype=torch.long,
        device=primary_path_mask.device,
    )
    for batch_idx in range(batch_size):
        for block_idx in range(num_blocks):
            keep = torch.nonzero(primary_path_mask[batch_idx, block_idx], as_tuple=False).squeeze(-1)
            if keep.numel() > 0:
                primary_indices[batch_idx, block_idx, : keep.numel()] = keep
    return primary_indices


def build_dynamic_tree_info_from_batch(
    *,
    tree_parent_indices: torch.Tensor,
    tree_depths: torch.Tensor,
    tree_node_ranks: torch.Tensor,
    tree_position_ids: torch.Tensor,
    tree_valid_mask: torch.Tensor,
    tree_primary_path_mask: torch.Tensor,
) -> TreeInfo:
    """Build batch-local tree metadata from padded subtree tensors.

    Args:
        `tree_parent_indices`: `(batch_size, num_anchors, tree_size)`
        `tree_depths`: `(batch_size, num_anchors, tree_size)`
        `tree_node_ranks`: `(batch_size, num_anchors, tree_size)`
        `tree_position_ids`: `(batch_size, num_anchors, tree_size)`
        `tree_valid_mask`: `(batch_size, num_anchors, tree_size)`
        `tree_primary_path_mask`: `(batch_size, num_anchors, tree_size)`
    """
    batch_size, num_blocks, block_size = tree_parent_indices.shape
    tree_masks = []
    relation_maps = []
    for batch_idx in range(batch_size):
        batch_masks = []
        batch_relations = []
        for block_idx in range(num_blocks):
            mask, relations = _build_single_tree_relations(
                parent_idx=tree_parent_indices[batch_idx, block_idx],
                node_ranks=tree_node_ranks[batch_idx, block_idx],
                valid_mask=tree_valid_mask[batch_idx, block_idx],
            )
            batch_masks.append(mask)
            batch_relations.append(relations)
        tree_masks.append(torch.stack(batch_masks, dim=0))
        relation_maps.append(torch.stack(batch_relations, dim=0))

    tree_mask = torch.stack(tree_masks, dim=0)
    relation_map = torch.stack(relation_maps, dim=0)
    non_root_mask = (tree_parent_indices >= 0) & tree_valid_mask
    primary_path_mask = tree_primary_path_mask & tree_valid_mask
    primary_path_indices = _build_primary_path_indices(primary_path_mask)
    return TreeInfo(
        tree_mask=tree_mask,
        parent_idx=tree_parent_indices,
        depth=tree_depths,
        relation_map=relation_map,
        tree_position_ids=tree_position_ids,
        non_root_mask=non_root_mask,
        primary_path_mask=primary_path_mask,
        primary_path_indices=primary_path_indices,
        block_size=block_size,
        num_blocks=num_blocks,
    )


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    """Pick evenly spaced target layers whose activations feed the drafter."""
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def extract_context_feature(
    hidden_states: list[torch.Tensor] | tuple[torch.Tensor, ...],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    """Concatenate selected target hidden states into one context feature tensor.

    Input:
    - `hidden_states`: tuple/list of tensors shaped `(batch_size, seq_len, hidden_size)`
    - `layer_ids`: list of selected hidden-state layer indices

    Output:
    - `(batch_size, seq_len, len(layer_ids) * hidden_size)`
    """
    if layer_ids is None:
        raise ValueError("layer_ids must be provided to extract context features.")
    offset = 1
    selected_states = [hidden_states[layer_id + offset] for layer_id in layer_ids]
    return torch.cat(selected_states, dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen rotary embeddings to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3TreeAttention(nn.Module):
    """Tree-aware multi-head attention used inside the cleaned-up drafter."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        if not hasattr(config, "use_additive_tree_pos_bias"):
            config.use_additive_tree_pos_bias = False
        if config.use_additive_tree_pos_bias:
            self.tree_pos_bias = nn.Embedding(RELATION_VOCAB_SIZE, config.num_attention_heads)
            self.tree_pos_bias.weight.data.zero_()

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: Optional[torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        tree_info: TreeInfo,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run tree-aware attention.

        Input shapes:
        - `hidden_states`: `(batch_size, tree_tokens, hidden_size)`
        - `target_hidden`: `(batch_size, ctx_len, hidden_size)` or `None`
        - `tree_info.relation_map`: `(batch_size, num_anchors, tree_size, tree_size)`

        Output:
        - attention output `(batch_size, tree_tokens, hidden_size)`
        """
        noise_len = hidden_states.shape[1]
        batch_size, query_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1] if target_hidden is not None else 0

        q = self.q_proj(hidden_states).view(batch_size, query_len, -1, self.head_dim)
        k_noise = self.k_proj(hidden_states)
        v_noise = self.v_proj(hidden_states)
        if target_hidden is not None:
            k_ctx = self.k_proj(target_hidden)
            v_ctx = self.v_proj(target_hidden)
            k = torch.cat([k_ctx, k_noise], dim=1)
            v = torch.cat([v_ctx, v_noise], dim=1)
        else:
            k = k_noise
            v = v_noise

        k = k.view(batch_size, ctx_len + query_len, -1, self.head_dim)
        v = v.view(batch_size, ctx_len + query_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2).to(v.dtype)
        q = self.q_norm(q).transpose(1, 2).to(v.dtype)
        v = v.transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        attention_fn = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        total_ctx_len = k.shape[2] - noise_len
        existing_score_mod = kwargs.get("score_mod", lambda score, B, H, Q, KV: score)
        score_mod = existing_score_mod
        if hasattr(self, "tree_pos_bias"):
            batch_size, num_blocks, tree_size, _ = tree_info.relation_map.shape
            bias = self.tree_pos_bias(tree_info.relation_map).view(batch_size, num_blocks, tree_size, tree_size, -1)

            def score_mod(score, B, H, Q, KV):
                kv_tree = (KV - total_ctx_len) // tree_size
                q_tree = Q // tree_size
                q_tree_pos = Q % tree_size
                kv_tree_pos = (KV - total_ctx_len) % tree_size
                same_tree = q_tree == kv_tree
                is_tree = KV >= total_ctx_len
                bias_term = bias[B, q_tree, q_tree_pos, kv_tree_pos, H] * same_tree * is_tree
                return existing_score_mod(score + bias_term, B, H, Q, KV)

        attn_output, attn_weights = attention_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            score_mod=score_mod,
            **kwargs,
        )
        attn_output = attn_output.reshape(batch_size, query_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3TreeDecoderLayer(GradientCheckpointingLayer):
    """One cleaned-up DFlash decoder layer."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3TreeAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        tree_info: TreeInfo,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            tree_info=tree_info,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class DFlashDraftModel(Qwen3PreTrainedModel):
    """Reduced local copy of the DFlash drafter model.

    Retained behaviors:
    - checkpoint loading with `from_pretrained`
    - target-context projection
    - relation-aware attention bias
    - q-head logits

    Removed behaviors:
    - AR head
    - AR helper methods
    - AR-specific forward arguments
    """

    config_class = Qwen3Config
    _no_split_modules = ["Qwen3TreeDecoderLayer"]

    def __init__(self, config) -> None:
        if isinstance(config, dict):
            config = Qwen3Config(**config)
        super().__init__(config)
        dflash_config = getattr(config, "dflash_config", {}) or {}
        self.config._attn_implementation = "flex_attention"
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3TreeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.target_layer_ids = dflash_config.get(
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = getattr(config, "block_size", getattr(config, "max_tree_size", 0))
        self.mask_token_id = dflash_config.get("mask_token_id", None)

        if not hasattr(config, "use_tree_pos_emb"):
            self.config.use_tree_pos_emb = False
        if self.config.use_tree_pos_emb:
            self.tree_pos_embd = nn.Embedding(config.max_tree_size, config.hidden_size)

        if not hasattr(config, "use_q_head"):
            self.config.use_q_head = False
        self.q_head = nn.Linear(config.hidden_size, 1, bias=False) if self.config.use_q_head else None

        self.post_init()

    def extract_ctx_features(
        self,
        hidden_states: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor,
    ) -> torch.Tensor:
        """Extract target-context features for the drafter.

        Input:
        - tuple/list of target hidden states, each `(batch_size, seq_len, hidden_size)`
        - or a packed tensor `(batch_size, seq_len, n_layers, hidden_size)`

        Output:
        - `(batch_size, seq_len, target_feature_dim)`
        """
        if isinstance(hidden_states, (list, tuple)):
            return extract_context_feature(hidden_states, self.target_layer_ids)
        if hidden_states.ndim == 4:
            batch_size, seq_len, n_layers, hidden_size = hidden_states.shape
            return hidden_states.view(batch_size, seq_len, n_layers * hidden_size)
        return hidden_states

    def encode_target_ctx(self, target_ctx_features: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Project target features into the drafter hidden space."""
        if target_ctx_features is None:
            return None
        return self.hidden_norm(self.fc(target_ctx_features))

    def compute_q_logits(self, backbone_hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute q-head logits of shape `(batch_size, tree_tokens)`."""
        if self.q_head is None:
            return backbone_hidden_states.new_zeros(backbone_hidden_states.shape[:2])
        return self.q_head(backbone_hidden_states).squeeze(-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        tree_info: TreeInfo,
        attention_mask: Optional[torch.Tensor] = None,
        target_ctx_features: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        return_aux: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        """Run the cleaned-up drafter forward pass.

        Input shapes:
        - `hidden_states`: `(batch_size, tree_tokens, hidden_size)`
        - `position_ids`: `(batch_size, ctx_len + tree_tokens)` during training
          or `(batch_size, tree_tokens)` during draft-only decoding
        - `tree_info.relation_map`: `(batch_size, num_anchors, tree_size, tree_size)`
        - `target_ctx_features`: `(batch_size, ctx_len, target_feature_dim)` or `None`

        Outputs:
        - when `return_aux=False`:
          - final hidden states `(batch_size, tree_tokens, hidden_size)`
          - backbone hidden states `(batch_size, tree_tokens, hidden_size)`
        - when `return_aux=True`:
          - final hidden states `(batch_size, tree_tokens, hidden_size)`
          - backbone hidden states `(batch_size, tree_tokens, hidden_size)`
          - q logits `(batch_size, tree_tokens)`
        """
        encoded_target_ctx = self.encode_target_ctx(target_ctx_features)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        if self.config.use_tree_pos_emb:
            tree_pos_ids = tree_info.tree_position_ids
            batch_size, num_blocks, block_size = tree_pos_ids.shape
            tree_position_embeddings = self.tree_pos_embd(tree_pos_ids.reshape(batch_size, num_blocks * block_size))
            hidden_states = hidden_states + tree_position_embeddings

        backbone_hidden = None
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=encoded_target_ctx,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                tree_info=tree_info,
                **kwargs,
            )
            if layer_idx == len(self.layers) - 2:
                backbone_hidden = hidden_states
        if backbone_hidden is None:
            backbone_hidden = hidden_states

        hidden_states = self.norm(hidden_states)
        backbone_hidden = self.norm(backbone_hidden)
        if not return_aux:
            return hidden_states, backbone_hidden

        q_logits = self.compute_q_logits(backbone_hidden)
        return hidden_states, backbone_hidden, q_logits


def load_drafter_model(source: dict[str, Any] | str, *, torch_dtype: torch.dtype) -> DFlashDraftModel:
    """Load the single supported drafter variant.

    The cleaned-up path requires:
    - q-head enabled
    - no AR head support in the local implementation
    """
    if isinstance(source, str):
        model = DFlashDraftModel.from_pretrained(source, torch_dtype=torch_dtype)
    else:
        model = DFlashDraftModel(source)
    raw = model
    if getattr(raw, "q_head", None) is None:
        raise ValueError("The cleaned-up trainer requires a drafter with `use_q_head=True`.")
    return model


def build_inference_tree(
    *,
    tree_seq_depth: int,
    branching_pattern: Sequence[Sequence[int]],
    candidate_tree_size: int,
    sub_tree_paths: Sequence[str] | None = None,
):
    """Build the single supported inference tree.

    The cleaned-up inference path always uses:
    - a fixed branch-off layout for drafting
    - q-head pruning to `candidate_tree_size`
    """
    return PrunableTreeProcessor(
        tree_seq_depth=tree_seq_depth,
        base_tree_type="branch_off",
        candidate_tree_size=int(candidate_tree_size),
        branching_pattern=branching_pattern,
        sub_tree_paths=sub_tree_paths,
    )
