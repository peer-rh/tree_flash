from typing import Optional, Callable
from typing_extensions import Unpack, Tuple
import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    Qwen3Config,
    Qwen3PreTrainedModel,
    Qwen3MLP,
    GradientCheckpointingLayer,
    FlashAttentionKwargs,
    rotate_half,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)
from transformers.cache_utils import Cache

from ..trees import TreeInfo


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [(num_target_layers // 2)]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]
    return target_layer_ids

def extract_context_feature(
    hidden_states: list[torch.Tensor] | tuple[torch.Tensor, ...],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    if layer_ids is None:
        raise ValueError("layer_ids must be provided to extract context features")
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Qwen3DFlashAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False  
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        if not hasattr(config, "use_additive_tree_pos_bias"):
            config.use_additive_tree_pos_bias = False
        if config.use_additive_tree_pos_bias:
            self.tree_pos_bias = nn.Embedding(64, config.num_attention_heads)
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
        noise_len = hidden_states.shape[1]
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1] if target_hidden is not None else 0
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
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
        k = k.view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = v.view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2).to(v.dtype)
        q = self.q_norm(q).transpose(1, 2).to(v.dtype)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        
        total_len = k.shape[2]
        total_ctx_len = total_len - noise_len
        exitsing_score_mod = kwargs.get("score_mod", lambda s, B, H, Q, KV: s)
        score_mod = exitsing_score_mod
        if hasattr(self, "tree_pos_bias"):
            B, N_B, T, _ = tree_info.relation_map.shape
            bias = self.tree_pos_bias(tree_info.relation_map).view(B, N_B, T, T, -1) # [B, N_B * T, T, num_heads]
            def score_mod(score, B, H, Q, KV):
                KV_TREE = (KV - total_ctx_len) // T
                Q_TREE = Q // T
                Q_TREE_POS = Q % T
                KV_TREE_POS = (KV - total_ctx_len) % T
                same_tree = (Q_TREE == KV_TREE)
                is_tree = KV >= total_ctx_len
                bias_ = bias[B, Q_TREE, Q_TREE_POS, KV_TREE_POS, H] * same_tree * is_tree
                score = exitsing_score_mod(score + bias_, B, H, Q, KV)
                return score

        attn_output, attn_weights = attn_fn(
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
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
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
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
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
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        if isinstance(config, dict):
            config = Qwen3Config(**config)
        super().__init__(config)
        dflash_config = getattr(config, "dflash_config", {}) or {}
        self.config._attn_implementation = "flex_attention"
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3DFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
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
        
        self.q_head = None
        if not hasattr(config, "use_q_head"):
            self.config.use_q_head = False
        if self.config.use_q_head:
            self.q_head = nn.Linear(config.hidden_size, 1, bias=False)
        
        self.post_init()

    def extract_ctx_features(
        self,
        hidden_states: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(hidden_states, (list, tuple)):
            return extract_context_feature(hidden_states, self.target_layer_ids)
        if hidden_states.ndim == 4:
            batch_size, seq_len, n_layers, hidden_size = hidden_states.shape
            return hidden_states.view(batch_size, seq_len, n_layers * hidden_size)
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        tree_info: TreeInfo,  
        attention_mask: Optional[torch.Tensor] = None,
        target_ctx_features: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if target_ctx_features is not None:
            target_ctx_features = self.hidden_norm(self.fc(target_ctx_features))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        if self.config.use_tree_pos_emb:
            tree_pos_ids = tree_info.tree_position_ids
            batch_size, num_blocks, block_size = tree_pos_ids.shape
            tree_position_embeddings = self.tree_pos_embd(tree_pos_ids.reshape(batch_size, num_blocks * block_size))
            hidden_states = hidden_states + tree_position_embeddings

        backbone_hs = None
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_ctx_features,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                tree_info=tree_info,
                **kwargs,
            )
            if i == len(self.layers) - 2:
                backbone_hs = hidden_states
        if backbone_hs is None:
            backbone_hs = hidden_states
        return self.norm(hidden_states), self.norm(backbone_hs)
