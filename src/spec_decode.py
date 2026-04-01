from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache

from .data import load_and_process_eval_dataset
from .models import DFlashDraftModel
from .trainer import has_pruning_head, unwrap_model
from .trees import (
    BlockTreeProcessor,
    BranchOffTreeProcessor,
    PrunableTreeProcessor,
    TreeInfo,
    subset_tree_info,
)


@dataclass
class SpecDecodeResult:
    token_ids: torch.Tensor
    continuation_ids: torch.Tensor
    text: str
    acceptance_lengths: list[int]
    off_main_path_last_accept_flags: list[bool]
    drafted_tokens: int
    committed_tokens: int


@dataclass
class SpecEvalAggregateResult:
    total_examples: int
    total_committed_tokens: int
    total_steps: int
    total_acceptance_length: int
    acceptance_length_histogram: list[int]
    off_main_path_last_accept_count: int
    elapsed_time_sec: float


@dataclass
class ComparedToken:
    token_id: int
    token_text: str
    status: str
    expected_token_id: int | None = None
    expected_token_text: str | None = None


@dataclass
class EvalExampleResult:
    index: int
    prompt: str
    reference_text: str | None
    generated_text: str
    comparisons: list[ComparedToken]
    missing_reference_tokens: list[ComparedToken]
    accepted_count: int
    rejected_count: int
    extra_count: int
    missing_count: int
    exact_match: bool
    drafted_tokens: int
    committed_tokens: int
    mean_acceptance_length: float


@dataclass
class EvalSuiteResult:
    examples: list[EvalExampleResult]
    total_examples: int
    exact_matches: int
    accepted_count: int
    rejected_count: int
    extra_count: int
    missing_count: int


OFFICIAL_DFLASH_MODELS = {
    "qwen3-8b": "z-lab/Qwen3-8B-DFlash-b16",
    "qwen3-4b": "z-lab/Qwen3-4B-DFlash-b16",
    "qwen3-coder-30b-a3b": "z-lab/Qwen3-Coder-30B-A3B-DFlash-b16",
    "z-lab/qwen3-8b-dflash-b16": "z-lab/Qwen3-8B-DFlash-b16",
    "z-lab/qwen3-coder-30b-a3b-dflash-b16": "z-lab/Qwen3-Coder-30B-A3B-DFlash-b16",
}

OFFICIAL_DFLASH_TARGETS = {
    "z-lab/Qwen3-8B-DFlash-b16": "Qwen/Qwen3-8B",
    "z-lab/Qwen3-4B-DFlash-b16": "Qwen/Qwen3-4B",
    "z-lab/Qwen3-Coder-30B-A3B-DFlash-b16": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
}


def sample_from_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Sample or greedily pick token ids from logits."""
    if temperature <= 0:
        return logits.argmax(dim=-1)
    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def gather_token_probability(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Return model probabilities assigned to specific token ids."""
    if temperature > 0:
        logits = logits / temperature
    probs = torch.softmax(logits.float(), dim=-1)
    return probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1).to(torch.float32)


def build_tree_processor(
    *,
    tree_type: str,
    tree_seq_depth: int,
    sub_tree_paths: Sequence[str] | None = None,
    tree_args: dict[str, Any] | None = None,
):
    """Construct the tree processor used by speculative decoding."""
    tree_args = tree_args or {}
    if tree_type in {"fixed", "block"}:
        return BlockTreeProcessor(
            tree_seq_depth=tree_seq_depth,
            sub_tree_paths=sub_tree_paths or tree_args.get("sub_tree_paths"),
        )
    if tree_type == "branch_off":
        return BranchOffTreeProcessor(
            tree_seq_depth=tree_seq_depth,
            sub_tree_paths=sub_tree_paths or tree_args.get("sub_tree_paths"),
            branching_pattern=tree_args.get("branching_pattern"),
        )
    if tree_type == "prunable":
        legacy_prune_topk = tree_args.get("prune_topk")
        candidate_tree_size = tree_args.get("candidate_tree_size")
        if candidate_tree_size is None:
            candidate_tree_size = 1 if legacy_prune_topk in {None, 0} else int(legacy_prune_topk)
        return PrunableTreeProcessor(
            tree_seq_depth=tree_seq_depth,
            base_tree_type=tree_args.get("base_tree_type", "block"),
            candidate_tree_size=int(candidate_tree_size),
            sub_tree_paths=sub_tree_paths or tree_args.get("sub_tree_paths"),
            branching_pattern=tree_args.get("branching_pattern"),
        )
    if tree_type in {"var", "vartree"}:
        raise NotImplementedError("tree_type='vartree' is training-only and not implemented for speculative decoding.")
    raise NotImplementedError(
        f"tree_type={tree_type!r} is not implemented for speculative decoding."
    )

def build_verifier_score_mod(
    *,
    tree_info: TreeInfo,
    prefix_len: int,
):
    """Build a verifier mask that only allows prefix and ancestor attention."""
    tree_mask = tree_info.tree_mask
    block_size = tree_info.block_size

    def score_mod(score, B, H, Q, KV):
        is_prefix = KV < prefix_len
        tree_kv = KV - prefix_len
        valid_tree = (tree_kv >= 0) & (tree_kv < block_size)
        q_idx = Q.clamp(0, block_size - 1)
        kv_idx = tree_kv.clamp(0, block_size - 1)
        legal_tree = valid_tree & tree_mask[q_idx, kv_idx]
        return torch.where(is_prefix | legal_tree, score, torch.full_like(score, float("-inf")))

    return score_mod


def trim_dynamic_cache(
    past_key_values: Cache,
    *,
    prefix_len: int,
    keep_tree_indices: list[int],
) -> Cache:
    """Keep only the accepted tree nodes in a DynamicCache."""
    keep = torch.tensor(keep_tree_indices, dtype=torch.long)
    for layer in past_key_values.layers:
        if not getattr(layer, "is_initialized", False) or layer.keys is None or layer.values is None:
            continue
        device = layer.keys.device
        select = keep.to(device)
        prefix_keys = layer.keys[:, :, :prefix_len, :]
        prefix_values = layer.values[:, :, :prefix_len, :]
        tree_keys = layer.keys[:, :, prefix_len:, :]
        tree_values = layer.values[:, :, prefix_len:, :]
        layer.keys = torch.cat([prefix_keys, tree_keys.index_select(2, select)], dim=2)
        layer.values = torch.cat([prefix_values, tree_values.index_select(2, select)], dim=2)
        if hasattr(layer, "cumulative_length"):
            layer.cumulative_length = int(layer.keys.shape[-2])
    return past_key_values


def choose_deepest_valid_node(
    accepted_mask: torch.Tensor,
    tree_info: TreeInfo,
) -> int:
    """Choose the deepest drafted node whose full path was accepted."""
    valid_nodes = []
    for node_idx in range(tree_info.block_size):
        path_mask = tree_info.tree_mask[node_idx]
        if bool(accepted_mask[path_mask].all().item()):
            valid_nodes.append(node_idx)
    if not valid_nodes:
        return 0
    return max(valid_nodes, key=lambda idx: (int(tree_info.depth[idx].item()), -idx))


def gather_path_indices(
    node_idx: int,
    tree_info: TreeInfo,
) -> list[int]:
    """Return the path from root to a node, inclusive."""
    path = []
    cur = node_idx
    while cur >= 0:
        path.append(cur)
        cur = int(tree_info.parent_idx[cur].item())
    return list(reversed(path))


def _compute_path_correctness_probabilities(
    *,
    tree_info: TreeInfo,
    node_correctness_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute expected-depth contribution scores by cumulative ancestor products."""
    path_probs = torch.zeros_like(node_correctness_probs, dtype=torch.float32)
    path_probs[0] = 1.0
    for node_idx in range(1, tree_info.block_size):
        parent_idx = int(tree_info.parent_idx[node_idx].item())
        path_probs[node_idx] = path_probs[parent_idx] * node_correctness_probs[node_idx].to(torch.float32)
    return path_probs


def select_pruned_keep_indices(
    *,
    tree_info: TreeInfo,
    node_correctness_probs: torch.Tensor,
    candidate_tree_size: int,
) -> list[int]:
    """Keep the rooted subtree of fixed size that maximizes expected depth."""
    if candidate_tree_size <= 0:
        raise ValueError(f"candidate_tree_size must be > 0, got {candidate_tree_size}.")
    if tree_info.block_size <= 1 or candidate_tree_size == 1:
        return [0]
    path_probs = _compute_path_correctness_probabilities(
        tree_info=tree_info,
        node_correctness_probs=node_correctness_probs,
    )
    ranked_nodes = sorted(
        range(tree_info.block_size),
        key=lambda idx: (
            -float(path_probs[idx].item()),
            int(tree_info.depth[idx].item()),
            idx,
        ),
    )
    kept = ranked_nodes[: min(int(candidate_tree_size), tree_info.block_size)]
    return sorted(kept)


def prune_drafted_tree(
    *,
    tree_token_ids: torch.Tensor,
    draft_logits: torch.Tensor,
    draft_token_probs: torch.Tensor,
    tree_info: TreeInfo,
    node_correctness_probs: torch.Tensor,
    candidate_tree_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, TreeInfo]:
    """Prune a drafted tree down to a smaller ancestor-closed subset."""
    keep_indices = select_pruned_keep_indices(
        tree_info=tree_info,
        node_correctness_probs=node_correctness_probs,
        candidate_tree_size=candidate_tree_size,
    )
    if len(keep_indices) == tree_info.block_size:
        return tree_token_ids, draft_logits, draft_token_probs, tree_info

    keep = torch.tensor(keep_indices, dtype=torch.long, device=tree_token_ids.device)
    pruned_tree_info = subset_tree_info(tree_info, keep)
    return (
        tree_token_ids.index_select(0, keep),
        draft_logits.index_select(0, keep),
        draft_token_probs.index_select(0, keep),
        pruned_tree_info,
    )


def build_tree_parent_token_ids(
    tree_token_ids: torch.Tensor,
    tree_info: TreeInfo,
) -> torch.Tensor:
    """Map each tree node to its parent token id."""
    parent_token_ids = tree_token_ids.clone()
    if tree_info.block_size <= 1:
        return parent_token_ids
    parent_idx = tree_info.parent_idx.to(tree_token_ids.device)
    parent_token_ids[1:] = tree_token_ids.index_select(0, parent_idx[1:])
    return parent_token_ids


def build_pruning_scores(
    *,
    raw_drafter,
    backbone_hidden: torch.Tensor,
    tree_token_ids: torch.Tensor,
    tree_info: TreeInfo,
    target_ctx_features: torch.Tensor,
    position_ids: torch.Tensor,
    target_embeddings,
    target_lm_head,
    temperature: float,
    ar_cache: Cache | None = None,
) -> torch.Tensor | None:
    """Score drafted nodes for pruning using the available drafter head."""
    if getattr(raw_drafter, "ar_block", None) is not None:
        parent_token_ids = build_tree_parent_token_ids(tree_token_ids, tree_info)
        parent_embeddings = target_embeddings(parent_token_ids.unsqueeze(0))
        prefix_position_ids = None
        use_cache = False
        pre_len = 0
        if ar_cache is not None:
            pre_len = ar_cache.get_seq_length()
            prefix_position_ids = torch.arange(
                pre_len,
                pre_len + target_ctx_features.shape[1],
                device=backbone_hidden.device,
            ).unsqueeze(0)
            use_cache = True
        ar_hidden_states = raw_drafter.build_ar_hidden_states(
            backbone_hidden,
            parent_embeddings,
            target_ctx_features=target_ctx_features,
            tree_info=tree_info,
            position_ids=position_ids,
            past_key_values=ar_cache,
            use_cache=use_cache,
            prefix_position_ids=prefix_position_ids,
        )
        if ar_cache is not None:
            ar_cache.crop(pre_len + target_ctx_features.shape[1])
        ar_logits = target_lm_head(ar_hidden_states)[0]
        ar_scores = gather_token_probability(ar_logits, tree_token_ids, temperature)
        ar_scores[0] = 1.0
        return ar_scores
    if getattr(raw_drafter, "q_head", None) is not None:
        compute_q_logits = getattr(raw_drafter, "compute_q_logits", None)
        if compute_q_logits is not None:
            q_logits = compute_q_logits(backbone_hidden)
        else:
            q_logits = raw_drafter.q_head(backbone_hidden).squeeze(-1)
        return torch.sigmoid(q_logits[0]).to(torch.float32)
    return None


def draft_tree(
    *,
    drafter_model,
    raw_drafter,
    target_embeddings,
    target_lm_head,
    tree_processor,
    target_ctx_features,
    drafter_cache: Cache,
    ar_cache: Cache | None,
    current_root_token: int,
    root_position: int,
    temperature: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, TreeInfo, torch.Tensor | None]:
    """Draft a speculative tree from the current root token."""
    tree_info = tree_processor.build_tree_info(batch_size=1, num_blocks=1, device=device)
    tree_len = tree_info.block_size
    pre_len = drafter_cache.get_seq_length()
    noise_ids = torch.full((1, tree_len), raw_drafter.mask_token_id, dtype=torch.long, device=device)
    noise_ids[0, 0] = current_root_token
    noise_embeddings = target_embeddings(noise_ids)
    tree_position_ids = (tree_info.depth.view(1, -1) + root_position).to(device)
    position_ids = torch.cat([
        torch.arange(pre_len, pre_len + target_ctx_features.shape[1], device=device),
        tree_position_ids.squeeze(0),
    ], dim=1)
    draft_hidden_states, backbone_hidden = drafter_model(
        hidden_states=noise_embeddings,
        position_ids=position_ids,
        tree_info=tree_info,
        target_ctx_features=target_ctx_features,
        past_key_values=drafter_cache,
        use_cache=True,
    )
    drafter_cache.crop(
        pre_len + target_ctx_features.shape[1],
    )
    draft_logits = target_lm_head(draft_hidden_states)[0]
    tree_token_ids = sample_from_logits(draft_logits, temperature)
    tree_token_ids[0] = current_root_token
    draft_token_probs = gather_token_probability(draft_logits, tree_token_ids, temperature)
    draft_token_probs[0] = 1.0
    pruning_scores = build_pruning_scores(
        raw_drafter=raw_drafter,
        backbone_hidden=backbone_hidden,
        tree_token_ids=tree_token_ids,
        tree_info=tree_info,
        target_ctx_features=target_ctx_features,
        position_ids=tree_position_ids,
        target_embeddings=target_embeddings,
        target_lm_head=target_lm_head,
        temperature=temperature,
        ar_cache=ar_cache,
    )
    return tree_token_ids, draft_logits, draft_token_probs, tree_info, pruning_scores


def verify_tree(
    *,
    target_model,
    raw_drafter,
    tree_token_ids: torch.Tensor,
    tree_info: TreeInfo,
    root_position: int,
    target_cache: Cache,
    temperature: float,
) -> tuple[Cache, torch.Tensor, torch.Tensor]:
    """Verify drafted tokens with the target model under tree constraints."""
    prefix_len = target_cache.get_seq_length()
    score_mod = build_verifier_score_mod(
        tree_info=tree_info,
        prefix_len=prefix_len,
    )
    device = tree_token_ids.device
    position_ids = (tree_info.depth.view(1, -1) + root_position).to(device)
    cache_position = torch.arange(prefix_len, prefix_len + tree_info.block_size, device=device)
    outputs = target_model(
        input_ids=tree_token_ids.unsqueeze(0),
        position_ids=position_ids,
        attention_mask=None,
        past_key_values=target_cache,
        use_cache=True,
        cache_position=cache_position,
        output_hidden_states=True,
        score_mod=score_mod,
    )
    tree_ctx_features = raw_drafter.extract_ctx_features(outputs.hidden_states)
    return outputs.past_key_values, outputs.logits[0], tree_ctx_features


def build_acceptance_mask(
    *,
    draft_logits: torch.Tensor,
    draft_token_probs: torch.Tensor,
    verifier_logits: torch.Tensor,
    tree_token_ids: torch.Tensor,
    tree_info: TreeInfo,
    temperature: float,
) -> torch.Tensor:
    """Decide which drafted nodes are accepted by the verifier."""
    accepted = torch.zeros((tree_info.block_size,), dtype=torch.bool, device=tree_token_ids.device)
    accepted[0] = True
    if temperature <= 0:
        for node_idx in range(1, tree_info.block_size):
            parent_idx = int(tree_info.parent_idx[node_idx].item())
            target_token = verifier_logits[parent_idx].argmax(dim=-1)
            accepted[node_idx] = tree_token_ids[node_idx] == target_token
        return accepted

    for node_idx in range(1, tree_info.block_size):
        parent_idx = int(tree_info.parent_idx[node_idx].item())
        target_prob = gather_token_probability(
            verifier_logits[parent_idx].unsqueeze(0),
            tree_token_ids[node_idx].view(1),
            temperature,
        )[0]
        draft_prob = draft_token_probs[node_idx].clamp_min(1e-8)
        accept_prob = torch.clamp(target_prob / draft_prob, max=1.0)
        accepted[node_idx] = torch.rand((), device=tree_token_ids.device) <= accept_prob
    return accepted


@torch.inference_mode()
def speculative_generate_from_ids(
    *,
    target_model,
    drafter_model,
    tokenizer,
    prompt_ids: torch.Tensor,
    tree_processor,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> SpecDecodeResult:
    """Run tree speculative decoding from already-tokenized prompt ids."""
    device = next(target_model.parameters()).device
    raw_drafter = unwrap_model(drafter_model)
    if isinstance(tree_processor, PrunableTreeProcessor) and not has_pruning_head(raw_drafter):
        raise ValueError(
            "tree_type='prunable' requires a drafter checkpoint/config with use_q_head=True or use_ar_head=True."
        )
    prompt_ids = prompt_ids.to(device).view(1, -1)
    target_cache = DynamicCache()
    prefill_outputs = target_model(
        input_ids=prompt_ids,
        output_hidden_states=True,
        use_cache=True,
        past_key_values=target_cache,
    )
    target_cache = prefill_outputs.past_key_values
    target_ctx_features = raw_drafter.extract_ctx_features(prefill_outputs.hidden_states)
    drafter_cache = DynamicCache()
    ar_cache = DynamicCache() if getattr(raw_drafter, "ar_block", None) is not None else None
    current_root_token = int(sample_from_logits(prefill_outputs.logits[:, -1, :], temperature)[0].item())
    output_ids = torch.cat(
        [prompt_ids[0], torch.tensor([current_root_token], dtype=torch.long, device=device)],
        dim=0,
    )
    input_len = output_ids.shape[0]
    acceptance_lengths: list[int] = []
    off_main_path_last_accept_flags: list[bool] = []
    drafted_tokens = 0
    committed_tokens = 1
    eos_token_id = tokenizer.eos_token_id

    target_embeddings = target_model.get_input_embeddings()
    target_lm_head = target_model.get_output_embeddings()
    if target_lm_head is None:
        raise ValueError("Target model must expose an LM head for speculative decoding.")

    while committed_tokens < max_new_tokens:
        if eos_token_id is not None and current_root_token == eos_token_id:
            break
            
        prefix_len = target_cache.get_seq_length()

        root_position = output_ids.shape[0] - 1
        tree_token_ids, draft_logits, draft_token_probs, tree_info, pruning_scores = draft_tree(
            drafter_model=drafter_model,
            raw_drafter=raw_drafter,
            target_embeddings=target_embeddings,
            target_lm_head=target_lm_head,
            tree_processor=tree_processor,
            drafter_cache=drafter_cache,
            ar_cache=ar_cache,
            current_root_token=current_root_token,
            root_position=root_position,
            temperature=temperature,
            target_ctx_features=target_ctx_features,
            device=device,
        )
        drafted_tokens += tree_info.block_size - 1
        if isinstance(tree_processor, PrunableTreeProcessor):
            if pruning_scores is None:
                raise ValueError("Prunable speculative decoding requires q_head or AR-head scores.")
            tree_token_ids, draft_logits, draft_token_probs, tree_info = prune_drafted_tree(
                tree_token_ids=tree_token_ids,
                draft_logits=draft_logits,
                draft_token_probs=draft_token_probs,
                tree_info=tree_info,
                node_correctness_probs=pruning_scores,
                candidate_tree_size=tree_processor.candidate_tree_size,
            )

        updated_cache, verifier_logits, tree_ctx_features = verify_tree(
            target_model=target_model,
            raw_drafter=raw_drafter,
            tree_token_ids=tree_token_ids,
            tree_info=tree_info,
            root_position=root_position,
            target_cache=target_cache,
            temperature=temperature,
        )
        accepted_mask = build_acceptance_mask(
            draft_logits=draft_logits,
            draft_token_probs=draft_token_probs,
            verifier_logits=verifier_logits,
            tree_token_ids=tree_token_ids,
            tree_info=tree_info,
            temperature=temperature,
        )
        deepest_idx = choose_deepest_valid_node(accepted_mask, tree_info)
        accepted_path = gather_path_indices(deepest_idx, tree_info)
        acceptance_lengths.append(len(accepted_path) - 1)
        off_main_path_last_accept_flags.append(not bool(tree_info.primary_path_mask[deepest_idx].item()))

        bonus_token = int(sample_from_logits(verifier_logits[deepest_idx].unsqueeze(0), temperature)[0].item())
        committed_path_tokens = tree_token_ids[accepted_path[1:]]
        step_tokens = torch.cat(
            [
                committed_path_tokens,
                torch.tensor([bonus_token], dtype=torch.long, device=device),
            ],
            dim=0,
        )
        remaining = max_new_tokens - committed_tokens
        if step_tokens.numel() > remaining:
            step_tokens = step_tokens[:remaining]
        output_ids = torch.cat([output_ids, step_tokens], dim=0)
        committed_tokens += int(step_tokens.numel())

        target_ctx_features = tree_ctx_features
        
        target_cache = trim_dynamic_cache(
            updated_cache,
            prefix_len=prefix_len,
            keep_tree_indices=accepted_path,
        )
        current_root_token = bonus_token
        if eos_token_id is not None and (output_ids[input_len:] == eos_token_id).any():
            break

    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    continuation_ids = output_ids[input_len:].detach().cpu()
    return SpecDecodeResult(
        token_ids=output_ids.detach().cpu(),
        continuation_ids=continuation_ids,
        text=text,
        acceptance_lengths=acceptance_lengths,
        off_main_path_last_accept_flags=off_main_path_last_accept_flags,
        drafted_tokens=drafted_tokens,
        committed_tokens=committed_tokens,
    )


@torch.inference_mode()
def speculative_generate(
    *,
    target_model,
    drafter_model,
    tokenizer,
    prompt: str,
    tree_processor,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> SpecDecodeResult:
    """Run tree speculative decoding from a text prompt."""
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    return speculative_generate_from_ids(
        target_model=target_model,
        drafter_model=drafter_model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        tree_processor=tree_processor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


def resolve_official_dflash_model(model_name: str) -> str:
    """Resolve a short official DFlash alias to its Hugging Face repo id."""
    return OFFICIAL_DFLASH_MODELS.get(model_name.lower(), model_name)


def infer_target_model_for_draft_model(draft_model: str) -> str | None:
    """Infer the base target model for a known official DFlash checkpoint."""
    return OFFICIAL_DFLASH_TARGETS.get(draft_model)


def apply_chat_template_if_requested(
    *,
    tokenizer,
    prompt: str,
    enabled: bool,
) -> str:
    """Apply the tokenizer chat template when requested."""
    if not enabled:
        return prompt
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            enable_thinking=False,
            **template_kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            **template_kwargs,
        )


def _format_token_text(tokenizer, token_id: int) -> str:
    """Render a token as a compact string for HTML output."""
    try:
        text = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        text = tokenizer.decode([token_id], skip_special_tokens=False)
    if not text:
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
        text = f"<{token}>"
    return text.replace(" ", "·").replace("\n", "↵\n").replace("\t", "⇥")


def compare_generation_to_reference(
    *,
    tokenizer,
    reference_ids: torch.Tensor,
    generated_ids: torch.Tensor,
) -> tuple[list[ComparedToken], list[ComparedToken], dict[str, int | bool]]:
    """Label generated tokens as accepted, rejected, extra, or missing."""
    comparisons: list[ComparedToken] = []
    missing_reference_tokens: list[ComparedToken] = []
    accepted_count = 0
    rejected_count = 0
    extra_count = 0

    reference_len = int(reference_ids.numel())
    generated_len = int(generated_ids.numel())
    overlap = min(reference_len, generated_len)

    for idx in range(overlap):
        generated_id = int(generated_ids[idx].item())
        expected_id = int(reference_ids[idx].item())
        if generated_id == expected_id:
            status = "accepted"
            accepted_count += 1
        else:
            status = "rejected"
            rejected_count += 1
        comparisons.append(
            ComparedToken(
                token_id=generated_id,
                token_text=_format_token_text(tokenizer, generated_id),
                status=status,
                expected_token_id=expected_id,
                expected_token_text=_format_token_text(tokenizer, expected_id),
            )
        )

    for idx in range(overlap, generated_len):
        generated_id = int(generated_ids[idx].item())
        comparisons.append(
            ComparedToken(
                token_id=generated_id,
                token_text=_format_token_text(tokenizer, generated_id),
                status="extra",
            )
        )
        extra_count += 1

    for idx in range(overlap, reference_len):
        expected_id = int(reference_ids[idx].item())
        missing_reference_tokens.append(
            ComparedToken(
                token_id=expected_id,
                token_text=_format_token_text(tokenizer, expected_id),
                status="missing",
            )
        )

    metrics = {
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "extra_count": extra_count,
        "missing_count": len(missing_reference_tokens),
        "exact_match": generated_len == reference_len and rejected_count == 0,
    }
    return comparisons, missing_reference_tokens, metrics


def get_eval_prompt(sample: dict[str, Any]) -> str:
    """Extract a single prompt string from an eval dataset sample."""
    turns = sample.get("turns")
    if not turns:
        raise ValueError("Expected eval dataset samples to contain a non-empty 'turns' field.")
    return "\n\n".join(str(turn) for turn in turns)


def build_dflash_sequence_tree_processor(
    *,
    block_size: int,
):
    """Represent a DFlash block as a single linear chain in the tree decoder."""
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}.")
    return build_tree_processor(
        tree_type="block",
        tree_seq_depth=block_size,
        sub_tree_paths=[],
    )

def evaluate_prompt_suite(
    *,
    data_name: str,
    tokenizer,
    generate_fn,
    max_examples: int | None = None,
    max_new_tokens: int = 256,
) -> EvalSuiteResult:
    """Run generation over prompts from load_and_process_eval_dataset."""
    dataset = load_and_process_eval_dataset(data_name)
    examples: list[EvalExampleResult] = []

    for sample_idx in range(len(dataset)):
        if max_examples is not None and sample_idx >= max_examples:
            break
        sample = dataset[sample_idx]
        prompt = get_eval_prompt(sample)
        result = generate_fn(prompt=prompt, max_new_tokens=max_new_tokens)
        generated_ids = result.continuation_ids
        comparisons = [
            ComparedToken(
                token_id=int(token_id.item()),
                token_text=_format_token_text(tokenizer, int(token_id.item())),
                status="extra",
            )
            for token_id in generated_ids
        ]
        examples.append(
            EvalExampleResult(
                index=sample_idx,
                prompt=prompt,
                reference_text=None,
                generated_text=tokenizer.decode(generated_ids, skip_special_tokens=False),
                comparisons=comparisons,
                missing_reference_tokens=[],
                accepted_count=0,
                rejected_count=0,
                extra_count=len(comparisons),
                missing_count=0,
                exact_match=False,
                drafted_tokens=result.drafted_tokens,
                committed_tokens=result.committed_tokens,
                mean_acceptance_length=(
                    sum(result.acceptance_lengths) / len(result.acceptance_lengths)
                    if result.acceptance_lengths
                    else 0.0
                ),
            )
        )

    return EvalSuiteResult(
        examples=examples,
        total_examples=len(examples),
        exact_matches=0,
        accepted_count=0,
        rejected_count=0,
        extra_count=sum(example.extra_count for example in examples),
        missing_count=0,
    )


def aggregate_speculative_eval(
    *,
    generate_fn,
    data_name: str | None = None,
    dataset: Sequence[dict[str, Any]] | None = None,
    sample_indices: Sequence[int] | None = None,
    max_examples: int | None = None,
    max_new_tokens: int = 256,
    acceptance_length_max: int | None = None,
) -> SpecEvalAggregateResult:
    """Aggregate speculative-decoding metrics over a fixed eval subset."""
    if dataset is None:
        if data_name is None:
            raise ValueError("data_name is required when dataset is not provided.")
        dataset = load_and_process_eval_dataset(data_name)

    if sample_indices is None:
        limit = len(dataset) if max_examples is None else min(len(dataset), max_examples)
        sample_indices = list(range(limit))
    else:
        sample_indices = list(sample_indices)
        if max_examples is not None:
            sample_indices = sample_indices[:max_examples]

    histogram = [0] * (acceptance_length_max + 1) if acceptance_length_max is not None else []
    total_examples = 0
    total_committed_tokens = 0
    total_steps = 0
    total_acceptance_length = 0
    off_main_path_last_accept_count = 0

    start_time = time.perf_counter()
    for sample_idx in sample_indices:
        sample = dataset[int(sample_idx)]
        prompt = get_eval_prompt(sample)
        result = generate_fn(prompt=prompt, max_new_tokens=max_new_tokens)
        off_main_flags = getattr(
            result,
            "off_main_path_last_accept_flags",
            [False] * len(result.acceptance_lengths),
        )
        if len(off_main_flags) != len(result.acceptance_lengths):
            raise ValueError(
                "off_main_path_last_accept_flags must align with acceptance_lengths "
                f"for sample index {sample_idx}."
            )

        total_examples += 1
        total_committed_tokens += int(result.committed_tokens)
        total_steps += len(result.acceptance_lengths)
        total_acceptance_length += sum(int(length) for length in result.acceptance_lengths)
        off_main_path_last_accept_count += sum(bool(flag) for flag in off_main_flags)

        for length in result.acceptance_lengths:
            accept_length = int(length)
            if accept_length < 0:
                raise ValueError(f"acceptance length must be non-negative, got {accept_length}.")
            if acceptance_length_max is None:
                if accept_length >= len(histogram):
                    histogram.extend([0] * (accept_length + 1 - len(histogram)))
            elif accept_length > acceptance_length_max:
                raise ValueError(
                    f"acceptance length {accept_length} exceeds histogram max "
                    f"{acceptance_length_max}."
                )
            histogram[accept_length] += 1

    return SpecEvalAggregateResult(
        total_examples=total_examples,
        total_committed_tokens=total_committed_tokens,
        total_steps=total_steps,
        total_acceptance_length=total_acceptance_length,
        acceptance_length_histogram=histogram,
        off_main_path_last_accept_count=off_main_path_last_accept_count,
        elapsed_time_sec=time.perf_counter() - start_time,
    )


def render_eval_suite_html(report: EvalSuiteResult) -> str:
    """Render a minimal HTML report with colored token spans."""
    def render_token_spans(tokens: Sequence[ComparedToken]) -> str:
        if not tokens:
            return '<span class="empty">None</span>'
        return "".join(
            '<span class="token {status}" title="{title}">{text}</span>'.format(
                status=escape(token.status),
                title=escape(f"id={token.token_id}"),
                text=escape(token.token_text),
            )
            for token in tokens
        )

    sections = []
    for example in report.examples:
        reference_block = ""
        if example.reference_text is not None:
            reference_block = "<h3>Reference</h3><pre>{}</pre>".format(escape(example.reference_text))
        sections.append(
            """
            <section class="example">
              <h2>Example {index}</h2>
              <p>
                match={match_label}
                accepted={accepted}
                rejected={rejected}
                extra={extra}
                missing={missing}
                drafted={drafted}
                committed={committed}
                mean_acceptance={mean_acceptance:.2f}
              </p>
              <h3>Prompt</h3>
              <pre>{prompt}</pre>
              {reference_block}
              <h3>Generated</h3>
              <pre>{generated_text}</pre>
              <h3>Accepted / Rejected / Extra Tokens</h3>
              <div>{comparison_tokens}</div>
              <h3>Missing Reference Tokens</h3>
              <div>{missing_tokens}</div>
            </section>
            """.format(
                index=example.index,
                match_label="exact" if example.exact_match else "mismatch",
                accepted=example.accepted_count,
                rejected=example.rejected_count,
                extra=example.extra_count,
                missing=example.missing_count,
                drafted=example.drafted_tokens,
                committed=example.committed_tokens,
                mean_acceptance=example.mean_acceptance_length,
                prompt=escape(example.prompt),
                reference_block=reference_block,
                generated_text=escape(example.generated_text),
                comparison_tokens=render_token_spans(example.comparisons),
                missing_tokens=render_token_spans(example.missing_reference_tokens),
            )
        )

    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <title>Spec Decode Eval Report</title>
        <style>
          body {{ font-family: sans-serif; line-height: 1.5; }}
          .example {{ margin: 1.5rem 0; }}
          .token {{ padding: 0.1rem 0.35rem; margin-right: 0.2rem; }}
          .accepted {{ background: #c8f7c5; }}
          .rejected {{ background: #f7c5c5; }}
          .extra, .missing {{ background: #c5d9f7; }}
          pre {{ white-space: pre-wrap; }}
        </style>
      </head>
      <body>
        <h1>Spec Decode Validation Report</h1>
        <p>
          examples={total_examples}
          exact_matches={exact_matches}
          accepted={accepted_count}
          rejected={rejected_count}
          extra={extra_count}
          missing={missing_count}
        </p>
        {sections}
      </body>
    </html>
    """.format(
        total_examples=report.total_examples,
        exact_matches=report.exact_matches,
        accepted_count=report.accepted_count,
        rejected_count=report.rejected_count,
        extra_count=report.extra_count,
        missing_count=report.missing_count,
        sections="".join(sections),
    )


@torch.inference_mode()
def official_dflash_generate_from_ids(
    *,
    target_model,
    drafter_model,
    tokenizer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
) -> SpecDecodeResult:
    """Run generation through the official DFlash remote-code backend."""
    device = next(target_model.parameters()).device
    prompt_ids = prompt_ids.to(device).view(1, -1)
    eos_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    output_ids = drafter_model.spec_generate(
        input_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        target=target_model,
        stop_token_ids=eos_token_ids,
    )[0]
    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    continuation_ids = output_ids[prompt_ids.shape[1] :].detach().cpu()
    return SpecDecodeResult(
        token_ids=output_ids.detach().cpu(),
        continuation_ids=continuation_ids,
        text=text,
        acceptance_lengths=[],
        off_main_path_last_accept_flags=[],
        drafted_tokens=0,
        committed_tokens=int(continuation_ids.numel()),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for prompt and eval-dataset modes."""
    parser = argparse.ArgumentParser(description="Tree speculative decoding.")
    parser.add_argument("--target-model")
    parser.add_argument("--draft-model")
    parser.add_argument("--official-dflash-model", default=None)
    parser.add_argument("--backend", choices=["auto", "tree_flash", "official_dflash"], default="auto")
    parser.add_argument("--prompt")
    parser.add_argument("--eval-data", default=None)
    parser.add_argument("--eval-max-examples", type=int, default=None)
    parser.add_argument("--html-report", default=None)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument(
        "--tree-type",
        default="block",
        choices=["fixed", "block", "branch_off", "prunable"],
    )
    parser.add_argument("--tree-seq-depth", type=int, default=4)
    parser.add_argument("--sub-tree-paths", nargs="*", default=None)
    parser.add_argument("--tree-args", type=str, default="{}")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    return parser


def main() -> None:
    """Run prompt decoding or eval-dataset generation from the command line."""
    parser = build_parser()
    args = parser.parse_args()
    if args.prompt is None and args.eval_data is None:
        parser.error("one of --prompt or --eval-data is required")
    tree_args = json.loads(args.tree_args)
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    backend = args.backend
    draft_model_name = args.draft_model
    if args.official_dflash_model is not None:
        draft_model_name = resolve_official_dflash_model(args.official_dflash_model)
        if backend == "auto":
            backend = "official_dflash"
    elif backend == "auto":
        backend = "tree_flash"
    if draft_model_name is None:
        parser.error("a draft model is required via --draft-model or --official-dflash-model")
    target_model_name = args.target_model or infer_target_model_for_draft_model(draft_model_name)
    if target_model_name is None:
        parser.error("target model is required when it cannot be inferred from the draft model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer_kwargs = {"trust_remote_code": True} if backend == "official_dflash" else {}
    tokenizer = AutoTokenizer.from_pretrained(target_model_name, **tokenizer_kwargs)
    target_model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype_map[args.dtype],
    }
    if backend == "tree_flash":
        target_model_kwargs["attn_implementation"] = "flex_attention"
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        **target_model_kwargs,
    ).to(device)
    target_model.eval()

    if backend == "official_dflash":
        drafter_model = AutoModel.from_pretrained(
            draft_model_name,
            trust_remote_code=True,
            torch_dtype=dtype_map[args.dtype],
        ).to(device)
    else:
        drafter_model = DFlashDraftModel.from_pretrained(
            draft_model_name,
            torch_dtype=dtype_map[args.dtype],
        ).to(device)
    drafter_model.eval()

    if args.eval_data is not None:
        if backend == "tree_flash":
            tree_processor = build_tree_processor(
                tree_type=args.tree_type,
                tree_seq_depth=args.tree_seq_depth,
                sub_tree_paths=args.sub_tree_paths,
                tree_args=tree_args,
            )

            def generate_fn(*, prompt: str, max_new_tokens: int) -> SpecDecodeResult:
                return speculative_generate(
                    target_model=target_model,
                    drafter_model=drafter_model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    tree_processor=tree_processor,
                    max_new_tokens=max_new_tokens,
                    temperature=args.temperature,
                )
        else:
            def generate_fn(*, prompt: str, max_new_tokens: int) -> SpecDecodeResult:
                prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
                return official_dflash_generate_from_ids(
                    target_model=target_model,
                    drafter_model=drafter_model,
                    tokenizer=tokenizer,
                    prompt_ids=prompt_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=args.temperature,
                )

        report = evaluate_prompt_suite(
            data_name=args.eval_data,
            tokenizer=tokenizer,
            generate_fn=generate_fn,
            max_examples=args.eval_max_examples,
            max_new_tokens=args.max_new_tokens,
        )
        print(
            json.dumps(
                {
                    "total_examples": report.total_examples,
                    "exact_matches": report.exact_matches,
                    "accepted_count": report.accepted_count,
                    "rejected_count": report.rejected_count,
                    "extra_count": report.extra_count,
                    "missing_count": report.missing_count,
                },
                indent=2,
            ),
            flush=True,
        )
        if args.html_report is not None:
            report_path = Path(args.html_report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(render_eval_suite_html(report), encoding="utf-8")
        return

    prompt = apply_chat_template_if_requested(
        tokenizer=tokenizer,
        prompt=args.prompt,
        enabled=args.apply_chat_template,
    )
    if backend == "official_dflash":
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
        result = official_dflash_generate_from_ids(
            target_model=target_model,
            drafter_model=drafter_model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    else:
        tree_processor = build_tree_processor(
            tree_type=args.tree_type,
            tree_seq_depth=args.tree_seq_depth,
            sub_tree_paths=args.sub_tree_paths,
            tree_args=tree_args,
        )
        result = speculative_generate(
            target_model=target_model,
            drafter_model=drafter_model,
            tokenizer=tokenizer,
            prompt=prompt,
            tree_processor=tree_processor,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    print(result.text, flush=True)


if __name__ == "__main__":
    main()
