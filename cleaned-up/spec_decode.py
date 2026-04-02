from __future__ import annotations

import sys
from dataclasses import dataclass

import torch
from transformers.cache_utils import Cache, DynamicCache

from src.trees import PrunableTreeProcessor, TreeInfo, subset_tree_info
from utils import gather_token_probability, sample_from_logits, unwrap_model

sys.modules.setdefault("spec_decode", sys.modules[__name__])


@dataclass
class SpecDecodeResult:
    token_ids: torch.Tensor
    continuation_ids: torch.Tensor
    text: str
    acceptance_lengths: list[int]
    off_main_path_last_accept_flags: list[bool]
    drafted_tokens: int
    committed_tokens: int


def build_verifier_score_mod(
    *,
    tree_info: TreeInfo,
    prefix_len: int,
):
    """Allow all prefix tokens and only ancestor/self links inside the tree."""
    tree_mask = tree_info.tree_mask
    block_size = tree_info.block_size

    def score_mod(score, B, H, Q, KV):
        del B, H
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
    """Keep the prefix and only the accepted drafted tree nodes in the cache."""
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


def gather_path_indices(
    node_idx: int,
    tree_info: TreeInfo,
) -> list[int]:
    """Return the root-to-node path, inclusive."""
    path = []
    cur = node_idx
    while cur >= 0:
        path.append(cur)
        cur = int(tree_info.parent_idx[cur].item())
    return list(reversed(path))


def choose_deepest_valid_node(
    accepted_mask: torch.Tensor,
    tree_info: TreeInfo,
) -> int:
    """Return the deepest node whose full path was accepted."""
    valid_nodes = []
    for node_idx in range(tree_info.block_size):
        path_mask = tree_info.tree_mask[node_idx]
        if bool(accepted_mask[path_mask].all().item()):
            valid_nodes.append(node_idx)
    if not valid_nodes:
        return 0
    return max(valid_nodes, key=lambda idx: (int(tree_info.depth[idx].item()), -idx))


def _compute_path_correctness_probabilities(
    *,
    tree_info: TreeInfo,
    node_correctness_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute cumulative ancestor correctness probabilities for each node."""
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, TreeInfo]:
    """Prune a drafted tree down to a smaller ancestor-closed subset."""
    keep_indices = select_pruned_keep_indices(
        tree_info=tree_info,
        node_correctness_probs=node_correctness_probs,
        candidate_tree_size=candidate_tree_size,
    )
    if len(keep_indices) == tree_info.block_size:
        return tree_token_ids, draft_logits, draft_token_probs, node_correctness_probs, tree_info

    keep = torch.tensor(keep_indices, dtype=torch.long, device=tree_token_ids.device)
    pruned_tree_info = subset_tree_info(tree_info, keep)
    return (
        tree_token_ids.index_select(0, keep),
        draft_logits.index_select(0, keep),
        draft_token_probs.index_select(0, keep),
        node_correctness_probs.index_select(0, keep),
        pruned_tree_info,
    )


def draft_tree(
    *,
    drafter_model,
    raw_drafter,
    target_embeddings,
    target_lm_head,
    tree_processor,
    target_ctx_features: torch.Tensor,
    drafter_cache: Cache,
    current_root_token: int,
    root_position: int,
    temperature: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, TreeInfo, torch.Tensor]:
    """Draft one tree from the current root token and score it with the q-head."""
    tree_info = tree_processor.build_tree_info(batch_size=1, num_blocks=1, device=device)
    tree_len = tree_info.block_size
    pre_len = drafter_cache.get_seq_length()
    noise_ids = torch.full((1, tree_len), raw_drafter.mask_token_id, dtype=torch.long, device=device)
    noise_ids[0, 0] = current_root_token
    noise_embeddings = target_embeddings(noise_ids)
    tree_position_ids = (tree_info.depth.view(1, -1) + root_position).to(device)
    position_ids = torch.cat(
        [
            torch.arange(pre_len, pre_len + target_ctx_features.shape[1], device=device),
            tree_position_ids.squeeze(0),
        ],
        dim=0,
    ).unsqueeze(0)
    draft_hidden_states, _, q_logits = drafter_model(
        hidden_states=noise_embeddings,
        position_ids=position_ids,
        tree_info=tree_info,
        target_ctx_features=target_ctx_features,
        past_key_values=drafter_cache,
        use_cache=True,
        return_aux=True,
    )
    drafter_cache.crop(pre_len + target_ctx_features.shape[1])
    draft_logits = target_lm_head(draft_hidden_states)[0]
    tree_token_ids = sample_from_logits(draft_logits, temperature)
    tree_token_ids[0] = current_root_token
    draft_token_probs = gather_token_probability(draft_logits, tree_token_ids, temperature)
    draft_token_probs[0] = 1.0
    pruning_scores = torch.sigmoid(q_logits[0]).to(torch.float32)
    pruning_scores[0] = 1.0
    return tree_token_ids, draft_logits, draft_token_probs, tree_info, pruning_scores


def verify_tree(
    *,
    target_model,
    raw_drafter,
    tree_token_ids: torch.Tensor,
    tree_info: TreeInfo,
    root_position: int,
    target_cache: Cache,
) -> tuple[Cache, torch.Tensor, torch.Tensor]:
    """Verify drafted tokens with the target model under tree constraints."""
    prefix_len = target_cache.get_seq_length()
    score_mod = build_verifier_score_mod(tree_info=tree_info, prefix_len=prefix_len)
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
    q_scores: torch.Tensor,
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

    proposal_probs = q_scores.to(device=tree_token_ids.device, dtype=torch.float32).clone()
    proposal_probs[0] = 1.0
    for parent_idx in range(tree_info.block_size):
        child_mask = tree_info.parent_idx == parent_idx
        child_indices = torch.nonzero(child_mask, as_tuple=False).squeeze(-1)
        if child_indices.numel() == 0:
            continue
        child_scores = proposal_probs.index_select(0, child_indices).clamp_min(0.0)
        score_sum = child_scores.sum()
        if float(score_sum.item()) <= 0.0:
            child_scores = torch.full_like(child_scores, 1.0 / float(child_indices.numel()))
        else:
            child_scores = child_scores / score_sum
        proposal_probs.index_copy_(0, child_indices, child_scores)

    for node_idx in range(1, tree_info.block_size):
        parent_idx = int(tree_info.parent_idx[node_idx].item())
        target_prob = gather_token_probability(
            verifier_logits[parent_idx].unsqueeze(0),
            tree_token_ids[node_idx].view(1),
            temperature,
        )[0]
        proposal_prob = proposal_probs[node_idx].clamp_min(1e-8)
        accept_prob = torch.clamp(target_prob / proposal_prob, max=1.0)
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
    if getattr(raw_drafter, "q_head", None) is None:
        raise ValueError("The cleaned-up speculative decoder requires a drafter with `use_q_head=True`.")

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
            target_ctx_features=target_ctx_features,
            drafter_cache=drafter_cache,
            current_root_token=current_root_token,
            root_position=root_position,
            temperature=temperature,
            device=device,
        )
        drafted_tokens += tree_info.block_size - 1
        if isinstance(tree_processor, PrunableTreeProcessor):
            tree_token_ids, draft_logits, draft_token_probs, pruning_scores, tree_info = prune_drafted_tree(
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
        )
        accepted_mask = build_acceptance_mask(
            q_scores=pruning_scores,
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
            [committed_path_tokens, torch.tensor([bonus_token], dtype=torch.long, device=device)],
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
