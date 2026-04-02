from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


IGNORE_IDX = -1


@dataclass
class SequenceTreeNode:
    token_id: int
    parent_index: int
    depth: int
    local_prob: float
    path_prob: float
    rank: int
    main_path_position: int
    is_main_path: bool
    child_indices: list[int]


@dataclass
class GeneratedAnchorTree:
    anchor_main_path_position: int
    anchor_next_token_prob: float
    nodes: list[SequenceTreeNode]


@dataclass
class GeneratedSequenceTree:
    record_idx: int
    main_path_ids: list[int]
    response_start_position: int
    anchors: list[GeneratedAnchorTree]


def _list_jsonl_files(datapath: str) -> list[Path]:
    path = Path(datapath)
    if path.is_file():
        if path.suffix != ".jsonl":
            raise ValueError(f"Expected a .jsonl file, got {path}.")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    files = sorted(path.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in {path}")
    return files


def get_dataloader(datapath: str, tokenizer, seq_len: int, batch_size: int):
    try:
        from datasets import load_dataset
        records = load_dataset(datapath, split="train")
    except Exception:
        records: list[dict[str, Any]] = []
        for file_path in _list_jsonl_files(datapath):
            with file_path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    if not raw_line.strip():
                        continue
                    row = json.loads(raw_line)
                    if "prompt" not in row or "response" not in row:
                        raise ValueError(f"Missing prompt/response columns in {file_path}")
                    prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
                    response_ids = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
                    if not prompt_ids or not response_ids:
                        continue
                    total_ids = prompt_ids + response_ids
                    if len(total_ids) > seq_len:
                        continue
                    records.append(
                        {
                            "record_idx": len(records),
                            "input_ids": total_ids,
                            "response_interval": [len(prompt_ids), len(total_ids)],
                        }
                    )

        if not records:
            raise ValueError("No usable prompt/response records were found.")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id before batching.")

    def collate_fn(batch_rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(row["input_ids"]) for row in batch_rows)
        input_ids = torch.full((len(batch_rows), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch_rows), max_len), dtype=torch.long)
        response_interval = torch.zeros((len(batch_rows), 2), dtype=torch.long)
        record_idx = torch.zeros((len(batch_rows),), dtype=torch.long)

        for row_idx, row in enumerate(batch_rows):
            ids = torch.tensor(row["input_ids"], dtype=torch.long)
            seq_len_row = ids.numel()
            input_ids[row_idx, :seq_len_row] = ids
            attention_mask[row_idx, :seq_len_row] = 1
            response_interval[row_idx] = torch.tensor(row["response_interval"], dtype=torch.long)
            record_idx[row_idx] = int(row["record_idx"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_interval": response_interval,
            "record_idx": record_idx,
        }

    return DataLoader(records, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def _get_base_model(model):
    prefix = getattr(model, "base_model_prefix", None)
    if prefix and hasattr(model, prefix):
        return getattr(model, prefix)
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "base_model"):
        return model.base_model
    return model


def _call_with_supported_kwargs(module, **kwargs):
    try:
        allowed = inspect.signature(module.forward).parameters
    except (TypeError, ValueError):
        allowed = inspect.signature(module).parameters
    filtered = {key: value for key, value in kwargs.items() if key in allowed and value is not None}
    return module(**filtered)


def _extract_hidden_and_cache(outputs) -> tuple[torch.Tensor, Any]:
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state, getattr(outputs, "past_key_values", None)
    if isinstance(outputs, tuple):
        hidden_states = outputs[0]
        cache = outputs[1] if len(outputs) > 1 else None
        return hidden_states, cache
    raise TypeError(f"Unsupported model output type: {type(outputs)!r}")


def _score_hidden_states(hidden_states: torch.Tensor, lm_head) -> tuple[torch.Tensor, torch.Tensor]:
    logits = lm_head(hidden_states).float()
    probs = torch.softmax(logits, dim=-1)
    sorted_token_ids = torch.argsort(logits, dim=-1, descending=True, stable=True)
    sorted_token_probs = probs.gather(-1, sorted_token_ids)
    return sorted_token_ids.to(torch.long), sorted_token_probs.to(torch.float32)


def _select_children(
    *,
    sorted_token_ids: torch.Tensor,
    sorted_token_probs: torch.Tensor,
    coverage_alpha: float,
    forced_token_ids: torch.Tensor,
    forced_main_positions: torch.Tensor,
    forced_mask: torch.Tensor,
    max_children_per_node: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if max_children_per_node <= 0:
        raise ValueError("max_children_per_node must be positive.")

    num_queries, vocab_size = sorted_token_ids.shape
    device = sorted_token_ids.device
    cap = int(max_children_per_node)
    ranks = torch.arange(1, vocab_size + 1, device=device, dtype=torch.long).unsqueeze(0).expand(num_queries, -1)

    token_out = torch.full((num_queries, cap), IGNORE_IDX, dtype=torch.long, device=device)
    prob_out = torch.zeros((num_queries, cap), dtype=torch.float32, device=device)
    rank_out = torch.zeros((num_queries, cap), dtype=torch.long, device=device)
    main_out = torch.zeros((num_queries, cap), dtype=torch.bool, device=device)
    main_pos_out = torch.full((num_queries, cap), IGNORE_IDX, dtype=torch.long, device=device)
    selected_count = torch.zeros((num_queries,), dtype=torch.long, device=device)

    if num_queries == 0:
        return token_out, prob_out, rank_out, main_out, main_pos_out, selected_count

    forced_match = forced_mask.unsqueeze(-1) & sorted_token_ids.eq(forced_token_ids.unsqueeze(-1))
    forced_found = forced_match.any(dim=-1)
    if bool((forced_mask & ~forced_found).any().item()):
        raise ValueError("Forced child token is missing from candidate list.")

    forced_rank_idx = forced_match.to(torch.int64).argmax(dim=-1)
    forced_prob = sorted_token_probs.gather(1, forced_rank_idx.unsqueeze(-1)).squeeze(-1)
    forced_rank = forced_rank_idx + 1
    forced_count = forced_mask.to(torch.long)

    forced_rows = torch.nonzero(forced_mask, as_tuple=False).squeeze(-1)
    if forced_rows.numel() > 0:
        token_out[forced_rows, 0] = forced_token_ids[forced_rows]
        prob_out[forced_rows, 0] = forced_prob[forced_rows]
        rank_out[forced_rows, 0] = forced_rank[forced_rows]
        main_out[forced_rows, 0] = True
        main_pos_out[forced_rows, 0] = forced_main_positions[forced_rows]

    non_forced = ~forced_match
    non_forced_prob = torch.where(non_forced, sorted_token_probs, torch.zeros_like(sorted_token_probs))
    non_forced_seen = non_forced.to(torch.long).cumsum(dim=-1)
    count_before = forced_count.unsqueeze(-1) + non_forced_seen - 1
    prob_before = forced_prob.unsqueeze(-1) * forced_mask.unsqueeze(-1).to(sorted_token_probs.dtype)
    prob_before = prob_before + non_forced_prob.cumsum(dim=-1) - non_forced_prob
    keep_non_forced = (
        non_forced
        & (count_before < cap)
        & ((count_before == 0) | (prob_before < float(coverage_alpha)))
    )

    keep_slot = forced_count.unsqueeze(-1) + keep_non_forced.to(torch.long).cumsum(dim=-1) - 1
    keep_valid = keep_non_forced & (keep_slot >= 0) & (keep_slot < cap)
    if bool(keep_valid.any().item()):
        query_idx = torch.arange(num_queries, device=device).unsqueeze(-1).expand(-1, vocab_size)[keep_valid]
        slot_idx = keep_slot[keep_valid]
        token_out[query_idx, slot_idx] = sorted_token_ids[keep_valid]
        prob_out[query_idx, slot_idx] = sorted_token_probs[keep_valid]
        rank_out[query_idx, slot_idx] = ranks[keep_valid]

    selected_count = forced_count + keep_non_forced.to(torch.long).sum(dim=-1)
    fallback = selected_count.eq(0)
    fallback_rows = torch.nonzero(fallback, as_tuple=False).squeeze(-1)
    if fallback_rows.numel() > 0:
        token_out[fallback_rows, 0] = sorted_token_ids[fallback_rows, 0]
        prob_out[fallback_rows, 0] = sorted_token_probs[fallback_rows, 0]
        rank_out[fallback_rows, 0] = 1
        selected_count[fallback_rows] = 1

    return token_out, prob_out, rank_out, main_out, main_pos_out, selected_count


def _append_children(
    *,
    batch_idx: torch.Tensor,
    tree_idx: torch.Tensor,
    parent_idx: torch.Tensor,
    child_token_ids: torch.Tensor,
    child_local_probs: torch.Tensor,
    child_ranks: torch.Tensor,
    child_is_main: torch.Tensor,
    child_main_pos: torch.Tensor,
    child_count_per_parent: torch.Tensor,
    node_token_ids: torch.Tensor,
    node_parent_indices: torch.Tensor,
    node_depths: torch.Tensor,
    node_local_probs: torch.Tensor,
    node_path_probs: torch.Tensor,
    node_ranks: torch.Tensor,
    node_main_pos: torch.Tensor,
    node_is_main: torch.Tensor,
    node_valid: torch.Tensor,
    frontier: torch.Tensor,
    first_child: torch.Tensor,
    child_count: torch.Tensor,
    node_creation_index: torch.Tensor,
    next_free_node_idx: torch.Tensor,
    next_creation_index: torch.Tensor,
    ancestor_self: torch.Tensor,
) -> None:
    if batch_idx.numel() == 0:
        return

    device = batch_idx.device
    cap = child_token_ids.shape[-1]
    max_nodes = node_token_ids.shape[-1]
    slot_offsets = torch.arange(cap, device=device, dtype=torch.long).unsqueeze(0)
    base_slots = next_free_node_idx[batch_idx, tree_idx]
    child_slots = base_slots.unsqueeze(-1) + slot_offsets
    valid_child = slot_offsets < child_count_per_parent.unsqueeze(-1)

    if bool(valid_child.any().item()):
        max_required = int(child_slots[valid_child].max().item())
        if max_required >= max_nodes:
            raise ValueError("Tree tensor capacity is too small for the requested expansion.")

    parent_depth = node_depths[batch_idx, tree_idx, parent_idx]
    parent_path_prob = node_path_probs[batch_idx, tree_idx, parent_idx]
    parent_ancestor = ancestor_self[batch_idx, tree_idx, parent_idx]

    batch_grid = batch_idx.unsqueeze(-1).expand(-1, cap)
    tree_grid = tree_idx.unsqueeze(-1).expand(-1, cap)
    parent_grid = parent_idx.unsqueeze(-1).expand(-1, cap)
    depth_grid = (parent_depth + 1).unsqueeze(-1).expand(-1, cap)
    path_prob_grid = parent_path_prob.unsqueeze(-1) * child_local_probs
    creation_grid = next_creation_index[batch_idx, tree_idx].unsqueeze(-1) + slot_offsets
    ancestor_grid = parent_ancestor.unsqueeze(1).expand(-1, cap, -1)

    flat_batch = batch_grid[valid_child]
    flat_tree = tree_grid[valid_child]
    flat_parent = parent_grid[valid_child]
    flat_slot = child_slots[valid_child]

    node_token_ids[flat_batch, flat_tree, flat_slot] = child_token_ids[valid_child]
    node_parent_indices[flat_batch, flat_tree, flat_slot] = flat_parent
    node_depths[flat_batch, flat_tree, flat_slot] = depth_grid[valid_child]
    node_local_probs[flat_batch, flat_tree, flat_slot] = child_local_probs[valid_child]
    node_path_probs[flat_batch, flat_tree, flat_slot] = path_prob_grid[valid_child]
    node_ranks[flat_batch, flat_tree, flat_slot] = child_ranks[valid_child]
    node_main_pos[flat_batch, flat_tree, flat_slot] = child_main_pos[valid_child]
    node_is_main[flat_batch, flat_tree, flat_slot] = child_is_main[valid_child]
    node_valid[flat_batch, flat_tree, flat_slot] = True
    frontier[flat_batch, flat_tree, flat_slot] = True
    node_creation_index[flat_batch, flat_tree, flat_slot] = creation_grid[valid_child]
    ancestor_self[flat_batch, flat_tree, flat_slot] = ancestor_grid[valid_child]
    ancestor_self[flat_batch, flat_tree, flat_slot, flat_slot] = True

    has_children = child_count_per_parent > 0
    if bool(has_children.any().item()):
        first_child[
            batch_idx[has_children],
            tree_idx[has_children],
            parent_idx[has_children],
        ] = base_slots[has_children]
        child_count[
            batch_idx[has_children],
            tree_idx[has_children],
            parent_idx[has_children],
        ] = child_count_per_parent[has_children]

    next_free_node_idx[batch_idx, tree_idx] = base_slots + child_count_per_parent
    next_creation_index[batch_idx, tree_idx] = next_creation_index[batch_idx, tree_idx] + child_count_per_parent


def _build_flex_attention_mask(
    *,
    anchor_positions: torch.Tensor,
    query_valid: torch.Tensor,
    tree_key_valid: torch.Tensor,
    tree_can_attend: torch.Tensor,
    valid_tokens: torch.Tensor,
    context_len: int,
):
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError as exc:
        raise RuntimeError("Flex attention is not available in this PyTorch build.") from exc

    batch_size, query_count = anchor_positions.shape
    total_tree_keys = tree_key_valid.shape[-1]
    ctx_max = max(context_len - 1, 0)
    tree_max = max(total_tree_keys - 1, 0)

    def mask_mod(batch, _head, query, key):
        in_context = key < context_len
        ctx_idx = key.clamp(0, ctx_max)
        tree_idx = (key - context_len).clamp(0, tree_max)
        ctx_ok = (
            in_context
            & query_valid[batch, query]
            & valid_tokens[batch, ctx_idx]
            & (ctx_idx <= anchor_positions[batch, query])
        )
        tree_ok = (
            (~in_context)
            & query_valid[batch, query]
            & tree_key_valid[batch, tree_idx]
            & tree_can_attend[batch, query, tree_idx]
        )
        return ctx_ok | tree_ok

    return create_block_mask(
        mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=query_count,
        KV_LEN=context_len + total_tree_keys,
        device=anchor_positions.device,
        BLOCK_SIZE=128,
    )


def process_batch(batch, model, alpha, num_attend_tokens, depth, max_top_k):
    if max_top_k <= 0:
        raise ValueError("max_top_k must be positive.")
    if depth < 0:
        raise ValueError("depth must be non-negative.")

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    response_interval = batch["response_interval"]
    record_idx = batch["record_idx"]
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    response_interval = response_interval.to(device)
    record_idx = record_idx.to(device)

    base_model = _get_base_model(model)
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise ValueError("Model must expose get_output_embeddings().")

    base_out = _call_with_supported_kwargs(
        base_model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    hidden_states, kv_cache = _extract_hidden_and_cache(base_out)
    logits = lm_head(hidden_states).float()
    log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
    next_ids = input_ids[:, 1:]
    next_token_probs = torch.ones_like(input_ids, dtype=torch.float32)
    next_token_probs[:, 1:] = log_probs.gather(-1, next_ids.unsqueeze(-1)).squeeze(-1).exp().to(torch.float32)

    batch_size, seq_len = input_ids.shape
    valid_tokens = attention_mask.bool()
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    response_start = response_interval[:, 0].unsqueeze(-1)
    response_end = response_interval[:, 1].unsqueeze(-1)
    is_response = (positions >= response_start) & (positions < response_end)
    anchor_mask = is_response[:, :-1] & valid_tokens[:, 1:] & (next_token_probs[:, 1:] <= float(alpha))
    anchor_counts = anchor_mask.sum(dim=-1)
    max_trees = max(int(anchor_counts.max().item()), 1)
    anchor_order = anchor_mask.to(torch.long).cumsum(dim=-1) - 1

    anchor_positions = torch.zeros((batch_size, max_trees), dtype=torch.long, device=device)
    anchor_probs = torch.zeros((batch_size, max_trees), dtype=torch.float32, device=device)
    if bool(anchor_mask.any().item()):
        batch_grid = torch.arange(batch_size, device=device).unsqueeze(-1).expand(-1, seq_len - 1)
        slot_idx = anchor_order[anchor_mask]
        anchor_positions[batch_grid[anchor_mask], slot_idx] = positions[:, :-1][anchor_mask]
        anchor_probs[batch_grid[anchor_mask], slot_idx] = next_token_probs[:, 1:][anchor_mask]

    tree_active = torch.arange(max_trees, device=device).unsqueeze(0) < anchor_counts.unsqueeze(-1)
    max_nodes = 1 + (num_attend_tokens + 1) * max_top_k

    node_token_ids = torch.full((batch_size, max_trees, max_nodes), IGNORE_IDX, dtype=torch.long, device=device)
    node_parent_indices = torch.full((batch_size, max_trees, max_nodes), -1, dtype=torch.long, device=device)
    node_depths = torch.zeros((batch_size, max_trees, max_nodes), dtype=torch.long, device=device)
    node_local_probs = torch.zeros((batch_size, max_trees, max_nodes), dtype=torch.float32, device=device)
    node_path_probs = torch.zeros((batch_size, max_trees, max_nodes), dtype=torch.float32, device=device)
    node_ranks = torch.zeros((batch_size, max_trees, max_nodes), dtype=torch.long, device=device)
    node_main_pos = torch.full((batch_size, max_trees, max_nodes), IGNORE_IDX, dtype=torch.long, device=device)
    node_is_main = torch.zeros((batch_size, max_trees, max_nodes), dtype=torch.bool, device=device)
    node_valid = torch.zeros((batch_size, max_trees, max_nodes), dtype=torch.bool, device=device)
    node_expanded = torch.zeros((batch_size, max_trees, max_nodes), dtype=torch.bool, device=device)
    frontier = torch.zeros((batch_size, max_trees, max_nodes), dtype=torch.bool, device=device)
    first_child = torch.full((batch_size, max_trees, max_nodes), -1, dtype=torch.long, device=device)
    child_count = torch.zeros((batch_size, max_trees, max_nodes), dtype=torch.long, device=device)
    node_creation_index = torch.full((batch_size, max_trees, max_nodes), max_nodes * 2, dtype=torch.long, device=device)
    ancestor_self = torch.zeros((batch_size, max_trees, max_nodes, max_nodes), dtype=torch.bool, device=device)
    next_free_node_idx = tree_active.to(torch.long)
    next_creation_index = torch.zeros((batch_size, max_trees), dtype=torch.long, device=device)
    expanded_count = torch.zeros((batch_size, max_trees), dtype=torch.long, device=device)
    expansion_history = torch.full((batch_size, num_attend_tokens, max_trees), -1, dtype=torch.long, device=device)

    node_valid[:, :, 0] = tree_active
    node_local_probs[:, :, 0] = tree_active.to(torch.float32)
    node_path_probs[:, :, 0] = tree_active.to(torch.float32)
    ancestor_self[:, :, 0, 0] = tree_active

    root_batch_idx, root_tree_idx = torch.nonzero(tree_active, as_tuple=True)
    if root_batch_idx.numel() > 0:
        root_anchor_pos = anchor_positions[root_batch_idx, root_tree_idx]
        root_hidden = hidden_states[root_batch_idx, root_anchor_pos]
        root_sorted_ids, root_sorted_probs = _score_hidden_states(root_hidden, lm_head)
        forced_pos = root_anchor_pos + 1
        selected = _select_children(
            sorted_token_ids=root_sorted_ids,
            sorted_token_probs=root_sorted_probs,
            coverage_alpha=alpha,
            forced_token_ids=input_ids[root_batch_idx, forced_pos],
            forced_main_positions=forced_pos,
            forced_mask=torch.ones_like(root_batch_idx, dtype=torch.bool, device=device),
            max_children_per_node=max_top_k,
        )
        _append_children(
            batch_idx=root_batch_idx,
            tree_idx=root_tree_idx,
            parent_idx=torch.zeros_like(root_batch_idx),
            child_token_ids=selected[0],
            child_local_probs=selected[1],
            child_ranks=selected[2],
            child_is_main=selected[3],
            child_main_pos=selected[4],
            child_count_per_parent=selected[5],
            node_token_ids=node_token_ids,
            node_parent_indices=node_parent_indices,
            node_depths=node_depths,
            node_local_probs=node_local_probs,
            node_path_probs=node_path_probs,
            node_ranks=node_ranks,
            node_main_pos=node_main_pos,
            node_is_main=node_is_main,
            node_valid=node_valid,
            frontier=frontier,
            first_child=first_child,
            child_count=child_count,
            node_creation_index=node_creation_index,
            next_free_node_idx=next_free_node_idx,
            next_creation_index=next_creation_index,
            ancestor_self=ancestor_self,
        )

    fallback_token = input_ids[:, :1].expand(-1, max_trees)
    eye_trees = torch.eye(max_trees, dtype=torch.bool, device=device).unsqueeze(0)

    for step in range(num_attend_tokens):
        eligible = (
            tree_active.unsqueeze(-1)
            & frontier
            & ~node_expanded
            & (node_depths < depth)
            & (expanded_count < num_attend_tokens).unsqueeze(-1)
        )
        query_valid = eligible.any(dim=-1)
        if not bool(query_valid.any().item()):
            break

        masked_path = torch.where(eligible, node_path_probs, torch.full_like(node_path_probs, float("-inf")))
        best_path = masked_path.max(dim=-1).values
        candidate = eligible & node_path_probs.eq(best_path.unsqueeze(-1))
        masked_creation = torch.where(
            candidate,
            node_creation_index,
            torch.full_like(node_creation_index, node_creation_index.max().item() + 1),
        )
        chosen_node_idx = masked_creation.argmin(dim=-1)
        expansion_history[:, step, :] = torch.where(
            query_valid,
            chosen_node_idx,
            torch.full_like(chosen_node_idx, -1),
        )

        query_ids = torch.where(
            query_valid,
            node_token_ids.gather(2, chosen_node_idx.unsqueeze(-1)).squeeze(-1),
            fallback_token,
        )
        query_depths = node_depths.gather(2, chosen_node_idx.unsqueeze(-1)).squeeze(-1)
        query_pos = torch.where(query_valid, anchor_positions + query_depths, torch.zeros_like(anchor_positions))

        total_tree_keys = (step + 1) * max_trees
        tree_key_valid = torch.zeros((batch_size, total_tree_keys), dtype=torch.bool, device=device)
        tree_can_attend = torch.zeros((batch_size, max_trees, total_tree_keys), dtype=torch.bool, device=device)

        if step > 0:
            cached_idx = expansion_history[:, :step, :].permute(0, 2, 1)
            cached_valid = cached_idx.ge(0)
            cached_idx_clamped = cached_idx.clamp(min=0)
            ancestor_rows = ancestor_self.gather(
                2,
                chosen_node_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, max_nodes),
            ).squeeze(2)
            cached_is_ancestor = ancestor_rows.gather(2, cached_idx_clamped)
            prev_attend = (cached_valid & cached_is_ancestor).unsqueeze(-1) & eye_trees.unsqueeze(2)
            tree_key_valid[:, : step * max_trees] = expansion_history[:, :step, :].permute(0, 2, 1).reshape(batch_size, -1).ge(0)
            tree_can_attend[:, :, : step * max_trees] = prev_attend.reshape(batch_size, max_trees, step * max_trees)

        tree_key_valid[:, step * max_trees : (step + 1) * max_trees] = query_valid
        tree_can_attend[:, :, step * max_trees : (step + 1) * max_trees] = query_valid.unsqueeze(-1) & eye_trees
        flex_mask = _build_flex_attention_mask(
            anchor_positions=anchor_positions,
            query_valid=query_valid,
            tree_key_valid=tree_key_valid,
            tree_can_attend=tree_can_attend,
            valid_tokens=valid_tokens,
            context_len=seq_len,
        )

        cache_position = torch.arange(
            seq_len + step * max_trees,
            seq_len + (step + 1) * max_trees,
            device=device,
        )
        step_out = _call_with_supported_kwargs(
            base_model,
            input_ids=query_ids,
            attention_mask=flex_mask,
            position_ids=query_pos,
            past_key_values=kv_cache,
            cache_position=cache_position,
            use_cache=True,
        )
        step_hidden, kv_cache = _extract_hidden_and_cache(step_out)

        active_batch_idx, active_tree_idx = torch.nonzero(query_valid, as_tuple=True)
        active_parent_idx = chosen_node_idx[query_valid]
        if active_batch_idx.numel() == 0:
            continue

        node_expanded[active_batch_idx, active_tree_idx, active_parent_idx] = True
        frontier[active_batch_idx, active_tree_idx, active_parent_idx] = False
        expanded_count[active_batch_idx, active_tree_idx] = expanded_count[active_batch_idx, active_tree_idx] + 1

        active_hidden = step_hidden[query_valid]
        sorted_ids, sorted_probs = _score_hidden_states(active_hidden, lm_head)
        parent_main = node_is_main[active_batch_idx, active_tree_idx, active_parent_idx]
        forced_pos = node_main_pos[active_batch_idx, active_tree_idx, active_parent_idx] + 1
        forced_ok = parent_main & (forced_pos >= 0) & (forced_pos < seq_len) & valid_tokens[active_batch_idx, forced_pos.clamp(min=0, max=seq_len - 1)]
        forced_ids = input_ids[active_batch_idx, forced_pos.clamp(min=0, max=seq_len - 1)]
        selected = _select_children(
            sorted_token_ids=sorted_ids,
            sorted_token_probs=sorted_probs,
            coverage_alpha=alpha,
            forced_token_ids=forced_ids,
            forced_main_positions=forced_pos,
            forced_mask=forced_ok,
            max_children_per_node=max_top_k,
        )
        _append_children(
            batch_idx=active_batch_idx,
            tree_idx=active_tree_idx,
            parent_idx=active_parent_idx,
            child_token_ids=selected[0],
            child_local_probs=selected[1],
            child_ranks=selected[2],
            child_is_main=selected[3],
            child_main_pos=selected[4],
            child_count_per_parent=selected[5],
            node_token_ids=node_token_ids,
            node_parent_indices=node_parent_indices,
            node_depths=node_depths,
            node_local_probs=node_local_probs,
            node_path_probs=node_path_probs,
            node_ranks=node_ranks,
            node_main_pos=node_main_pos,
            node_is_main=node_is_main,
            node_valid=node_valid,
            frontier=frontier,
            first_child=first_child,
            child_count=child_count,
            node_creation_index=node_creation_index,
            next_free_node_idx=next_free_node_idx,
            next_creation_index=next_creation_index,
            ancestor_self=ancestor_self,
        )

    sequences: list[GeneratedSequenceTree] = []
    lengths = attention_mask.sum(dim=-1).tolist()
    for batch_idx in range(batch_size):
        row_len = int(lengths[batch_idx])
        row_main_path = input_ids[batch_idx, :row_len].tolist()
        row_response_start = int(response_interval[batch_idx, 0].item())
        row_anchors: list[GeneratedAnchorTree] = []
        for tree_idx in range(int(anchor_counts[batch_idx].item())):
            node_count = int(next_free_node_idx[batch_idx, tree_idx].item())
            nodes: list[SequenceTreeNode] = []
            for node_idx in range(node_count):
                nodes.append(
                    SequenceTreeNode(
                        token_id=int(node_token_ids[batch_idx, tree_idx, node_idx].item()),
                        parent_index=int(node_parent_indices[batch_idx, tree_idx, node_idx].item()),
                        depth=int(node_depths[batch_idx, tree_idx, node_idx].item()),
                        local_prob=float(node_local_probs[batch_idx, tree_idx, node_idx].item()),
                        path_prob=float(node_path_probs[batch_idx, tree_idx, node_idx].item()),
                        rank=int(node_ranks[batch_idx, tree_idx, node_idx].item()),
                        main_path_position=int(node_main_pos[batch_idx, tree_idx, node_idx].item()),
                        is_main_path=bool(node_is_main[batch_idx, tree_idx, node_idx].item()),
                        child_indices=[
                            int(first_child[batch_idx, tree_idx, node_idx].item()),
                            int(child_count[batch_idx, tree_idx, node_idx].item()),
                        ],
                    )
                )
            row_anchors.append(
                GeneratedAnchorTree(
                    anchor_main_path_position=int(anchor_positions[batch_idx, tree_idx].item()),
                    anchor_next_token_prob=float(anchor_probs[batch_idx, tree_idx].item()),
                    nodes=nodes,
                )
            )
        sequences.append(
            GeneratedSequenceTree(
                record_idx=int(record_idx[batch_idx].item()),
                main_path_ids=[int(token_id) for token_id in row_main_path],
                response_start_position=row_response_start,
                anchors=row_anchors,
            )
        )
    return sequences


def initialize_hdf5(hf: h5py.File) -> None:
    hf.create_dataset("main_path_ids", shape=(0,), maxshape=(None,), dtype="int32")
    hf.create_dataset("main_path_offsets", shape=(1,), maxshape=(None,), dtype="int64")
    hf["main_path_offsets"][0] = 0
    hf.create_dataset("response_start_positions", shape=(0,), maxshape=(None,), dtype="int32")
    hf.create_dataset("record_idx", shape=(0,), maxshape=(None,), dtype="int64")
    hf.create_dataset("sequence_anchor_offsets", shape=(1,), maxshape=(None,), dtype="int64")
    hf["sequence_anchor_offsets"][0] = 0
    hf.create_dataset("anchor_main_path_positions", shape=(0,), maxshape=(None,), dtype="int32")
    hf.create_dataset("anchor_next_token_probs", shape=(0,), maxshape=(None,), dtype="float32")
    hf.create_dataset("anchor_node_offsets", shape=(1,), maxshape=(None,), dtype="int64")
    hf["anchor_node_offsets"][0] = 0
    hf.create_dataset("node_token_ids", shape=(0,), maxshape=(None,), dtype="int32")
    hf.create_dataset("node_parent_indices", shape=(0,), maxshape=(None,), dtype="int32")
    hf.create_dataset("node_depths", shape=(0,), maxshape=(None,), dtype="int16")
    hf.create_dataset("node_local_probs", shape=(0,), maxshape=(None,), dtype="float32")
    hf.create_dataset("node_path_probs", shape=(0,), maxshape=(None,), dtype="float32")
    hf.create_dataset("node_ranks", shape=(0,), maxshape=(None,), dtype="int32")
    hf.create_dataset("node_main_path_positions", shape=(0,), maxshape=(None,), dtype="int32")
    hf.create_dataset("node_is_main_path", shape=(0,), maxshape=(None,), dtype="bool")
    hf.create_dataset("node_first_child", shape=(0,), maxshape=(None,), dtype="int32")
    hf.create_dataset("node_child_count", shape=(0,), maxshape=(None,), dtype="int32")
    hf.attrs["format_version"] = "stage2"
    hf.attrs["attn_implementation"] = "flex_attention"


def _append_1d(hf: h5py.File, name: str, values, dtype=None) -> None:
    array = np.asarray(list(values), dtype=dtype)
    dataset = hf[name]
    start = dataset.shape[0]
    dataset.resize(start + len(array), axis=0)
    if len(array):
        dataset[start:] = array


def flush_hdf5(hf: h5py.File, sequences: list[GeneratedSequenceTree]) -> None:
    main_path_ids: list[int] = []
    main_path_offsets = [int(hf["main_path_offsets"][-1])]
    response_start_positions: list[int] = []
    record_idx: list[int] = []
    anchor_positions: list[int] = []
    anchor_probs: list[float] = []
    sequence_anchor_offsets = [int(hf["sequence_anchor_offsets"][-1])]
    anchor_node_offsets = [int(hf["anchor_node_offsets"][-1])]
    node_token_ids: list[int] = []
    node_parent_indices: list[int] = []
    node_depths: list[int] = []
    node_local_probs: list[float] = []
    node_path_probs: list[float] = []
    node_ranks: list[int] = []
    node_main_pos: list[int] = []
    node_is_main: list[bool] = []
    node_first_child: list[int] = []
    node_child_count: list[int] = []

    for sequence in sequences:
        main_path_ids.extend(sequence.main_path_ids)
        main_path_offsets.append(main_path_offsets[-1] + len(sequence.main_path_ids))
        response_start_positions.append(sequence.response_start_position)
        record_idx.append(sequence.record_idx)
        sequence_anchor_offsets.append(sequence_anchor_offsets[-1] + len(sequence.anchors))
        for anchor in sequence.anchors:
            anchor_positions.append(anchor.anchor_main_path_position)
            anchor_probs.append(anchor.anchor_next_token_prob)
            anchor_node_offsets.append(anchor_node_offsets[-1] + len(anchor.nodes))
            for node in anchor.nodes:
                node_token_ids.append(node.token_id)
                node_parent_indices.append(node.parent_index)
                node_depths.append(node.depth)
                node_local_probs.append(node.local_prob)
                node_path_probs.append(node.path_prob)
                node_ranks.append(node.rank)
                node_main_pos.append(node.main_path_position)
                node_is_main.append(node.is_main_path)
                node_first_child.append(node.child_indices[0])
                node_child_count.append(node.child_indices[1])

    _append_1d(hf, "main_path_ids", main_path_ids, dtype=np.int32)
    _append_1d(hf, "response_start_positions", response_start_positions, dtype=np.int32)
    _append_1d(hf, "record_idx", record_idx, dtype=np.int64)
    _append_1d(hf, "anchor_main_path_positions", anchor_positions, dtype=np.int32)
    _append_1d(hf, "anchor_next_token_probs", anchor_probs, dtype=np.float32)
    _append_1d(hf, "node_token_ids", node_token_ids, dtype=np.int32)
    _append_1d(hf, "node_parent_indices", node_parent_indices, dtype=np.int32)
    _append_1d(hf, "node_depths", node_depths, dtype=np.int16)
    _append_1d(hf, "node_local_probs", node_local_probs, dtype=np.float32)
    _append_1d(hf, "node_path_probs", node_path_probs, dtype=np.float32)
    _append_1d(hf, "node_ranks", node_ranks, dtype=np.int32)
    _append_1d(hf, "node_main_path_positions", node_main_pos, dtype=np.int32)
    _append_1d(hf, "node_is_main_path", node_is_main, dtype=np.bool_)
    _append_1d(hf, "node_first_child", node_first_child, dtype=np.int32)
    _append_1d(hf, "node_child_count", node_child_count, dtype=np.int32)

    hf["main_path_offsets"].resize(len(hf["main_path_offsets"]) + len(main_path_offsets) - 1, axis=0)
    hf["main_path_offsets"][-(len(main_path_offsets) - 1) :] = np.asarray(main_path_offsets[1:], dtype=np.int64)
    hf["sequence_anchor_offsets"].resize(len(hf["sequence_anchor_offsets"]) + len(sequence_anchor_offsets) - 1, axis=0)
    hf["sequence_anchor_offsets"][-(len(sequence_anchor_offsets) - 1) :] = np.asarray(sequence_anchor_offsets[1:], dtype=np.int64)
    hf["anchor_node_offsets"].resize(len(hf["anchor_node_offsets"]) + len(anchor_node_offsets) - 1, axis=0)
    hf["anchor_node_offsets"][-(len(anchor_node_offsets) - 1) :] = np.asarray(anchor_node_offsets[1:], dtype=np.int64)


def _parse_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def main():
    parser = argparse.ArgumentParser(description="Generate Stage 2 continuation trees from prompt/response JSONL.")
    parser.add_argument("--input", required=True, help="A .jsonl file or directory of .jsonl files.")
    parser.add_argument("--output", required=True, help="Output HDF5 path.")
    parser.add_argument("--model", required=True, help="Model name or path.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer name or path. Defaults to --model.")
    parser.add_argument("--seq-len", type=int, default=3072, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--num-attend-tokens", type=int, default=512)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--max-top-k", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=_parse_dtype(args.dtype),
        attn_implementation="flex_attention",
    ).to(args.device)
    model.eval()

    if getattr(model.config, "_attn_implementation", None) != "flex_attention":
        raise ValueError("The loaded model is not configured for flex_attention.")

    dataloader = get_dataloader(args.input, tokenizer, args.seq_len, args.batch_size)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as hf:
        initialize_hdf5(hf)
        with torch.no_grad():
            for batch in dataloader:
                sequences = process_batch(
                    batch=batch,
                    model=model,
                    alpha=args.alpha,
                    num_attend_tokens=args.num_attend_tokens,
                    depth=args.depth,
                    max_top_k=args.max_top_k,
                )
                flush_hdf5(hf, sequences)


if __name__ == "__main__":
    main()
