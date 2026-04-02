from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm


IGNORE_IDX = -1
TOKENIZATION_BATCH_SIZE = 1024


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


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: str | None = None

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def log(self, message: str) -> None:
        if self.is_primary:
            print(message, flush=True)

    def barrier(self) -> None:
        if self.is_distributed and dist.is_initialized():
            dist.barrier()

    def shutdown(self) -> None:
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()


@dataclass(frozen=True)
class HDF5MergeEntry:
    record_idx: int
    part_path: str
    seq_idx: int
    main_start: int
    main_end: int
    anchor_start: int
    anchor_end: int
    node_start: int
    node_end: int


def init_distributed_context(device_name: str | None = None) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        if device_name is None:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        return DistributedContext(
            rank=0,
            world_size=1,
            local_rank=0,
            device=torch.device(device_name),
            backend=None,
        )

    requested_device = torch.device(device_name or "cuda")
    if requested_device.type != "cuda":
        raise ValueError("Torchrun Stage 2 device parallelism requires CUDA devices.")
    if not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU Stage 2 requires CUDA when launched with torchrun.")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable in this PyTorch build.")

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=torch.device("cuda", local_rank),
        backend="nccl",
    )


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


def _normalize_input_record(
    *,
    input_ids: list[int],
    response_interval: list[int],
    seq_len: int,
    record_idx: int,
) -> dict[str, Any] | None:
    input_ids = [int(token_id) for token_id in input_ids]
    response_interval = [int(position) for position in response_interval]
    if len(response_interval) != 2:
        raise ValueError("response_interval must contain exactly two positions.")
    if not input_ids or response_interval[0] >= response_interval[1]:
        return None
    if response_interval[0] < 0 or response_interval[1] > len(input_ids):
        raise ValueError("response_interval is out of bounds for input_ids.")
    if len(input_ids) > seq_len:
        return None
    return {
        "record_idx": record_idx,
        "input_ids": input_ids,
        "response_interval": response_interval,
    }


def _normalize_prompt_response_ids(
    *,
    prompt_ids: list[int],
    response_ids: list[int],
    seq_len: int,
    record_idx: int,
) -> dict[str, Any] | None:
    if not prompt_ids or not response_ids:
        return None
    total_ids = prompt_ids + response_ids
    if len(total_ids) > seq_len:
        return None
    return {
        "record_idx": record_idx,
        "input_ids": total_ids,
        "response_interval": [len(prompt_ids), len(total_ids)],
    }


def _flush_pending_prompt_rows(
    pending_rows: list[dict[str, Any]],
    *,
    tokenizer,
    seq_len: int,
    records: list[dict[str, Any]],
) -> None:
    if not pending_rows:
        return
    prompts = [str(row["prompt"]) for row in pending_rows]
    responses = [str(row["response"]) for row in pending_rows]
    prompt_ids_batch = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    response_ids_batch = tokenizer(responses, add_special_tokens=False)["input_ids"]
    for prompt_ids, response_ids in zip(prompt_ids_batch, response_ids_batch, strict=True):
        normalized = _normalize_prompt_response_ids(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            seq_len=seq_len,
            record_idx=len(records),
        )
        if normalized is not None:
            records.append(normalized)
    pending_rows.clear()


def load_records(datapath: str, tokenizer, seq_len: int) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
        source_rows = load_dataset(datapath, split="train")
    except Exception:
        source_rows = []
        for file_path in _list_jsonl_files(datapath):
            with file_path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    if not raw_line.strip():
                        continue
                    source_rows.append(json.loads(raw_line))

    records: list[dict[str, Any]] = []
    pending_prompt_rows: list[dict[str, Any]] = []
    for row in source_rows:
        if "input_ids" in row and "response_interval" in row:
            _flush_pending_prompt_rows(
                pending_prompt_rows,
                tokenizer=tokenizer,
                seq_len=seq_len,
                records=records,
            )
            normalized = _normalize_input_record(
                input_ids=row["input_ids"],
                response_interval=row["response_interval"],
                seq_len=seq_len,
                record_idx=len(records),
            )
            if normalized is not None:
                records.append(normalized)
            continue

        if "prompt" not in row or "response" not in row:
            raise ValueError("Expected either input_ids/response_interval or prompt/response columns.")
        pending_prompt_rows.append(row)
        if len(pending_prompt_rows) >= TOKENIZATION_BATCH_SIZE:
            _flush_pending_prompt_rows(
                pending_prompt_rows,
                tokenizer=tokenizer,
                seq_len=seq_len,
                records=records,
            )

    _flush_pending_prompt_rows(
        pending_prompt_rows,
        tokenizer=tokenizer,
        seq_len=seq_len,
        records=records,
    )
    if not records:
        raise ValueError("No usable prompt/response records were found.")
    return records


def shard_records_for_rank(records: list[dict[str, Any]], ctx: DistributedContext) -> list[dict[str, Any]]:
    if not ctx.is_distributed:
        return records
    return [record for idx, record in enumerate(records) if idx % ctx.world_size == ctx.rank]


def build_dataloader(records: list[dict[str, Any]], tokenizer, batch_size: int):
    if not records:
        raise ValueError("No usable prompt/response records were found for this rank.")

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


def get_dataloader(
    datapath: str,
    tokenizer,
    seq_len: int,
    batch_size: int,
    *,
    ctx: DistributedContext | None = None,
):
    records = load_records(datapath, tokenizer, seq_len)
    if ctx is not None:
        records = shard_records_for_rank(records, ctx)
    return build_dataloader(records, tokenizer, batch_size)


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


def select_anchor_positions(
    *,
    is_response: torch.Tensor,
    valid_tokens: torch.Tensor,
    next_token_probs: torch.Tensor,
    alpha: float,
    max_trees: int,
) -> tuple[list[int], list[float]]:
    if max_trees <= 0:
        raise ValueError("max_trees must be positive.")
    if is_response.ndim != 1 or valid_tokens.ndim != 1 or next_token_probs.ndim != 1:
        raise ValueError("Expected 1D tensors for anchor selection.")
    if is_response.numel() < 2:
        return [], []

    anchor_mask = is_response[:-1] & valid_tokens[1:] & (next_token_probs[1:] <= float(alpha))
    candidate_positions = torch.nonzero(anchor_mask, as_tuple=False).squeeze(-1)
    if candidate_positions.numel() == 0:
        return [], []

    candidate_probs = next_token_probs[1:].gather(0, candidate_positions).to(torch.float32)
    if int(candidate_positions.numel()) > max_trees:
        hardest = torch.argsort(candidate_probs, descending=False, stable=True)[:max_trees]
        candidate_positions = candidate_positions[hardest]
        candidate_probs = candidate_probs[hardest]
        in_order = torch.argsort(candidate_positions, stable=True)
        candidate_positions = candidate_positions[in_order]
        candidate_probs = candidate_probs[in_order]

    return candidate_positions.tolist(), [float(prob) for prob in candidate_probs.tolist()]


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

    batch_grid = batch_idx.unsqueeze(-1).expand(-1, cap)
    tree_grid = tree_idx.unsqueeze(-1).expand(-1, cap)
    parent_grid = parent_idx.unsqueeze(-1).expand(-1, cap)
    depth_grid = (parent_depth + 1).unsqueeze(-1).expand(-1, cap)
    path_prob_grid = parent_path_prob.unsqueeze(-1) * child_local_probs
    creation_grid = next_creation_index[batch_idx, tree_idx].unsqueeze(-1) + slot_offsets

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


def _compute_cached_ancestor_mask(
    *,
    query_node_idx: torch.Tensor,
    cached_idx: torch.Tensor,
    node_parent_indices: torch.Tensor,
    query_depths: torch.Tensor,
) -> torch.Tensor:
    if cached_idx.numel() == 0:
        return torch.zeros_like(cached_idx, dtype=torch.bool)

    current = query_node_idx
    ancestor_mask = cached_idx.eq(current.unsqueeze(-1))
    max_steps = int(query_depths.max().item())
    for _ in range(max_steps):
        parent_idx = node_parent_indices.gather(2, current.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        valid_parent = parent_idx.ge(0)
        if not bool(valid_parent.any().item()):
            break
        current = torch.where(valid_parent, parent_idx, current)
        ancestor_mask = ancestor_mask | (valid_parent.unsqueeze(-1) & cached_idx.eq(current.unsqueeze(-1)))
    return ancestor_mask


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


def process_batch(batch, model, alpha, num_attend_tokens, max_trees, max_top_k):
    if max_top_k <= 0:
        raise ValueError("max_top_k must be positive.")
    if max_trees <= 0:
        raise ValueError("max_trees must be positive.")

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
    anchor_positions_per_row: list[list[int]] = []
    anchor_probs_per_row: list[list[float]] = []
    anchor_count_list: list[int] = []
    for row_idx in range(batch_size):
        row_anchor_positions, row_anchor_probs = select_anchor_positions(
            is_response=is_response[row_idx],
            valid_tokens=valid_tokens[row_idx],
            next_token_probs=next_token_probs[row_idx],
            alpha=alpha,
            max_trees=max_trees,
        )
        anchor_positions_per_row.append(row_anchor_positions)
        anchor_probs_per_row.append(row_anchor_probs)
        anchor_count_list.append(len(row_anchor_positions))

    anchor_counts = torch.tensor(anchor_count_list, dtype=torch.long, device=device)
    batch_max_trees = max(max(anchor_count_list, default=0), 1)

    anchor_positions = torch.zeros((batch_size, batch_max_trees), dtype=torch.long, device=device)
    anchor_probs = torch.zeros((batch_size, batch_max_trees), dtype=torch.float32, device=device)
    for row_idx, (row_positions, row_probs) in enumerate(zip(anchor_positions_per_row, anchor_probs_per_row, strict=True)):
        if not row_positions:
            continue
        count = len(row_positions)
        anchor_positions[row_idx, :count] = torch.tensor(row_positions, dtype=torch.long, device=device)
        anchor_probs[row_idx, :count] = torch.tensor(row_probs, dtype=torch.float32, device=device)

    tree_active = torch.arange(batch_max_trees, device=device).unsqueeze(0) < anchor_counts.unsqueeze(-1)
    max_nodes = 1 + (num_attend_tokens + 1) * max_top_k

    node_token_ids = torch.full((batch_size, batch_max_trees, max_nodes), IGNORE_IDX, dtype=torch.long, device=device)
    node_parent_indices = torch.full((batch_size, batch_max_trees, max_nodes), -1, dtype=torch.long, device=device)
    node_depths = torch.zeros((batch_size, batch_max_trees, max_nodes), dtype=torch.long, device=device)
    node_local_probs = torch.zeros((batch_size, batch_max_trees, max_nodes), dtype=torch.float32, device=device)
    node_path_probs = torch.zeros((batch_size, batch_max_trees, max_nodes), dtype=torch.float32, device=device)
    node_ranks = torch.zeros((batch_size, batch_max_trees, max_nodes), dtype=torch.long, device=device)
    node_main_pos = torch.full((batch_size, batch_max_trees, max_nodes), IGNORE_IDX, dtype=torch.long, device=device)
    node_is_main = torch.zeros((batch_size, batch_max_trees, max_nodes), dtype=torch.bool, device=device)
    node_valid = torch.zeros((batch_size, batch_max_trees, max_nodes), dtype=torch.bool, device=device)
    node_expanded = torch.zeros((batch_size, batch_max_trees, max_nodes), dtype=torch.bool, device=device)
    frontier = torch.zeros((batch_size, batch_max_trees, max_nodes), dtype=torch.bool, device=device)
    first_child = torch.full((batch_size, batch_max_trees, max_nodes), -1, dtype=torch.long, device=device)
    child_count = torch.zeros((batch_size, batch_max_trees, max_nodes), dtype=torch.long, device=device)
    node_creation_index = torch.full((batch_size, batch_max_trees, max_nodes), max_nodes * 2, dtype=torch.long, device=device)
    next_free_node_idx = tree_active.to(torch.long)
    next_creation_index = torch.zeros((batch_size, batch_max_trees), dtype=torch.long, device=device)
    expanded_count = torch.zeros((batch_size, batch_max_trees), dtype=torch.long, device=device)
    expansion_history = torch.full((batch_size, num_attend_tokens, batch_max_trees), -1, dtype=torch.long, device=device)

    node_valid[:, :, 0] = tree_active
    node_local_probs[:, :, 0] = tree_active.to(torch.float32)
    node_path_probs[:, :, 0] = tree_active.to(torch.float32)

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
        )

    fallback_token = input_ids[:, :1].expand(-1, batch_max_trees)
    eye_trees = torch.eye(batch_max_trees, dtype=torch.bool, device=device).unsqueeze(0)

    for step in range(num_attend_tokens):
        eligible = (
            tree_active.unsqueeze(-1)
            & frontier
            & ~node_expanded
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

        total_tree_keys = (step + 1) * batch_max_trees
        tree_key_valid = torch.zeros((batch_size, total_tree_keys), dtype=torch.bool, device=device)
        tree_can_attend = torch.zeros((batch_size, batch_max_trees, total_tree_keys), dtype=torch.bool, device=device)

        if step > 0:
            cached_idx = expansion_history[:, :step, :].permute(0, 2, 1)
            cached_valid = cached_idx.ge(0)
            cached_idx_clamped = cached_idx.clamp(min=0)
            cached_is_ancestor = _compute_cached_ancestor_mask(
                query_node_idx=chosen_node_idx,
                cached_idx=cached_idx_clamped,
                node_parent_indices=node_parent_indices,
                query_depths=query_depths,
            )
            prev_attend = (cached_valid & cached_is_ancestor).unsqueeze(-1) & eye_trees.unsqueeze(2)
            tree_key_valid[:, : step * batch_max_trees] = expansion_history[:, :step, :].permute(0, 2, 1).reshape(batch_size, -1).ge(0)
            tree_can_attend[:, :, : step * batch_max_trees] = prev_attend.reshape(batch_size, batch_max_trees, step * batch_max_trees)

        tree_key_valid[:, step * batch_max_trees : (step + 1) * batch_max_trees] = query_valid
        tree_can_attend[:, :, step * batch_max_trees : (step + 1) * batch_max_trees] = query_valid.unsqueeze(-1) & eye_trees
        flex_mask = _build_flex_attention_mask(
            anchor_positions=anchor_positions,
            query_valid=query_valid,
            tree_key_valid=tree_key_valid,
            tree_can_attend=tree_can_attend,
            valid_tokens=valid_tokens,
            context_len=seq_len,
        )

        cache_position = torch.arange(
            seq_len + step * batch_max_trees,
            seq_len + (step + 1) * batch_max_trees,
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


def initialize_hdf5(hf: h5py.File, attrs: dict[str, Any] | None = None) -> None:
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
    if attrs:
        for key, value in attrs.items():
            hf.attrs[key] = value


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


def build_parts_dir(output_path: Path) -> Path:
    return Path(f"{output_path}.parts")


def build_rank_part_path(output_path: Path, rank: int) -> Path:
    return build_parts_dir(output_path) / f"rank_{rank:05d}.h5"


def prepare_parts_dir(parts_dir: Path, ctx: DistributedContext) -> None:
    if not ctx.is_distributed:
        if parts_dir.exists():
            raise FileExistsError(
                f"Temporary Stage 2 parts directory already exists: {parts_dir}. "
                "Remove it or choose a different --output path before rerunning."
            )
        parts_dir.mkdir(parents=True, exist_ok=False)
        return

    status = torch.ones(1, device=ctx.device, dtype=torch.int32)
    if ctx.is_primary:
        try:
            if parts_dir.exists():
                raise FileExistsError(
                    f"Temporary Stage 2 parts directory already exists: {parts_dir}. "
                    "Remove it or choose a different --output path before rerunning."
                )
            parts_dir.mkdir(parents=True, exist_ok=False)
        except Exception:
            status.zero_()
    dist.broadcast(status, src=0)
    if int(status.item()) != 1:
        raise RuntimeError(f"Could not prepare temporary Stage 2 parts directory: {parts_dir}")


def collect_merge_manifest(part_paths: list[Path]) -> list[HDF5MergeEntry]:
    manifest: list[HDF5MergeEntry] = []
    for part_path in part_paths:
        with h5py.File(part_path, "r") as hf:
            record_idx = hf["record_idx"][:]
            main_offsets = hf["main_path_offsets"][:]
            seq_anchor_offsets = hf["sequence_anchor_offsets"][:]
            anchor_node_offsets = hf["anchor_node_offsets"][:]
            if main_offsets.shape[0] != record_idx.shape[0] + 1:
                raise ValueError(
                    f"Mismatched main_path_offsets and record_idx lengths in {part_path}: "
                    f"{main_offsets.shape[0]} vs {record_idx.shape[0]}"
                )
            if seq_anchor_offsets.shape[0] != record_idx.shape[0] + 1:
                raise ValueError(
                    f"Mismatched sequence_anchor_offsets and record_idx lengths in {part_path}: "
                    f"{seq_anchor_offsets.shape[0]} vs {record_idx.shape[0]}"
                )
            for seq_idx, seq_record_idx in enumerate(record_idx.tolist()):
                anchor_start = int(seq_anchor_offsets[seq_idx])
                anchor_end = int(seq_anchor_offsets[seq_idx + 1])
                manifest.append(
                    HDF5MergeEntry(
                        record_idx=int(seq_record_idx),
                        part_path=str(part_path),
                        seq_idx=seq_idx,
                        main_start=int(main_offsets[seq_idx]),
                        main_end=int(main_offsets[seq_idx + 1]),
                        anchor_start=anchor_start,
                        anchor_end=anchor_end,
                        node_start=int(anchor_node_offsets[anchor_start]),
                        node_end=int(anchor_node_offsets[anchor_end]),
                    )
                )
    manifest.sort(key=lambda entry: entry.record_idx)
    return manifest


def _load_sequence_from_hdf5(hf: h5py.File, entry: HDF5MergeEntry) -> GeneratedSequenceTree:
    anchors: list[GeneratedAnchorTree] = []
    anchor_node_offsets = hf["anchor_node_offsets"]
    for anchor_idx in range(entry.anchor_start, entry.anchor_end):
        node_start = int(anchor_node_offsets[anchor_idx])
        node_end = int(anchor_node_offsets[anchor_idx + 1])
        nodes: list[SequenceTreeNode] = []
        for node_idx in range(node_start, node_end):
            nodes.append(
                SequenceTreeNode(
                    token_id=int(hf["node_token_ids"][node_idx]),
                    parent_index=int(hf["node_parent_indices"][node_idx]),
                    depth=int(hf["node_depths"][node_idx]),
                    local_prob=float(hf["node_local_probs"][node_idx]),
                    path_prob=float(hf["node_path_probs"][node_idx]),
                    rank=int(hf["node_ranks"][node_idx]),
                    main_path_position=int(hf["node_main_path_positions"][node_idx]),
                    is_main_path=bool(hf["node_is_main_path"][node_idx]),
                    child_indices=[
                        int(hf["node_first_child"][node_idx]),
                        int(hf["node_child_count"][node_idx]),
                    ],
                )
            )
        anchors.append(
            GeneratedAnchorTree(
                anchor_main_path_position=int(hf["anchor_main_path_positions"][anchor_idx]),
                anchor_next_token_prob=float(hf["anchor_next_token_probs"][anchor_idx]),
                nodes=nodes,
            )
        )
    return GeneratedSequenceTree(
        record_idx=entry.record_idx,
        main_path_ids=[int(token_id) for token_id in hf["main_path_ids"][entry.main_start : entry.main_end]],
        response_start_position=int(hf["response_start_positions"][entry.seq_idx]),
        anchors=anchors,
    )


def merge_hdf5_parts(
    *,
    part_paths: list[Path],
    output_path: Path,
    log_fn,
) -> int:
    manifest = collect_merge_manifest(part_paths)
    sequence_buf: list[GeneratedSequenceTree] = []
    flush_every = 128

    with ExitStack() as stack:
        part_handles = {
            str(path): stack.enter_context(h5py.File(path, "r"))
            for path in part_paths
        }
        merged_attrs: dict[str, Any] = {}
        if part_paths:
            first_part = part_handles[str(part_paths[0])]
            merged_attrs = {str(key): first_part.attrs[key] for key in first_part.attrs.keys()}
        with h5py.File(output_path, "w") as hf:
            initialize_hdf5(hf, attrs=merged_attrs)
            written = 0
            for entry in manifest:
                sequence_buf.append(_load_sequence_from_hdf5(part_handles[entry.part_path], entry))
                if len(sequence_buf) >= flush_every:
                    flush_hdf5(hf, sequence_buf)
                    written += len(sequence_buf)
                    sequence_buf.clear()
                    log_fn(f"Merged sequences: {written}")

            if sequence_buf:
                flush_hdf5(hf, sequence_buf)
                written += len(sequence_buf)
                sequence_buf.clear()
    return len(manifest)


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
    parser.add_argument("--max-trees", type=int, default=512)
    parser.add_argument("--max-top-k", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()
    ctx = init_distributed_context(args.device)
    try:
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
        ).to(ctx.device)
        model.eval()

        if getattr(model.config, "_attn_implementation", None) != "flex_attention":
            raise ValueError("The loaded model is not configured for flex_attention.")

        dataloader = get_dataloader(
            args.input,
            tokenizer,
            args.seq_len,
            args.batch_size,
            ctx=ctx,
        )
        hdf5_attrs = {
            "tokenizer_name_or_path": tokenizer_name,
            "model_name_or_path": args.model,
            "alpha": float(args.alpha),
            "num_attend_tokens": int(args.num_attend_tokens),
            "max_trees": int(args.max_trees),
            "max_top_k": int(args.max_top_k),
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = output_path
        if ctx.is_distributed:
            parts_dir = build_parts_dir(output_path)
            prepare_parts_dir(parts_dir, ctx)
            part_path = build_rank_part_path(output_path, ctx.rank)
            ctx.log(f"Running Stage 2 with torchrun across {ctx.world_size} ranks.")

        with h5py.File(part_path, "w") as hf:
            initialize_hdf5(hf, attrs=hdf5_attrs)
            with torch.no_grad():
                for batch in tqdm.tqdm(dataloader):
                    sequences = process_batch(
                        batch=batch,
                        model=model,
                        alpha=args.alpha,
                        num_attend_tokens=args.num_attend_tokens,
                        max_trees=args.max_trees,
                        max_top_k=args.max_top_k,
                    )
                    flush_hdf5(hf, sequences)

        if ctx.is_distributed:
            ctx.barrier()
            if ctx.is_primary:
                part_paths = [build_rank_part_path(output_path, rank) for rank in range(ctx.world_size)]
                ctx.log(f"Merging {len(part_paths)} Stage 2 rank shards into {output_path}")
                total_sequences = merge_hdf5_parts(
                    part_paths=part_paths,
                    output_path=output_path,
                    log_fn=ctx.log,
                )
                shutil.rmtree(build_parts_dir(output_path))
                ctx.log(f"Done. Sequences: {total_sequences}")
        else:
            ctx.log(f"Done. Wrote Stage 2 output to {output_path}")
    finally:
        ctx.shutdown()


if __name__ == "__main__":
    main()
