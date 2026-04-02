"""Cleaned-up Stage 2 v2 sequence-tree generation.

This module is a local, standalone rewrite of the Stage 2 v2 generator used by
the cleaned-up path. It preserves the Stage 2 v2 HDF5 schema and tree
semantics, but intentionally narrows the implementation to:

- flex attention only
- single-process generation
- local JSONL or Hugging Face dataset input
- compile-friendly fixed-shape tree expansion
- StaticCache-backed verifier KV reuse
"""

from __future__ import annotations

import argparse
import inspect
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence

import h5py
import numpy as np
import torch

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache

if TYPE_CHECKING:
    from datasets import Dataset
else:
    Dataset = Any


IGNORE_IDX = -1

sys.modules.setdefault("stage2_v2", sys.modules[__name__])


@dataclass
class CompiledCallable:
    name: str
    eager_fn: Callable
    compiled_fn: Callable | None = None
    fallback_reason: str | None = None
    log_enabled: bool = True

    def __call__(self, *args, **kwargs):
        if self.compiled_fn is None:
            return self.eager_fn(*args, **kwargs)
        try:
            return self.compiled_fn(*args, **kwargs)
        except Exception as exc:
            self.compiled_fn = None
            self.fallback_reason = f"{type(exc).__name__}: {exc}"
            if self.log_enabled:
                print(
                    f"Stage 2 v2 compile fallback for {self.name}: {self.fallback_reason}",
                    flush=True,
                )
            return self.eager_fn(*args, **kwargs)


@dataclass
class Stage2V2Runtime:
    base_model_forward: CompiledCallable
    flex_mask_builder: CompiledCallable
    compile_enabled: bool = False
    compile_mode: str | None = None


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
    child_indices: list[int] = field(default_factory=list)
    expanded: bool = False


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


@dataclass
class AnchorTreeState:
    anchor_main_path_position: int
    anchor_next_token_prob: float
    main_path_ids: list[int]
    response_start_position: int
    nodes: list[SequenceTreeNode] = field(default_factory=list)
    frontier: list[tuple[float, int, int]] = field(default_factory=list)
    expanded_token_nodes: int = 0
    next_heap_tiebreak: int = 0


def _accumulate_profile(profile: dict[str, float] | None, key: str, start_time: float) -> None:
    if profile is not None:
        profile[key] = profile.get(key, 0.0) + (time.perf_counter() - start_time)


def iter_position_chunks(total_positions: int, chunk_size: int) -> Iterator[slice]:
    if total_positions <= 0:
        return
    if chunk_size <= 0 or chunk_size >= total_positions:
        yield slice(0, total_positions)
        return
    for start in range(0, total_positions, chunk_size):
        yield slice(start, min(start + chunk_size, total_positions))


def _extract_hidden_and_cache(model_outputs) -> tuple[torch.Tensor, object | None]:
    if hasattr(model_outputs, "last_hidden_state"):
        return model_outputs.last_hidden_state, getattr(model_outputs, "past_key_values", None)
    if isinstance(model_outputs, tuple):
        hidden_states = model_outputs[0]
        past_key_values = model_outputs[1] if len(model_outputs) > 1 else None
        return hidden_states, past_key_values
    raise TypeError(f"Unsupported base model output type: {type(model_outputs)!r}")


def _run_base_model_forward(
    forward_fn: Callable,
    *,
    profile: dict[str, float] | None,
    profile_key: str,
    **kwargs,
):
    start_time = time.perf_counter()
    outputs = forward_fn(**kwargs)
    _accumulate_profile(profile, profile_key, start_time)
    return outputs


def build_batch(
    examples: Sequence[tuple[list[int], list[int]]],
    pad_token_id: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    max_len = max(len(prompt_ids) + len(response_ids) for prompt_ids, response_ids in examples)
    max_len = math.ceil(max_len / 128) * 128
    batch_size = len(examples)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    is_response = torch.zeros((batch_size, max_len), dtype=torch.bool)
    document_mask = torch.full((batch_size, max_len), -1, dtype=torch.long)

    for row_idx, (prompt_ids, response_ids) in enumerate(examples):
        full_ids = prompt_ids + response_ids
        seq_len = len(full_ids)
        prompt_len = len(prompt_ids)
        input_ids[row_idx, :seq_len] = torch.tensor(full_ids, dtype=torch.long)
        is_response[row_idx, prompt_len:seq_len] = True
        document_mask[row_idx, :seq_len] = 0

    return {
        "input_ids": input_ids.to(device),
        "is_response": is_response.to(device),
        "document_mask": document_mask.to(device),
    }


def load_tokenized_records(
    tokenizer,
    max_len: int | None,
    *,
    data_dir: Path | None = None,
    hf_dataset: str | None = None,
    hf_config: str | None = None,
    hf_split: str = "train",
    prompt_column: str = "prompt",
    response_column: str = "response",
    sort_descending: bool,
    num_proc: int | None = None,
) -> Dataset:
    if load_dataset is None:
        raise RuntimeError(
            "Stage 2 v2 generation requires Hugging Face `datasets` for JSONL/HF dataset loading."
        )
    if (data_dir is None) == (hf_dataset is None):
        raise ValueError("Pass exactly one input source: --data-dir or --hf-dataset.")

    if data_dir is not None:
        files = sorted(data_dir.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No .jsonl files found in {data_dir}")
        dataset = load_dataset(
            "json",
            data_files=[str(path) for path in files],
            split="train",
        )
    else:
        dataset = load_dataset(hf_dataset, name=hf_config, split=hf_split)

    missing_columns = [
        column for column in (prompt_column, response_column)
        if column not in dataset.column_names
    ]
    if missing_columns:
        raise ValueError(
            f"Input dataset is missing required columns {missing_columns}; "
            f"available columns: {dataset.column_names}"
        )

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]] | list[int]]:
        prompt_ids = tokenizer(batch[prompt_column], add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(batch[response_column], add_special_tokens=False)["input_ids"]
        total_len = [len(prompt) + len(response) for prompt, response in zip(prompt_ids, response_ids)]
        return {
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "total_len": total_len,
        }

    dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Tokenizing prompt/response pairs",
    )

    def keep_batch(
        prompt_ids: list[list[int]],
        response_ids: list[list[int]],
        total_len: list[int],
    ) -> list[bool]:
        keep = [len(prompt) > 0 and len(response) > 0 for prompt, response in zip(prompt_ids, response_ids)]
        if max_len is not None:
            keep = [flag and length <= max_len for flag, length in zip(keep, total_len)]
        return keep

    dataset = dataset.filter(
        keep_batch,
        batched=True,
        input_columns=["prompt_ids", "response_ids", "total_len"],
        num_proc=num_proc,
        desc="Filtering tokenized prompt/response pairs",
    )
    return dataset.sort("total_len", reverse=sort_descending)


def add_stable_record_idx(records: Dataset) -> Dataset:
    def assign_record_idx(_batch, indices: list[int]) -> dict[str, list[int]]:
        return {"record_idx": indices}

    return records.map(
        assign_record_idx,
        batched=True,
        with_indices=True,
        desc="Assigning stable record indices",
    )


def compute_exact_token_rank(logits: torch.Tensor, token_id: int) -> int:
    token_id = int(token_id)
    target_logit = logits[token_id]
    vocab_ids = torch.arange(logits.shape[-1], device=logits.device)
    strictly_greater = logits > target_logit
    same_logit_smaller_id = (logits == target_logit) & (vocab_ids < token_id)
    return 1 + int((strictly_greater | same_logit_smaller_id).sum().item())


def _select_anchor_positions_for_sequence(
    *,
    is_response: torch.Tensor,
    valid_tokens: torch.Tensor,
    next_token_probs: torch.Tensor,
    alpha: float,
    max_anchors_per_sequence: int,
) -> tuple[list[int], list[float]]:
    if is_response.ndim != 1 or valid_tokens.ndim != 1 or next_token_probs.ndim != 1:
        raise ValueError("Expected 1D tensors for anchor selection.")
    if is_response.numel() < 2:
        return [], []

    candidate_positions = torch.arange(is_response.numel() - 1, device=is_response.device)
    candidate_mask = is_response[:-1] & valid_tokens[1:]
    candidate_probs = next_token_probs[1:]
    eligible_mask = candidate_mask & (candidate_probs <= float(alpha))
    if not bool(eligible_mask.any().item()):
        return [], []

    positions = candidate_positions[eligible_mask]
    probs = candidate_probs[eligible_mask]
    if max_anchors_per_sequence > 0 and positions.numel() > max_anchors_per_sequence:
        keep = torch.argsort(probs, stable=True)[:max_anchors_per_sequence]
        positions = positions.index_select(0, keep)
        probs = probs.index_select(0, keep)
    order = torch.argsort(positions)
    positions = positions.index_select(0, order)
    probs = probs.index_select(0, order)
    return [int(value) for value in positions.tolist()], [float(value) for value in probs.tolist()]


def _stable_descending_token_order(logits: torch.Tensor) -> torch.Tensor:
    return torch.argsort(logits, dim=-1, descending=True, stable=True)


def _score_hidden_states_with_candidates(
    hidden_states: torch.Tensor,
    lm_head: torch.nn.Module,
    *,
    logit_chunk_size: int,
    profile: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden_states.ndim != 2:
        raise ValueError("Expected hidden_states with shape (num_queries, hidden_size).")
    if hidden_states.shape[0] == 0:
        return (
            torch.empty((0, 0), dtype=torch.long, device=hidden_states.device),
            torch.empty((0, 0), dtype=torch.float32, device=hidden_states.device),
        )

    sorted_ids_rows: list[torch.Tensor] = []
    sorted_probs_rows: list[torch.Tensor] = []
    score_start = time.perf_counter()
    weight_dtype = getattr(lm_head, "weight", hidden_states).dtype
    for chunk in iter_position_chunks(hidden_states.shape[0], logit_chunk_size):
        logits = lm_head(hidden_states[chunk].to(weight_dtype)).float()
        log_denom = torch.logsumexp(logits, dim=-1, keepdim=True)
        probs = (logits - log_denom).exp()
        sorted_ids = _stable_descending_token_order(logits)
        sorted_probs = probs.gather(-1, sorted_ids)
        sorted_ids_rows.append(sorted_ids.to(torch.long))
        sorted_probs_rows.append(sorted_probs.to(torch.float32))
    _accumulate_profile(profile, "candidate_score_s", score_start)
    return torch.cat(sorted_ids_rows, dim=0), torch.cat(sorted_probs_rows, dim=0)


def _find_forced_candidate(
    sorted_token_ids: torch.Tensor,
    sorted_token_probs: torch.Tensor,
    forced_token_id: int,
) -> tuple[float, int]:
    matches = torch.nonzero(sorted_token_ids.eq(int(forced_token_id)), as_tuple=False)
    if matches.numel() == 0:
        raise ValueError(f"Forced token {forced_token_id} does not appear in the candidate list.")
    first_idx = int(matches[0].item())
    return float(sorted_token_probs[first_idx].item()), first_idx + 1


def _select_children_for_storage_tensor(
    *,
    sorted_token_ids: torch.Tensor,
    sorted_token_probs: torch.Tensor,
    coverage_alpha: float,
    forced_token_ids: torch.Tensor | None,
    forced_main_path_positions: torch.Tensor | None,
    forced_child_mask: torch.Tensor | None,
    max_children_per_parent: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized child selection that preserves existing exact semantics.

    Returns:
    - `token_ids`: `(num_queries, max_children_per_parent)`
    - `local_probs`: `(num_queries, max_children_per_parent)`
    - `ranks`: `(num_queries, max_children_per_parent)`
    - `is_main_path`: `(num_queries, max_children_per_parent)`
    - `main_path_positions`: `(num_queries, max_children_per_parent)`
    - `selected_count`: `(num_queries,)`
    """

    if max_children_per_parent <= 0:
        raise ValueError(
            f"max_children_per_parent must be positive, got {max_children_per_parent}."
        )
    if sorted_token_ids.ndim != 2 or sorted_token_probs.ndim != 2:
        raise ValueError("Expected 2D candidate tensors.")
    if sorted_token_ids.shape != sorted_token_probs.shape:
        raise ValueError("Candidate id/prob tensors must have the same shape.")

    num_queries, vocab_window = sorted_token_ids.shape
    device = sorted_token_ids.device
    cap = int(max_children_per_parent)

    token_out = torch.full((num_queries, cap), IGNORE_IDX, dtype=torch.long, device=device)
    prob_out = torch.zeros((num_queries, cap), dtype=torch.float32, device=device)
    rank_out = torch.zeros((num_queries, cap), dtype=torch.long, device=device)
    is_main_out = torch.zeros((num_queries, cap), dtype=torch.bool, device=device)
    main_pos_out = torch.full((num_queries, cap), IGNORE_IDX, dtype=torch.long, device=device)
    selected_count = torch.zeros((num_queries,), dtype=torch.long, device=device)

    if num_queries == 0 or vocab_window == 0:
        return token_out, prob_out, rank_out, is_main_out, main_pos_out, selected_count

    if forced_child_mask is None:
        forced_child_mask = torch.zeros((num_queries,), dtype=torch.bool, device=device)
    else:
        forced_child_mask = forced_child_mask.to(device=device, dtype=torch.bool)
    if forced_token_ids is None:
        forced_token_ids = torch.full((num_queries,), IGNORE_IDX, dtype=torch.long, device=device)
    else:
        forced_token_ids = forced_token_ids.to(device=device, dtype=torch.long)
    if forced_main_path_positions is None:
        forced_main_path_positions = torch.full((num_queries,), IGNORE_IDX, dtype=torch.long, device=device)
    else:
        forced_main_path_positions = forced_main_path_positions.to(device=device, dtype=torch.long)

    candidate_ranks = torch.arange(1, vocab_window + 1, dtype=torch.long, device=device).unsqueeze(0).expand(num_queries, -1)
    forced_match_mask = forced_child_mask.unsqueeze(-1) & sorted_token_ids.eq(forced_token_ids.unsqueeze(-1))
    if bool(forced_child_mask.any().item()):
        matches = forced_match_mask.any(dim=-1)
        if not bool(matches[forced_child_mask].all().item()):
            raise ValueError("Forced child token does not appear in the candidate list.")

    forced_rank_idx = forced_match_mask.to(torch.int64).argmax(dim=-1)
    forced_prob = sorted_token_probs.gather(1, forced_rank_idx.unsqueeze(-1)).squeeze(-1)
    forced_rank = forced_rank_idx + 1
    forced_count = forced_child_mask.to(torch.long)

    if bool(forced_child_mask.any().item()):
        forced_rows = torch.nonzero(forced_child_mask, as_tuple=False).squeeze(-1)
        token_out[forced_rows, 0] = forced_token_ids[forced_rows]
        prob_out[forced_rows, 0] = forced_prob[forced_rows].to(torch.float32)
        rank_out[forced_rows, 0] = forced_rank[forced_rows]
        is_main_out[forced_rows, 0] = True
        main_pos_out[forced_rows, 0] = forced_main_path_positions[forced_rows]

    non_forced_mask = ~forced_match_mask
    non_forced_prob = torch.where(non_forced_mask, sorted_token_probs, torch.zeros_like(sorted_token_probs))
    non_forced_seen = non_forced_mask.to(torch.long).cumsum(dim=-1)
    count_before = forced_count.unsqueeze(-1) + non_forced_seen - 1
    prob_before = forced_prob.unsqueeze(-1) * forced_child_mask.unsqueeze(-1).to(sorted_token_probs.dtype)
    prob_before = prob_before + non_forced_prob.cumsum(dim=-1) - non_forced_prob
    keep_non_forced = (
        non_forced_mask
        & (count_before < cap)
        & ((count_before == 0) | (prob_before < float(coverage_alpha)))
    )

    keep_slot = forced_count.unsqueeze(-1) + keep_non_forced.to(torch.long).cumsum(dim=-1) - 1
    keep_valid = keep_non_forced & (keep_slot >= 0) & (keep_slot < cap)
    if bool(keep_valid.any().item()):
        q_idx = torch.arange(num_queries, device=device).unsqueeze(-1).expand(-1, vocab_window)[keep_valid]
        slot_idx = keep_slot[keep_valid]
        token_out[q_idx, slot_idx] = sorted_token_ids[keep_valid]
        prob_out[q_idx, slot_idx] = sorted_token_probs[keep_valid].to(torch.float32)
        rank_out[q_idx, slot_idx] = candidate_ranks[keep_valid]

    selected_count = forced_count + keep_non_forced.to(torch.long).sum(dim=-1)
    fallback_mask = selected_count.eq(0)
    if bool(fallback_mask.any().item()):
        fallback_rows = torch.nonzero(fallback_mask, as_tuple=False).squeeze(-1)
        token_out[fallback_rows, 0] = sorted_token_ids[fallback_rows, 0]
        prob_out[fallback_rows, 0] = sorted_token_probs[fallback_rows, 0].to(torch.float32)
        rank_out[fallback_rows, 0] = 1
        selected_count[fallback_rows] = 1

    return token_out, prob_out, rank_out, is_main_out, main_pos_out, selected_count


def _empty_anchor_tree(
    *,
    anchor_main_path_position: int,
    anchor_next_token_prob: float,
    main_path_ids: Sequence[int],
    response_start_position: int,
) -> AnchorTreeState:
    return AnchorTreeState(
        anchor_main_path_position=int(anchor_main_path_position),
        anchor_next_token_prob=float(anchor_next_token_prob),
        main_path_ids=[int(token_id) for token_id in main_path_ids],
        response_start_position=int(response_start_position),
        nodes=[
            SequenceTreeNode(
                token_id=IGNORE_IDX,
                parent_index=-1,
                depth=0,
                local_prob=1.0,
                path_prob=1.0,
                rank=0,
                main_path_position=IGNORE_IDX,
                is_main_path=False,
            )
        ],
    )


def _expected_main_path_child(
    state: AnchorTreeState,
    parent_idx: int,
) -> tuple[int, int] | None:
    if parent_idx == 0:
        next_pos = state.anchor_main_path_position + 1
        if next_pos >= len(state.main_path_ids):
            return None
        return int(state.main_path_ids[next_pos]), next_pos

    parent = state.nodes[parent_idx]
    if not parent.is_main_path:
        return None
    next_pos = parent.main_path_position + 1
    if next_pos >= len(state.main_path_ids):
        return None
    return int(state.main_path_ids[next_pos]), next_pos


def _select_children_for_storage(
    *,
    sorted_token_ids: torch.Tensor,
    sorted_token_probs: torch.Tensor,
    coverage_alpha: float,
    forced_child: tuple[int, int] | None,
    max_children_per_parent: int,
) -> list[tuple[int, float, int, bool, int]]:
    forced_token_ids = None
    forced_main_positions = None
    forced_mask = None
    if forced_child is not None:
        forced_token_ids = torch.tensor([forced_child[0]], dtype=torch.long, device=sorted_token_ids.device)
        forced_main_positions = torch.tensor([forced_child[1]], dtype=torch.long, device=sorted_token_ids.device)
        forced_mask = torch.tensor([True], dtype=torch.bool, device=sorted_token_ids.device)

    token_ids, probs, ranks, is_main_path, main_positions, selected_count = _select_children_for_storage_tensor(
        sorted_token_ids=sorted_token_ids.unsqueeze(0),
        sorted_token_probs=sorted_token_probs.unsqueeze(0),
        coverage_alpha=coverage_alpha,
        forced_token_ids=forced_token_ids,
        forced_main_path_positions=forced_main_positions,
        forced_child_mask=forced_mask,
        max_children_per_parent=max_children_per_parent,
    )
    count = int(selected_count[0].item())
    return [
        (
            int(token_ids[0, idx].item()),
            float(probs[0, idx].item()),
            int(ranks[0, idx].item()),
            bool(is_main_path[0, idx].item()),
            int(main_positions[0, idx].item()),
        )
        for idx in range(count)
    ]


def _append_children_to_tree(
    state: AnchorTreeState,
    *,
    parent_idx: int,
    sorted_token_ids: torch.Tensor,
    sorted_token_probs: torch.Tensor,
    child_coverage_alpha: float,
    max_children_per_parent: int,
) -> None:
    forced_child = _expected_main_path_child(state, parent_idx)
    selected_children = _select_children_for_storage(
        sorted_token_ids=sorted_token_ids,
        sorted_token_probs=sorted_token_probs,
        coverage_alpha=child_coverage_alpha,
        forced_child=forced_child,
        max_children_per_parent=max_children_per_parent,
    )

    parent_path_prob = state.nodes[parent_idx].path_prob
    parent_depth = state.nodes[parent_idx].depth
    for token_id, local_prob, rank, is_main_path, main_path_position in selected_children:
        node_idx = len(state.nodes)
        state.nodes.append(
            SequenceTreeNode(
                token_id=int(token_id),
                parent_index=int(parent_idx),
                depth=int(parent_depth + 1),
                local_prob=float(local_prob),
                path_prob=float(parent_path_prob * local_prob),
                rank=int(rank),
                main_path_position=int(main_path_position),
                is_main_path=bool(is_main_path),
            )
        )
        state.nodes[parent_idx].child_indices.append(node_idx)
        import heapq

        heapq.heappush(
            state.frontier,
            (-float(parent_path_prob * local_prob), state.next_heap_tiebreak, node_idx),
        )
        state.next_heap_tiebreak += 1


def finalize_anchor_tree(state: AnchorTreeState) -> GeneratedAnchorTree:
    finalized_nodes: list[SequenceTreeNode] = []
    for node in state.nodes:
        first_child = node.child_indices[0] if node.child_indices else -1
        child_count = len(node.child_indices)
        finalized_nodes.append(
            SequenceTreeNode(
                token_id=node.token_id,
                parent_index=node.parent_index,
                depth=node.depth,
                local_prob=node.local_prob,
                path_prob=node.path_prob,
                rank=node.rank,
                main_path_position=node.main_path_position,
                is_main_path=node.is_main_path,
                child_indices=[first_child, child_count],
                expanded=node.expanded,
            )
        )
    return GeneratedAnchorTree(
        anchor_main_path_position=state.anchor_main_path_position,
        anchor_next_token_prob=state.anchor_next_token_prob,
        nodes=finalized_nodes,
    )


def build_anchor_tree_from_candidate_provider(
    *,
    anchor_main_path_position: int,
    anchor_next_token_prob: float,
    main_path_ids: Sequence[int],
    response_start_position: int,
    num_attend_tokens_per_anchor: int,
    child_coverage_alpha: float,
    root_sorted_token_ids: Sequence[int],
    root_sorted_token_probs: Sequence[float],
    candidate_provider: Callable[[SequenceTreeNode], tuple[Sequence[int], Sequence[float]]],
    max_children_per_parent: int = 8,
) -> GeneratedAnchorTree:
    import heapq

    state = _empty_anchor_tree(
        anchor_main_path_position=anchor_main_path_position,
        anchor_next_token_prob=anchor_next_token_prob,
        main_path_ids=main_path_ids,
        response_start_position=response_start_position,
    )
    _append_children_to_tree(
        state,
        parent_idx=0,
        sorted_token_ids=torch.as_tensor(root_sorted_token_ids, dtype=torch.long),
        sorted_token_probs=torch.as_tensor(root_sorted_token_probs, dtype=torch.float32),
        child_coverage_alpha=child_coverage_alpha,
        max_children_per_parent=max_children_per_parent,
    )

    while state.frontier and state.expanded_token_nodes < num_attend_tokens_per_anchor:
        _, _, node_idx = heapq.heappop(state.frontier)
        node = state.nodes[node_idx]
        if node.expanded:
            continue
        node.expanded = True
        state.expanded_token_nodes += 1
        child_token_ids, child_probs = candidate_provider(node)
        _append_children_to_tree(
            state,
            parent_idx=node_idx,
            sorted_token_ids=torch.as_tensor(child_token_ids, dtype=torch.long),
            sorted_token_probs=torch.as_tensor(child_probs, dtype=torch.float32),
            child_coverage_alpha=child_coverage_alpha,
            max_children_per_parent=max_children_per_parent,
        )
    return finalize_anchor_tree(state)


def _build_dynamic_flex_block_mask(
    *,
    query_anchor_positions: torch.Tensor,
    query_valid_mask: torch.Tensor,
    tree_can_attend: torch.Tensor,
    tree_key_valid_mask: torch.Tensor,
    document_mask: torch.Tensor,
    valid_tokens: torch.Tensor,
    ctx_len: int,
):
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError as exc:
        raise RuntimeError(
            "Flex attention is not available in this PyTorch build. "
            "Install a PyTorch build with torch.nn.attention.flex_attention."
        ) from exc

    batch_size, q_count = query_anchor_positions.shape
    total_tree_keys = tree_can_attend.shape[-1]
    ctx_clamp_max = max(ctx_len - 1, 0)
    tree_clamp_max = max(total_tree_keys - 1, 0)

    def mask_mod(b, h, q_idx, kv_idx):
        del h
        q_valid = query_valid_mask[b, q_idx]

        in_ctx = kv_idx < ctx_len
        ctx_idx = kv_idx.clamp(0, ctx_clamp_max)
        q_anchor = query_anchor_positions[b, q_idx]
        same_doc = document_mask[b, ctx_idx] == document_mask[b, q_anchor]
        causal_ctx = ctx_idx <= q_anchor
        ctx_ok = in_ctx & q_valid & valid_tokens[b, ctx_idx] & same_doc & causal_ctx

        tree_idx = (kv_idx - ctx_len).clamp(0, tree_clamp_max)
        tree_ok = (
            (~in_ctx)
            & q_valid
            & tree_key_valid_mask[b, tree_idx]
            & tree_can_attend[b, q_idx, tree_idx]
        )
        return ctx_ok | tree_ok

    return create_block_mask(
        mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=q_count,
        KV_LEN=ctx_len + total_tree_keys,
        device=query_anchor_positions.device,
        BLOCK_SIZE=128,
    )


def build_stage2_v2_runtime(
    model,
    *,
    compile_enabled: bool = False,
    compile_mode: str = "reduce-overhead",
    log_enabled: bool = True,
) -> Stage2V2Runtime:
    if getattr(getattr(model, "config", None), "_attn_implementation", None) != "flex_attention":
        raise ValueError("Cleaned-up Stage 2 v2 only supports flex_attention.")

    base_model_forward = CompiledCallable(
        name="stage2_v2_base_model_forward",
        eager_fn=model.base_model,
        log_enabled=log_enabled,
    )
    flex_mask_builder = CompiledCallable(
        name="stage2_v2_flex_block_mask_builder",
        eager_fn=_build_dynamic_flex_block_mask,
        log_enabled=log_enabled,
    )
    runtime = Stage2V2Runtime(
        base_model_forward=base_model_forward,
        flex_mask_builder=flex_mask_builder,
        compile_enabled=False,
        compile_mode=None,
    )

    if not compile_enabled:
        return runtime
    if not hasattr(torch, "compile"):
        if log_enabled:
            print("Stage 2 v2 compile requested, but torch.compile is unavailable; continuing eagerly.", flush=True)
        return runtime

    base_model_forward.compiled_fn = torch.compile(model.base_model, dynamic=True, mode=compile_mode)
    flex_mask_builder.compiled_fn = torch.compile(
        _build_dynamic_flex_block_mask,
        dynamic=True,
        mode=compile_mode,
    )
    runtime.compile_enabled = True
    runtime.compile_mode = compile_mode
    if log_enabled:
        print(f"Stage 2 v2 compile enabled for flex_attention (mode={compile_mode}).", flush=True)
    return runtime


def _compute_next_token_stats(
    *,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    valid_tokens: torch.Tensor,
    lm_head: torch.nn.Module,
    logit_chunk_size: int,
    profile: dict[str, float] | None = None,
) -> torch.Tensor:
    del valid_tokens
    batch_size, seq_len = input_ids.shape
    token_probs = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=input_ids.device)
    if seq_len == 0:
        return token_probs
    token_probs[:, 0] = 1.0
    score_start = time.perf_counter()
    weight_dtype = getattr(lm_head, "weight", hidden_states).dtype
    for pos_chunk in iter_position_chunks(seq_len, logit_chunk_size):
        start = pos_chunk.start
        stop = pos_chunk.stop
        next_len = max(0, min(stop, seq_len - 1) - start)
        if next_len <= 0:
            continue
        logits = lm_head(hidden_states[:, pos_chunk, :].to(weight_dtype)).float()
        next_token_ids = input_ids[:, start + 1 : start + 1 + next_len]
        next_logits = logits[:, :next_len].gather(-1, next_token_ids.unsqueeze(-1)).squeeze(-1)
        log_denom = torch.logsumexp(logits[:, :next_len], dim=-1)
        token_probs[:, start + 1 : start + 1 + next_len] = (next_logits - log_denom).exp().to(torch.float32)
    _accumulate_profile(profile, "initial_summary_s", score_start)
    return token_probs


def _ancestor_node_indices(state: AnchorTreeState, node_idx: int) -> set[int]:
    ancestors: set[int] = set()
    cur = state.nodes[node_idx].parent_index
    while cur > 0:
        ancestors.add(cur)
        cur = state.nodes[cur].parent_index
    return ancestors


def _dummy_anchor_position_for_row(
    *,
    valid_tokens_row: torch.Tensor,
    preferred_anchor_positions: Sequence[int],
) -> int:
    row_len = int(valid_tokens_row.numel())
    if row_len <= 0:
        return 0
    for position in preferred_anchor_positions:
        position = int(position)
        if 0 <= position < row_len:
            return position
    valid_positions = torch.nonzero(valid_tokens_row, as_tuple=False).squeeze(-1)
    if valid_positions.numel() > 0:
        return int(valid_positions[0].item())
    return 0


def _build_static_cache(model, *, batch_size: int, max_cache_len: int):
    signature = inspect.signature(StaticCache)
    first_param = next(model.parameters(), None)
    device = first_param.device if first_param is not None else torch.device("cpu")
    dtype = getattr(model, "dtype", None)
    if dtype is None:
        dtype = first_param.dtype if first_param is not None else torch.float32
    kwargs: dict[str, Any] = {
        "config": model.config,
        "max_cache_len": int(max_cache_len),
        "device": device,
        "dtype": dtype,
    }
    if "max_batch_size" in signature.parameters:
        kwargs["max_batch_size"] = int(batch_size)
    elif "batch_size" in signature.parameters:
        kwargs["batch_size"] = int(batch_size)
    else:
        raise TypeError("Unsupported StaticCache signature.")
    return StaticCache(**kwargs)


def _append_children_to_tensor_state(
    *,
    batch_indices: torch.Tensor,
    tree_indices: torch.Tensor,
    parent_indices: torch.Tensor,
    selected_token_ids: torch.Tensor,
    selected_local_probs: torch.Tensor,
    selected_ranks: torch.Tensor,
    selected_is_main_path: torch.Tensor,
    selected_main_path_positions: torch.Tensor,
    selected_count: torch.Tensor,
    node_token_ids: torch.Tensor,
    node_parent_indices: torch.Tensor,
    node_depths: torch.Tensor,
    node_local_probs: torch.Tensor,
    node_path_probs: torch.Tensor,
    node_ranks: torch.Tensor,
    node_main_path_positions: torch.Tensor,
    node_is_main_path: torch.Tensor,
    node_valid_mask: torch.Tensor,
    frontier_mask: torch.Tensor,
    first_child: torch.Tensor,
    child_count: torch.Tensor,
    node_creation_index: torch.Tensor,
    next_free_node_idx: torch.Tensor,
    next_creation_index: torch.Tensor,
    ancestor_self_mask: torch.Tensor,
) -> None:
    if batch_indices.numel() == 0:
        return

    device = batch_indices.device
    num_queries, cap = selected_token_ids.shape
    max_nodes_per_tree = node_token_ids.shape[-1]
    slot_offsets = torch.arange(cap, dtype=torch.long, device=device).unsqueeze(0)
    base_slots = next_free_node_idx[batch_indices, tree_indices]
    child_slots = base_slots.unsqueeze(-1) + slot_offsets
    valid_child_mask = slot_offsets < selected_count.unsqueeze(-1)

    if bool(valid_child_mask.any().item()):
        max_required = int(child_slots[valid_child_mask].max().item())
        if max_required >= max_nodes_per_tree:
            raise ValueError(
                "Stage 2 v2 tensor state is too small for the requested tree budget. "
                f"Need node index {max_required}, but max_nodes_per_tree={max_nodes_per_tree}."
            )

    parent_depth = node_depths[batch_indices, tree_indices, parent_indices]
    parent_path_prob = node_path_probs[batch_indices, tree_indices, parent_indices]
    parent_ancestor_rows = ancestor_self_mask[batch_indices, tree_indices, parent_indices]

    batch_grid = batch_indices.unsqueeze(-1).expand(-1, cap)
    tree_grid = tree_indices.unsqueeze(-1).expand(-1, cap)
    parent_grid = parent_indices.unsqueeze(-1).expand(-1, cap)
    depth_grid = (parent_depth + 1).unsqueeze(-1).expand(-1, cap)
    path_prob_grid = parent_path_prob.unsqueeze(-1) * selected_local_probs
    creation_grid = next_creation_index[batch_indices, tree_indices].unsqueeze(-1) + slot_offsets
    ancestor_row_grid = parent_ancestor_rows.unsqueeze(1).expand(-1, cap, -1)

    flat_batch = batch_grid[valid_child_mask]
    flat_tree = tree_grid[valid_child_mask]
    flat_parent = parent_grid[valid_child_mask]
    flat_slot = child_slots[valid_child_mask]

    node_token_ids[flat_batch, flat_tree, flat_slot] = selected_token_ids[valid_child_mask]
    node_parent_indices[flat_batch, flat_tree, flat_slot] = flat_parent
    node_depths[flat_batch, flat_tree, flat_slot] = depth_grid[valid_child_mask]
    node_local_probs[flat_batch, flat_tree, flat_slot] = selected_local_probs[valid_child_mask]
    node_path_probs[flat_batch, flat_tree, flat_slot] = path_prob_grid[valid_child_mask]
    node_ranks[flat_batch, flat_tree, flat_slot] = selected_ranks[valid_child_mask]
    node_main_path_positions[flat_batch, flat_tree, flat_slot] = selected_main_path_positions[valid_child_mask]
    node_is_main_path[flat_batch, flat_tree, flat_slot] = selected_is_main_path[valid_child_mask]
    node_valid_mask[flat_batch, flat_tree, flat_slot] = True
    frontier_mask[flat_batch, flat_tree, flat_slot] = True
    node_creation_index[flat_batch, flat_tree, flat_slot] = creation_grid[valid_child_mask]
    ancestor_self_mask[flat_batch, flat_tree, flat_slot] = ancestor_row_grid[valid_child_mask]
    ancestor_self_mask[flat_batch, flat_tree, flat_slot, flat_slot] = True

    parent_has_children = selected_count > 0
    if bool(parent_has_children.any().item()):
        first_child[
            batch_indices[parent_has_children],
            tree_indices[parent_has_children],
            parent_indices[parent_has_children],
        ] = base_slots[parent_has_children]
        child_count[
            batch_indices[parent_has_children],
            tree_indices[parent_has_children],
            parent_indices[parent_has_children],
        ] = selected_count[parent_has_children]

    next_free_node_idx[batch_indices, tree_indices] = base_slots + selected_count
    next_creation_index[batch_indices, tree_indices] = next_creation_index[batch_indices, tree_indices] + selected_count


def _materialize_generated_trees(
    *,
    tree_active: torch.Tensor,
    anchor_positions: torch.Tensor,
    anchor_probs: torch.Tensor,
    node_token_ids: torch.Tensor,
    node_parent_indices: torch.Tensor,
    node_depths: torch.Tensor,
    node_local_probs: torch.Tensor,
    node_path_probs: torch.Tensor,
    node_ranks: torch.Tensor,
    node_main_path_positions: torch.Tensor,
    node_is_main_path: torch.Tensor,
    first_child: torch.Tensor,
    child_count: torch.Tensor,
    node_expanded_mask: torch.Tensor,
    next_free_node_idx: torch.Tensor,
) -> list[list[GeneratedAnchorTree]]:
    batch_size, max_trees = tree_active.shape
    materialized: list[list[GeneratedAnchorTree]] = []
    for batch_idx in range(batch_size):
        row_anchors: list[GeneratedAnchorTree] = []
        for tree_idx in range(max_trees):
            if not bool(tree_active[batch_idx, tree_idx].item()):
                continue
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
                        main_path_position=int(node_main_path_positions[batch_idx, tree_idx, node_idx].item()),
                        is_main_path=bool(node_is_main_path[batch_idx, tree_idx, node_idx].item()),
                        child_indices=[
                            int(first_child[batch_idx, tree_idx, node_idx].item()),
                            int(child_count[batch_idx, tree_idx, node_idx].item()),
                        ],
                        expanded=bool(node_expanded_mask[batch_idx, tree_idx, node_idx].item()),
                    )
                )
            row_anchors.append(
                GeneratedAnchorTree(
                    anchor_main_path_position=int(anchor_positions[batch_idx, tree_idx].item()),
                    anchor_next_token_prob=float(anchor_probs[batch_idx, tree_idx].item()),
                    nodes=nodes,
                )
            )
        materialized.append(row_anchors)
    return materialized


def _generate_sequence_trees_with_verifier(
    *,
    model,
    runtime: Stage2V2Runtime | None,
    input_ids: torch.Tensor,
    valid_tokens: torch.Tensor,
    document_mask: torch.Tensor,
    main_path_ids: Sequence[int],
    response_start_position: int,
    anchor_positions: list[int],
    anchor_next_token_probs: list[float],
    hidden_states: torch.Tensor,
    kv_cache,
    logit_chunk_size: int,
    num_attend_tokens_per_anchor: int,
    child_coverage_alpha: float,
    max_children_per_node: int = 8,
    max_trees_per_row: int | None = None,
    profile: dict[str, float] | None = None,
) -> list[GeneratedAnchorTree]:
    return _generate_sequence_trees_with_verifier_batch(
        model=model,
        runtime=runtime,
        input_ids=input_ids,
        valid_tokens=valid_tokens,
        document_mask=document_mask,
        main_path_ids_per_row=[main_path_ids],
        response_start_positions=[response_start_position],
        anchor_positions_per_row=[anchor_positions],
        anchor_next_token_probs_per_row=[anchor_next_token_probs],
        hidden_states=hidden_states,
        kv_cache=kv_cache,
        logit_chunk_size=logit_chunk_size,
        num_attend_tokens_per_anchor=num_attend_tokens_per_anchor,
        child_coverage_alpha=child_coverage_alpha,
        max_children_per_node=max_children_per_node,
        max_trees_per_row=max_trees_per_row,
        profile=profile,
    )[0]


def _generate_sequence_trees_with_verifier_batch(
    *,
    model,
    runtime: Stage2V2Runtime | None,
    input_ids: torch.Tensor,
    valid_tokens: torch.Tensor,
    document_mask: torch.Tensor,
    main_path_ids_per_row: Sequence[Sequence[int]],
    response_start_positions: Sequence[int],
    anchor_positions_per_row: Sequence[list[int]],
    anchor_next_token_probs_per_row: Sequence[list[float]],
    hidden_states: torch.Tensor,
    kv_cache,
    logit_chunk_size: int,
    num_attend_tokens_per_anchor: int,
    child_coverage_alpha: float,
    max_children_per_node: int,
    max_trees_per_row: int | None = None,
    profile: dict[str, float] | None = None,
) -> list[list[GeneratedAnchorTree]]:
    batch_size = int(input_ids.shape[0])
    if not (
        len(main_path_ids_per_row)
        == len(response_start_positions)
        == len(anchor_positions_per_row)
        == len(anchor_next_token_probs_per_row)
        == batch_size
    ):
        raise ValueError("Per-row Stage 2 v2 metadata must match the input batch size.")

    device = input_ids.device
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise ValueError("Model must expose output embeddings via get_output_embeddings().")

    max_trees = max_trees_per_row
    if max_trees is None:
        max_trees = max((len(anchor_positions) for anchor_positions in anchor_positions_per_row), default=0)
    max_trees = max(int(max_trees), 1)
    max_nodes_per_tree = 1 + ((int(num_attend_tokens_per_anchor) + 1) * int(max_children_per_node))

    if kv_cache is None:
        max_cache_len = int(input_ids.shape[1]) + (max_trees * max(num_attend_tokens_per_anchor, 1))
        kv_cache = _build_static_cache(model, batch_size=batch_size, max_cache_len=max_cache_len)

    response_start_tensor = torch.as_tensor(response_start_positions, dtype=torch.long, device=device)
    tree_active = torch.zeros((batch_size, max_trees), dtype=torch.bool, device=device)
    anchor_positions = torch.zeros((batch_size, max_trees), dtype=torch.long, device=device)
    anchor_probs = torch.zeros((batch_size, max_trees), dtype=torch.float32, device=device)
    for row_idx, (main_path_ids_row, response_start_position, row_anchor_positions, row_anchor_probs) in enumerate(
        zip(
            main_path_ids_per_row,
            response_start_positions,
            anchor_positions_per_row,
            anchor_next_token_probs_per_row,
            strict=True,
        )
    ):
        del main_path_ids_row, response_start_position
        real_count = min(len(row_anchor_positions), max_trees)
        if real_count > 0:
            tree_active[row_idx, :real_count] = True
            anchor_positions[row_idx, :real_count] = torch.tensor(row_anchor_positions[:real_count], dtype=torch.long, device=device)
            anchor_probs[row_idx, :real_count] = torch.tensor(row_anchor_probs[:real_count], dtype=torch.float32, device=device)

    node_token_ids = torch.full((batch_size, max_trees, max_nodes_per_tree), IGNORE_IDX, dtype=torch.long, device=device)
    node_parent_indices = torch.full((batch_size, max_trees, max_nodes_per_tree), -1, dtype=torch.long, device=device)
    node_depths = torch.zeros((batch_size, max_trees, max_nodes_per_tree), dtype=torch.long, device=device)
    node_local_probs = torch.zeros((batch_size, max_trees, max_nodes_per_tree), dtype=torch.float32, device=device)
    node_path_probs = torch.zeros((batch_size, max_trees, max_nodes_per_tree), dtype=torch.float32, device=device)
    node_ranks = torch.zeros((batch_size, max_trees, max_nodes_per_tree), dtype=torch.long, device=device)
    node_main_path_positions = torch.full((batch_size, max_trees, max_nodes_per_tree), IGNORE_IDX, dtype=torch.long, device=device)
    node_is_main_path = torch.zeros((batch_size, max_trees, max_nodes_per_tree), dtype=torch.bool, device=device)
    node_valid_mask = torch.zeros((batch_size, max_trees, max_nodes_per_tree), dtype=torch.bool, device=device)
    node_expanded_mask = torch.zeros((batch_size, max_trees, max_nodes_per_tree), dtype=torch.bool, device=device)
    frontier_mask = torch.zeros((batch_size, max_trees, max_nodes_per_tree), dtype=torch.bool, device=device)
    first_child = torch.full((batch_size, max_trees, max_nodes_per_tree), -1, dtype=torch.long, device=device)
    child_count = torch.zeros((batch_size, max_trees, max_nodes_per_tree), dtype=torch.long, device=device)
    node_creation_index = torch.full((batch_size, max_trees, max_nodes_per_tree), max_nodes_per_tree * (num_attend_tokens_per_anchor + 2), dtype=torch.long, device=device)
    ancestor_self_mask = torch.zeros((batch_size, max_trees, max_nodes_per_tree, max_nodes_per_tree), dtype=torch.bool, device=device)
    next_free_node_idx = tree_active.to(torch.long)
    next_creation_index = torch.zeros((batch_size, max_trees), dtype=torch.long, device=device)
    expanded_token_counts = torch.zeros((batch_size, max_trees), dtype=torch.long, device=device)
    expansion_history = torch.full((batch_size, num_attend_tokens_per_anchor, max_trees), -1, dtype=torch.long, device=device)

    node_valid_mask[:, :, 0] = tree_active
    node_local_probs[:, :, 0] = tree_active.to(torch.float32)
    node_path_probs[:, :, 0] = tree_active.to(torch.float32)
    ancestor_self_mask[:, :, 0, 0] = tree_active

    root_batch_idx, root_tree_idx = torch.nonzero(tree_active, as_tuple=True)
    if root_batch_idx.numel() > 0:
        root_anchor_positions = anchor_positions[root_batch_idx, root_tree_idx]
        root_hidden_states = hidden_states[root_batch_idx, root_anchor_positions]
        root_sorted_ids, root_sorted_probs = _score_hidden_states_with_candidates(
            root_hidden_states,
            lm_head,
            logit_chunk_size=logit_chunk_size,
            profile=profile,
        )
        forced_main_positions = root_anchor_positions + 1
        root_selected = _select_children_for_storage_tensor(
            sorted_token_ids=root_sorted_ids,
            sorted_token_probs=root_sorted_probs,
            coverage_alpha=child_coverage_alpha,
            forced_token_ids=input_ids[root_batch_idx, forced_main_positions],
            forced_main_path_positions=forced_main_positions,
            forced_child_mask=torch.ones_like(root_batch_idx, dtype=torch.bool, device=device),
            max_children_per_parent=max_children_per_node,
        )
        _append_children_to_tensor_state(
            batch_indices=root_batch_idx,
            tree_indices=root_tree_idx,
            parent_indices=torch.zeros_like(root_batch_idx, dtype=torch.long, device=device),
            selected_token_ids=root_selected[0],
            selected_local_probs=root_selected[1],
            selected_ranks=root_selected[2],
            selected_is_main_path=root_selected[3],
            selected_main_path_positions=root_selected[4],
            selected_count=root_selected[5],
            node_token_ids=node_token_ids,
            node_parent_indices=node_parent_indices,
            node_depths=node_depths,
            node_local_probs=node_local_probs,
            node_path_probs=node_path_probs,
            node_ranks=node_ranks,
            node_main_path_positions=node_main_path_positions,
            node_is_main_path=node_is_main_path,
            node_valid_mask=node_valid_mask,
            frontier_mask=frontier_mask,
            first_child=first_child,
            child_count=child_count,
            node_creation_index=node_creation_index,
            next_free_node_idx=next_free_node_idx,
            next_creation_index=next_creation_index,
            ancestor_self_mask=ancestor_self_mask,
        )

    base_model_forward = runtime.base_model_forward if runtime is not None else model.base_model
    flex_mask_builder = runtime.flex_mask_builder if runtime is not None else _build_dynamic_flex_block_mask
    ctx_len = int(document_mask.shape[1])
    tree_slot_ids = torch.arange(max_trees, dtype=torch.long, device=device)
    current_self_mask = tree_slot_ids.view(1, max_trees, 1) == tree_slot_ids.view(1, 1, max_trees)
    fallback_anchor_positions = torch.where(
        tree_active.any(dim=-1),
        anchor_positions[:, 0],
        response_start_tensor,
    )
    fallback_token_ids = input_ids.gather(1, fallback_anchor_positions.unsqueeze(-1)).squeeze(-1)

    for step_idx in range(num_attend_tokens_per_anchor):
        total_tree_keys = (step_idx + 1) * max_trees
        eligible_frontier = tree_active.unsqueeze(-1) & frontier_mask & ~node_expanded_mask
        eligible_frontier = eligible_frontier & (expanded_token_counts < num_attend_tokens_per_anchor).unsqueeze(-1)
        query_valid_mask = eligible_frontier.any(dim=-1)
        if not bool(query_valid_mask.any().item()):
            break

        masked_path_prob = torch.where(
            eligible_frontier,
            node_path_probs,
            torch.full_like(node_path_probs, float("-inf")),
        )
        best_path_prob = masked_path_prob.max(dim=-1).values
        candidate_mask = eligible_frontier & node_path_probs.eq(best_path_prob.unsqueeze(-1))
        masked_creation_index = torch.where(
            candidate_mask,
            node_creation_index,
            torch.full_like(node_creation_index, node_creation_index.max().item() + 1),
        )
        chosen_node_idx = masked_creation_index.argmin(dim=-1)
        chosen_node_idx = torch.where(query_valid_mask, chosen_node_idx, torch.zeros_like(chosen_node_idx))

        expansion_history[:, step_idx, :] = torch.where(
            query_valid_mask,
            chosen_node_idx,
            torch.full_like(chosen_node_idx, -1),
        )

        gathered_token_ids = node_token_ids.gather(2, chosen_node_idx.unsqueeze(-1)).squeeze(-1)
        gathered_depths = node_depths.gather(2, chosen_node_idx.unsqueeze(-1)).squeeze(-1)
        query_ids = torch.where(query_valid_mask, gathered_token_ids, fallback_token_ids.unsqueeze(-1))
        query_positions = torch.where(
            query_valid_mask,
            anchor_positions + gathered_depths,
            fallback_anchor_positions.unsqueeze(-1),
        )
        query_anchor_positions = torch.where(
            query_valid_mask,
            anchor_positions,
            fallback_anchor_positions.unsqueeze(-1),
        )

        tree_key_valid_mask = torch.zeros((batch_size, total_tree_keys), dtype=torch.bool, device=device)
        tree_can_attend = torch.zeros((batch_size, max_trees, total_tree_keys), dtype=torch.bool, device=device)
        if step_idx > 0:
            cached_node_idx = expansion_history[:, :step_idx, :].reshape(batch_size, -1)
            cached_valid = cached_node_idx.ge(0)
            tree_key_valid_mask[:, : step_idx * max_trees] = cached_valid

            current_ancestor_rows = torch.take_along_dim(
                ancestor_self_mask,
                chosen_node_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, max_nodes_per_tree),
                dim=2,
            ).squeeze(2)
            flat_cached_idx = cached_node_idx.clamp(min=0)
            gathered_ancestors = current_ancestor_rows.reshape(batch_size * max_trees, max_nodes_per_tree).gather(
                1,
                flat_cached_idx.unsqueeze(1).expand(-1, max_trees, -1).reshape(batch_size * max_trees, -1),
            ).reshape(batch_size, max_trees, -1)
            cached_tree_ids = tree_slot_ids.unsqueeze(0).expand(step_idx, -1).reshape(-1)
            same_tree_mask = tree_slot_ids.view(1, max_trees, 1) == cached_tree_ids.view(1, 1, -1)
            tree_can_attend[:, :, : step_idx * max_trees] = (
                query_valid_mask.unsqueeze(-1)
                & cached_valid.unsqueeze(1)
                & same_tree_mask
                & gathered_ancestors
            )

        tree_key_valid_mask[:, step_idx * max_trees : (step_idx + 1) * max_trees] = query_valid_mask
        tree_can_attend[:, :, step_idx * max_trees : (step_idx + 1) * max_trees] = (
            query_valid_mask.unsqueeze(-1) & current_self_mask
        )

        mask_start = time.perf_counter()
        attention_mask = flex_mask_builder(
            query_anchor_positions=query_anchor_positions,
            query_valid_mask=query_valid_mask,
            tree_can_attend=tree_can_attend,
            tree_key_valid_mask=tree_key_valid_mask,
            document_mask=document_mask,
            valid_tokens=valid_tokens,
            ctx_len=ctx_len,
        )
        _accumulate_profile(profile, "mask_build_s", mask_start)

        cache_position = torch.arange(
            ctx_len + (step_idx * max_trees),
            ctx_len + ((step_idx + 1) * max_trees),
            device=device,
        )
        base_out = _run_base_model_forward(
            base_model_forward,
            profile=profile,
            profile_key="tree_forward_s",
            input_ids=query_ids,
            position_ids=query_positions,
            attention_mask=attention_mask,
            past_key_values=kv_cache,
            cache_position=cache_position,
            use_cache=True,
        )
        new_hidden_states, kv_cache = _extract_hidden_and_cache(base_out)
        real_hidden_states = new_hidden_states[query_valid_mask]
        query_batch_idx, query_tree_idx = torch.nonzero(query_valid_mask, as_tuple=True)
        parent_node_idx = chosen_node_idx[query_valid_mask]

        node_expanded_mask[query_batch_idx, query_tree_idx, parent_node_idx] = True
        frontier_mask[query_batch_idx, query_tree_idx, parent_node_idx] = False
        expanded_token_counts[query_batch_idx, query_tree_idx] = expanded_token_counts[query_batch_idx, query_tree_idx] + 1

        if real_hidden_states.shape[0] > 0:
            child_sorted_ids, child_sorted_probs = _score_hidden_states_with_candidates(
                real_hidden_states,
                lm_head,
                logit_chunk_size=logit_chunk_size,
                profile=profile,
            )

            parent_is_root = parent_node_idx.eq(0)
            parent_is_main_path = node_is_main_path[query_batch_idx, query_tree_idx, parent_node_idx]
            forced_main_positions = torch.where(
                parent_is_root,
                anchor_positions[query_batch_idx, query_tree_idx] + 1,
                node_main_path_positions[query_batch_idx, query_tree_idx, parent_node_idx] + 1,
            )
            forced_in_range = (forced_main_positions >= 0) & (forced_main_positions < ctx_len)
            clamped_forced_positions = forced_main_positions.clamp(min=0, max=max(ctx_len - 1, 0))
            forced_child_mask = (
                (parent_is_root | parent_is_main_path)
                & forced_in_range
                & valid_tokens[query_batch_idx, clamped_forced_positions]
            )
            forced_selected = _select_children_for_storage_tensor(
                sorted_token_ids=child_sorted_ids,
                sorted_token_probs=child_sorted_probs,
                coverage_alpha=child_coverage_alpha,
                forced_token_ids=input_ids[query_batch_idx, clamped_forced_positions],
                forced_main_path_positions=forced_main_positions,
                forced_child_mask=forced_child_mask,
                max_children_per_parent=max_children_per_node,
            )
            _append_children_to_tensor_state(
                batch_indices=query_batch_idx,
                tree_indices=query_tree_idx,
                parent_indices=parent_node_idx,
                selected_token_ids=forced_selected[0],
                selected_local_probs=forced_selected[1],
                selected_ranks=forced_selected[2],
                selected_is_main_path=forced_selected[3],
                selected_main_path_positions=forced_selected[4],
                selected_count=forced_selected[5],
                node_token_ids=node_token_ids,
                node_parent_indices=node_parent_indices,
                node_depths=node_depths,
                node_local_probs=node_local_probs,
                node_path_probs=node_path_probs,
                node_ranks=node_ranks,
                node_main_path_positions=node_main_path_positions,
                node_is_main_path=node_is_main_path,
                node_valid_mask=node_valid_mask,
                frontier_mask=frontier_mask,
                first_child=first_child,
                child_count=child_count,
                node_creation_index=node_creation_index,
                next_free_node_idx=next_free_node_idx,
                next_creation_index=next_creation_index,
                ancestor_self_mask=ancestor_self_mask,
            )

    return _materialize_generated_trees(
        tree_active=tree_active,
        anchor_positions=anchor_positions,
        anchor_probs=anchor_probs,
        node_token_ids=node_token_ids,
        node_parent_indices=node_parent_indices,
        node_depths=node_depths,
        node_local_probs=node_local_probs,
        node_path_probs=node_path_probs,
        node_ranks=node_ranks,
        node_main_path_positions=node_main_path_positions,
        node_is_main_path=node_is_main_path,
        first_child=first_child,
        child_count=child_count,
        node_expanded_mask=node_expanded_mask,
        next_free_node_idx=next_free_node_idx,
    )


def initialize_stage2_v2_hdf5(
    hf: h5py.File,
    *,
    prob_dtype: np.dtype,
    attrs: dict[str, Any],
) -> None:
    hf.create_dataset("main_path_ids", shape=(0,), maxshape=(None,), dtype="int32", chunks=(4096,), compression="lzf")
    hf.create_dataset("main_path_offsets", shape=(1,), maxshape=(None,), dtype="int64")
    hf["main_path_offsets"][0] = 0
    hf.create_dataset("response_start_positions", shape=(0,), maxshape=(None,), dtype="int32")
    hf.create_dataset("record_idx", shape=(0,), maxshape=(None,), dtype="int64")
    hf.create_dataset("sequence_anchor_offsets", shape=(1,), maxshape=(None,), dtype="int64")
    hf["sequence_anchor_offsets"][0] = 0

    hf.create_dataset(
        "anchor_main_path_positions",
        shape=(0,),
        maxshape=(None,),
        dtype="int32",
        chunks=(4096,),
        compression="lzf",
    )
    hf.create_dataset(
        "anchor_next_token_probs",
        shape=(0,),
        maxshape=(None,),
        dtype=prob_dtype,
        chunks=(4096,),
        compression="lzf",
    )
    hf.create_dataset("anchor_node_offsets", shape=(1,), maxshape=(None,), dtype="int64")
    hf["anchor_node_offsets"][0] = 0

    hf.create_dataset("node_token_ids", shape=(0,), maxshape=(None,), dtype="int32", chunks=(4096,), compression="lzf")
    hf.create_dataset(
        "node_parent_indices",
        shape=(0,),
        maxshape=(None,),
        dtype="int32",
        chunks=(4096,),
        compression="lzf",
    )
    hf.create_dataset("node_depths", shape=(0,), maxshape=(None,), dtype="int16", chunks=(4096,), compression="lzf")
    hf.create_dataset("node_local_probs", shape=(0,), maxshape=(None,), dtype=prob_dtype, chunks=(4096,), compression="lzf")
    hf.create_dataset("node_path_probs", shape=(0,), maxshape=(None,), dtype=prob_dtype, chunks=(4096,), compression="lzf")
    hf.create_dataset("node_ranks", shape=(0,), maxshape=(None,), dtype="int32", chunks=(4096,), compression="lzf")
    hf.create_dataset(
        "node_main_path_positions",
        shape=(0,),
        maxshape=(None,),
        dtype="int32",
        chunks=(4096,),
        compression="lzf",
    )
    hf.create_dataset("node_is_main_path", shape=(0,), maxshape=(None,), dtype="bool", chunks=(4096,), compression="lzf")
    hf.create_dataset("node_first_child", shape=(0,), maxshape=(None,), dtype="int32", chunks=(4096,), compression="lzf")
    hf.create_dataset("node_child_count", shape=(0,), maxshape=(None,), dtype="int32", chunks=(4096,), compression="lzf")

    for key, value in attrs.items():
        hf.attrs[key] = value


def flush_stage2_v2_hdf5(
    hf: h5py.File,
    sequence_buf: list[GeneratedSequenceTree],
    *,
    n_sequences_written: int,
    n_main_path_ids_written: int,
    n_anchors_written: int,
    n_nodes_written: int,
    prob_dtype: np.dtype,
) -> tuple[int, int, int, int]:
    if not sequence_buf:
        return n_sequences_written, n_main_path_ids_written, n_anchors_written, n_nodes_written

    ds_main_ids = hf["main_path_ids"]
    ds_main_offsets = hf["main_path_offsets"]
    ds_resp_start = hf["response_start_positions"]
    ds_record_idx = hf["record_idx"]
    ds_seq_anchor_offsets = hf["sequence_anchor_offsets"]
    ds_anchor_positions = hf["anchor_main_path_positions"]
    ds_anchor_probs = hf["anchor_next_token_probs"]
    ds_anchor_node_offsets = hf["anchor_node_offsets"]
    ds_node_token_ids = hf["node_token_ids"]
    ds_node_parent_indices = hf["node_parent_indices"]
    ds_node_depths = hf["node_depths"]
    ds_node_local_probs = hf["node_local_probs"]
    ds_node_path_probs = hf["node_path_probs"]
    ds_node_ranks = hf["node_ranks"]
    ds_node_main_path_positions = hf["node_main_path_positions"]
    ds_node_is_main_path = hf["node_is_main_path"]
    ds_node_first_child = hf["node_first_child"]
    ds_node_child_count = hf["node_child_count"]

    main_path_ids = np.concatenate(
        [np.asarray(sequence.main_path_ids, dtype=np.int32) for sequence in sequence_buf],
        axis=0,
    )
    anchor_positions = np.concatenate(
        [
            np.asarray([anchor.anchor_main_path_position for anchor in sequence.anchors], dtype=np.int32)
            for sequence in sequence_buf
        ]
        or [np.empty((0,), dtype=np.int32)],
        axis=0,
    )
    anchor_probs = np.concatenate(
        [
            np.asarray([anchor.anchor_next_token_prob for anchor in sequence.anchors], dtype=prob_dtype)
            for sequence in sequence_buf
        ]
        or [np.empty((0,), dtype=prob_dtype)],
        axis=0,
    )

    flat_nodes = [node for sequence in sequence_buf for anchor in sequence.anchors for node in anchor.nodes]
    node_token_ids = np.asarray([node.token_id for node in flat_nodes], dtype=np.int32)
    node_parent_indices = np.asarray([node.parent_index for node in flat_nodes], dtype=np.int32)
    node_depths = np.asarray([node.depth for node in flat_nodes], dtype=np.int16)
    node_local_probs = np.asarray([node.local_prob for node in flat_nodes], dtype=prob_dtype)
    node_path_probs = np.asarray([node.path_prob for node in flat_nodes], dtype=prob_dtype)
    node_ranks = np.asarray([node.rank for node in flat_nodes], dtype=np.int32)
    node_main_path_positions = np.asarray([node.main_path_position for node in flat_nodes], dtype=np.int32)
    node_is_main_path = np.asarray([node.is_main_path for node in flat_nodes], dtype=np.bool_)
    node_first_child = np.asarray(
        [node.child_indices[0] if node.child_indices else -1 for node in flat_nodes],
        dtype=np.int32,
    )
    node_child_count = np.asarray(
        [node.child_indices[1] if node.child_indices else 0 for node in flat_nodes],
        dtype=np.int32,
    )

    new_sequence_total = n_sequences_written + len(sequence_buf)
    new_main_total = n_main_path_ids_written + int(main_path_ids.shape[0])
    new_anchor_total = n_anchors_written + int(anchor_positions.shape[0])
    new_node_total = n_nodes_written + int(node_token_ids.shape[0])

    ds_record_idx.resize(new_sequence_total, axis=0)
    ds_record_idx[n_sequences_written:new_sequence_total] = np.asarray(
        [sequence.record_idx for sequence in sequence_buf],
        dtype=np.int64,
    )
    ds_resp_start.resize(new_sequence_total, axis=0)
    ds_resp_start[n_sequences_written:new_sequence_total] = np.asarray(
        [sequence.response_start_position for sequence in sequence_buf],
        dtype=np.int32,
    )

    ds_main_ids.resize(new_main_total, axis=0)
    ds_main_ids[n_main_path_ids_written:new_main_total] = main_path_ids
    main_offsets = [n_main_path_ids_written]
    for sequence in sequence_buf:
        main_offsets.append(main_offsets[-1] + len(sequence.main_path_ids))
    ds_main_offsets.resize(new_sequence_total + 1, axis=0)
    ds_main_offsets[n_sequences_written + 1 : new_sequence_total + 1] = np.asarray(main_offsets[1:], dtype=np.int64)

    ds_anchor_positions.resize(new_anchor_total, axis=0)
    ds_anchor_positions[n_anchors_written:new_anchor_total] = anchor_positions
    ds_anchor_probs.resize(new_anchor_total, axis=0)
    ds_anchor_probs[n_anchors_written:new_anchor_total] = anchor_probs
    seq_anchor_offsets = [n_anchors_written]
    for sequence in sequence_buf:
        seq_anchor_offsets.append(seq_anchor_offsets[-1] + len(sequence.anchors))
    ds_seq_anchor_offsets.resize(new_sequence_total + 1, axis=0)
    ds_seq_anchor_offsets[n_sequences_written + 1 : new_sequence_total + 1] = np.asarray(
        seq_anchor_offsets[1:],
        dtype=np.int64,
    )

    ds_node_token_ids.resize(new_node_total, axis=0)
    ds_node_parent_indices.resize(new_node_total, axis=0)
    ds_node_depths.resize(new_node_total, axis=0)
    ds_node_local_probs.resize(new_node_total, axis=0)
    ds_node_path_probs.resize(new_node_total, axis=0)
    ds_node_ranks.resize(new_node_total, axis=0)
    ds_node_main_path_positions.resize(new_node_total, axis=0)
    ds_node_is_main_path.resize(new_node_total, axis=0)
    ds_node_first_child.resize(new_node_total, axis=0)
    ds_node_child_count.resize(new_node_total, axis=0)

    ds_node_token_ids[n_nodes_written:new_node_total] = node_token_ids
    ds_node_parent_indices[n_nodes_written:new_node_total] = node_parent_indices
    ds_node_depths[n_nodes_written:new_node_total] = node_depths
    ds_node_local_probs[n_nodes_written:new_node_total] = node_local_probs
    ds_node_path_probs[n_nodes_written:new_node_total] = node_path_probs
    ds_node_ranks[n_nodes_written:new_node_total] = node_ranks
    ds_node_main_path_positions[n_nodes_written:new_node_total] = node_main_path_positions
    ds_node_is_main_path[n_nodes_written:new_node_total] = node_is_main_path
    ds_node_first_child[n_nodes_written:new_node_total] = node_first_child
    ds_node_child_count[n_nodes_written:new_node_total] = node_child_count

    anchor_node_offsets = [n_nodes_written]
    for sequence in sequence_buf:
        for anchor in sequence.anchors:
            anchor_node_offsets.append(anchor_node_offsets[-1] + len(anchor.nodes))
    ds_anchor_node_offsets.resize(new_anchor_total + 1, axis=0)
    ds_anchor_node_offsets[n_anchors_written + 1 : new_anchor_total + 1] = np.asarray(
        anchor_node_offsets[1:],
        dtype=np.int64,
    )

    sequence_buf.clear()
    return new_sequence_total, new_main_total, new_anchor_total, new_node_total


def _generate_sequences_for_pending_batch(
    *,
    pending: Sequence[tuple[int, list[int], list[int]]],
    model,
    runtime: Stage2V2Runtime | None,
    pad_token_id: int,
    device: torch.device,
    alpha: float,
    max_anchors_per_sequence: int,
    logit_chunk_size: int,
    num_attend_tokens_per_anchor: int,
    child_coverage_alpha: float,
    max_children_per_node: int,
    profile: dict[str, float] | None = None,
) -> list[GeneratedSequenceTree]:
    if not pending:
        return []

    build_batch_start = time.perf_counter()
    batch = build_batch(
        [(prompt_ids, response_ids) for _, prompt_ids, response_ids in pending],
        pad_token_id,
        device,
    )
    _accumulate_profile(profile, "build_batch_s", build_batch_start)

    input_ids = batch["input_ids"]
    valid_tokens = batch["document_mask"] >= 0
    is_response = batch["is_response"]
    document_mask = batch["document_mask"]

    if getattr(getattr(model, "config", None), "_attn_implementation", None) != "flex_attention":
        raise ValueError("Cleaned-up Stage 2 v2 only supports flex_attention.")

    base_model_forward = runtime.base_model_forward if runtime is not None else model.base_model
    static_cache = _build_static_cache(
        model,
        batch_size=int(input_ids.shape[0]),
        max_cache_len=int(input_ids.shape[1]) + (max(max_anchors_per_sequence, 1) * max(num_attend_tokens_per_anchor, 1)),
    )
    cache_position = torch.arange(input_ids.shape[1], device=device)
    base_out = _run_base_model_forward(
        base_model_forward,
        profile=profile,
        profile_key="initial_forward_s",
        input_ids=input_ids,
        attention_mask=valid_tokens.long(),
        past_key_values=static_cache,
        cache_position=cache_position,
        use_cache=True,
    )
    hidden_states, kv_cache = _extract_hidden_and_cache(base_out)
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise ValueError("Model must expose output embeddings via get_output_embeddings().")
    next_token_probs = _compute_next_token_stats(
        hidden_states=hidden_states,
        input_ids=input_ids,
        valid_tokens=valid_tokens,
        lm_head=lm_head,
        logit_chunk_size=logit_chunk_size,
        profile=profile,
    )

    main_path_ids_per_row: list[list[int]] = []
    response_start_positions: list[int] = []
    anchor_positions_per_row: list[list[int]] = []
    anchor_probs_per_row: list[list[float]] = []
    for row_idx, (_, prompt_ids, response_ids) in enumerate(pending):
        response_start_position = len(prompt_ids)
        main_path_ids = [int(token_id) for token_id in (prompt_ids + response_ids)]
        anchor_positions, anchor_probs = _select_anchor_positions_for_sequence(
            is_response=is_response[row_idx],
            valid_tokens=valid_tokens[row_idx],
            next_token_probs=next_token_probs[row_idx],
            alpha=alpha,
            max_anchors_per_sequence=max_anchors_per_sequence,
        )
        main_path_ids_per_row.append(main_path_ids)
        response_start_positions.append(response_start_position)
        anchor_positions_per_row.append(anchor_positions)
        anchor_probs_per_row.append(anchor_probs)

    anchors_per_row = _generate_sequence_trees_with_verifier_batch(
        model=model,
        runtime=runtime,
        input_ids=input_ids,
        valid_tokens=valid_tokens,
        document_mask=document_mask,
        main_path_ids_per_row=main_path_ids_per_row,
        response_start_positions=response_start_positions,
        anchor_positions_per_row=anchor_positions_per_row,
        anchor_next_token_probs_per_row=anchor_probs_per_row,
        hidden_states=hidden_states,
        kv_cache=kv_cache,
        logit_chunk_size=logit_chunk_size,
        num_attend_tokens_per_anchor=num_attend_tokens_per_anchor,
        child_coverage_alpha=child_coverage_alpha,
        max_children_per_node=max_children_per_node,
        max_trees_per_row=max_anchors_per_sequence,
        profile=profile,
    )

    sequences: list[GeneratedSequenceTree] = []
    for (record_idx, _, _), main_path_ids, response_start_position, anchors in zip(
        pending,
        main_path_ids_per_row,
        response_start_positions,
        anchors_per_row,
        strict=True,
    ):
        sequences.append(
            GeneratedSequenceTree(
                record_idx=int(record_idx),
                main_path_ids=main_path_ids,
                response_start_position=int(response_start_position),
                anchors=anchors,
            )
        )
    return sequences


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cleaned-up Stage 2 v2: generate dynamic sequence-tree training data.")
    parser.add_argument("--model", required=True, help="HF model name or local path")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--data-dir", help="Directory of Stage 1 JSONL shards")
    input_group.add_argument("--hf-dataset", help="HF dataset id containing prompt/response rows")
    parser.add_argument("--hf-config", default=None, help="Optional HF dataset config name")
    parser.add_argument("--hf-split", default="train", help="HF dataset split to load")
    parser.add_argument("--prompt-column", default="prompt", help="Dataset column containing prompt text")
    parser.add_argument("--response-column", default="response", help="Dataset column containing response text")
    parser.add_argument("--output", required=True, help="Output HDF5 path")
    parser.add_argument("--alpha", type=float, default=0.9, help="Anchor-selection threshold on p(x_{t+1}|x_{1:t}).")
    parser.add_argument("--max-anchors-per-sequence", type=int, default=512, help="Maximum anchors kept per sequence.")
    parser.add_argument(
        "--num-attend-tokens-per-anchor",
        type=int,
        default=16,
        help="Maximum non-root token nodes to expand per anchor.",
    )
    parser.add_argument(
        "--child-coverage-alpha",
        type=float,
        default=0.95,
        help="Keep children until cumulative local probability mass reaches this threshold.",
    )
    parser.add_argument(
        "--max-children-per-node",
        type=int,
        default=8,
        help="Maximum children stored for any parent, including the forced main-path child.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model load dtype.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Records per prefill batch.")
    parser.add_argument("--max-sequences", type=int, default=None, help="Optional cap on processed sequences.")
    parser.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="Optional max prompt+response length; longer rows are skipped.",
    )
    parser.add_argument(
        "--logit-chunk-size",
        type=int,
        default=0,
        help="Project at most this many positions through the LM head at once; <= 0 disables chunking.",
    )
    parser.add_argument("--compile", action="store_true", help="Compile supported flex-attention helpers.")
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
        help="torch.compile mode for supported Stage 2 v2 helpers.",
    )
    parser.add_argument(
        "--dataset-num-proc",
        type=int,
        default=None,
        help="Optional worker count for dataset tokenization/filtering.",
    )
    parser.add_argument(
        "--profile-every",
        type=int,
        default=0,
        help="Print lightweight timing stats every N processed batches; 0 disables profiling logs.",
    )
    parser.add_argument(
        "--store-probs-fp16",
        action="store_true",
        help="Store probability tensors as float16 on disk instead of float32.",
    )
    return parser


@torch.inference_mode()
def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]
    disk_prob_dtype = np.float16 if args.store_probs_fp16 else np.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        attn_implementation="flex_attention",
    )
    if tokenizer.pad_token_id is not None and getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "resize_token_embeddings") and len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    if getattr(getattr(model, "config", None), "_attn_implementation", None) != "flex_attention":
        raise ValueError("Cleaned-up Stage 2 v2 only supports flex_attention.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    runtime = build_stage2_v2_runtime(
        model,
        compile_enabled=args.compile,
        compile_mode=args.compile_mode,
        log_enabled=True,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attrs = {
        "format_version": "stage2_v2",
        "attn_implementation": "flex_attention",
        "alpha": float(args.alpha),
        "max_anchors_per_sequence": int(args.max_anchors_per_sequence),
        "num_attend_tokens_per_anchor": int(args.num_attend_tokens_per_anchor),
        "child_coverage_alpha": float(args.child_coverage_alpha),
        "max_children_per_node": int(args.max_children_per_node),
        "model_name_or_path": args.model,
        "tokenizer_name_or_path": args.model,
    }

    dataset_prep_start = time.perf_counter()
    records = load_tokenized_records(
        tokenizer,
        args.max_len,
        data_dir=Path(args.data_dir) if args.data_dir is not None else None,
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        hf_split=args.hf_split,
        prompt_column=args.prompt_column,
        response_column=args.response_column,
        sort_descending=True,
        num_proc=args.dataset_num_proc,
    )
    if args.max_sequences is not None:
        records = records.select(range(min(args.max_sequences, len(records))))
    records = add_stable_record_idx(records)
    dataset_prep_s = time.perf_counter() - dataset_prep_start

    sequence_buf: list[GeneratedSequenceTree] = []
    flush_every = 128
    batch_count = 0
    timings: dict[str, float] = {"dataset_prep_s": dataset_prep_s}

    with h5py.File(output_path, "w") as hf:
        initialize_stage2_v2_hdf5(hf, prob_dtype=disk_prob_dtype, attrs=attrs)
        n_sequences_written = 0
        n_main_written = 0
        n_anchor_written = 0
        n_node_written = 0

        pending: list[tuple[int, list[int], list[int]]] = []
        for record in records:
            pending.append((int(record["record_idx"]), record["prompt_ids"], record["response_ids"]))
            if len(pending) < args.batch_size:
                continue

            sequence_buf.extend(
                _generate_sequences_for_pending_batch(
                    pending=pending,
                    model=model,
                    runtime=runtime,
                    pad_token_id=tokenizer.pad_token_id,
                    device=device,
                    alpha=args.alpha,
                    max_anchors_per_sequence=args.max_anchors_per_sequence,
                    logit_chunk_size=args.logit_chunk_size,
                    num_attend_tokens_per_anchor=args.num_attend_tokens_per_anchor,
                    child_coverage_alpha=args.child_coverage_alpha,
                    max_children_per_node=args.max_children_per_node,
                    profile=timings,
                )
            )
            pending.clear()
            batch_count += 1

            if len(sequence_buf) >= flush_every:
                flush_start = time.perf_counter()
                n_sequences_written, n_main_written, n_anchor_written, n_node_written = flush_stage2_v2_hdf5(
                    hf,
                    sequence_buf,
                    n_sequences_written=n_sequences_written,
                    n_main_path_ids_written=n_main_written,
                    n_anchors_written=n_anchor_written,
                    n_nodes_written=n_node_written,
                    prob_dtype=disk_prob_dtype,
                )
                _accumulate_profile(timings, "flush_s", flush_start)
                print(
                    f"Sequences: {n_sequences_written} "
                    f"main_path_tokens: {n_main_written} anchors: {n_anchor_written} nodes: {n_node_written}",
                    flush=True,
                )

            if args.profile_every > 0 and batch_count % args.profile_every == 0:
                print(
                    f"[profile] batches={batch_count} "
                    f"dataset_prep={timings.get('dataset_prep_s', 0.0):.3f}s "
                    f"build_batch={timings.get('build_batch_s', 0.0):.3f}s "
                    f"initial_forward={timings.get('initial_forward_s', 0.0):.3f}s "
                    f"initial_summary={timings.get('initial_summary_s', 0.0):.3f}s "
                    f"candidate_score={timings.get('candidate_score_s', 0.0):.3f}s "
                    f"mask_build={timings.get('mask_build_s', 0.0):.3f}s "
                    f"tree_forward={timings.get('tree_forward_s', 0.0):.3f}s "
                    f"flush={timings.get('flush_s', 0.0):.3f}s",
                    flush=True,
                )
                timings.clear()

        if pending:
            sequence_buf.extend(
                _generate_sequences_for_pending_batch(
                    pending=pending,
                    model=model,
                    runtime=runtime,
                    pad_token_id=tokenizer.pad_token_id,
                    device=device,
                    alpha=args.alpha,
                    max_anchors_per_sequence=args.max_anchors_per_sequence,
                    logit_chunk_size=args.logit_chunk_size,
                    num_attend_tokens_per_anchor=args.num_attend_tokens_per_anchor,
                    child_coverage_alpha=args.child_coverage_alpha,
                    max_children_per_node=args.max_children_per_node,
                    profile=timings,
                )
            )

        n_sequences_written, n_main_written, n_anchor_written, n_node_written = flush_stage2_v2_hdf5(
            hf,
            sequence_buf,
            n_sequences_written=n_sequences_written,
            n_main_path_ids_written=n_main_written,
            n_anchors_written=n_anchor_written,
            n_nodes_written=n_node_written,
            prob_dtype=disk_prob_dtype,
        )

    print(
        f"Done. Sequences: {n_sequences_written} "
        f"main_path_tokens: {n_main_written} anchors: {n_anchor_written} nodes: {n_node_written}",
        flush=True,
    )


if __name__ == "__main__":
    main()
