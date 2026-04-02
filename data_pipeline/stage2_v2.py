"""Stage 2 v2 sequence-tree generation.

Overview
========
This module builds a richer Stage 2 dataset than the original fixed-grid
subtree format.

The core idea is:

1. Run the verifier once on the full prompt+response main path.
2. Identify response positions whose true next token has low verifier
   probability. These positions become anchors.
3. For each anchor, build a dynamic tree rooted at that anchor position:
   - the synthetic root corresponds to "continue from main-path position t"
   - the true next token on the main path is always stored
   - additional children are stored in descending probability order until the
     retained local child mass reaches ``child_coverage_alpha`` or
     ``max_children_per_node`` children have been written
   - the next node to expand is chosen by highest cumulative ``path_prob``,
     i.e. the probability of reaching that node by a random walk from the
     anchor root
   - expansion stops once ``num_attend_tokens_per_anchor`` token nodes have
     been expanded
4. Persist the result as a sequence-tree dataset with explicit sequence,
   anchor, and node tables. Each stored node records its token id, local
   probability, cumulative path probability, exact verifier rank, main-path
   membership, and parent/child structure.

The verifier-backed expansion path uses flex attention when available. Each
tree query attends only to:
  - valid main-path context up to the anchor position
  - ancestor nodes within the same anchor tree

This makes the stored branching structure match the intended tree semantics:
siblings and unrelated branches do not leak information into one another.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import shutil
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from .stage2 import (
    CompiledCallable,
    DistributedContext,
    add_stable_record_idx,
    build_batch,
    build_parts_dir,
    build_rank_part_path,
    init_distributed_context,
    iter_position_chunks,
    load_tokenized_records,
    prepare_parts_dir,
    shard_records_for_rank,
)

if TYPE_CHECKING:
    from datasets import Dataset
else:
    Dataset = Any

IGNORE_IDX = -1


@dataclass
class Stage2V2Runtime:
    """Runtime wrappers for the verifier forward path and flex mask builder.

    The wrappers optionally hold ``torch.compile``-produced callables while
    preserving eager fallbacks when compilation or runtime execution fails.
    """

    base_model_forward: CompiledCallable
    flex_mask_builder: CompiledCallable
    compile_enabled: bool = False
    compile_mode: str | None = None


@dataclass(frozen=True)
class Stage2V2MergeEntry:
    record_idx: int
    part_path: str
    seq_idx: int
    main_start: int
    main_end: int
    anchor_start: int
    anchor_end: int
    node_start: int
    node_end: int


@dataclass
class SequenceTreeNode:
    """One stored node inside an anchor-local sequence tree.

    ``local_prob`` is the verifier probability of this token given its parent
    context. ``path_prob`` is the product of local probabilities from the
    synthetic anchor root to this node and is used to decide which frontier node
    should be expanded next.
    """

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
    """Final materialized tree for a single anchor position."""

    anchor_main_path_position: int
    anchor_next_token_prob: float
    nodes: list[SequenceTreeNode]


@dataclass
class GeneratedSequenceTree:
    """One fully processed prompt+response sequence plus all of its anchors."""

    record_idx: int
    main_path_ids: list[int]
    response_start_position: int
    anchors: list[GeneratedAnchorTree]


@dataclass
class AnchorTreeState:
    """Mutable state used while growing one anchor-local tree."""

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


def _empty_anchor_tree(
    *,
    anchor_main_path_position: int,
    anchor_next_token_prob: float,
    main_path_ids: Sequence[int],
    response_start_position: int,
) -> AnchorTreeState:
    """Create the mutable tree state for one anchor.

    The tree starts with a synthetic root node whose token id is ``IGNORE_IDX``.
    That root represents the decision point at anchor position ``t`` before any
    continuation token has been emitted.
    """

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


def compute_exact_token_rank(logits: torch.Tensor, token_id: int) -> int:
    """Return the exact 1-based rank of ``token_id`` under ``logits``.

    Ranks are deterministic even under equal logits: ties are broken by token id
    ascending, so a smaller token id wins the tie.
    """

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
    """Select hard response positions to use as anchors.

    Anchors are positions ``t`` on the response main path such that the verifier
    assigns low probability to the true next token ``x_(t+1)``. Only response
    positions are eligible. If more than ``max_anchors_per_sequence`` positions
    satisfy the threshold, the lowest-probability anchors are kept and then
    sorted back into sequence order.
    """

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
    """Project hidden states through the LM head and return sorted candidates.

    The result is a pair ``(sorted_token_ids, sorted_token_probs)`` where each
    row is ordered from most likely to least likely according to the verifier.
    """

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
    for chunk in iter_position_chunks(hidden_states.shape[0], logit_chunk_size):
        logits = lm_head(hidden_states[chunk].to(getattr(lm_head, "weight", hidden_states).dtype)).float()
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
    """Locate a forced child token inside a sorted candidate list.

    This is used to guarantee that the true main-path continuation is stored
    even when the tree is otherwise built from alternative high-probability
    continuations.
    """

    matches = torch.nonzero(sorted_token_ids.eq(int(forced_token_id)), as_tuple=False)
    if matches.numel() == 0:
        raise ValueError(f"Forced token {forced_token_id} does not appear in the candidate list.")
    first_idx = int(matches[0].item())
    return float(sorted_token_probs[first_idx].item()), first_idx + 1


def _expected_main_path_child(
    state: AnchorTreeState,
    parent_idx: int,
) -> tuple[int, int] | None:
    """Return the forced main-path child for a parent, if one exists.

    The synthetic root forces ``x_(t+1)``. Any node already marked as part of
    the main path forces the next token along that path as well. Non-main-path
    nodes do not have a forced child.
    """

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
    """Choose which children of a parent should be written to the tree.

    Selection happens in two stages:

    1. If ``forced_child`` exists, insert it first and mark it as a main-path
       node.
    2. Walk the verifier candidates in descending probability order and keep
       adding unseen children until the retained local child mass reaches
       ``coverage_alpha`` or ``max_children_per_parent`` children have been
       stored.

    The returned tuples contain:
      ``(token_id, local_prob, rank, is_main_path, main_path_position)``.
    """

    if max_children_per_parent <= 0:
        raise ValueError(
            f"max_children_per_parent must be positive, got {max_children_per_parent}."
        )

    selected: list[tuple[int, float, int, bool, int]] = []
    selected_ids: set[int] = set()
    cumulative_prob = 0.0

    if forced_child is not None:
        forced_token_id, forced_main_path_position = forced_child
        forced_prob, forced_rank = _find_forced_candidate(
            sorted_token_ids,
            sorted_token_probs,
            forced_token_id,
        )
        selected.append(
            (
                int(forced_token_id),
                float(forced_prob),
                int(forced_rank),
                True,
                int(forced_main_path_position),
            )
        )
        selected_ids.add(int(forced_token_id))
        cumulative_prob += float(forced_prob)
        if len(selected) >= max_children_per_parent:
            return selected

    for rank_idx, token_id in enumerate(sorted_token_ids.tolist(), start=1):
        token_id = int(token_id)
        if token_id in selected_ids:
            continue
        prob = float(sorted_token_probs[rank_idx - 1].item())
        selected.append((token_id, prob, rank_idx, False, IGNORE_IDX))
        selected_ids.add(token_id)
        cumulative_prob += prob
        if len(selected) >= max_children_per_parent or cumulative_prob >= float(coverage_alpha):
            break

    if not selected and sorted_token_ids.numel() > 0:
        selected.append(
            (
                int(sorted_token_ids[0].item()),
                float(sorted_token_probs[0].item()),
                1,
                False,
                IGNORE_IDX,
            )
        )
    return selected


def _append_children_to_tree(
    state: AnchorTreeState,
    *,
    parent_idx: int,
    sorted_token_ids: torch.Tensor,
    sorted_token_probs: torch.Tensor,
    child_coverage_alpha: float,
    max_children_per_parent: int,
) -> None:
    """Materialize one parent's selected children and queue them for expansion.

    Every child receives both its local probability and cumulative ``path_prob``.
    The frontier heap is keyed by ``-path_prob`` so later expansions always pick
    the currently most reachable node first.
    """

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
        heapq.heappush(
            state.frontier,
            (-float(parent_path_prob * local_prob), state.next_heap_tiebreak, node_idx),
        )
        state.next_heap_tiebreak += 1


def finalize_anchor_tree(state: AnchorTreeState) -> GeneratedAnchorTree:
    """Freeze a mutable anchor tree into its storage form.

    During generation ``child_indices`` stores the explicit child node indices.
    On disk we store a more compact ``(first_child, child_count)`` pair instead.
    """

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
    """Pure helper used by tests to exercise the branching policy.

    This function runs the same tree-growth logic as the verifier-backed path,
    but gets child candidate lists from ``candidate_provider`` instead of a real
    model forward pass.
    """

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
    """Build the flex BlockMask used for tree expansion.

    Each tree query can attend to valid main-path tokens up to its anchor and to
    ancestor nodes from the same anchor tree. It cannot attend to siblings or to
    nodes from other anchor trees.
    """

    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError as exc:
        raise RuntimeError(
            "Flex attention is not available in this PyTorch build. "
            "Install a PyTorch build with torch.nn.attention.flex_attention "
            "or rerun with --attn-implementation sdpa."
        ) from exc

    batch_size, q_count = query_anchor_positions.shape
    total_tree_keys = tree_can_attend.shape[-1]
    ctx_clamp_max = max(ctx_len - 1, 0)
    tree_clamp_max = max(total_tree_keys - 1, 0)

    def mask_mod(b, h, q_idx, kv_idx):
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


def _build_dynamic_dense_step_attention_mask(
    *,
    query_anchor_positions: torch.Tensor,
    query_valid_mask: torch.Tensor,
    tree_can_attend: torch.Tensor,
    tree_key_valid_mask: torch.Tensor,
    document_mask: torch.Tensor,
    valid_tokens: torch.Tensor,
    ctx_len: int,
) -> torch.Tensor:
    """Dense fallback version of the tree-expansion attention mask."""

    device = query_anchor_positions.device
    batch_size, q_count = query_anchor_positions.shape
    total_tree_keys = tree_can_attend.shape[-1]

    ctx_positions = torch.arange(ctx_len, device=device).view(1, 1, ctx_len)
    query_docs = document_mask.gather(1, query_anchor_positions.clamp(min=0, max=max(ctx_len - 1, 0)))
    ctx_can_attend = (
        query_valid_mask.unsqueeze(-1)
        & valid_tokens.unsqueeze(1)
        & (document_mask.unsqueeze(1) == query_docs.unsqueeze(-1))
        & (ctx_positions <= query_anchor_positions.unsqueeze(-1))
    )
    tree_can_attend = (
        query_valid_mask.unsqueeze(-1)
        & tree_key_valid_mask.unsqueeze(1)
        & tree_can_attend
    )
    attend = torch.cat([ctx_can_attend, tree_can_attend], dim=-1)
    mask = torch.zeros((batch_size, 1, q_count, ctx_len + total_tree_keys), dtype=torch.float32, device=device)
    mask.masked_fill_(~attend.unsqueeze(1), float("-inf"))
    return mask


def build_stage2_v2_runtime(
    model,
    *,
    compile_enabled: bool = False,
    compile_mode: str = "reduce-overhead",
    log_enabled: bool = True,
) -> Stage2V2Runtime:
    """Create the optional compile-aware runtime wrappers for Stage 2 v2."""

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

    if getattr(getattr(model, "config", None), "_attn_implementation", None) != "flex_attention":
        if log_enabled:
            print(
                "Stage 2 v2 compile currently targets flex_attention; continuing without compile.",
                flush=True,
            )
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
    """Compute verifier probabilities for the true next token along the main path.

    The returned tensor is aligned to ``input_ids`` such that position ``i``
    stores the probability assigned to token ``input_ids[i]`` when predicting it
    from prefix ``input_ids[:i]``.
    """

    batch_size, seq_len = input_ids.shape
    token_probs = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=input_ids.device)
    if seq_len == 0:
        return token_probs
    token_probs[:, 0] = 1.0
    score_start = time.perf_counter()
    for pos_chunk in iter_position_chunks(seq_len, logit_chunk_size):
        start = pos_chunk.start
        stop = pos_chunk.stop
        next_len = max(0, min(stop, seq_len - 1) - start)
        if next_len <= 0:
            continue
        logits = lm_head(hidden_states[:, pos_chunk, :].to(getattr(lm_head, "weight", hidden_states).dtype)).float()
        next_token_ids = input_ids[:, start + 1 : start + 1 + next_len]
        next_logits = logits[:, :next_len].gather(-1, next_token_ids.unsqueeze(-1)).squeeze(-1)
        log_denom = torch.logsumexp(logits[:, :next_len], dim=-1)
        token_probs[:, start + 1 : start + 1 + next_len] = (next_logits - log_denom).exp().to(torch.float32)
    _accumulate_profile(profile, "initial_summary_s", score_start)
    return token_probs


def _build_single_sequence_attention_mask(
    *,
    query_anchor_positions: torch.Tensor,
    query_valid_mask: torch.Tensor,
    tree_can_attend: torch.Tensor,
    tree_key_valid_mask: torch.Tensor,
    document_mask: torch.Tensor,
    valid_tokens: torch.Tensor,
    use_flex: bool,
    runtime: Stage2V2Runtime | None,
) -> torch.Tensor:
    """Dispatch to either the flex-attention or dense tree mask builder."""

    builder = runtime.flex_mask_builder if runtime is not None else _build_dynamic_flex_block_mask
    if use_flex:
        return builder(
            query_anchor_positions=query_anchor_positions,
            query_valid_mask=query_valid_mask,
            tree_can_attend=tree_can_attend,
            tree_key_valid_mask=tree_key_valid_mask,
            document_mask=document_mask,
            valid_tokens=valid_tokens,
            ctx_len=document_mask.shape[1],
        )
    return _build_dynamic_dense_step_attention_mask(
        query_anchor_positions=query_anchor_positions,
        query_valid_mask=query_valid_mask,
        tree_can_attend=tree_can_attend,
        tree_key_valid_mask=tree_key_valid_mask,
        document_mask=document_mask,
        valid_tokens=valid_tokens,
        ctx_len=document_mask.shape[1],
    )


def _ancestor_node_indices(state: AnchorTreeState, node_idx: int) -> set[int]:
    """Return the strict ancestor set of a node within one anchor tree."""

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
    """Pick an in-range fallback position for dummy batch slots."""

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
    profile: dict[str, float] | None = None,
) -> list[list[GeneratedAnchorTree]]:
    """Grow anchor-local trees for a padded batch of sequences."""

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

    states_per_row: list[list[AnchorTreeState]] = []
    root_row_indices: list[int] = []
    root_anchor_positions: list[int] = []
    root_targets: list[tuple[int, int]] = []
    for row_idx, (main_path_ids_row, response_start_position, anchor_positions, anchor_probs) in enumerate(
        zip(
            main_path_ids_per_row,
            response_start_positions,
            anchor_positions_per_row,
            anchor_next_token_probs_per_row,
            strict=True,
        )
    ):
        row_states = [
            _empty_anchor_tree(
                anchor_main_path_position=anchor_position,
                anchor_next_token_prob=anchor_prob,
                main_path_ids=[int(token_id) for token_id in main_path_ids_row],
                response_start_position=response_start_position,
            )
            for anchor_position, anchor_prob in zip(anchor_positions, anchor_probs, strict=True)
        ]
        states_per_row.append(row_states)
        for tree_id, anchor_position in enumerate(anchor_positions):
            root_row_indices.append(row_idx)
            root_anchor_positions.append(int(anchor_position))
            root_targets.append((row_idx, tree_id))

    if root_targets:
        root_hidden_states = hidden_states[
            torch.as_tensor(root_row_indices, dtype=torch.long, device=device),
            torch.as_tensor(root_anchor_positions, dtype=torch.long, device=device),
        ]
        root_sorted_ids, root_sorted_probs = _score_hidden_states_with_candidates(
            root_hidden_states,
            lm_head,
            logit_chunk_size=logit_chunk_size,
            profile=profile,
        )
        for root_idx, (row_idx, tree_id) in enumerate(root_targets):
            _append_children_to_tree(
                states_per_row[row_idx][tree_id],
                parent_idx=0,
                sorted_token_ids=root_sorted_ids[root_idx],
                sorted_token_probs=root_sorted_probs[root_idx],
                child_coverage_alpha=child_coverage_alpha,
                max_children_per_parent=max_children_per_node,
            )

    cached_tree_meta: list[list[tuple[int, int] | None]] = [[] for _ in range(batch_size)]
    cached_slot_count = 0
    use_flex = getattr(getattr(model, "config", None), "_attn_implementation", None) == "flex_attention"
    base_model_forward = runtime.base_model_forward if runtime is not None else model.base_model

    while True:
        selected_per_row: list[list[tuple[int, int]]] = []
        for row_states in states_per_row:
            row_selected: list[tuple[int, int]] = []
            for tree_id, state in enumerate(row_states):
                if state.expanded_token_nodes >= num_attend_tokens_per_anchor:
                    continue
                while state.frontier:
                    _, _, node_idx = heapq.heappop(state.frontier)
                    if not state.nodes[node_idx].expanded:
                        row_selected.append((tree_id, node_idx))
                        break
            selected_per_row.append(row_selected)

        q_count = max((len(row_selected) for row_selected in selected_per_row), default=0)
        if q_count == 0:
            break

        total_tree_keys = cached_slot_count + q_count
        query_ids = torch.zeros((batch_size, q_count), dtype=torch.long, device=device)
        query_positions = torch.zeros((batch_size, q_count), dtype=torch.long, device=device)
        query_anchor_positions = torch.zeros((batch_size, q_count), dtype=torch.long, device=device)
        query_valid_mask = torch.zeros((batch_size, q_count), dtype=torch.bool, device=device)
        tree_can_attend = torch.zeros((batch_size, q_count, total_tree_keys), dtype=torch.bool, device=device)
        tree_key_valid_mask = torch.zeros((batch_size, total_tree_keys), dtype=torch.bool, device=device)

        for row_idx, row_selected in enumerate(selected_per_row):
            fallback_anchor_position = _dummy_anchor_position_for_row(
                valid_tokens_row=valid_tokens[row_idx],
                preferred_anchor_positions=(
                    anchor_positions_per_row[row_idx]
                    or [response_start_positions[row_idx]]
                ),
            )
            fallback_token_id = int(input_ids[row_idx, fallback_anchor_position].item())

            for key_idx, cached_meta in enumerate(cached_tree_meta[row_idx]):
                if cached_meta is not None:
                    tree_key_valid_mask[row_idx, key_idx] = True

            for query_idx in range(q_count):
                query_ids[row_idx, query_idx] = fallback_token_id
                query_positions[row_idx, query_idx] = fallback_anchor_position
                query_anchor_positions[row_idx, query_idx] = fallback_anchor_position
                if query_idx >= len(row_selected):
                    continue

                tree_id, node_idx = row_selected[query_idx]
                state = states_per_row[row_idx][tree_id]
                node = state.nodes[node_idx]
                query_valid_mask[row_idx, query_idx] = True
                query_ids[row_idx, query_idx] = int(node.token_id)
                query_positions[row_idx, query_idx] = int(state.anchor_main_path_position + node.depth)
                query_anchor_positions[row_idx, query_idx] = int(state.anchor_main_path_position)

                ancestors = _ancestor_node_indices(state, node_idx)
                for key_idx, cached_meta in enumerate(cached_tree_meta[row_idx]):
                    if cached_meta is None:
                        continue
                    cached_tree_id, cached_node_idx = cached_meta
                    if cached_tree_id == tree_id and cached_node_idx in ancestors:
                        tree_can_attend[row_idx, query_idx, key_idx] = True

                current_key_idx = cached_slot_count + query_idx
                tree_can_attend[row_idx, query_idx, current_key_idx] = True
                tree_key_valid_mask[row_idx, current_key_idx] = True

        mask_start = time.perf_counter()
        print(
            "running model..."
        )
        attention_mask = _build_single_sequence_attention_mask(
            query_anchor_positions=query_anchor_positions,
            query_valid_mask=query_valid_mask,
            tree_can_attend=tree_can_attend,
            tree_key_valid_mask=tree_key_valid_mask,
            document_mask=document_mask,
            valid_tokens=valid_tokens,
            use_flex=use_flex,
            runtime=runtime,
        )
        _accumulate_profile(profile, "mask_build_s", mask_start)

        base_out = _run_base_model_forward(
            base_model_forward,
            profile=profile,
            profile_key="tree_forward_s",
            input_ids=query_ids,
            position_ids=query_positions,
            attention_mask=attention_mask,
            past_key_values=kv_cache,
            use_cache=True,
        )
        new_hidden_states, kv_cache = _extract_hidden_and_cache(base_out)
        print(
            "ran model"
        )
        real_hidden_states = new_hidden_states[query_valid_mask]
        child_sorted_ids, child_sorted_probs = _score_hidden_states_with_candidates(
            real_hidden_states,
            lm_head,
            logit_chunk_size=logit_chunk_size,
            profile=profile,
        )

        real_query_idx = 0
        for row_idx, row_selected in enumerate(selected_per_row):
            cache_step_entries: list[tuple[int, int] | None] = []
            for query_idx in range(q_count):
                if query_idx >= len(row_selected):
                    cache_step_entries.append(None)
                    continue

                tree_id, node_idx = row_selected[query_idx]
                state = states_per_row[row_idx][tree_id]
                state.nodes[node_idx].expanded = True
                state.expanded_token_nodes += 1
                _append_children_to_tree(
                    state,
                    parent_idx=node_idx,
                    sorted_token_ids=child_sorted_ids[real_query_idx],
                    sorted_token_probs=child_sorted_probs[real_query_idx],
                    child_coverage_alpha=child_coverage_alpha,
                    max_children_per_parent=max_children_per_node,
                )
                real_query_idx += 1
                cache_step_entries.append((tree_id, node_idx))
            cached_tree_meta[row_idx].extend(cache_step_entries)
        cached_slot_count += q_count

    return [[finalize_anchor_tree(state) for state in row_states] for row_states in states_per_row]


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
    profile: dict[str, float] | None = None,
) -> list[GeneratedAnchorTree]:
    """Backward-compatible single-sequence wrapper over the batched generator."""

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
        profile=profile,
    )[0]


def initialize_stage2_v2_hdf5(
    hf: h5py.File,
    *,
    prob_dtype: np.dtype,
    attrs: dict[str, Any],
) -> None:
    """Create an empty Stage 2 v2 HDF5 file with the full schema.

    The file is split into sequence-level, anchor-level, and node-level tables
    with offset arrays linking those tables together.
    """

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
    """Append buffered sequence trees to disk and update all offset tables."""

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


def collect_stage2_v2_merge_manifest(part_paths: Sequence[Path]) -> list[Stage2V2MergeEntry]:
    """Build the per-sequence merge manifest for distributed Stage 2 v2 output."""

    manifest: list[Stage2V2MergeEntry] = []
    for part_path in part_paths:
        with h5py.File(part_path, "r") as hf:
            record_idx = hf["record_idx"][:]
            main_offsets = hf["main_path_offsets"][:]
            seq_anchor_offsets = hf["sequence_anchor_offsets"][:]
            anchor_node_offsets = hf["anchor_node_offsets"][:]
            for seq_idx, seq_record_idx in enumerate(record_idx.tolist()):
                anchor_start = int(seq_anchor_offsets[seq_idx])
                anchor_end = int(seq_anchor_offsets[seq_idx + 1])
                node_start = int(anchor_node_offsets[anchor_start])
                node_end = int(anchor_node_offsets[anchor_end])
                manifest.append(
                    Stage2V2MergeEntry(
                        record_idx=int(seq_record_idx),
                        part_path=str(part_path),
                        seq_idx=seq_idx,
                        main_start=int(main_offsets[seq_idx]),
                        main_end=int(main_offsets[seq_idx + 1]),
                        anchor_start=anchor_start,
                        anchor_end=anchor_end,
                        node_start=node_start,
                        node_end=node_end,
                    )
                )
    manifest.sort(key=lambda entry: entry.record_idx)
    return manifest


def merge_stage2_v2_parts(
    *,
    part_paths: Sequence[Path],
    output_path: Path,
    prob_dtype: np.dtype,
    attrs: dict[str, Any],
    log_fn: Callable[[str], None],
) -> tuple[int, int, int, int]:
    """Merge rank-local Stage 2 v2 HDF5 shards back into one ordered file."""

    manifest = collect_stage2_v2_merge_manifest(part_paths)
    seq_buf: list[GeneratedSequenceTree] = []
    flush_every = 128
    n_sequences_written = 0
    n_main_written = 0
    n_anchor_written = 0
    n_node_written = 0

    with ExitStack() as stack:
        handles = {str(path): stack.enter_context(h5py.File(path, "r")) for path in part_paths}
        with h5py.File(output_path, "w") as hf:
            initialize_stage2_v2_hdf5(hf, prob_dtype=prob_dtype, attrs=attrs)
            for entry in manifest:
                src = handles[entry.part_path]
                anchors: list[GeneratedAnchorTree] = []
                anchor_positions = src["anchor_main_path_positions"][entry.anchor_start : entry.anchor_end]
                anchor_probs = src["anchor_next_token_probs"][entry.anchor_start : entry.anchor_end]
                anchor_node_offsets = src["anchor_node_offsets"][entry.anchor_start : entry.anchor_end + 1]
                for local_anchor_idx, (anchor_position, anchor_prob) in enumerate(zip(anchor_positions, anchor_probs, strict=True)):
                    node_start = int(anchor_node_offsets[local_anchor_idx])
                    node_end = int(anchor_node_offsets[local_anchor_idx + 1])
                    nodes: list[SequenceTreeNode] = []
                    for node_idx in range(node_start, node_end):
                        nodes.append(
                            SequenceTreeNode(
                                token_id=int(src["node_token_ids"][node_idx]),
                                parent_index=int(src["node_parent_indices"][node_idx]),
                                depth=int(src["node_depths"][node_idx]),
                                local_prob=float(src["node_local_probs"][node_idx]),
                                path_prob=float(src["node_path_probs"][node_idx]),
                                rank=int(src["node_ranks"][node_idx]),
                                main_path_position=int(src["node_main_path_positions"][node_idx]),
                                is_main_path=bool(src["node_is_main_path"][node_idx]),
                                child_indices=[
                                    int(src["node_first_child"][node_idx]),
                                    int(src["node_child_count"][node_idx]),
                                ],
                            )
                        )
                    anchors.append(
                        GeneratedAnchorTree(
                            anchor_main_path_position=int(anchor_position),
                            anchor_next_token_prob=float(anchor_prob),
                            nodes=nodes,
                        )
                    )

                seq_buf.append(
                    GeneratedSequenceTree(
                        record_idx=entry.record_idx,
                        main_path_ids=src["main_path_ids"][entry.main_start : entry.main_end].astype(np.int32).tolist(),
                        response_start_position=int(src["response_start_positions"][entry.seq_idx]),
                        anchors=anchors,
                    )
                )
                if len(seq_buf) >= flush_every:
                    n_sequences_written, n_main_written, n_anchor_written, n_node_written = flush_stage2_v2_hdf5(
                        hf,
                        seq_buf,
                        n_sequences_written=n_sequences_written,
                        n_main_path_ids_written=n_main_written,
                        n_anchors_written=n_anchor_written,
                        n_nodes_written=n_node_written,
                        prob_dtype=prob_dtype,
                    )
                    log_fn(
                        f"Merged sequences: {n_sequences_written} "
                        f"main_path_tokens: {n_main_written} anchors: {n_anchor_written} nodes: {n_node_written}"
                    )

            n_sequences_written, n_main_written, n_anchor_written, n_node_written = flush_stage2_v2_hdf5(
                hf,
                seq_buf,
                n_sequences_written=n_sequences_written,
                n_main_path_ids_written=n_main_written,
                n_anchors_written=n_anchor_written,
                n_nodes_written=n_node_written,
                prob_dtype=prob_dtype,
            )
    return n_sequences_written, n_main_written, n_anchor_written, n_node_written


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
    """Generate Stage 2 v2 sequence trees for one padded pending batch."""

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

    base_model_forward = runtime.base_model_forward if runtime is not None else model.base_model
    base_out = _run_base_model_forward(
        base_model_forward,
        profile=profile,
        profile_key="initial_forward_s",
        input_ids=input_ids,
        attention_mask=valid_tokens.long(),
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
    """Build the CLI parser for Stage 2 v2 dataset generation."""

    parser = argparse.ArgumentParser(description="Stage 2 v2: generate dynamic sequence-tree training data.")
    parser.add_argument("--model", required=True, help="HF model name or local path")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--data-dir", help="Directory of JSONL shards")
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
        "--attn-implementation",
        choices=["flex_attention", "sdpa"],
        default="flex_attention",
        help="Attention backend for the verifier model.",
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
    """CLI entrypoint for Stage 2 v2 generation."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = build_arg_parser()
    args = parser.parse_args()
    ctx = init_distributed_context()
    try:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

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
            attn_implementation=args.attn_implementation,
        )
        if tokenizer.pad_token_id is not None and getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(model, "resize_token_embeddings") and len(tokenizer) != model.get_input_embeddings().num_embeddings:
            model.resize_token_embeddings(len(tokenizer))
        model.to(ctx.device)
        model.eval()
        runtime = build_stage2_v2_runtime(
            model,
            compile_enabled=args.compile,
            compile_mode=args.compile_mode,
            log_enabled=ctx.is_primary,
        )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = output_path
        if ctx.is_distributed:
            parts_dir = build_parts_dir(output_path)
            prepare_parts_dir(parts_dir, ctx)
            part_path = build_rank_part_path(output_path, ctx.rank)

        attrs = {
            "format_version": "stage2_v2",
            "attn_implementation": args.attn_implementation,
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
        records = shard_records_for_rank(records, ctx)
        dataset_prep_s = time.perf_counter() - dataset_prep_start

        sequence_buf: list[GeneratedSequenceTree] = []
        flush_every = 128
        batch_count = 0
        timings: dict[str, float] = {"dataset_prep_s": dataset_prep_s}

        with h5py.File(part_path, "w") as hf:
            initialize_stage2_v2_hdf5(hf, prob_dtype=disk_prob_dtype, attrs=attrs)
            n_sequences_written = 0
            n_main_written = 0
            n_anchor_written = 0
            n_node_written = 0

            pending: list[tuple[int, list[int], list[int]]] = []
            for record in tqdm.tqdm(records):
                pending.append((int(record["record_idx"]), record["prompt_ids"], record["response_ids"]))
                if len(pending) < args.batch_size:
                    continue

                sequence_buf.extend(
                    _generate_sequences_for_pending_batch(
                        pending=pending,
                        model=model,
                        runtime=runtime,
                        pad_token_id=tokenizer.pad_token_id,
                        device=ctx.device,
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
                    ctx.log(
                        f"Sequences: {n_sequences_written} "
                        f"main_path_tokens: {n_main_written} anchors: {n_anchor_written} nodes: {n_node_written}"
                    )

                if args.profile_every > 0 and batch_count % args.profile_every == 0 and ctx.is_primary:
                    ctx.log(
                        f"[profile] batches={batch_count} "
                        f"dataset_prep={timings.get('dataset_prep_s', 0.0):.3f}s "
                        f"build_batch={timings.get('build_batch_s', 0.0):.3f}s "
                        f"initial_forward={timings.get('initial_forward_s', 0.0):.3f}s "
                        f"initial_summary={timings.get('initial_summary_s', 0.0):.3f}s "
                        f"candidate_score={timings.get('candidate_score_s', 0.0):.3f}s "
                        f"mask_build={timings.get('mask_build_s', 0.0):.3f}s "
                        f"tree_forward={timings.get('tree_forward_s', 0.0):.3f}s "
                        f"flush={timings.get('flush_s', 0.0):.3f}s"
                    )
                    timings.clear()

            if pending:
                sequence_buf.extend(
                    _generate_sequences_for_pending_batch(
                        pending=pending,
                        model=model,
                        runtime=runtime,
                        pad_token_id=tokenizer.pad_token_id,
                        device=ctx.device,
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

        if ctx.is_distributed:
            ctx.barrier()
            if ctx.is_primary:
                part_paths = [build_rank_part_path(output_path, rank) for rank in range(ctx.world_size)]
                ctx.log(f"Merging {len(part_paths)} Stage 2 v2 rank shards into {output_path}")
                final_counts = merge_stage2_v2_parts(
                    part_paths=part_paths,
                    output_path=output_path,
                    prob_dtype=disk_prob_dtype,
                    attrs=attrs,
                    log_fn=ctx.log,
                )
                shutil.rmtree(build_parts_dir(output_path))
                ctx.log(
                    "Done. Sequences: "
                    f"{final_counts[0]} main_path_tokens: {final_counts[1]} anchors: {final_counts[2]} nodes: {final_counts[3]}"
                )
        else:
            ctx.log(
                f"Done. Sequences: {n_sequences_written} "
                f"main_path_tokens: {n_main_written} anchors: {n_anchor_written} nodes: {n_node_written}"
            )
    finally:
        ctx.shutdown()


if __name__ == "__main__":
    main()
