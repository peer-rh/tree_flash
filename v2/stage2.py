from __future__ import annotations
import math
import os
import shutil
import time

import argparse
import random
from contextlib import ExitStack
from collections import defaultdict
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

if TYPE_CHECKING:
    from datasets import Dataset
else:
    Dataset = Any

IGNORE_IDX = -1
DEFAULT_SUB_TREE_PATHS = ["0-1", "0-2", "0-3", "1-4", "1-5", "2-6", "2-7"]


@dataclass
class SubTreeInfo:
    edge_list: InitVar[Sequence[str | tuple[int, int]]]

    paths: list[tuple[int, int]] = field(init=False)
    size: int = field(init=False)
    ancestor_map: torch.Tensor = field(init=False)  # [size, size], ancestor-or-self
    parent_map: dict[int, int] = field(init=False)
    children_map: dict[int, list[int]] = field(init=False)
    depth_of: list[int] = field(init=False)
    nodes_at_depth: dict[int, list[int]] = field(init=False)
    non_leaf_at_depth: dict[int, list[int]] = field(init=False)
    max_depth: int = field(init=False)

    def __post_init__(self, edge_list: Sequence[str | tuple[int, int]]) -> None:
        parsed_paths: list[tuple[int, int]] = []
        parent_map: dict[int, int] = {}
        all_nodes: set[int] = {0}

        for edge in edge_list:
            if isinstance(edge, str):
                parts = edge.split("-")
                if len(parts) != 2:
                    raise ValueError(f"Subtree edge must be 'X-Y', got {edge!r}")
                parent, child = int(parts[0]), int(parts[1])
            else:
                parent, child = int(edge[0]), int(edge[1])

            if child == 0:
                raise ValueError("Subtree root must remain node 0")
            if child in parent_map:
                raise ValueError(f"Node {child} has multiple parents")

            parsed_paths.append((parent, child))
            parent_map[child] = parent
            all_nodes.add(parent)
            all_nodes.add(child)

        sorted_nodes = sorted(all_nodes)
        expected_nodes = list(range(len(sorted_nodes)))
        if sorted_nodes != expected_nodes:
            raise ValueError(
                f"Subtree node ids must be contiguous from 0, got {sorted_nodes}"
            )

        self.paths = parsed_paths
        self.parent_map = parent_map
        self.size = len(sorted_nodes)

        children_map: dict[int, list[int]] = defaultdict(list)
        for parent, child in parsed_paths:
            children_map[parent].append(child)
        self.children_map = {node: sorted(children_map.get(node, [])) for node in range(self.size)}

        depth_of = [0] * self.size
        for node in range(1, self.size):
            cur = node
            depth = 0
            seen: set[int] = set()
            while cur != 0:
                if cur in seen:
                    raise ValueError("Cycle detected in subtree edge list")
                seen.add(cur)
                if cur not in parent_map:
                    raise ValueError(f"Node {cur} is disconnected from root 0")
                cur = parent_map[cur]
                depth += 1
            depth_of[node] = depth
        self.depth_of = depth_of
        self.max_depth = max(depth_of, default=0)

        nodes_at_depth: dict[int, list[int]] = defaultdict(list)
        non_leaf_at_depth: dict[int, list[int]] = defaultdict(list)
        for node, depth in enumerate(depth_of):
            nodes_at_depth[depth].append(node)
            if self.children_map.get(node):
                non_leaf_at_depth[depth].append(node)
        self.nodes_at_depth = {depth: sorted(nodes) for depth, nodes in nodes_at_depth.items()}
        self.non_leaf_at_depth = {
            depth: sorted(nodes) for depth, nodes in non_leaf_at_depth.items()
        }

        anc = torch.zeros(self.size, self.size, dtype=torch.bool)
        for q in range(self.size):
            cur = q
            while True:
                anc[cur, q] = True
                if cur == 0:
                    break
                cur = parent_map[cur]
        self.ancestor_map = anc


@dataclass
class LMHeadChunkOutputs:
    log_denom: torch.Tensor
    gathered_logits: torch.Tensor | None = None
    top_vals: torch.Tensor | None = None
    top_ids: torch.Tensor | None = None
    top_probs: torch.Tensor | None = None


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
                    f"Stage 2 compile fallback for {self.name}: {self.fallback_reason}",
                    flush=True,
                )
            return self.eager_fn(*args, **kwargs)


@dataclass
class Stage2Runtime:
    base_model_forward: CompiledCallable
    flex_mask_builder: CompiledCallable
    compile_enabled: bool = False
    compile_mode: str | None = None


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
    row_start: int
    row_end: int


def init_distributed_context() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return DistributedContext(rank=0, world_size=1, local_rank=0, device=device, backend=None)

    if not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU Stage 2 requires CUDA when launched with torchrun.")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable in this PyTorch build.")

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device("cuda", local_rank)
    return DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        backend="nccl",
    )


def iter_position_chunks(total_positions: int, chunk_size: int) -> Iterator[slice]:
    if total_positions <= 0:
        return
    if chunk_size <= 0 or chunk_size >= total_positions:
        yield slice(0, total_positions)
        return
    for start in range(0, total_positions, chunk_size):
        yield slice(start, min(start + chunk_size, total_positions))


def summarize_lm_head_chunk(
    hidden_states_chunk: torch.Tensor,
    lm_head: torch.nn.Module,
    *,
    gather_token_ids: torch.Tensor | None = None,
    topk: int | None = None,
    compute_top_probs: bool = False,
) -> LMHeadChunkOutputs:
    logits_chunk = lm_head(hidden_states_chunk)
    gathered_logits = None
    top_vals = None
    top_ids = None
    top_probs = None

    if gather_token_ids is not None:
        gathered_logits = logits_chunk.gather(-1, gather_token_ids.unsqueeze(-1)).squeeze(-1)

    if topk is not None and topk > 0:
        top_vals, top_ids = logits_chunk.topk(topk, dim=-1)
        if compute_top_probs:
            top_probs = F.softmax(logits_chunk.float(), dim=-1).gather(-1, top_ids).to(torch.float32)

    return LMHeadChunkOutputs(
        log_denom=torch.logsumexp(logits_chunk, dim=-1),
        gathered_logits=gathered_logits,
        top_vals=top_vals,
        top_ids=top_ids,
        top_probs=top_probs,
    )


def _extract_hidden_and_cache(model_outputs) -> tuple[torch.Tensor, object | None]:
    if hasattr(model_outputs, "last_hidden_state"):
        return model_outputs.last_hidden_state, getattr(model_outputs, "past_key_values", None)
    if isinstance(model_outputs, tuple):
        hidden_states = model_outputs[0]
        past_key_values = model_outputs[1] if len(model_outputs) > 1 else None
        return hidden_states, past_key_values
    raise TypeError(f"Unsupported base model output type: {type(model_outputs)!r}")


def _accumulate_profile(profile: dict[str, float] | None, key: str, start_time: float) -> None:
    if profile is not None:
        profile[key] = profile.get(key, 0.0) + (time.perf_counter() - start_time)


def _build_flex_block_mask(
    *,
    root_positions: torch.Tensor,
    query_vertex_ids: torch.Tensor,
    tree_root_positions: torch.Tensor,
    tree_vertex_ids: torch.Tensor,
    document_mask: torch.Tensor,
    valid_tokens: torch.Tensor,
    ancestor_map: torch.Tensor,
    ctx_len: int,
):
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError as exc:
        raise RuntimeError(
            "Flex attention is not available in this PyTorch build. "
            "Use a PyTorch version with torch.nn.attention.flex_attention "
            "or rerun with --attn-implementation sdpa."
        ) from exc

    device = root_positions.device
    bsz, q_count = root_positions.shape
    k_count = tree_root_positions.shape[1]
    ctx_clamp_max = max(ctx_len - 1, 0)
    tree_clamp_max = max(k_count - 1, 0)

    def mask_mod(b, h, q_idx, kv_idx):
        q_root = root_positions[b, q_idx]

        in_ctx = kv_idx < ctx_len
        ctx_idx = kv_idx.clamp(0, ctx_clamp_max)
        same_doc = document_mask[b, ctx_idx] == document_mask[b, q_root]
        causal_ctx = ctx_idx <= q_root
        ctx_mask = in_ctx & same_doc & causal_ctx & valid_tokens[b, ctx_idx]

        tree_idx = (kv_idx - ctx_len).clamp(0, tree_clamp_max)
        same_tree = tree_root_positions[b, tree_idx] == q_root
        key_vertex = tree_vertex_ids[b, tree_idx]
        query_vertex = query_vertex_ids[q_idx]
        tree_mask = (~in_ctx) & same_tree & ancestor_map[key_vertex, query_vertex]
        return ctx_mask | tree_mask

    return create_block_mask(
        mask_mod,
        B=bsz,
        H=None,
        Q_LEN=q_count,
        KV_LEN=ctx_len + k_count,
        device=device,
        BLOCK_SIZE=128,
    )


def _build_dense_step_attention_mask(
    *,
    root_positions: torch.Tensor,
    query_vertex_ids: torch.Tensor,
    tree_root_positions: torch.Tensor,
    tree_vertex_ids: torch.Tensor,
    document_mask: torch.Tensor,
    valid_tokens: torch.Tensor,
    ancestor_map: torch.Tensor,
    ctx_len: int,
):
    device = root_positions.device
    bsz, q_count = root_positions.shape
    k_count = tree_root_positions.shape[1]

    ctx_pos = torch.arange(ctx_len, device=device).view(1, 1, ctx_len)
    root_docs = document_mask.gather(1, root_positions)
    ctx_attend = (
        (document_mask.unsqueeze(1) == root_docs.unsqueeze(-1))
        & (ctx_pos <= root_positions.unsqueeze(-1))
        & valid_tokens.unsqueeze(1)
    )

    key_vertices = tree_vertex_ids.unsqueeze(1).expand(bsz, q_count, k_count)
    query_vertices = query_vertex_ids.view(1, q_count, 1).expand(bsz, q_count, k_count)
    same_tree = tree_root_positions.unsqueeze(1) == root_positions.unsqueeze(-1)
    tree_attend = same_tree & ancestor_map[key_vertices, query_vertices]

    attend = torch.cat([ctx_attend, tree_attend], dim=-1)
    mask_4d = torch.zeros(
        (bsz, 1, q_count, ctx_len + k_count),
        dtype=torch.float32,
        device=device,
    )
    mask_4d.masked_fill_(~attend.unsqueeze(1), float("-inf"))
    return mask_4d


def build_stage2_runtime(
    model,
    *,
    compile_enabled: bool = False,
    compile_mode: str = "reduce-overhead",
    log_enabled: bool = True,
) -> Stage2Runtime:
    base_model_forward = CompiledCallable(
        name="base_model_forward",
        eager_fn=model.base_model,
        log_enabled=log_enabled,
    )
    flex_mask_builder = CompiledCallable(
        name="flex_block_mask_builder",
        eager_fn=_build_flex_block_mask,
        log_enabled=log_enabled,
    )
    runtime = Stage2Runtime(
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
                "Stage 2 compile currently targets flex_attention; continuing without compile.",
                flush=True,
            )
        return runtime

    if not hasattr(torch, "compile"):
        if log_enabled:
            print("Stage 2 compile requested, but torch.compile is unavailable; continuing eagerly.", flush=True)
        return runtime

    base_model_forward.compiled_fn = torch.compile(
        model.base_model,
        dynamic=True,
        mode=compile_mode,
    )
    flex_mask_builder.compiled_fn = torch.compile(
        _build_flex_block_mask,
        dynamic=True,
        mode=compile_mode,
    )
    runtime.compile_enabled = True
    runtime.compile_mode = compile_mode
    if log_enabled:
        print(
            f"Stage 2 compile enabled for flex_attention (mode={compile_mode}).",
            flush=True,
        )
    return runtime


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


def _summarize_initial_pass(
    *,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    valid_tokens: torch.Tensor,
    lm_head: torch.nn.Module,
    vocab_size: int,
    logit_chunk_size: int,
    profile: dict[str, float] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    bsz, seq_len = input_ids.shape
    device = input_ids.device
    proj_dtype = lm_head.weight.dtype if hasattr(lm_head, "weight") else hidden_states.dtype
    log_denom = torch.empty((bsz, seq_len), dtype=proj_dtype, device=device)
    token_probs = torch.zeros((bsz, seq_len), dtype=torch.float32, device=device)
    if seq_len > 0:
        token_probs[:, 0] = 1.0

    request_k = vocab_size
    cand_vals = None
    cand_ids = None
    if request_k > 0:
        cand_vals = torch.empty((bsz, seq_len, request_k), dtype=proj_dtype, device=device)
        cand_ids = torch.empty((bsz, seq_len, request_k), dtype=torch.long, device=device)

    summary_start = time.perf_counter()
    for pos_chunk in iter_position_chunks(seq_len, logit_chunk_size):
        start = pos_chunk.start
        stop = pos_chunk.stop
        chunk_len = stop - start
        gather_token_ids = None
        next_len = max(0, min(stop, seq_len - 1) - start)
        if next_len > 0:
            gather_token_ids = torch.zeros((bsz, chunk_len), dtype=input_ids.dtype, device=device)
            gather_token_ids[:, :next_len] = input_ids[:, start + 1 : start + 1 + next_len]

        chunk_outputs = summarize_lm_head_chunk(
            hidden_states[:, pos_chunk, :],
            lm_head,
            gather_token_ids=gather_token_ids,
            topk=request_k if request_k > 0 else None,
        )
        log_denom[:, pos_chunk] = chunk_outputs.log_denom

        if next_len > 0 and chunk_outputs.gathered_logits is not None:
            token_probs[:, start + 1 : start + 1 + next_len] = (
                chunk_outputs.gathered_logits[:, :next_len] - chunk_outputs.log_denom[:, :next_len]
            ).exp().to(torch.float32)

        if request_k > 0 and cand_vals is not None and cand_ids is not None:
            cand_vals[:, pos_chunk, :] = chunk_outputs.top_vals
            cand_ids[:, pos_chunk, :] = chunk_outputs.top_ids

    _accumulate_profile(profile, "initial_summary_s", summary_start)
    return log_denom, token_probs, cand_vals, cand_ids


def _select_subtree_anchors(
    *,
    valid_response: torch.Tensor,
    token_probs: torch.Tensor,
    n_subtrees: int,
) -> torch.Tensor:
    device = token_probs.device
    response_counts = valid_response.sum(dim=1)
    if int(response_counts.amin().item()) <= 0:
        return torch.empty((token_probs.shape[0], 0), dtype=torch.long, device=device)

    score = torch.where(
        valid_response,
        1.0 - token_probs,
        torch.full_like(token_probs, float("-inf")),
    )
    n_select = min(n_subtrees, int(response_counts.amin().item()))
    anchors =  score.topk(n_select, dim=1).indices.sort(dim=1).values
    anchors = anchors - 1
    return anchors


def _fill_depth1_subtrees(
    *,
    subtree_ids: torch.Tensor,
    subtree_ar_probs: torch.Tensor,
    cand_vals: torch.Tensor | None,
    cand_ids: torch.Tensor | None,
    log_denom: torch.Tensor,
    input_ids: torch.Tensor,
    valid_tokens: torch.Tensor,
    depth1_slots: Sequence[int],
) -> None:
    if not depth1_slots or cand_vals is None or cand_ids is None:
        return

    bsz, seq_len = input_ids.shape
    device = input_ids.device
    has_next = torch.zeros((bsz, seq_len), dtype=torch.bool, device=device)
    excluded_token = torch.full((bsz, seq_len), IGNORE_IDX, dtype=torch.long, device=device)
    if seq_len > 1:
        has_next[:, :-1] = valid_tokens[:, 1:]
        excluded_token[:, :-1] = torch.where(
            valid_tokens[:, 1:],
            input_ids[:, 1:],
            torch.full_like(input_ids[:, 1:], IGNORE_IDX),
        )

    valid_candidates = valid_tokens.unsqueeze(-1)
    valid_candidates = valid_candidates & (
        (~has_next.unsqueeze(-1)) | cand_ids.ne(excluded_token.unsqueeze(-1))
    )
    candidate_ranks = valid_candidates.to(torch.int64).cumsum(dim=-1)

    for rank_idx, slot in enumerate(depth1_slots, start=1):
        slot_match = valid_candidates & candidate_ranks.eq(rank_idx)
        has_match = slot_match.any(dim=-1)
        match_idx = slot_match.to(torch.int64).argmax(dim=-1)
        chosen_ids = cand_ids.gather(-1, match_idx.unsqueeze(-1)).squeeze(-1)
        chosen_vals = cand_vals.gather(-1, match_idx.unsqueeze(-1)).squeeze(-1)
        chosen_probs = (chosen_vals - log_denom).exp().to(torch.float32)

        subtree_ids[:, :, slot] = torch.where(has_match, chosen_ids, subtree_ids[:, :, slot])
        subtree_ar_probs[:, :, slot] = torch.where(
            has_match,
            chosen_probs,
            subtree_ar_probs[:, :, slot],
        )


def _write_children_and_build_frontier(
    *,
    new_hidden_states: torch.Tensor,
    root_positions: torch.Tensor,
    vertex_ids: torch.Tensor,
    lm_head: torch.nn.Module,
    st_info: SubTreeInfo,
    current_depth: int,
    logit_chunk_size: int,
    subtree_ids: torch.Tensor,
    subtree_ar_probs: torch.Tensor,
    device: torch.device,
    profile: dict[str, float] | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    batch_size = root_positions.shape[0]
    batch_rows_cache: dict[int, torch.Tensor] = {}
    next_tokens: list[torch.Tensor] = []
    next_roots: list[torch.Tensor] = []
    next_vertices: list[torch.Tensor] = []
    next_non_leaf = set(st_info.non_leaf_at_depth.get(current_depth + 1, []))
    write_start = time.perf_counter()

    for parent_vertex in st_info.non_leaf_at_depth.get(current_depth, []):
        children = st_info.children_map.get(parent_vertex, [])
        if not children:
            continue

        parent_cols = torch.nonzero(vertex_ids == parent_vertex, as_tuple=False).flatten()
        if parent_cols.numel() == 0:
            continue

        parent_hidden_states = new_hidden_states.index_select(1, parent_cols)
        parent_roots = root_positions.index_select(1, parent_cols)
        child_k = len(children)

        for col_chunk in iter_position_chunks(parent_cols.numel(), logit_chunk_size):
            chunk_outputs = summarize_lm_head_chunk(
                parent_hidden_states[:, col_chunk, :],
                lm_head,
                topk=child_k,
                compute_top_probs=True,
            )
            chunk_roots = parent_roots[:, col_chunk]
            chunk_width = chunk_roots.shape[1]
            batch_rows = batch_rows_cache.get(chunk_width)
            if batch_rows is None:
                batch_rows = torch.arange(batch_size, device=device).unsqueeze(-1).expand(batch_size, chunk_width)
                batch_rows_cache[chunk_width] = batch_rows

            for child_rank, child_vertex in enumerate(children):
                child_ids = chunk_outputs.top_ids[:, :, child_rank]
                child_probs = chunk_outputs.top_probs[:, :, child_rank]
                subtree_ids[batch_rows, chunk_roots, child_vertex] = child_ids
                subtree_ar_probs[batch_rows, chunk_roots, child_vertex] = child_probs
                if child_vertex in next_non_leaf:
                    next_tokens.append(child_ids)
                    next_roots.append(chunk_roots)
                    next_vertices.append(
                        torch.full((chunk_width,), child_vertex, dtype=torch.long, device=device)
                    )

    _accumulate_profile(profile, "tree_writeback_s", write_start)
    if not next_tokens:
        return None, None, None

    return (
        torch.cat(next_tokens, dim=1),
        torch.cat(next_roots, dim=1),
        torch.cat(next_vertices, dim=0),
    )



def build_step_attention_mask(
    *,
    root_positions: torch.Tensor,           # [B, Q]
    query_vertex_ids: torch.Tensor,        # [Q]
    tree_root_positions: torch.Tensor,     # [B, K_tree]
    tree_vertex_ids: torch.Tensor,         # [B, K_tree]
    document_mask: torch.Tensor,           # [B, S]
    valid_tokens: torch.Tensor,            # [B, S]
    ancestor_map: torch.Tensor,            # [st_size, st_size]
    ctx_len: int,
    use_flex: bool,
    flex_mask_builder: Callable | None = None,
):
    if use_flex:
        builder = flex_mask_builder or _build_flex_block_mask
        return builder(
            root_positions=root_positions,
            query_vertex_ids=query_vertex_ids,
            tree_root_positions=tree_root_positions,
            tree_vertex_ids=tree_vertex_ids,
            document_mask=document_mask,
            valid_tokens=valid_tokens,
            ancestor_map=ancestor_map,
            ctx_len=ctx_len,
        )
    return _build_dense_step_attention_mask(
        root_positions=root_positions,
        query_vertex_ids=query_vertex_ids,
        tree_root_positions=tree_root_positions,
        tree_vertex_ids=tree_vertex_ids,
        document_mask=document_mask,
        valid_tokens=valid_tokens,
        ancestor_map=ancestor_map,
        ctx_len=ctx_len,
    )


@torch.inference_mode()
def generate_trees(
    batch,
    model,
    n_subtrees: int,
    st_info: SubTreeInfo,
    logit_chunk_size: int = 128,
    runtime: Stage2Runtime | None = None,
    profile: dict[str, float] | None = None,
):
    input_ids = batch["input_ids"]  # [B, S]
    is_response = batch["is_response"]  # [B, S]
    document_mask = batch["document_mask"]  # [B, S], -1 for padding
    B, S = input_ids.shape
    device = input_ids.device
    valid_tokens = document_mask >= 0
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise ValueError("Model must expose output embeddings via get_output_embeddings()")

    subtree_ids = torch.full(
        (B, S, st_info.size),
        fill_value=IGNORE_IDX,
        dtype=torch.long,
        device=device,
    )
    subtree_ar_probs = torch.zeros((B, S, st_info.size), dtype=torch.float32, device=device)
    subtree_ids[:, :, 0] = torch.where(valid_tokens, input_ids, torch.full_like(input_ids, IGNORE_IDX))

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
    vocab_size = getattr(model.config, "vocab_size", None)
    if vocab_size is None:
        if not hasattr(lm_head, "weight"):
            raise ValueError("Could not infer vocab size from model config or LM head weights")
        vocab_size = lm_head.weight.shape[0]
    use_flex = getattr(getattr(model, "config", None), "_attn_implementation", None) == "flex_attention"
    depth1_slots = st_info.nodes_at_depth.get(1, [])
    request_k = min(len(depth1_slots) + 1, vocab_size) if depth1_slots else 0
    log_denom, token_probs, cand_vals, cand_ids = _summarize_initial_pass(
        hidden_states=hidden_states,
        input_ids=input_ids,
        valid_tokens=valid_tokens,
        lm_head=lm_head,
        vocab_size=request_k,
        logit_chunk_size=logit_chunk_size,
        profile=profile,
    )

    subtree_ar_probs[:, :, 0] = torch.where(valid_tokens, token_probs, torch.zeros_like(token_probs))

    valid_response = is_response & valid_tokens
    subtree_anchors = _select_subtree_anchors(
        valid_response=valid_response,
        token_probs=token_probs,
        n_subtrees=n_subtrees,
    )
    _fill_depth1_subtrees(
        subtree_ids=subtree_ids,
        subtree_ar_probs=subtree_ar_probs,
        cand_vals=cand_vals,
        cand_ids=cand_ids,
        log_denom=log_denom,
        input_ids=input_ids,
        valid_tokens=valid_tokens,
        depth1_slots=depth1_slots,
    )

    ancestor_map = st_info.ancestor_map.to(device)
    current_depth = 1
    current_vertices_list = st_info.non_leaf_at_depth.get(current_depth, [])
    if subtree_anchors.shape[1] > 0 and current_vertices_list:
        current_vertices = torch.tensor(current_vertices_list, dtype=torch.long, device=device)
        anchor_idx = subtree_anchors.unsqueeze(-1).expand(B, subtree_anchors.shape[1], len(current_vertices_list))
        vertex_idx = current_vertices.view(1, 1, -1).expand(B, subtree_anchors.shape[1], -1)
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand_as(anchor_idx)
        next_input_ids = subtree_ids[batch_idx, anchor_idx, vertex_idx].reshape(B, -1)
        root_positions = anchor_idx.reshape(B, -1)
        vertex_ids = current_vertices.repeat(subtree_anchors.shape[1])
        position_ids = root_positions + current_depth
        cached_root_positions: torch.Tensor | None = None
        cached_vertex_ids: torch.Tensor | None = None

        while next_input_ids.shape[1] > 0:
            current_vertex_ids = vertex_ids.unsqueeze(0).expand(B, -1)
            if cached_root_positions is None:
                all_tree_root_positions = root_positions
                all_tree_vertex_ids = current_vertex_ids
            else:
                all_tree_root_positions = torch.cat([cached_root_positions, root_positions], dim=1)
                all_tree_vertex_ids = torch.cat([cached_vertex_ids, current_vertex_ids], dim=1)
            mask_start = time.perf_counter()
            attention_mask = build_step_attention_mask(
                root_positions=root_positions,
                query_vertex_ids=vertex_ids,
                tree_root_positions=all_tree_root_positions,
                tree_vertex_ids=all_tree_vertex_ids,
                document_mask=document_mask,
                valid_tokens=valid_tokens,
                ancestor_map=ancestor_map,
                ctx_len=S,
                use_flex=use_flex,
                flex_mask_builder=runtime.flex_mask_builder if runtime is not None else None,
            )
            _accumulate_profile(profile, "mask_build_s", mask_start)

            base_out = _run_base_model_forward(
                base_model_forward,
                profile=profile,
                profile_key="tree_forward_s",
                input_ids=next_input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=kv_cache,
                use_cache=True,
                kernel_options={
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "BLOCK_M1": 32,
                    "BLOCK_N1": 64,
                    "BLOCK_M2": 64,
                    "BLOCK_N2": 32,
                }
            )
            new_hidden_states, kv_cache = _extract_hidden_and_cache(base_out)
            cached_root_positions = all_tree_root_positions
            cached_vertex_ids = all_tree_vertex_ids

            next_input_ids, root_positions, vertex_ids = _write_children_and_build_frontier(
                new_hidden_states=new_hidden_states,
                root_positions=root_positions,
                vertex_ids=vertex_ids,
                lm_head=lm_head,
                st_info=st_info,
                current_depth=current_depth,
                logit_chunk_size=logit_chunk_size,
                subtree_ids=subtree_ids,
                subtree_ar_probs=subtree_ar_probs,
                device=device,
                profile=profile,
            )
            if next_input_ids is None or root_positions is None or vertex_ids is None:
                break

            current_depth += 1
            position_ids = root_positions + current_depth

    subtree_ids = subtree_ids.masked_fill(~valid_tokens.unsqueeze(-1), IGNORE_IDX)
    subtree_ar_probs = subtree_ar_probs * valid_tokens.unsqueeze(-1)
    batch["subtree_ids"] = subtree_ids
    batch["subtree_ar_probs"] = subtree_ar_probs
    return batch


def load_tokenized_records(
    data_dir: Path,
    tokenizer,
    max_len: int | None,
    *,
    sort_descending: bool,
    num_proc: int | None = None,
) -> Dataset:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Stage 2 now requires Hugging Face `datasets` for JSONL loading, tokenization, and filtering."
        ) from exc

    files = sorted(data_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in {data_dir}")
    dataset = load_dataset(
        "json",
        data_files=[str(path) for path in files],
        split="train",
    )

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]] | list[int]]:
        prompt_ids = tokenizer(batch["prompt"], add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(batch["response"], add_special_tokens=False)["input_ids"]
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
        keep = [len(response) > 0 for response in response_ids]
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
    def assign_record_idx(batch, indices: list[int]) -> dict[str, list[int]]:
        return {"record_idx": indices}

    return records.map(
        assign_record_idx,
        batched=True,
        with_indices=True,
        desc="Assigning stable record indices",
    )


def shard_records_for_rank(records: Dataset, ctx: DistributedContext) -> Dataset:
    if not ctx.is_distributed:
        return records
    return records.shard(num_shards=ctx.world_size, index=ctx.rank, contiguous=False)


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
        raise RuntimeError(
            f"Could not prepare temporary Stage 2 parts directory: {parts_dir}"
        )


def initialize_stage2_hdf5(
    hf: h5py.File,
    *,
    st_info: SubTreeInfo,
    sub_tree_paths: Sequence[str],
    include_record_idx: bool,
) -> None:
    vlen_int64 = h5py.vlen_dtype(np.int64)
    if include_record_idx:
        hf.create_dataset("record_idx", shape=(0,), maxshape=(None,), dtype="int64")
    hf.create_dataset("prompt_ids", shape=(0,), maxshape=(None,), dtype=vlen_int64)
    hf.create_dataset("response_ids", shape=(0,), maxshape=(None,), dtype=vlen_int64)
    hf.create_dataset(
        "sub_trees",
        shape=(0, st_info.size),
        maxshape=(None, st_info.size),
        dtype="int64",
        chunks=(512, st_info.size),
        compression="lzf",
    )
    hf.create_dataset(
        "sub_trees_ar_probs",
        shape=(0, st_info.size),
        maxshape=(None, st_info.size),
        dtype="float32",
        chunks=(512, st_info.size),
        compression="lzf",
    )
    hf.create_dataset("sequence_offsets", shape=(1,), maxshape=(None,), dtype="int64")
    hf["sequence_offsets"][0] = 0
    hf.attrs["sub_tree_paths"] = np.array(sub_tree_paths, dtype=h5py.string_dtype())


def build_batch(
    examples: Sequence[tuple[list[int], list[int]]],
    pad_token_id: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    max_len = max(len(prompt_ids) + len(response_ids) for prompt_ids, response_ids in examples)
    max_len = math.ceil(max_len / 128) * 128
    B = len(examples)

    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    is_response = torch.zeros((B, max_len), dtype=torch.bool)
    document_mask = torch.full((B, max_len), -1, dtype=torch.long)

    for row, (prompt_ids, response_ids) in enumerate(examples):
        full_ids = prompt_ids + response_ids
        seq_len = len(full_ids)
        prompt_len = len(prompt_ids)
        input_ids[row, :seq_len] = torch.tensor(full_ids, dtype=torch.long)
        is_response[row, prompt_len:seq_len] = True
        document_mask[row, :seq_len] = 0

    return {
        "input_ids": input_ids.to(device),
        "is_response": is_response.to(device),
        "document_mask": document_mask.to(device),
    }


def flush_hdf5(
    hf: h5py.File,
    prompt_buf: list[np.ndarray],
    response_buf: list[np.ndarray],
    tree_buf: list[np.ndarray],
    prob_buf: list[np.ndarray],
    record_idx_buf: list[int] | None,
    n_seqs_written: int,
    n_rows_written: int,
) -> tuple[int, int]:
    if not prompt_buf:
        return n_seqs_written, n_rows_written

    ds_record_idx = hf["record_idx"] if record_idx_buf is not None else None
    ds_prompt = hf["prompt_ids"]
    ds_response = hf["response_ids"]
    ds_trees = hf["sub_trees"]
    ds_probs = hf["sub_trees_ar_probs"]
    ds_offsets = hf["sequence_offsets"]

    n_new_seqs = len(prompt_buf)
    new_seq_total = n_seqs_written + n_new_seqs
    if ds_record_idx is not None:
        ds_record_idx.resize(new_seq_total, axis=0)
        ds_record_idx[n_seqs_written:new_seq_total] = np.asarray(record_idx_buf, dtype=np.int64)
    ds_prompt.resize(new_seq_total, axis=0)
    ds_response.resize(new_seq_total, axis=0)
    for idx, arr in enumerate(prompt_buf):
        ds_prompt[n_seqs_written + idx] = arr
    for idx, arr in enumerate(response_buf):
        ds_response[n_seqs_written + idx] = arr

    combined_trees = np.concatenate(tree_buf, axis=0)
    combined_probs = np.concatenate(prob_buf, axis=0)
    new_rows = combined_trees.shape[0]
    ds_trees.resize(n_rows_written + new_rows, axis=0)
    ds_probs.resize(n_rows_written + new_rows, axis=0)
    ds_trees[n_rows_written : n_rows_written + new_rows] = combined_trees
    ds_probs[n_rows_written : n_rows_written + new_rows] = combined_probs

    offsets = [n_rows_written]
    for arr in tree_buf:
        offsets.append(offsets[-1] + arr.shape[0])
    ds_offsets.resize(new_seq_total + 1, axis=0)
    ds_offsets[n_seqs_written + 1 : new_seq_total + 1] = np.array(offsets[1:], dtype=np.int64)

    prompt_buf.clear()
    response_buf.clear()
    tree_buf.clear()
    prob_buf.clear()
    if record_idx_buf is not None:
        record_idx_buf.clear()
    return new_seq_total, n_rows_written + new_rows


def collect_merge_manifest(part_paths: Sequence[Path]) -> list[HDF5MergeEntry]:
    manifest: list[HDF5MergeEntry] = []
    for part_path in part_paths:
        with h5py.File(part_path, "r") as hf:
            offsets = hf["sequence_offsets"][:]
            record_idx = hf["record_idx"][:]
            if offsets.shape[0] != record_idx.shape[0] + 1:
                raise ValueError(
                    f"Mismatched sequence_offsets and record_idx lengths in {part_path}: "
                    f"{offsets.shape[0]} vs {record_idx.shape[0]}"
                )
            for seq_idx, seq_record_idx in enumerate(record_idx.tolist()):
                manifest.append(
                    HDF5MergeEntry(
                        record_idx=int(seq_record_idx),
                        part_path=str(part_path),
                        seq_idx=seq_idx,
                        row_start=int(offsets[seq_idx]),
                        row_end=int(offsets[seq_idx + 1]),
                    )
                )
    manifest.sort(key=lambda entry: entry.record_idx)
    return manifest


def merge_hdf5_parts(
    *,
    part_paths: Sequence[Path],
    output_path: Path,
    st_info: SubTreeInfo,
    sub_tree_paths: Sequence[str],
    log_fn: Callable[[str], None],
) -> tuple[int, int]:
    manifest = collect_merge_manifest(part_paths)
    prompt_buf: list[np.ndarray] = []
    response_buf: list[np.ndarray] = []
    tree_buf: list[np.ndarray] = []
    prob_buf: list[np.ndarray] = []
    flush_every = 128

    with ExitStack() as stack:
        part_handles = {
            str(path): stack.enter_context(h5py.File(path, "r"))
            for path in part_paths
        }
        with h5py.File(output_path, "w") as hf:
            initialize_stage2_hdf5(
                hf,
                st_info=st_info,
                sub_tree_paths=sub_tree_paths,
                include_record_idx=False,
            )

            n_seqs_written = 0
            n_rows_written = 0
            for entry in manifest:
                src = part_handles[entry.part_path]
                prompt_buf.append(np.asarray(src["prompt_ids"][entry.seq_idx], dtype=np.int64))
                response_buf.append(np.asarray(src["response_ids"][entry.seq_idx], dtype=np.int64))
                tree_buf.append(
                    np.asarray(src["sub_trees"][entry.row_start:entry.row_end], dtype=np.int64)
                )
                prob_buf.append(
                    np.asarray(src["sub_trees_ar_probs"][entry.row_start:entry.row_end], dtype=np.float32)
                )

                if len(prompt_buf) >= flush_every:
                    n_seqs_written, n_rows_written = flush_hdf5(
                        hf,
                        prompt_buf,
                        response_buf,
                        tree_buf,
                        prob_buf,
                        record_idx_buf=None,
                        n_seqs_written=n_seqs_written,
                        n_rows_written=n_rows_written,
                    )
                    log_fn(f"Merged sequences: {n_seqs_written}  Response rows: {n_rows_written}")

            n_seqs_written, n_rows_written = flush_hdf5(
                hf,
                prompt_buf,
                response_buf,
                tree_buf,
                prob_buf,
                record_idx_buf=None,
                n_seqs_written=n_seqs_written,
                n_rows_written=n_rows_written,
            )
    return n_seqs_written, n_rows_written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 2 v2: batched subtree generation with KV caching.")
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument("--data-dir", required=True, help="Directory of stage-1 JSONL shards")
    parser.add_argument("--output", required=True, help="Output HDF5 path")
    parser.add_argument(
        "--sub-tree-paths",
        nargs="+",
        default=DEFAULT_SUB_TREE_PATHS,
        help='Subtree edges as "X-Y" strings',
    )
    parser.add_argument("--n-subtrees", type=int, default=512, help="Anchor positions per sequence")
    parser.add_argument("--batch-size", type=int, default=1, help="Sequences per batch")
    parser.add_argument("--max-sequences", type=int, default=None, help="Optional cap on sequences")
    parser.add_argument(
        "--attn-implementation",
        choices=["flex_attention", "sdpa"],
        default="flex_attention",
        help="Attention backend for the HF model",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model load dtype",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="Optional max prompt+response length; longer ones are skipped",
    )
    parser.add_argument(
        "--logit-chunk-size",
        type=int,
        default=128,
        help="Project at most this many positions through the LM head at once; <= 0 disables chunking.",
    )
    parser.add_argument("--compile", action="store_true", help="Compile supported Stage 2 flex-attention helpers.")
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
        help="torch.compile mode for supported Stage 2 helpers.",
    )
    parser.add_argument(
        "--profile-every",
        type=int,
        default=0,
        help="Print lightweight timing stats every N processed batches; 0 disables profiling logs.",
    )
    parser.add_argument(
        "--dataset-num-proc",
        type=int,
        default=None,
        help="Optional `datasets` worker count for tokenization/filtering.",
    )
    return parser


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = build_arg_parser()
    args = parser.parse_args()
    ctx = init_distributed_context()
    try:
        if args.batch_size > 1:
            ctx.log(
                "Note: batch_size > 1 uses padded sequence batches; anchor count is clipped "
                "to the minimum valid response-token count in the batch."
            )

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[args.dtype]

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
        runtime = build_stage2_runtime(
            model,
            compile_enabled=args.compile,
            compile_mode=args.compile_mode,
            log_enabled=ctx.is_primary,
        )

        st_info = SubTreeInfo(args.sub_tree_paths)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        part_path = output_path
        include_record_idx = False
        if ctx.is_distributed:
            parts_dir = build_parts_dir(output_path)
            prepare_parts_dir(parts_dir, ctx)
            part_path = build_rank_part_path(output_path, ctx.rank)
            include_record_idx = True

        prompt_buf: list[np.ndarray] = []
        response_buf: list[np.ndarray] = []
        tree_buf: list[np.ndarray] = []
        prob_buf: list[np.ndarray] = []
        record_idx_buf: list[int] | None = [] if include_record_idx else None
        pending: list[tuple[int, list[int], list[int]]] = []
        flush_every = 128
        batch_count = 0
        timings: dict[str, float] = {}

        with h5py.File(part_path, "w") as hf:
            initialize_stage2_hdf5(
                hf,
                st_info=st_info,
                sub_tree_paths=args.sub_tree_paths,
                include_record_idx=include_record_idx,
            )

            n_seqs_written = 0
            n_rows_written = 0

            dataset_prep_start = time.perf_counter()
            records = load_tokenized_records(
                Path(args.data_dir),
                tokenizer,
                args.max_len,
                sort_descending=True,
                num_proc=args.dataset_num_proc,
            )
            if args.max_sequences is not None:
                records = records.select(range(min(args.max_sequences, len(records))))
            records = add_stable_record_idx(records)
            records = shard_records_for_rank(records, ctx)
            _accumulate_profile(timings, "dataset_prep_s", dataset_prep_start)

            for record in records:
                record_idx = int(record["record_idx"])
                prompt_ids = record["prompt_ids"]
                response_ids = record["response_ids"]

                pending.append((record_idx, prompt_ids, response_ids))
                if len(pending) < args.batch_size:
                    continue

                build_batch_start = time.perf_counter()
                batch = build_batch(
                    [(prompt_ids_row, response_ids_row) for _, prompt_ids_row, response_ids_row in pending],
                    tokenizer.pad_token_id,
                    ctx.device,
                )
                _accumulate_profile(timings, "build_batch_s", build_batch_start)
                batch = generate_trees(
                    batch,
                    model,
                    args.n_subtrees,
                    st_info,
                    logit_chunk_size=args.logit_chunk_size,
                    runtime=runtime,
                    profile=timings,
                )
                batch_count += 1

                for row, (record_idx_row, prompt_ids_row, response_ids_row) in enumerate(pending):
                    prompt_len = len(prompt_ids_row)
                    response_len = len(response_ids_row)
                    seq_slice = slice(prompt_len, prompt_len + response_len)
                    subtrees = batch["subtree_ids"][row, seq_slice].detach().cpu().numpy().astype(np.int64, copy=False)
                    subtree_probs = (
                        batch["subtree_ar_probs"][row, seq_slice].detach().cpu().numpy().astype(np.float32, copy=False)
                    )

                    if record_idx_buf is not None:
                        record_idx_buf.append(record_idx_row)
                    prompt_buf.append(np.asarray(prompt_ids_row, dtype=np.int64))
                    response_buf.append(np.asarray(response_ids_row, dtype=np.int64))
                    tree_buf.append(subtrees)
                    prob_buf.append(subtree_probs)

                pending.clear()

                if len(prompt_buf) >= flush_every:
                    flush_start = time.perf_counter()
                    n_seqs_written, n_rows_written = flush_hdf5(
                        hf,
                        prompt_buf,
                        response_buf,
                        tree_buf,
                        prob_buf,
                        record_idx_buf=record_idx_buf,
                        n_seqs_written=n_seqs_written,
                        n_rows_written=n_rows_written,
                    )
                    _accumulate_profile(timings, "flush_s", flush_start)
                    ctx.log(f"Sequences: {n_seqs_written}  Response rows: {n_rows_written}")

                if args.profile_every > 0 and batch_count % args.profile_every == 0 and ctx.is_primary:
                    ctx.log(
                        f"[profile] batches={batch_count} "
                        f"dataset_prep={timings.get('dataset_prep_s', 0.0):.3f}s "
                        f"build_batch={timings.get('build_batch_s', 0.0):.3f}s "
                        f"initial_forward={timings.get('initial_forward_s', 0.0):.3f}s "
                        f"initial_summary={timings.get('initial_summary_s', 0.0):.3f}s "
                        f"mask_build={timings.get('mask_build_s', 0.0):.3f}s "
                        f"tree_forward={timings.get('tree_forward_s', 0.0):.3f}s "
                        f"tree_writeback={timings.get('tree_writeback_s', 0.0):.3f}s "
                        f"flush={timings.get('flush_s', 0.0):.3f}s"
                    )
                    timings.clear()

            if pending:
                build_batch_start = time.perf_counter()
                batch = build_batch(
                    [(prompt_ids_row, response_ids_row) for _, prompt_ids_row, response_ids_row in pending],
                    tokenizer.pad_token_id,
                    ctx.device,
                )
                _accumulate_profile(timings, "build_batch_s", build_batch_start)
                batch = generate_trees(
                    batch,
                    model,
                    args.n_subtrees,
                    st_info,
                    logit_chunk_size=args.logit_chunk_size,
                    runtime=runtime,
                    profile=timings,
                )

                for row, (record_idx_row, prompt_ids_row, response_ids_row) in enumerate(pending):
                    prompt_len = len(prompt_ids_row)
                    response_len = len(response_ids_row)
                    seq_slice = slice(prompt_len, prompt_len + response_len)
                    subtrees = batch["subtree_ids"][row, seq_slice].detach().cpu().numpy().astype(np.int64, copy=False)
                    subtree_probs = (
                        batch["subtree_ar_probs"][row, seq_slice].detach().cpu().numpy().astype(np.float32, copy=False)
                    )

                    if record_idx_buf is not None:
                        record_idx_buf.append(record_idx_row)
                    prompt_buf.append(np.asarray(prompt_ids_row, dtype=np.int64))
                    response_buf.append(np.asarray(response_ids_row, dtype=np.int64))
                    tree_buf.append(subtrees)
                    prob_buf.append(subtree_probs)

            n_seqs_written, n_rows_written = flush_hdf5(
                hf,
                prompt_buf,
                response_buf,
                tree_buf,
                prob_buf,
                record_idx_buf=record_idx_buf,
                n_seqs_written=n_seqs_written,
                n_rows_written=n_rows_written,
            )

        if ctx.is_distributed:
            ctx.barrier()
            if ctx.is_primary:
                part_paths = [build_rank_part_path(output_path, rank) for rank in range(ctx.world_size)]
                ctx.log(f"Merging {len(part_paths)} Stage 2 rank shards into {output_path}")
                final_seqs_written, final_rows_written = merge_hdf5_parts(
                    part_paths=part_paths,
                    output_path=output_path,
                    st_info=st_info,
                    sub_tree_paths=args.sub_tree_paths,
                    log_fn=ctx.log,
                )
                shutil.rmtree(build_parts_dir(output_path))
                ctx.log(f"Done. Sequences: {final_seqs_written}  Response rows: {final_rows_written}")
        else:
            ctx.log(f"Done. Sequences: {n_seqs_written}  Response rows: {n_rows_written}")
    finally:
        ctx.shutdown()


if __name__ == "__main__":
    main()
