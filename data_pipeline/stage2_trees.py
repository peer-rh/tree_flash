"""
Stage 2: Generate per-position subtrees for tree-flash training data.

Input
-----
A directory of JSONL files.  Each line must contain:
    {"prompt": "...", "response": "..."}

Output
------
A single HDF5 file with the following datasets:

    prompt_ids               : vlen int64   [N]          — tokenized prompt per sequence
    response_ids             : vlen int64   [N]          — tokenized response per sequence
    response_probs           : float32      [T]          — p(response[t] | context[:t])
                                                           for each response position t;
                                                           one entry per response token,
                                                           all sequences concatenated
    continuation_trees       : int64        [T, subtree_size] — subtree token ids
    continuation_trees_probs : float32      [T, subtree_size] — individual AR probs
    sequence_offsets         : int64        [N+1]        — row pointers into the above

For sequence n, its data lives at rows sequence_offsets[n] : sequence_offsets[n+1].
Each row corresponds to one response position t (0..S_R-1).

Only ``num_trees_per_seq`` rows per sequence are filled with real subtree data;
the rest contain IGNORE_IDX = -1 (and 0.0 for probs).  response_probs is always
populated for all response positions regardless of selection.

Subtree specification
---------------------
The subtree is specified by --sub-tree-paths, mirroring TrainConfig.sub_tree_paths.
subtree_size = number of non-root subtree nodes.

What the subtree stores at position t
--------------------------------------
The subtree at position t provides alternatives that **diverge** from the response
at position t+1.  It does NOT store the response continuation itself.

Subtree node layout (0-indexed within the subtree's subtree_size slots):
  depth-1 nodes (direct children of attachment point):
      top-k most likely tokens after response[t], EXCLUDING response[t+1]
  depth-2 nodes (children of depth-1 nodes):
      top-k most likely tokens after the depth-1 parent token (via extra forward pass)

Primary path (response[t : t+n_subtrees]) is assembled at training time from
response_ids and does not need to be stored here.

Position selection
------------------
For each response position t, the "value" of a subtree is
    1 - p_target(response[t] | prefix before t)
The top ``num_trees_per_seq`` positions by descending uncertainty are selected.
All others have IGNORE_IDX / 0.0 in continuation_trees / continuation_trees_probs,
but still have a valid entry in response_probs.

Valid range for selection: 0 <= t < S_R (any response position).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

IGNORE_IDX: int = -1


# ---------------------------------------------------------------------------
# Subtree structure helper
# ---------------------------------------------------------------------------

@dataclass
class SubTreeStructure:
    """
    Parsed subtree structure for tree generation.

    Derived from the --sub-tree-paths edge list (same format as TreeSpec).

    Attributes
    ----------
    non_root_nodes  : sorted list of non-zero subtree node indices
    parent_map      : {child: parent}  (0-indexed, 0 = attachment root)
    local_idx       : {subtree_node: 0-based rank in non_root_nodes}
    subtree_size    : len(non_root_nodes)
    depth1_nodes    : subtree nodes at depth 1 (direct children of root=0)
    depth2_by_d1    : {depth-1 node: sorted list of depth-2 children}
    max_depth       : maximum depth in the subtree
    """
    non_root_nodes: list[int]
    parent_map: dict[int, int]
    local_idx: dict[int, int]
    subtree_size: int
    depth1_nodes: list[int]
    depth2_by_d1: dict[int, list[int]]
    max_depth: int


def parse_sub_tree_structure(sub_tree_paths: list[str]) -> SubTreeStructure:
    """
    Parse sub_tree_paths (e.g. ["0-1","0-2","1-4"]) into SubTreeStructure.

    Compatible with tree/spec.py's _parse_sub_tree format.
    """
    parent_map: dict[int, int] = {}
    all_nodes: set[int] = {0}
    for path in sub_tree_paths:
        parts = path.split("-")
        assert len(parts) == 2, f"subtree edge must be 'X-Y', got {path!r}"
        p, c = int(parts[0]), int(parts[1])
        assert c not in parent_map, f"node {c} has duplicate parent in subtree"
        parent_map[c] = p
        all_nodes |= {p, c}

    non_root_nodes = sorted(all_nodes - {0})
    local_idx = {n: j for j, n in enumerate(non_root_nodes)}
    subtree_size = len(non_root_nodes)

    depth1_nodes = sorted(n for n, p in parent_map.items() if p == 0)

    depth2_by_d1: dict[int, list[int]] = defaultdict(list)
    depth1_set = set(depth1_nodes)
    for n, p in parent_map.items():
        if p in depth1_set:
            depth2_by_d1[p].append(n)
    for k in depth2_by_d1:
        depth2_by_d1[k].sort()

    # Compute max depth in subtree
    def _depth(node: int) -> int:
        d = 0
        cur = node
        while cur in parent_map:
            cur = parent_map[cur]
            d += 1
        return d

    max_depth = max((_depth(n) for n in non_root_nodes), default=0)

    return SubTreeStructure(
        non_root_nodes=non_root_nodes,
        parent_map=parent_map,
        local_idx=local_idx,
        subtree_size=subtree_size,
        depth1_nodes=depth1_nodes,
        depth2_by_d1=dict(depth2_by_d1),
        max_depth=max_depth,
    )


# ---------------------------------------------------------------------------
# Uncertainty scoring
# ---------------------------------------------------------------------------

@torch.inference_mode()
def compute_token_probs(
    model: AutoModelForCausalLM,
    full_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    """
    Run a single forward pass over full_ids and return logits [len(full_ids), V].

    The logit at position k predicts the token at position k+1.
    """
    input_tensor = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
    out = model(input_tensor, use_cache=False)
    return out.logits[0]  # [len(full_ids), V]


def compute_response_probs(
    base_logits: torch.Tensor,  # [S_P + S_R, V]
    response_ids: list[int],
    prompt_len: int,
) -> list[float]:
    """
    Compute p(response[t] | context[:t]) for all t in 0..S_R-1.

    Returns a list of length S_R.
    """
    S_R = len(response_ids)
    probs = []
    for t in range(S_R):
        t_abs = prompt_len + t
        logit_pos = t_abs - 1
        if logit_pos < 0:
            probs.append(1.0)
        else:
            lp = F.log_softmax(base_logits[logit_pos], dim=-1)
            probs.append(math.exp(float(lp[response_ids[t]].item())))
    return probs


def select_positions(
    response_probs: list[float],
    num_trees_per_seq: int,
) -> list[int]:
    """
    Select the top ``num_trees_per_seq`` response positions by descending
    uncertainty = 1 - p_target(response[t] | prefix before t).

    Valid range: all positions 0 <= t < S_R.

    Parameters
    ----------
    response_probs    : list of p(response[t]) for t in 0..S_R-1
    num_trees_per_seq : number of positions to select

    Returns
    -------
    selected : sorted list of selected response-relative positions (0-indexed)
    """
    scores = [(p, t) for t, p in enumerate(response_probs)]
    # Sort ascending by p (lowest prob = highest uncertainty = most valuable)
    scores.sort(key=lambda x: x[0])
    n_select = min(num_trees_per_seq, len(scores))
    selected = sorted(t for _, t in scores[:n_select])
    return selected


# ---------------------------------------------------------------------------
# Per-position subtree generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_subtree(
    model: AutoModelForCausalLM,
    full_ids: list[int],
    t: int,
    response_ids: list[int],
    prompt_len: int,
    sub_tree: SubTreeStructure,
    base_logits: torch.Tensor,  # [S_P + S_R, V]
    device: torch.device,
) -> tuple[list[int], list[float]]:
    """
    Generate subtree_tokens and subtree_probs for response position t.

    The subtree provides alternatives that diverge from the response at t+1.
    Only the first (depth-1) nodes need to exclude response[t+1]; deeper
    nodes sample top-k freely.

    Subtree node layout
    -------------------
    local_idx 0..subtree_size-1 follow the rank ordering of SubTreeStructure.non_root_nodes.
    Depth-1 nodes (children of attachment point 0):   top-k excluding response[t+1]
    Depth-2 nodes (children of depth-1 parents):      top-k from an extra forward pass

    Returns
    -------
    subtree_tokens : list[int]   length subtree_size (IGNORE_IDX for unfilled nodes)
    subtree_probs  : list[float] length subtree_size (individual AR probs; 0.0 for unfilled)
    """
    subtree_size = sub_tree.subtree_size
    subtree_tokens: list[int] = [IGNORE_IDX] * subtree_size
    subtree_probs: list[float] = [0.0] * subtree_size

    t_abs = prompt_len + t  # absolute position of response[t] in full_ids

    # Logits for what follows response[t] = base_logits[t_abs]
    # (logit at t_abs predicts position t_abs+1)
    if t_abs >= base_logits.shape[0]:
        return subtree_tokens, subtree_probs

    logits_at_t = base_logits[t_abs]    # [V]
    log_probs_at_t = F.log_softmax(logits_at_t, dim=-1)

    # Exclude the next response token (already captured by the primary path)
    response_next = response_ids[t + 1] if t + 1 < len(response_ids) else -1

    # ── Depth-1 subtree nodes ──────────────────────────────────────────────
    k_d1 = len(sub_tree.depth1_nodes)
    # Over-request by 1 so we can skip the excluded token
    topk_lp, topk_ids = torch.topk(
        log_probs_at_t, k=min(k_d1 + 1, log_probs_at_t.shape[0])
    )

    d1_tokens: list[int] = []
    d1_lps: list[float] = []
    for tok, lp in zip(topk_ids.tolist(), topk_lp.tolist()):
        if tok == response_next:
            continue
        d1_tokens.append(tok)
        d1_lps.append(lp)
        if len(d1_tokens) >= k_d1:
            break

    # Fill depth-1 nodes
    for rank, sub_node in enumerate(sub_tree.depth1_nodes):
        if rank >= len(d1_tokens):
            break
        idx = sub_tree.local_idx[sub_node]
        subtree_tokens[idx] = d1_tokens[rank]
        subtree_probs[idx] = math.exp(d1_lps[rank])

    # ── Depth-2 subtree nodes ──────────────────────────────────────────────
    # For each depth-1 parent with children, run a forward pass on
    # context[0..t_abs] + [d1_token] to predict depth-2 tokens.
    for d1_rank, d1_sub_node in enumerate(sub_tree.depth1_nodes):
        if d1_sub_node not in sub_tree.depth2_by_d1:
            continue
        if d1_rank >= len(d1_tokens):
            continue  # depth-1 node itself wasn't filled

        d2_children = sub_tree.depth2_by_d1[d1_sub_node]
        d1_token = d1_tokens[d1_rank]

        # Forward pass: context includes everything up to and including
        # response[t], then the depth-1 alternative token.
        context_ext = full_ids[: t_abs + 1] + [d1_token]
        ext_tensor = torch.tensor(
            context_ext, dtype=torch.long, device=device
        ).unsqueeze(0)
        ext_out = model(ext_tensor, use_cache=False)
        ext_logits = ext_out.logits[0, -1, :]       # [V]
        ext_log_probs = F.log_softmax(ext_logits, dim=-1)

        k_d2 = len(d2_children)
        topk2_lp, topk2_ids = torch.topk(ext_log_probs, k=k_d2)

        for child_rank, d2_sub_node in enumerate(d2_children):
            idx = sub_tree.local_idx[d2_sub_node]
            subtree_tokens[idx] = int(topk2_ids[child_rank].item())
            subtree_probs[idx] = math.exp(float(topk2_lp[child_rank].item()))

    return subtree_tokens, subtree_probs


# ---------------------------------------------------------------------------
# Per-sequence processing
# ---------------------------------------------------------------------------

@torch.inference_mode()
def process_sequence(
    model: AutoModelForCausalLM,
    prompt_ids: list[int],
    response_ids: list[int],
    sub_tree: SubTreeStructure,
    num_trees_per_seq: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """
    Process one (prompt, response) pair.

    Returns
    -------
    subtrees       : int64   [S_R, subtree_size] — IGNORE_IDX for non-selected positions
    subtree_probs  : float32 [S_R, subtree_size] — 0.0 for non-selected positions
    resp_probs     : float32 [S_R]               — p(response[t] | context[:t]) for all t
    selected       : list[int]                   — response-relative positions selected
    """
    S_R = len(response_ids)
    subtree_size = sub_tree.subtree_size
    subtrees = np.full((S_R, subtree_size), IGNORE_IDX, dtype=np.int64)
    subtree_probs_arr = np.zeros((S_R, subtree_size), dtype=np.float32)

    full_ids = prompt_ids + response_ids
    prompt_len = len(prompt_ids)

    # Single forward pass for all logits
    base_logits = compute_token_probs(model, full_ids, device)
    # base_logits: [S_P + S_R, V]

    # Response probs for all positions (needed for primary path cumprod in training)
    resp_probs_list = compute_response_probs(base_logits, response_ids, prompt_len)
    resp_probs = np.array(resp_probs_list, dtype=np.float32)  # [S_R]

    # Select positions by uncertainty
    selected = select_positions(resp_probs_list, num_trees_per_seq)

    # Generate subtrees for selected positions
    for t in selected:
        tok_list, prob_list = generate_subtree(
            model=model,
            full_ids=full_ids,
            t=t,
            response_ids=response_ids,
            prompt_len=prompt_len,
            sub_tree=sub_tree,
            base_logits=base_logits,
            device=device,
        )
        subtrees[t] = tok_list
        subtree_probs_arr[t] = prob_list

    return subtrees, subtree_probs_arr, resp_probs, selected


# ---------------------------------------------------------------------------
# JSONL reader
# ---------------------------------------------------------------------------

def iter_jsonl_dir(data_dir: Path) -> Iterator[dict]:
    """Yield parsed JSON objects from all *.jsonl files in data_dir."""
    files = sorted(data_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in {data_dir}")
    for path in files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    sub_tree: SubTreeStructure,
    num_trees_per_seq: int,
    data_dir: Path,
    output_path: Path,
    device: torch.device,
    max_sequences: int | None,
) -> None:
    """
    Iterate over all JSONL records, generate subtrees, write to HDF5.

    HDF5 layout
    -----------
    prompt_ids               : vlen int64 [N]
    response_ids             : vlen int64 [N]
    response_probs           : float32 [T]             (flat, one entry per response token)
    continuation_trees       : int64 [T, subtree_size] (flat, all sequences concatenated)
    continuation_trees_probs : float32 [T, subtree_size]
    selected_positions       : vlen int64 [N]          (response-relative t indices)
    sequence_offsets         : int64 [N+1]

    For sequence n:  rows offsets[n]:offsets[n+1] cover all S_R response positions.
    """
    subtree_size = sub_tree.subtree_size
    vlen_int64 = h5py.vlen_dtype(np.int64)
    FLUSH_EVERY = 128

    with h5py.File(output_path, "w") as hf:
        ds_prompt = hf.create_dataset(
            "prompt_ids", shape=(0,), maxshape=(None,),
            dtype=vlen_int64,
        )
        ds_response = hf.create_dataset(
            "response_ids", shape=(0,), maxshape=(None,),
            dtype=vlen_int64,
        )
        ds_resp_probs = hf.create_dataset(
            "response_probs",
            shape=(0,), maxshape=(None,),
            dtype="float32", chunks=(4096,), compression="lzf",
        )
        ds_trees = hf.create_dataset(
            "continuation_trees",
            shape=(0, subtree_size), maxshape=(None, subtree_size),
            dtype="int64", chunks=(512, subtree_size), compression="lzf",
        )
        ds_probs = hf.create_dataset(
            "continuation_trees_probs",
            shape=(0, subtree_size), maxshape=(None, subtree_size),
            dtype="float32", chunks=(512, subtree_size), compression="lzf",
        )
        ds_offsets = hf.create_dataset(
            "sequence_offsets",
            shape=(1,), maxshape=(None,),
            dtype="int64",
        )
        ds_offsets[0] = 0
        ds_selected = hf.create_dataset(
            "selected_positions", shape=(0,), maxshape=(None,),
            dtype=vlen_int64,
        )

        # Buffers
        buf_prompt:      list[np.ndarray] = []
        buf_response:    list[np.ndarray] = []
        buf_resp_probs:  list[np.ndarray] = []   # each [S_R]
        buf_trees:       list[np.ndarray] = []   # each [S_R, subtree_size]
        buf_probs:       list[np.ndarray] = []
        buf_selected:    list[list[int]]  = []
        n_seqs_written = 0
        n_tokens_written = 0

        def _flush():
            nonlocal n_seqs_written, n_tokens_written
            if not buf_prompt:
                return
            n = len(buf_prompt)
            new_seq_size = n_seqs_written + n

            # vlen datasets
            ds_prompt.resize(new_seq_size, axis=0)
            ds_response.resize(new_seq_size, axis=0)
            ds_selected.resize(new_seq_size, axis=0)
            for k, (p, r) in enumerate(zip(buf_prompt, buf_response)):
                ds_prompt[n_seqs_written + k] = p
                ds_response[n_seqs_written + k] = r
            for k, sel in enumerate(buf_selected):
                ds_selected[n_seqs_written + k] = np.array(sel, dtype=np.int64)

            # Flat datasets (concatenate along axis 0)
            combined_resp_probs = np.concatenate(buf_resp_probs, axis=0)  # [sum(S_R)]
            combined_trees = np.concatenate(buf_trees, axis=0)            # [sum(S_R), subtree_size]
            combined_probs = np.concatenate(buf_probs, axis=0)
            total_new = combined_trees.shape[0]

            ds_resp_probs.resize(n_tokens_written + total_new, axis=0)
            ds_resp_probs[n_tokens_written: n_tokens_written + total_new] = combined_resp_probs

            ds_trees.resize(n_tokens_written + total_new, axis=0)
            ds_probs.resize(n_tokens_written + total_new, axis=0)
            ds_trees[n_tokens_written: n_tokens_written + total_new] = combined_trees
            ds_probs[n_tokens_written: n_tokens_written + total_new] = combined_probs

            # Append offsets (one per new sequence)
            new_offsets = np.array(
                [n_tokens_written + sum(arr.shape[0] for arr in buf_trees[:k+1])
                 for k in range(n)],
                dtype=np.int64,
            )
            ds_offsets.resize(new_seq_size + 1, axis=0)
            ds_offsets[n_seqs_written + 1: new_seq_size + 1] = new_offsets

            n_seqs_written = new_seq_size
            n_tokens_written += total_new
            buf_prompt.clear()
            buf_response.clear()
            buf_resp_probs.clear()
            buf_trees.clear()
            buf_probs.clear()
            buf_selected.clear()

        for record in iter_jsonl_dir(data_dir):
            if max_sequences is not None and n_seqs_written + len(buf_prompt) >= max_sequences:
                break

            prompt_text   = record["prompt"]
            response_text = record["response"]

            prompt_ids_list   = tokenizer.encode(prompt_text,   add_special_tokens=False)
            response_ids_list = tokenizer.encode(response_text, add_special_tokens=False)

            if len(response_ids_list) == 0:
                continue

            subtrees_arr, subtree_probs_arr, resp_probs_arr, selected = process_sequence(
                model=model,
                prompt_ids=prompt_ids_list,
                response_ids=response_ids_list,
                sub_tree=sub_tree,
                num_trees_per_seq=num_trees_per_seq,
                device=device,
            )

            buf_prompt.append(np.array(prompt_ids_list, dtype=np.int64))
            buf_response.append(np.array(response_ids_list, dtype=np.int64))
            buf_resp_probs.append(resp_probs_arr)
            buf_trees.append(subtrees_arr)
            buf_probs.append(subtree_probs_arr)
            buf_selected.append(selected)

            if len(buf_prompt) >= FLUSH_EVERY:
                _flush()
                print(
                    f"Sequences: {n_seqs_written}  "
                    f"Response tokens: {n_tokens_written}",
                    flush=True,
                )

        _flush()

    print(f"Done.  Sequences: {n_seqs_written}  Response tokens: {n_tokens_written}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: generate per-position subtrees from prompt/response JSONL data."
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model name or local path (e.g. Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Directory containing *.jsonl files with 'prompt' and 'response' fields",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output HDF5 file path (e.g. data/stage2.h5)",
    )
    parser.add_argument(
        "--sub-tree-paths", nargs="+",
        default=["0-1", "0-2", "0-3", "1-4", "1-5", "2-6", "2-7"],
        help=(
            'Subtree edges as "X-Y" strings (default: '
            '"0-1 0-2 0-3 1-4 1-5 2-6 2-7")'
        ),
    )
    parser.add_argument(
        "--num-trees-per-seq", type=int, default=64,
        help="Number of subtrees to generate per sequence (default: 64)",
    )
    parser.add_argument(
        "--max-sequences", type=int, default=None,
        help="Stop after processing this many sequences (default: unlimited)",
    )
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }

    sub_tree = parse_sub_tree_structure(args.sub_tree_paths)
    print(f"Subtree size: {sub_tree.subtree_size}")
    print(f"Subtree paths: {args.sub_tree_paths}")
    print(f"Depth-1 nodes: {sub_tree.depth1_nodes}")
    print(f"Depth-2 groups: {dict(sub_tree.depth2_by_d1)}")
    print(f"Trees per sequence: {args.num_trees_per_seq}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation="sdpa",
        torch_dtype=dtype_map[args.dtype],
        device_map=str(device),
    ).eval()

    process(
        model=model,
        tokenizer=tokenizer,
        sub_tree=sub_tree,
        num_trees_per_seq=args.num_trees_per_seq,
        data_dir=Path(args.data_dir),
        output_path=Path(args.output),
        device=device,
        max_sequences=args.max_sequences,
    )


if __name__ == "__main__":
    main()
