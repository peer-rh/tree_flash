"""
Stage 2 v2: Per-position subtree generation with Flex Attention batching.

Input
-----
A directory of JSONL files.  Each line must contain:
    {"prompt": "...", "response": "..."}

Output
------
HDF5 file with datasets:

    prompt_ids          : vlen int64   [N]
    response_ids        : vlen int64   [N]
    sub_trees           : int64        [T, sub_tree_size]
    sub_trees_ar_probs  : float32      [T, sub_tree_size]
    sequence_offsets    : int64        [N+1]

For sequence n, rows sequence_offsets[n]:sequence_offsets[n+1] cover all
S_R response positions.

sub_tree_size includes root (node 0):
    sub_trees[offset+t, 0]          = response_ids[n][t]   (always)
    sub_trees[offset+t, 1:]         = alternative tokens    (IGNORE_IDX=-1 if not selected)
    sub_trees_ar_probs[offset+t, 0] = p(response[t]|ctx)   (always)
    sub_trees_ar_probs[offset+t, 1] = individual AR probs   (0.0 if not selected)

Generation
----------
Depth-0 (root): base forward pass, top-k excluding response[t+1].
Depth d >= 1  : one flex-attention forward pass packing all non-leaf internal
                nodes at that depth across all selected positions.
Leaf nodes never trigger a forward pass (no children to predict).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

IGNORE_IDX: int = -1


# ---------------------------------------------------------------------------
# Subtree structure
# ---------------------------------------------------------------------------

@dataclass
class SubTree:
    """
    Parsed subtree specification.

    sub_tree_size includes root (node 0).
    Slot mapping: node 0 -> slot 0; non-root nodes sorted ascending -> slots 1..
    """
    sub_tree_size: int                          # total nodes including root
    local_idx: dict[int, int]                   # sub_node -> slot index
    parent_map: dict[int, int]                  # child -> parent
    children: dict[int, list[int]]              # parent -> sorted children
    node_depth: dict[int, int]                  # sub_node -> depth (root=0)
    leaves: set[int]                            # nodes with no children
    internal_non_root: set[int]                 # non-root nodes that have children
    nodes_by_depth: dict[int, list[int]]        # depth -> sorted node list
    max_depth: int


def parse_sub_tree(sub_tree_paths: list[str]) -> SubTree:
    """
    Parse edge list (e.g. ["0-1","0-2","1-3"]) into a SubTree.

    sub_tree_size = 1 + len(non_root_nodes)  (root node 0 is always slot 0).
    """
    parent_map: dict[int, int] = {}
    all_nodes: set[int] = {0}
    for edge in sub_tree_paths:
        parts = edge.split("-")
        assert len(parts) == 2, f"edge must be 'X-Y', got {edge!r}"
        p, c = int(parts[0]), int(parts[1])
        assert c not in parent_map, f"node {c} has duplicate parent"
        parent_map[c] = p
        all_nodes |= {p, c}

    non_root_nodes = sorted(all_nodes - {0})
    # slot 0 = root, slots 1.. = non-root nodes in ascending order
    local_idx: dict[int, int] = {0: 0}
    for j, n in enumerate(non_root_nodes, start=1):
        local_idx[n] = j
    sub_tree_size = 1 + len(non_root_nodes)

    # Build children map
    children: dict[int, list[int]] = defaultdict(list)
    for c, p in parent_map.items():
        children[p].append(c)
    for k in children:
        children[k].sort()
    children = dict(children)

    # Compute depths
    node_depth: dict[int, int] = {0: 0}
    for node in non_root_nodes:
        d = 0
        cur = node
        while cur in parent_map:
            cur = parent_map[cur]
            d += 1
        node_depth[node] = d

    max_depth = max(node_depth.values(), default=0)

    nodes_by_depth: dict[int, list[int]] = defaultdict(list)
    for n, d in node_depth.items():
        nodes_by_depth[d].append(n)
    for d in nodes_by_depth:
        nodes_by_depth[d].sort()
    nodes_by_depth = dict(nodes_by_depth)

    leaves = {n for n in all_nodes if n not in children}
    internal_non_root = {n for n in non_root_nodes if n in children}

    return SubTree(
        sub_tree_size=sub_tree_size,
        local_idx=local_idx,
        parent_map=parent_map,
        children=children,
        node_depth=node_depth,
        leaves=leaves,
        internal_non_root=internal_non_root,
        nodes_by_depth=nodes_by_depth,
        max_depth=max_depth,
    )


# ---------------------------------------------------------------------------
# Ancestor path helper
# ---------------------------------------------------------------------------

def ancestor_path_tokens(
    sub_node: int,
    subtree: SubTree,
    node_tokens: dict[int, int],
) -> list[int]:
    """
    Return the list of tokens along the path from depth-1 to sub_node (inclusive),
    i.e. [token_at_d1_ancestor, ..., token_at_sub_node].
    """
    path: list[int] = []
    cur = sub_node
    while cur != 0:
        path.append(node_tokens[cur])
        cur = subtree.parent_map[cur]
    path.reverse()  # root -> sub_node order, drop root (depth-0)
    return path


# ---------------------------------------------------------------------------
# Flex Attention batch forward pass
# ---------------------------------------------------------------------------

@torch.inference_mode()
def flex_batch_forward(
    model: AutoModelForCausalLM,
    queries: list[tuple[list[int], int]],  # (context_ids, query_key)
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """
    Run a single packed forward pass over all queries using a document-causal
    block mask (Flex Attention).  Returns {query_key: last_position_logits [V]}.

    Each query is (context_ids, query_key).  The context attends causally only
    within its own document; position_ids reset per document for correct RoPE.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    packed_ids: list[int] = []
    pos_ids: list[int] = []
    doc_ids_list: list[int] = []
    last_positions: list[int] = []
    keys: list[int] = []

    for doc_idx, (ctx, qkey) in enumerate(queries):
        packed_ids.extend(ctx)
        pos_ids.extend(range(len(ctx)))
        doc_ids_list.extend([doc_idx] * len(ctx))
        last_positions.append(len(packed_ids) - 1)
        keys.append(qkey)

    S = len(packed_ids)
    doc_ids_t = torch.tensor(doc_ids_list, dtype=torch.int32, device=device)

    def _mask(b, h, q, k):
        return (doc_ids_t[q] == doc_ids_t[k]) & (q >= k)

    block_mask = create_block_mask(_mask, B=None, H=None, Q_LEN=S, KV_LEN=S, device=device)

    input_t = torch.tensor(packed_ids, dtype=torch.long, device=device).unsqueeze(0)
    pos_t = torch.tensor(pos_ids, dtype=torch.long, device=device).unsqueeze(0)

    out = model(input_t, position_ids=pos_t, attention_mask=block_mask, use_cache=False)
    logits = out.logits[0]  # [S, V]

    return {qkey: logits[last_positions[i]] for i, qkey in enumerate(keys)}


@torch.inference_mode()
def eager_batch_forward(
    model: AutoModelForCausalLM,
    queries: list[tuple[list[int], int]],
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """
    Fallback: run one forward pass per query (no packing).
    """
    results: dict[int, torch.Tensor] = {}
    for ctx, qkey in queries:
        input_t = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)
        out = model(input_t, use_cache=False)
        results[qkey] = out.logits[0, -1, :]
    return results


# ---------------------------------------------------------------------------
# Per-sequence processing
# ---------------------------------------------------------------------------

@torch.inference_mode()
def process_sequence(
    model: AutoModelForCausalLM,
    prompt_ids: list[int],
    response_ids: list[int],
    subtree: SubTree,
    n_subtrees_per_seq: int,
    device: torch.device,
    use_flex: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Process one (prompt, response) pair.

    Returns
    -------
    sub_trees_arr      : int64   [S_R, sub_tree_size]
    sub_trees_probs_arr: float32 [S_R, sub_tree_size]
    """
    S_R = len(response_ids)
    ST = subtree.sub_tree_size
    sub_trees_arr = np.full((S_R, ST), IGNORE_IDX, dtype=np.int64)
    sub_trees_probs_arr = np.zeros((S_R, ST), dtype=np.float32)

    prompt_len = len(prompt_ids)
    full_ids = prompt_ids + response_ids

    # ── Base forward pass ──────────────────────────────────────────────────
    input_t = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
    base_out = model(input_t, use_cache=False)
    base_logits = base_out.logits[0]  # [S_P+S_R, V]

    # ── Fill root slot (slot 0) for ALL positions ──────────────────────────
    for t in range(S_R):
        sub_trees_arr[t, 0] = response_ids[t]
        t_abs = prompt_len + t
        logit_pos = t_abs - 1
        if logit_pos >= 0:
            lp = F.log_softmax(base_logits[logit_pos], dim=-1)
            sub_trees_probs_arr[t, 0] = math.exp(float(lp[response_ids[t]].item()))
        else:
            sub_trees_probs_arr[t, 0] = 1.0

    # ── Select positions by uncertainty ───────────────────────────────────
    root_probs = [float(sub_trees_probs_arr[t, 0]) for t in range(S_R)]
    scores = sorted(enumerate(root_probs), key=lambda x: x[1])  # ascending p
    selected = sorted(t for t, _ in scores[:min(n_subtrees_per_seq, S_R)])

    if not selected or subtree.max_depth == 0:
        return sub_trees_arr, sub_trees_probs_arr

    # ── Depth 0 → fill depth-1 children from base_logits ──────────────────
    # node_tokens[t][sub_node] = token assigned to sub_node at position t
    node_tokens: dict[int, dict[int, int]] = {t: {} for t in selected}

    if 0 in subtree.children:
        d1_children = subtree.children[0]
        k_d1 = len(d1_children)
        for t in selected:
            t_abs = prompt_len + t
            if t_abs >= base_logits.shape[0]:
                continue
            logits_at_t = base_logits[t_abs]  # predicts position t_abs+1
            log_probs = F.log_softmax(logits_at_t, dim=-1)

            # Exclude the next response token (already on the primary path)
            response_next = response_ids[t + 1] if t + 1 < S_R else -1
            topk_lp, topk_ids = torch.topk(log_probs, k=min(k_d1 + 1, log_probs.shape[0]))

            filled = 0
            for tok, lp in zip(topk_ids.tolist(), topk_lp.tolist()):
                if tok == response_next:
                    continue
                child = d1_children[filled]
                slot = subtree.local_idx[child]
                sub_trees_arr[t, slot] = tok
                sub_trees_probs_arr[t, slot] = math.exp(lp)
                node_tokens[t][child] = tok
                filled += 1
                if filled >= k_d1:
                    break

    # ── Depth d >= 1: batch all non-leaf internal nodes ───────────────────
    batch_fn = flex_batch_forward if use_flex else eager_batch_forward

    for d in range(1, subtree.max_depth):
        nodes_at_d = subtree.nodes_by_depth.get(d, [])
        internal_at_d = [n for n in nodes_at_d if n in subtree.internal_non_root]
        if not internal_at_d:
            continue

        # Build queries: one per (t, sub_node) where sub_node was filled
        queries: list[tuple[list[int], int]] = []
        query_meta: list[tuple[int, int]] = []  # (t, sub_node)
        qkey = 0

        for t in selected:
            t_abs = prompt_len + t
            for sub_node in internal_at_d:
                if sub_node not in node_tokens[t]:
                    continue  # this node wasn't filled (parent missing or skipped)
                path_toks = ancestor_path_tokens(sub_node, subtree, node_tokens[t])
                ctx = full_ids[: t_abs + 1] + path_toks
                queries.append((ctx, qkey))
                query_meta.append((t, sub_node))
                qkey += 1

        if not queries:
            continue

        logits_map = batch_fn(model, queries, device)

        # Fill children of each queried node
        for (t, sub_node), qk in zip(query_meta, range(len(queries))):
            child_logits = logits_map[qk]  # [V]
            children = subtree.children[sub_node]
            k = len(children)
            lp, ids = torch.topk(F.log_softmax(child_logits, dim=-1), k=k)
            for rank, child in enumerate(children):
                slot = subtree.local_idx[child]
                tok = int(ids[rank].item())
                sub_trees_arr[t, slot] = tok
                sub_trees_probs_arr[t, slot] = math.exp(float(lp[rank].item()))
                node_tokens[t][child] = tok

    return sub_trees_arr, sub_trees_probs_arr


# ---------------------------------------------------------------------------
# JSONL reader
# ---------------------------------------------------------------------------

def iter_jsonl_dir(data_dir: Path) -> Iterator[dict]:
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
    subtree: SubTree,
    n_subtrees_per_seq: int,
    data_dir: Path,
    output_path: Path,
    device: torch.device,
    max_sequences: int | None,
    use_flex: bool,
) -> None:
    ST = subtree.sub_tree_size
    vlen_int64 = h5py.vlen_dtype(np.int64)
    FLUSH_EVERY = 128

    with h5py.File(output_path, "w") as hf:
        ds_prompt = hf.create_dataset("prompt_ids", shape=(0,), maxshape=(None,), dtype=vlen_int64)
        ds_response = hf.create_dataset("response_ids", shape=(0,), maxshape=(None,), dtype=vlen_int64)
        ds_trees = hf.create_dataset(
            "sub_trees",
            shape=(0, ST), maxshape=(None, ST),
            dtype="int64", chunks=(512, ST), compression="lzf",
        )
        ds_probs = hf.create_dataset(
            "sub_trees_ar_probs",
            shape=(0, ST), maxshape=(None, ST),
            dtype="float32", chunks=(512, ST), compression="lzf",
        )
        ds_offsets = hf.create_dataset("sequence_offsets", shape=(1,), maxshape=(None,), dtype="int64")
        ds_offsets[0] = 0

        # Buffers
        buf_prompt:   list[np.ndarray] = []
        buf_response: list[np.ndarray] = []
        buf_trees:    list[np.ndarray] = []  # each [S_R, ST]
        buf_probs:    list[np.ndarray] = []
        n_seqs = 0
        n_toks = 0  # total response tokens written so far

        def _flush() -> None:
            nonlocal n_seqs, n_toks
            if not buf_prompt:
                return
            n = len(buf_prompt)
            new_n_seqs = n_seqs + n

            ds_prompt.resize(new_n_seqs, axis=0)
            ds_response.resize(new_n_seqs, axis=0)
            for k in range(n):
                ds_prompt[n_seqs + k] = buf_prompt[k]
                ds_response[n_seqs + k] = buf_response[k]

            combined_trees = np.concatenate(buf_trees, axis=0)
            combined_probs = np.concatenate(buf_probs, axis=0)
            total_new = combined_trees.shape[0]

            ds_trees.resize(n_toks + total_new, axis=0)
            ds_probs.resize(n_toks + total_new, axis=0)
            ds_trees[n_toks: n_toks + total_new] = combined_trees
            ds_probs[n_toks: n_toks + total_new] = combined_probs

            # Offsets: cumulative sum of S_R lengths
            new_offsets = np.array(
                [n_toks + sum(buf_trees[j].shape[0] for j in range(k + 1)) for k in range(n)],
                dtype=np.int64,
            )
            ds_offsets.resize(new_n_seqs + 1, axis=0)
            ds_offsets[n_seqs + 1: new_n_seqs + 1] = new_offsets

            n_seqs = new_n_seqs
            n_toks += total_new
            buf_prompt.clear(); buf_response.clear()
            buf_trees.clear(); buf_probs.clear()

        for record in iter_jsonl_dir(data_dir):
            if max_sequences is not None and n_seqs + len(buf_prompt) >= max_sequences:
                break

            prompt_ids = tokenizer.encode(record["prompt"], add_special_tokens=False)
            response_ids = tokenizer.encode(record["response"], add_special_tokens=False)
            if not response_ids:
                continue

            trees_arr, probs_arr = process_sequence(
                model=model,
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                subtree=subtree,
                n_subtrees_per_seq=n_subtrees_per_seq,
                device=device,
                use_flex=use_flex,
            )

            buf_prompt.append(np.array(prompt_ids, dtype=np.int64))
            buf_response.append(np.array(response_ids, dtype=np.int64))
            buf_trees.append(trees_arr)
            buf_probs.append(probs_arr)

            if len(buf_prompt) >= FLUSH_EVERY:
                _flush()
                print(f"Sequences: {n_seqs}  Response tokens: {n_toks}", flush=True)

        _flush()

    print(f"Done.  Sequences: {n_seqs}  Response tokens: {n_toks}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2 v2: generate per-position subtrees with Flex Attention batching."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--sub-tree-paths", nargs="+",
        default=["0-1", "0-2", "0-3", "1-4", "1-5", "2-6", "2-7"],
        help='Subtree edges as "X-Y" strings',
    )
    parser.add_argument("--n-subtrees-per-seq", type=int, default=64,
                        help="Number of positions to generate subtrees for per sequence")
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--attn-impl", choices=["flex_attention", "sdpa", "eager"],
                        default="flex_attention")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"],
                        default="bfloat16")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    subtree = parse_sub_tree(args.sub_tree_paths)
    print(f"sub_tree_size : {subtree.sub_tree_size}  (includes root)")
    print(f"max_depth     : {subtree.max_depth}")
    print(f"leaves        : {sorted(subtree.leaves)}")
    print(f"internal nodes: {sorted(subtree.internal_non_root)}")
    print(f"attn_impl     : {args.attn_impl}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation=args.attn_impl,
        torch_dtype=dtype_map[args.dtype],
        device_map=str(device),
    ).eval()

    process(
        model=model,
        tokenizer=tokenizer,
        subtree=subtree,
        n_subtrees_per_seq=args.n_subtrees_per_seq,
        data_dir=Path(args.data_dir),
        output_path=Path(args.output),
        device=device,
        max_sequences=args.max_sequences,
        use_flex=(args.attn_impl == "flex_attention"),
    )


if __name__ == "__main__":
    main()
