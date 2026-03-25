"""
Stage 2: Tree token generation for tree-flash training data.

Input
-----
A directory of JSONL files.  Each line must contain:
    {"prompt": "...", "response": "..."}

Output
------
A single HDF5 file with three datasets:
    context_ids      [N, ctx_len]    int64   — context window (prompt+response prefix)
    tree_tokens      [N, tree_size]  int64   — target model's predicted tokens at each node
    cumprod_weights  [N, tree_size]  float32 — ∏ p_target along path from root to each node

Tree specification
------------------
Pass --tree-edges as a space-separated list of "parent,child" integer pairs:

    --tree-edges "0,1 0,2 1,3 1,4 2,5 2,6"

defines root 0 with children [1,2], node 1 with children [3,4], node 2
with children [5,6].  tree_size = number of unique nodes = 7.

The root is the unique node that never appears as a child.
Depths and ancestor relationships are derived automatically.

Token assignment
----------------
The tree is filled level by level.  At each depth level, one Flex Attention
forward pass is run over [context tokens] + [all nodes at depths 0..prev].
For each parent node p with k children (sorted ascending by node id):
    child[0] ← top-1 prediction from p's logit position
    child[1] ← top-2 prediction
    ...
    child[k-1] ← top-k prediction

The root token is the top-1 prediction from the last context position.

CumProd weights
---------------
w[root] = 1.0
w[node] = w[parent] * p_target(token[node] | context + path_to_parent)
where p_target is read from the same Flex Attention forward pass used to
predict the node's children.  This matches the loss scaling used by the trainer.

Scope
-----
Trees are generated only for response positions.  Each valid window gives
one training example:
    context_ids = tokenised(prompt + response_prefix)[-ctx_len:]
    The first ctx_len tokens are taken; if the prompt+response prefix is
    shorter the example is skipped.  --stride controls how many tokens to
    advance between consecutive windows within a single response.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Tree structure from edges
# ---------------------------------------------------------------------------

@dataclass
class TreeInfo:
    """
    Describes an arbitrary rooted tree derived from a parent,child edge list.

    All lists/tensors are indexed 0..tree_size-1 in the original node numbering
    given by the user; the root is node 0 (reindexed if necessary).

    Attributes
    ----------
    tree_size     : int — total number of nodes
    parent_ids    : list[int] — parent_ids[i] = parent of node i (-1 for root)
    depths        : list[int] — depths[i] = depth of node i (root = 0)
    children      : list[list[int]] — children[i] = sorted list of children of i
    depth_to_nodes: list[list[int]] — depth_to_nodes[d] = nodes at depth d
    max_depth     : int
    """
    tree_size:      int
    parent_ids:     list[int]
    depths:         list[int]
    children:       list[list[int]]
    depth_to_nodes: list[list[int]]
    max_depth:      int


def parse_tree_edges(edge_str: str) -> TreeInfo:
    """
    Parse a space-separated list of "parent,child" pairs into a TreeInfo.

    Parameters
    ----------
    edge_str : e.g. "0,1 0,2 1,3 1,4 2,5 2,6"

    Returns
    -------
    TreeInfo with nodes reindexed so that the root is 0 and all nodes are
    numbered 0..tree_size-1 (original indices preserved if they already form
    a dense 0-based range).
    """
    if not edge_str.strip():
        raise ValueError("--tree-edges must not be empty")

    raw_edges: list[tuple[int, int]] = []
    all_nodes: set[int] = set()
    children_nodes: set[int] = set()

    for token in edge_str.split():
        parts = token.split(",")
        if len(parts) != 2:
            raise ValueError(f"Edge token must be 'parent,child', got {token!r}")
        p, c = int(parts[0]), int(parts[1])
        raw_edges.append((p, c))
        all_nodes |= {p, c}
        children_nodes.add(c)

    root_candidates = all_nodes - children_nodes
    if len(root_candidates) != 1:
        raise ValueError(
            f"Tree must have exactly one root (node not appearing as child). "
            f"Found: {root_candidates}"
        )
    root = next(iter(root_candidates))

    # Reindex nodes so root=0 and ids are dense 0..N-1 (BFS order)
    queue = [root]
    bfs_order: list[int] = []
    while queue:
        node = queue.pop(0)
        bfs_order.append(node)
        # collect children in sorted original-id order for reproducibility
        kids = sorted(c for (p, c) in raw_edges if p == node)
        queue.extend(kids)

    orig_to_new = {orig: new for new, orig in enumerate(bfs_order)}
    tree_size = len(bfs_order)

    parent_ids   = [-1] * tree_size
    children_map: list[list[int]] = [[] for _ in range(tree_size)]

    for orig_p, orig_c in raw_edges:
        p = orig_to_new[orig_p]
        c = orig_to_new[orig_c]
        parent_ids[c] = p
        children_map[p].append(c)

    # Sort children lists ascending
    for i in range(tree_size):
        children_map[i].sort()

    # Compute depths
    depths = [0] * tree_size
    for node in range(1, tree_size):  # BFS order ensures parent is already done
        depths[node] = depths[parent_ids[node]] + 1
    max_depth = max(depths)

    depth_to_nodes: list[list[int]] = [[] for _ in range(max_depth + 1)]
    for node, d in enumerate(depths):
        depth_to_nodes[d].append(node)

    return TreeInfo(
        tree_size=tree_size,
        parent_ids=parent_ids,
        depths=depths,
        children=children_map,
        depth_to_nodes=depth_to_nodes,
        max_depth=max_depth,
    )


# ---------------------------------------------------------------------------
# Attention table and Flex Attention block mask
# ---------------------------------------------------------------------------

def _build_attention_table(
    tree: TreeInfo,
    ctx_len: int,
    n_tree_nodes: int,          # nodes present in this forward pass
    device: torch.device,
) -> torch.Tensor:
    """
    Build a boolean attention table for a packed [context | tree_nodes] sequence.

    table[q, kv] = True means query q may attend to key kv.

    Rules:
        context q, context kv  — standard causal (kv <= q)
        tree q,    context kv  — always attend (kv < ctx_len)
        tree q,    tree kv     — kv is ancestor-or-self of q
        context q, tree kv     — never (future information)
    """
    total = ctx_len + n_tree_nodes
    table = torch.zeros(total, total, dtype=torch.bool, device=device)

    # Context → context: causal
    ctx_q  = torch.arange(ctx_len, device=device)
    ctx_kv = torch.arange(ctx_len, device=device)
    table[:ctx_len, :ctx_len] = ctx_kv.unsqueeze(0) <= ctx_q.unsqueeze(1)

    # Tree → context: full attention
    table[ctx_len:, :ctx_len] = True

    # Tree → tree: ancestor-or-self
    # Build ancestor matrix for the n_tree_nodes present
    # (nodes are in depth order so ancestor is always earlier in the list)
    anc = torch.zeros(n_tree_nodes, n_tree_nodes, dtype=torch.bool, device=device)
    for q in range(n_tree_nodes):
        cur = q
        while cur >= 0:
            anc[cur, q] = True
            cur = tree.parent_ids[cur] if cur > 0 else -1
    table[ctx_len:, ctx_len:] = anc

    return table


def _build_flex_block_mask(
    tree: TreeInfo,
    ctx_len: int,
    n_tree_nodes: int,
    device: torch.device,
) -> "BlockMask":
    total = ctx_len + n_tree_nodes
    table = _build_attention_table(tree, ctx_len, n_tree_nodes, device)

    def mask_mod(batch, head, q_idx, kv_idx):
        return table[q_idx, kv_idx]

    return create_block_mask(
        mask_mod,
        B=None, H=None,
        Q_LEN=total, KV_LEN=total,
        device=device,
        BLOCK_SIZE=(32, 32),
        _compile=True,
    )


@contextlib.contextmanager
def _flex_attention_context(block_mask):
    """Temporarily patch F.scaled_dot_product_attention with flex_attention."""
    original = F.scaled_dot_product_attention

    def _patched(query, key, value, **kwargs):
        return flex_attention(query, key, value, block_mask=block_mask,
                              scale=kwargs.get("scale"))

    F.scaled_dot_product_attention = _patched
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original


# ---------------------------------------------------------------------------
# Tree generation for one context window
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_tree_example(
    model: AutoModelForCausalLM,
    context_ids: list[int],           # [ctx_len]
    tree: TreeInfo,
    device: torch.device,
) -> tuple[list[int], list[float]]:
    """
    Generate tree_tokens and cumprod_weights for one context window.

    Returns
    -------
    tree_tokens      : list[int]   length tree_size — predicted token at each node
    cumprod_weights  : list[float] length tree_size — ∏ p_target along path to node
    """
    ctx_len    = len(context_ids)
    tree_size  = tree.tree_size
    tree_tokens  = [0] * tree_size
    log_weights  = [0.0] * tree_size   # log(cumprod_weight); root = log(1) = 0

    # ── Depth 0: root token from context ─────────────────────────────────────
    ctx_tensor = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)
    ctx_out    = model(ctx_tensor, use_cache=False)
    ctx_logits = ctx_out.logits[0, -1, :]              # [V] — prediction after context

    root_children = tree.children[0]
    if not root_children:
        # Degenerate tree with only a root and no children — unusual but handle it
        root_lp = F.log_softmax(ctx_logits, dim=-1)
        tree_tokens[0] = int(root_lp.argmax().item())
        return tree_tokens, [math.exp(lw) for lw in log_weights]

    # Assign top-k predictions to root's children (k = number of children)
    root_log_probs  = F.log_softmax(ctx_logits, dim=-1)   # [V]
    k = len(root_children)
    top_lp, top_ids = torch.topk(root_log_probs, k=k)    # [k]
    for rank, child in enumerate(root_children):
        tree_tokens[child]  = int(top_ids[rank].item())
        log_weights[child]  = float(top_lp[rank].item())

    # root itself gets the top-1 token (to feed as input for deeper levels)
    tree_tokens[0] = int(top_ids[0].item())
    # root weight stays 0.0 (= weight 1.0) — the root node is always included

    # ── Depths 1..max_depth ───────────────────────────────────────────────────
    # At each depth, run one Flex Attention forward pass over:
    #   [context tokens] [node 0 token] [node 1 token] ... [node prev_depth_last token]
    # and extract logits at depth-1 nodes to assign tokens to depth nodes.

    for depth in range(1, tree.max_depth + 1):
        nodes_so_far = [n for d in range(depth) for n in tree.depth_to_nodes[d]]
        # nodes in the input sequence (after context): nodes_so_far[0..M-1]
        M = len(nodes_so_far)

        packed_ids = context_ids + [tree_tokens[n] for n in nodes_so_far]
        input_tensor = torch.tensor(packed_ids, dtype=torch.long, device=device).unsqueeze(0)

        block_mask = _build_flex_block_mask(tree, ctx_len, M, device)

        with _flex_attention_context(block_mask):
            output = model(input_tensor, use_cache=False)

        logits = output.logits[0]   # [ctx_len + M, V]

        # Logits at each node position predict its children
        # node i is at sequence position ctx_len + nodes_so_far.index(i)
        node_seq_pos = {n: ctx_len + idx for idx, n in enumerate(nodes_so_far)}

        for parent in tree.depth_to_nodes[depth - 1]:
            children = tree.children[parent]
            if not children:
                continue

            parent_logits  = logits[node_seq_pos[parent]]    # [V]
            parent_log_probs = F.log_softmax(parent_logits, dim=-1)
            k = len(children)
            top_lp_c, top_ids_c = torch.topk(parent_log_probs, k=k)

            for rank, child in enumerate(children):
                tree_tokens[child] = int(top_ids_c[rank].item())
                # cumprod weight = parent_weight * p_target(this token)
                log_weights[child] = log_weights[parent] + float(top_lp_c[rank].item())

    cumprod_weights = [math.exp(lw) for lw in log_weights]
    return tree_tokens, cumprod_weights


# ---------------------------------------------------------------------------
# JSONL directory reader
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
    tree: TreeInfo,
    data_dir: Path,
    output_path: Path,
    ctx_len: int,
    stride: int,
    device: torch.device,
    max_examples: int | None,
) -> None:
    """
    Iterate over all JSONL records, generate tree examples, write to HDF5.

    Each record must have "prompt" and "response" string fields.
    Trees are generated only at positions within the response, never in the prompt.
    Each valid window advances by ``stride`` response tokens.
    """
    tree_size = tree.tree_size

    # Pre-allocate HDF5 with extensible datasets
    with h5py.File(output_path, "w") as hf:
        ds_ctx = hf.create_dataset(
            "context_ids",
            shape=(0, ctx_len),
            maxshape=(None, ctx_len),
            dtype="int64",
            chunks=(256, ctx_len),
            compression="lzf",
        )
        ds_tok = hf.create_dataset(
            "tree_tokens",
            shape=(0, tree_size),
            maxshape=(None, tree_size),
            dtype="int64",
            chunks=(256, tree_size),
            compression="lzf",
        )
        ds_wts = hf.create_dataset(
            "cumprod_weights",
            shape=(0, tree_size),
            maxshape=(None, tree_size),
            dtype="float32",
            chunks=(256, tree_size),
            compression="lzf",
        )

        # Write buffer: flush to HDF5 every FLUSH_EVERY examples
        FLUSH_EVERY = 512
        buf_ctx: list[list[int]]   = []
        buf_tok: list[list[int]]   = []
        buf_wts: list[list[float]] = []
        total_written = 0

        def _flush():
            nonlocal total_written
            if not buf_ctx:
                return
            n = len(buf_ctx)
            new_size = total_written + n
            for ds in (ds_ctx, ds_tok, ds_wts):
                ds.resize(new_size, axis=0)
            ds_ctx[total_written:new_size]  = np.array(buf_ctx,  dtype=np.int64)
            ds_tok[total_written:new_size]  = np.array(buf_tok,  dtype=np.int64)
            ds_wts[total_written:new_size]  = np.array(buf_wts,  dtype=np.float32)
            total_written = new_size
            buf_ctx.clear()
            buf_tok.clear()
            buf_wts.clear()

        n_records = 0
        for record in iter_jsonl_dir(data_dir):
            prompt   = record["prompt"]
            response = record["response"]

            # prompt is a chat-template-formatted string that already contains
            # all special tokens; encode with add_special_tokens=False to avoid
            # inserting a duplicate BOS token.
            prompt_ids   = tokenizer.encode(prompt,   add_special_tokens=False)
            response_ids = tokenizer.encode(response, add_special_tokens=False)

            full_ids   = prompt_ids + response_ids
            prompt_len = len(prompt_ids)

            # The minimum number of response tokens needed:
            #   ctx_len tokens for the context window
            #   tree_size tokens look-ahead (tree generation needs at least ctx_len tokens)
            # We require the window end to stay within the full sequence, so the
            # last window starts at most at full_ids[-tree_size] and must use
            # response tokens as the context anchor.
            #
            # Window [t - ctx_len : t] is valid when:
            #   t >= ctx_len                (enough tokens before t for context)
            #   t > prompt_len              (anchor token is in the response)

            t_min = max(ctx_len, prompt_len + 1)   # first valid anchor position
            t_max = len(full_ids)                  # exclusive

            positions = range(t_min, t_max, stride)
            for t in positions:
                if max_examples is not None and (total_written + len(buf_ctx)) >= max_examples:
                    _flush()
                    print(f"Reached max_examples={max_examples}. Done.")
                    return

                ctx_window = full_ids[t - ctx_len : t]   # [ctx_len]

                tree_tokens, cumprod_weights = generate_tree_example(
                    model, ctx_window, tree, device
                )

                buf_ctx.append(ctx_window)
                buf_tok.append(tree_tokens)
                buf_wts.append(cumprod_weights)

                if len(buf_ctx) >= FLUSH_EVERY:
                    _flush()

            n_records += 1
            if n_records % 100 == 0:
                _flush()
                print(f"Records: {n_records}  Examples: {total_written + len(buf_ctx)}",
                      flush=True)

        _flush()

    print(f"Done. Records processed: {n_records}  Examples written: {total_written}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: generate tree tokens from prompt/response JSONL data."
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
        "--tree-edges", required=True,
        help=(
            'Space-separated "parent,child" pairs defining the tree shape. '
            'Example: "0,1 0,2 1,3 1,4 2,5 2,6" '
            '(root 0 with children 1,2; node 1 with children 3,4; node 2 with 5,6)'
        ),
    )
    parser.add_argument(
        "--ctx-len", type=int, default=512,
        help="Context window length in tokens (default: 512)",
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Tokens to advance between consecutive windows per response (default: 1)",
    )
    parser.add_argument(
        "--max-examples", type=int, default=None,
        help="Stop after writing this many examples (default: unlimited)",
    )
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}

    tree = parse_tree_edges(args.tree_edges)
    print(f"Tree: {tree.tree_size} nodes, max depth {tree.max_depth}")
    print(f"  parent_ids : {tree.parent_ids}")
    print(f"  depths     : {tree.depths}")

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
        tree=tree,
        data_dir=Path(args.data_dir),
        output_path=Path(args.output),
        ctx_len=args.ctx_len,
        stride=args.stride,
        device=device,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()
