"""
Stage 2: Continuation tree generation.

Accepts JSONL records with or without pre-computed logprobs:
  - With logprobs (from Stage 1 with --top-b): anchor selection and level-1
    tree nodes are free — only d-1 forward passes needed.
  - Without logprobs: one extra causal forward pass on the full sequence
    to compute per-token uncertainty for anchor selection and level-1 tokens,
    then d-1 tree-masked passes. Total: d forward passes.

Uses Flex Attention for all tree-masked forward passes to avoid materialising
an O(seq²) float attention mask in GPU memory.

Output format (merged into input record):
{
    ...(input fields),
    "anchor_positions": List[int],            # relative to completion
    "tree_tokens":   List[List[int]],         # [completion_len, tree_size]
    "tree_logprobs": List[List[float]],       # [completion_len, tree_size]
}
Non-anchor positions have all tree entries set to ignore_idx (-100).

Tree node BFS ordering (branching factor b, depth d):
  depth-1:  b nodes      (indices 0 .. b-1)
  depth-2:  b² nodes     (indices b .. b+b²-1)
  ...
  tree_size = b + b² + ... + b^d

Path encoding at depth l:
  path_enc = p_1*b^(l-1) + ... + p_l   (p_i in 0..b-1, 0-indexed branches)
"""

import argparse
import contextlib
import json
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from transformers import AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Anchor selection
# ---------------------------------------------------------------------------

def select_anchors(
    chosen_logprobs: list[float],
    threshold: float = 0.02,
    max_fraction: float = 0.5,
) -> list[int]:
    """
    Return positions where the target model was uncertain (1 - p > threshold),
    capped at max_fraction * len(chosen_logprobs) positions.
    Positions are relative to the completion and sorted ascending.
    """
    uncertainty = [1.0 - math.exp(lp) for lp in chosen_logprobs]
    S = len(uncertainty)
    max_anchors = max(1, int(S * max_fraction))
    candidates = sorted(
        ((u, i) for i, u in enumerate(uncertainty) if u > threshold),
        reverse=True,
    )
    return sorted(i for _, i in candidates[:max_anchors])


# ---------------------------------------------------------------------------
# Tree layout
# ---------------------------------------------------------------------------

@dataclass
class TreeLayout:
    """
    Flat BFS description of tree nodes for a single (sequence, anchor-set) pair.

    Packed input tensor layout:
      [original tokens: 0..S-1] [tree nodes: S..S+num_tree_nodes-1]

    Tree nodes are in BFS order: all depth-1 nodes first, then depth-2, etc.
    Within each depth level, ordering is anchor_0_path_0, anchor_0_path_1, ...
    anchor_1_path_0, ...
    """
    S: int                            # full sequence length (prompt + completion)
    K: int                            # number of anchors
    b: int                            # branching factor
    d: int                            # tree depth (number of levels stored)
    anchor_seq_positions: list[int]   # anchor positions in full sequence (len K)
    tree_size: int                    # nodes per anchor = b + b² + ... + b^d
    num_tree_nodes: int               # K * tree_size
    node_anchor: torch.Tensor         # (num_tree_nodes,) int — which anchor
    node_depth: torch.Tensor          # (num_tree_nodes,) int — 1-indexed depth
    node_path: torch.Tensor           # (num_tree_nodes,) int — path encoding


def build_tree_layout(
    S: int,
    anchor_seq_positions: list[int],
    b: int,
    d: int,
    device: torch.device,
) -> TreeLayout:
    K = len(anchor_seq_positions)
    tree_size = sum(b**l for l in range(1, d + 1))
    num_tree_nodes = K * tree_size

    anchors, depths, paths = [], [], []
    for depth_l in range(1, d + 1):
        for anchor_i in range(K):
            for path_enc in range(b ** depth_l):
                anchors.append(anchor_i)
                depths.append(depth_l)
                paths.append(path_enc)

    return TreeLayout(
        S=S,
        K=K,
        b=b,
        d=d,
        anchor_seq_positions=anchor_seq_positions,
        tree_size=tree_size,
        num_tree_nodes=num_tree_nodes,
        node_anchor=torch.tensor(anchors, dtype=torch.long, device=device),
        node_depth=torch.tensor(depths, dtype=torch.long, device=device),
        node_path=torch.tensor(paths, dtype=torch.long, device=device),
    )


# ---------------------------------------------------------------------------
# Attention table & Flex Attention block mask
# ---------------------------------------------------------------------------

def _build_attention_table(layout: TreeLayout, total_len: int) -> torch.Tensor:
    """
    Build a (total_len, total_len) boolean attention table on CPU.
    table[q, kv] = True means position q may attend to position kv.

    Fully vectorised — no Python loops over node pairs.
    """
    S = layout.S
    N = layout.num_tree_nodes
    d = layout.d
    b = layout.b
    b_powers = torch.tensor([b**k for k in range(d + 1)], dtype=torch.long)

    table = torch.zeros(total_len, total_len, dtype=torch.bool)

    # Original q, original kv: standard causal
    q_idx = torch.arange(S)
    kv_idx = torch.arange(S)
    table[:S, :S] = kv_idx.unsqueeze(0) <= q_idx.unsqueeze(1)

    if N == 0:
        return table

    # Tree q, original kv: attend iff kv <= anchor_seq_pos
    anchor_seq_pos_t = torch.tensor(layout.anchor_seq_positions, dtype=torch.long)
    tree_anchor_seq_pos = anchor_seq_pos_t[layout.node_anchor]  # (N,)
    table[S:, :S] = kv_idx.unsqueeze(0) <= tree_anchor_seq_pos.unsqueeze(1)

    # Tree q, tree kv: ancestor check (vectorised over all N×N pairs)
    depth_i = layout.node_depth.unsqueeze(1)    # (N, 1)
    depth_j = layout.node_depth.unsqueeze(0)    # (1, N)
    path_i  = layout.node_path.unsqueeze(1)     # (N, 1)
    path_j  = layout.node_path.unsqueeze(0)     # (1, N)
    anch_i  = layout.node_anchor.unsqueeze(1)   # (N, 1)
    anch_j  = layout.node_anchor.unsqueeze(0)   # (1, N)

    same_anchor   = anch_i == anch_j
    j_shallower   = depth_j < depth_i
    depth_diff    = (depth_i - depth_j).clamp(0, d)
    ancestor_path = (path_i // b_powers[depth_diff]) == path_j

    table[S:, S:] = same_anchor & j_shallower & ancestor_path
    return table


def build_tree_block_mask(
    layout: TreeLayout,
    total_len: int,
    device: torch.device,
) -> "BlockMask":
    """
    Build a Flex Attention BlockMask for the tree attention pattern.

    The mask_mod closes over a precomputed boolean table — the canonical pattern
    from the PyTorch Flex Attention blog. BLOCK_SIZE=(32,32) is used because
    the default 128×128 blocks are too coarse for typical tree sizes (16-512 nodes)
    and would exhibit poor sparsity.
    """
    attn_table = _build_attention_table(layout, total_len).to(device)

    def mask_mod(batch, head, q_idx, kv_idx):
        return attn_table[q_idx, kv_idx]

    return create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=total_len,
        KV_LEN=total_len,
        device=device,
        BLOCK_SIZE=(32, 32),
        _compile=True,
    )


# ---------------------------------------------------------------------------
# Flex Attention context manager
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _flex_attention_context(block_mask):
    """
    Temporarily replace torch.nn.functional.scaled_dot_product_attention with
    flex_attention using the provided BlockMask.

    Works with any HF model loaded with attn_implementation="sdpa". The patch is
    applied at the F.scaled_dot_product_attention level so it intercepts all
    attention layers in the model without requiring model surgery.

    The block_mask encodes the full attention pattern (causal for original tokens,
    tree-ancestor for tree tokens), so is_causal and attn_mask from the model's
    standard attention paths are intentionally ignored here.
    """
    original_sdpa = F.scaled_dot_product_attention

    def _patched_sdpa(query, key, value, **kwargs):
        return flex_attention(query, key, value, block_mask=block_mask, scale=kwargs.get("scale"))

    F.scaled_dot_product_attention = _patched_sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original_sdpa


# ---------------------------------------------------------------------------
# Causal forward pass (for records without pre-computed logprobs)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _causal_forward(
    model: AutoModelForCausalLM,
    token_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Run a standard causal forward pass. Returns logits of shape (S, vocab)."""
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    output = model(input_ids, use_cache=False)
    return output.logits[0]  # (S, vocab)


def _compute_logprobs_from_logits(
    logits: torch.Tensor,   # (S, vocab)
    token_ids: list[int],   # length S, the actual tokens chosen
    b: int,
) -> tuple[list[float], list[list[int]], list[list[float]]]:
    """
    Extract:
      chosen_logprobs: log p(token_ids[t]) at each position  — length S
      top_token_ids:   top-b token IDs at each position      — (S, b)
      top_logprobs:    log probs of top-b tokens              — (S, b)
    """
    log_probs = F.log_softmax(logits, dim=-1)                        # (S, vocab)
    chosen = torch.tensor(token_ids, dtype=torch.long, device=logits.device)
    chosen_lp = log_probs[torch.arange(len(token_ids), device=logits.device), chosen]

    top_lp, top_ids = torch.topk(log_probs, k=b, dim=-1)            # (S, b)

    return (
        chosen_lp.tolist(),
        top_ids.tolist(),
        top_lp.tolist(),
    )


# ---------------------------------------------------------------------------
# Tree generation for one sequence
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_trees_for_sequence(
    model: AutoModelForCausalLM,
    record: dict,
    b: int,
    d: int,
    device: torch.device,
    anchor_threshold: float = 0.02,
    anchor_max_fraction: float = 0.5,
    ignore_idx: int = -100,
) -> dict:
    """
    Generate continuation trees for one sequence and merge results into record.

    Handles records with or without pre-computed logprobs.
    Uses Flex Attention for all tree-masked forward passes.
    """
    prompt_ids = record["prompt_token_ids"]
    completion_ids = record["completion_token_ids"]
    full_ids = prompt_ids + completion_ids
    S_prompt = len(prompt_ids)
    S_completion = len(completion_ids)
    S_full = len(full_ids)
    tree_size = sum(b**l for l in range(1, d + 1))

    # --- Resolve logprobs: compute from scratch if not stored ---
    if "chosen_logprobs" not in record or "top_token_ids" not in record:
        # One extra causal forward pass on the full sequence.
        # We only need logits at completion positions.
        logits = _causal_forward(model, full_ids, device)           # (S_full, vocab)
        completion_logits = logits[S_prompt:]                        # (S_completion, vocab)
        chosen_lp, top_ids_list, top_lp_list = _compute_logprobs_from_logits(
            completion_logits, completion_ids, b
        )
        record.setdefault("chosen_logprobs", chosen_lp)
        record.setdefault("top_token_ids", top_ids_list)
        record.setdefault("top_logprobs", top_lp_list)

    chosen_logprobs = record["chosen_logprobs"]
    top_token_ids_s1 = record["top_token_ids"]
    top_logprobs_s1 = record["top_logprobs"]

    # --- Anchor selection ---
    anchor_positions = select_anchors(
        chosen_logprobs, threshold=anchor_threshold, max_fraction=anchor_max_fraction
    )
    anchor_full_positions = [S_prompt + a for a in anchor_positions]
    K = len(anchor_positions)

    # Initialise output arrays (completion_len × tree_size), all ignore_idx
    tree_tokens = [[ignore_idx] * tree_size for _ in range(S_completion)]
    tree_logprobs_out = [[float("nan")] * tree_size for _ in range(S_completion)]

    if K == 0 or d == 0:
        record.update(
            anchor_positions=anchor_positions,
            tree_tokens=tree_tokens,
            tree_logprobs=tree_logprobs_out,
        )
        return record

    # --- Initialise tree node storage ---
    # tree_node_tokens[anchor_i][depth_l] = list of token_ids (length b^l)
    tree_node_tokens = [[[] for _ in range(d + 1)] for _ in range(K)]
    tree_node_logprobs = [[[] for _ in range(d + 1)] for _ in range(K)]

    # Level 1 is free from Stage 1 (or from the causal pass above)
    for anchor_i, comp_pos in enumerate(anchor_positions):
        tree_node_tokens[anchor_i][1] = list(top_token_ids_s1[comp_pos][:b])
        tree_node_logprobs[anchor_i][1] = list(top_logprobs_s1[comp_pos][:b])

    # --- Levels 2..d: one Flex Attention forward pass per level ---
    for current_depth in range(2, d + 1):
        prev_depth = current_depth - 1

        # Build packed input: full_ids followed by tree nodes at depths 1..prev_depth
        packed = list(full_ids)
        for depth_l in range(1, current_depth):
            for anchor_i in range(K):
                for path_enc in range(b ** depth_l):
                    packed.append(tree_node_tokens[anchor_i][depth_l][path_enc])

        total_len = len(packed)
        input_ids = torch.tensor(packed, dtype=torch.long, device=device).unsqueeze(0)

        # Build tree layout for the nodes present so far (depths 1..prev_depth)
        layout = build_tree_layout(S_full, anchor_full_positions, b, prev_depth, device)
        block_mask = build_tree_block_mask(layout, total_len, device)

        # Forward pass with Flex Attention tree mask
        with _flex_attention_context(block_mask):
            output = model(input_ids, use_cache=False)

        logits = output.logits[0]  # (total_len, vocab)

        # Extract logits at the prev_depth node positions to predict current_depth children
        depth_offset = S_full + sum(K * b**l for l in range(1, prev_depth))
        num_nodes_at_prev = K * (b ** prev_depth)
        node_logits = logits[depth_offset : depth_offset + num_nodes_at_prev]  # (num_nodes, vocab)

        log_probs = F.log_softmax(node_logits, dim=-1)
        top_lp, top_ids = torch.topk(log_probs, k=b, dim=-1)   # (num_nodes, b)

        # Unpack into tree_node_tokens[anchor_i][current_depth]
        for anchor_i in range(K):
            new_toks, new_lps = [], []
            for path_enc in range(b ** prev_depth):
                flat_in_level = anchor_i * (b ** prev_depth) + path_enc
                for branch_j in range(b):
                    new_toks.append(top_ids[flat_in_level, branch_j].item())
                    new_lps.append(top_lp[flat_in_level, branch_j].item())
            tree_node_tokens[anchor_i][current_depth] = new_toks
            tree_node_logprobs[anchor_i][current_depth] = new_lps

    # --- Pack results into [completion_len, tree_size] ---
    for anchor_i, comp_pos in enumerate(anchor_positions):
        flat_idx = 0
        for depth_l in range(1, d + 1):
            for path_enc in range(b ** depth_l):
                tree_tokens[comp_pos][flat_idx] = tree_node_tokens[anchor_i][depth_l][path_enc]
                tree_logprobs_out[comp_pos][flat_idx] = tree_node_logprobs[anchor_i][depth_l][path_enc]
                flat_idx += 1

    record.update(
        anchor_positions=anchor_positions,
        tree_tokens=tree_tokens,
        tree_logprobs=tree_logprobs_out,
    )
    return record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Target model (HF name or local path), e.g. Qwen/Qwen3-8B")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL (Stage 1 output, with or without logprobs)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL path")
    parser.add_argument("--branching-factor", type=int, default=2)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--anchor-threshold", type=float, default=0.02)
    parser.add_argument("--anchor-max-fraction", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # sdpa: required so that _flex_attention_context can patch F.scaled_dot_product_attention
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
        device_map=str(device),
    ).eval()

    written = 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            record = json.loads(line)
            record = generate_trees_for_sequence(
                model, record,
                b=args.branching_factor,
                d=args.depth,
                device=device,
                anchor_threshold=args.anchor_threshold,
                anchor_max_fraction=args.anchor_max_fraction,
            )
            fout.write(json.dumps(record) + "\n")
            written += 1
            if written % 100 == 0:
                print(f"Processed {written} sequences")

    print(f"Done. Wrote {written} records to {args.output}")


if __name__ == "__main__":
    main()
