"""
Quick visualisation of one subtree from a stage2 v2 HDF5 file.

Usage:
    python v2/viz_tree.py data/stage2_v2.h5 --model Qwen/Qwen3-8B
    python v2/viz_tree.py data/stage2_v2.h5 --model Qwen/Qwen3-8B --seq 3 --pos 12
"""

import argparse
import h5py
import numpy as np
from transformers import AutoTokenizer

IGNORE_IDX = -1


def pick_position(sub_trees: np.ndarray, seq_slice: slice, prefer_pos: int | None):
    """Find a response position that has at least one non-root alternative filled."""
    rows = sub_trees[seq_slice]  # [S_R, sub_tree_size]
    for t in range(rows.shape[0]):
        if prefer_pos is not None and t != prefer_pos:
            continue
        if np.any(rows[t, 1:] != IGNORE_IDX):
            return t
    # fallback: first row
    return 0


def render_tree(
    t: int,
    sub_trees_row: np.ndarray,       # [sub_tree_size]
    sub_trees_probs_row: np.ndarray,  # [sub_tree_size]
    sub_tree_paths: list[str],
    tokenizer,
    response_ids: np.ndarray,
    prompt_len: int,
) -> str:
    # Parse edge list into children map
    from collections import defaultdict
    parent_map: dict[int, int] = {}
    all_nodes: set[int] = {0}
    for edge in sub_tree_paths:
        p, c = map(int, edge.split("-"))
        parent_map[c] = p
        all_nodes |= {p, c}
    non_root = sorted(all_nodes - {0})
    local_idx = {0: 0}
    for j, n in enumerate(non_root, 1):
        local_idx[n] = j
    children: dict[int, list[int]] = defaultdict(list)
    for c, p in parent_map.items():
        children[p].append(c)

    def tok_str(node_id: int) -> str:
        slot = local_idx[node_id]
        tok = int(sub_trees_row[slot])
        prob = float(sub_trees_probs_row[slot])
        if tok == IGNORE_IDX:
            return f"<missing> (p={prob:.3f})"
        text = repr(tokenizer.decode([tok]))
        return f"{text} [{tok}] (p={prob:.3f})"

    lines: list[str] = []
    lines.append(f"Response position t={t}  (absolute={prompt_len + t})")
    lines.append(f"  Root (node 0): {tok_str(0)}")

    def _render(node: int, prefix: str, is_last: bool):
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}node {node}: {tok_str(node)}")
        child_list = sorted(children.get(node, []))
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(child_list):
            _render(child, child_prefix, i == len(child_list) - 1)

    root_children = sorted(children.get(0, []))
    for i, child in enumerate(root_children):
        _render(child, "  ", i == len(root_children) - 1)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5", help="Path to stage2 v2 HDF5 file")
    parser.add_argument("--model", required=True, help="HF model name for tokenizer")
    parser.add_argument("--seq", type=int, default=0, help="Sequence index (default: 0)")
    parser.add_argument("--pos", type=int, default=None, help="Response position t (default: first filled)")
    parser.add_argument(
        "--sub-tree-paths", nargs="+",
        default=["0-1", "0-2", "0-3", "1-4", "1-5", "2-6", "2-7"],
        help="Subtree edges (must match those used in stage2)",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    with h5py.File(args.hdf5, "r") as hf:
        N = hf["sequence_offsets"].shape[0] - 1
        seq = args.seq % N
        offsets = hf["sequence_offsets"][:]
        start, end = int(offsets[seq]), int(offsets[seq + 1])

        sub_trees = hf["sub_trees"][start:end]           # [S_R, ST]
        sub_trees_probs = hf["sub_trees_ar_probs"][start:end]
        prompt_ids = hf["prompt_ids"][seq]
        response_ids = hf["response_ids"][seq]
        prompt_len = len(prompt_ids)

    print(f"\n=== Sequence {seq} | prompt_len={prompt_len} | response_len={len(response_ids)} ===")
    print(f"sub_tree_size={sub_trees.shape[1]}\n")

    t = pick_position(sub_trees, slice(None), args.pos)
    tree_str = render_tree(
        t=t,
        sub_trees_row=sub_trees[t],
        sub_trees_probs_row=sub_trees_probs[t],
        sub_tree_paths=args.sub_tree_paths,
        tokenizer=tokenizer,
        response_ids=response_ids,
        prompt_len=prompt_len,
    )
    print(tree_str)

    # Also show context around t
    ctx_start = max(0, t - 3)
    ctx_end = min(len(response_ids), t + 4)
    ctx_toks = response_ids[ctx_start:ctx_end]
    ctx_text = tokenizer.decode(ctx_toks)
    print(f"\nContext around t={t}: ...{ctx_text!r}...")


if __name__ == "__main__":
    main()
