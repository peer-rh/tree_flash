"""
Tree structure definition for Tree-v1 speculative decoding.

Tree layout
-----------
The full tree is parameterised by ``seq_depth`` and ``sub_tree_paths``.

Primary path (the left-most path of the full tree):
    node 0 → node 1 → … → node seq_depth-1
    parent_ids[0] = -1  (root)
    parent_ids[i] = i-1  for i in 1..seq_depth-1

Attached sub_tree at primary-path node i:
    sub_tree_paths is a list of 2-char strings, e.g. ["01","02","14"].
    "XY" means sub_tree node X has child Y (single-digit indices, node 0 = attachment point).
    For each primary-path node i, the non-root sub_tree nodes (all nodes except 0)
    are placed at absolute indices:
        base_i + local_j   where base_i = seq_depth + i * n_sub
    and local_j is the 0-based rank of sub_tree node j among non-root nodes sorted ascending.

Total tree size:
    tree_size = seq_depth + seq_depth * n_sub
    where n_sub = number of non-root sub_tree nodes (= len(unique non-zero nodes in paths))
"""

from __future__ import annotations
from itertools import combinations
from dataclasses import dataclass, field
from typing import Optional

import torch


def _parse_sub_tree(paths: list[str]) -> tuple[list[int], dict[int, int]]:
    """
    Parse sub_tree path strings into a sorted node list and parent map.

    Parameters
    ----------
    paths : e.g. ["01", "02", "14"]

    Returns
    -------
    non_root_nodes : sorted list of non-zero sub_tree node indices
    parent_map     : {child_node: parent_node} (0-indexed, 0 = attachment root)
    """
    parent_map: dict[int, int] = {}
    all_nodes: set[int] = {0}
    for path in paths:
        assert len(path) == 2, f"sub_tree path must be 2 chars, got {path!r}"
        p, c = int(path[0]), int(path[1])
        assert c not in parent_map, f"node {c} has duplicate parent in sub_tree"
        parent_map[c] = p
        all_nodes |= {p, c}
    non_root_nodes = sorted(all_nodes - {0})
    return non_root_nodes, parent_map


def _sub_tree_depth(node: int, parent_map: dict[int, int]) -> int:
    """Depth of a sub_tree node relative to the sub_tree root (node 0)."""
    depth = 0
    cur = node
    while cur in parent_map:
        cur = parent_map[cur]
        depth += 1
    return depth


@dataclass
class TreeSpec:
    """
    Fully describes the verification tree and exposes derived tensors used
    throughout training, loss computation and validation.

    All tensors are CPU tensors; move them to the target device as needed.

    Parameters
    ----------
    seq_depth       : length of the primary (left-most) path
    sub_tree_paths  : list of 2-char strings defining the attached sub_tree
                      (empty list → pure chain, no branching)

    Derived attributes
    ------------------
    tree_size            : int  — total number of nodes
    parent_ids           : [tree_size]         long  — -1 for root
    depths               : [tree_size]         long  — 0 for root
    ancestor_matrix      : [tree_size, tree_size] bool
                           ancestor_matrix[k, q] = True iff k is an ancestor-or-self of q
                           (key k can be attended by query q in the verification pass)
    sibling_pairs        : [n_pairs, 2]        long  — pairs (i, j) with i<j, same parent
    adjusted_parent_ids  : [tree_size]         long
                           indexes into [anchor, tree_token_0, …, tree_token_{T-1}]
                           root → 0 (anchor slot), others → parent_ids[i] + 1
    position_ids         : [tree_size]         long
                           depth-based: position_ids[i] = depths[i]
                           (add ctx_len at the call site when building absolute positions)
    """

    seq_depth: int
    sub_tree_paths: list[str] = field(default_factory=list)

    # Derived — set in __post_init__
    tree_size: int = field(init=False)
    parent_ids: torch.Tensor = field(init=False)          # [tree_size]
    depths: torch.Tensor = field(init=False)              # [tree_size]
    ancestor_matrix: torch.Tensor = field(init=False)     # [tree_size, tree_size]
    sibling_pairs: torch.Tensor = field(init=False)       # [n_pairs, 2]
    adjusted_parent_ids: torch.Tensor = field(init=False) # [tree_size]
    position_ids: torch.Tensor = field(init=False)        # [tree_size]

    def __post_init__(self) -> None:
        non_root_nodes, parent_map = _parse_sub_tree(self.sub_tree_paths)
        n_sub = len(non_root_nodes)  # non-root sub_tree nodes per attachment point

        # local_idx: sub_tree node number → 0-based rank among non-root nodes
        local_idx: dict[int, int] = {n: i for i, n in enumerate(non_root_nodes)}

        self.tree_size = self.seq_depth + self.seq_depth * n_sub

        parent_ids_list = [-1] * self.tree_size
        depths_list = [0] * self.tree_size

        # ── Primary path ────────────────────────────────────────────────────
        for i in range(1, self.seq_depth):
            parent_ids_list[i] = i - 1
            depths_list[i] = i

        # ── Sub_tree copies attached at each primary-path node ───────────────
        for i in range(self.seq_depth):
            base = self.seq_depth + i * n_sub   # absolute index of first non-root node
            primary_depth = i                    # depth of attachment point

            for sub_node in non_root_nodes:
                j = local_idx[sub_node]          # 0-based rank
                abs_idx = base + j               # absolute index in full tree

                sub_parent = parent_map[sub_node]
                if sub_parent == 0:
                    # Parent is the primary-path node itself
                    parent_ids_list[abs_idx] = i
                else:
                    # Parent is another non-root sub_tree node
                    parent_ids_list[abs_idx] = base + local_idx[sub_parent]

                depths_list[abs_idx] = primary_depth + _sub_tree_depth(sub_node, parent_map)

        self.parent_ids = torch.tensor(parent_ids_list, dtype=torch.long)
        self.depths = torch.tensor(depths_list, dtype=torch.long)

        # ── Ancestor matrix ──────────────────────────────────────────────────
        # ancestor_matrix[k, q] = True iff k is on the path from root to q (inclusive)
        n = self.tree_size
        anc = torch.zeros(n, n, dtype=torch.bool)
        for q in range(n):
            cur = q
            while cur >= 0:
                anc[cur, q] = True
                cur = parent_ids_list[cur]
        self.ancestor_matrix = anc  # [tree_size, tree_size]

        # ── Sibling pairs ────────────────────────────────────────────────────
        # Group children by parent, then enumerate all pairs within each group
        from collections import defaultdict
        children_by_parent: dict[int, list[int]] = defaultdict(list)
        for node, par in enumerate(parent_ids_list):
            if par >= 0:
                children_by_parent[par].append(node)

        pairs: list[tuple[int, int]] = []
        for children in children_by_parent.values():
            pairs.extend(combinations(sorted(children), 2))

        if pairs:
            self.sibling_pairs = torch.tensor(sorted(pairs), dtype=torch.long)  # [n_pairs, 2]
        else:
            self.sibling_pairs = torch.zeros(0, 2, dtype=torch.long)

        # ── adjusted_parent_ids ──────────────────────────────────────────────
        # Used in DraftWrapper to index into [anchor_token, tree_token_0, …]:
        #   root (parent_ids = -1) → 0   (the anchor slot)
        #   others                 → parent_ids[i] + 1
        self.adjusted_parent_ids = self.parent_ids.clone()
        self.adjusted_parent_ids[self.adjusted_parent_ids == -1] = -1  # keep for clarity
        self.adjusted_parent_ids = self.adjusted_parent_ids + 1
        self.adjusted_parent_ids[0] = 0  # root → anchor at position 0

        # ── Depth-based position IDs ─────────────────────────────────────────
        # Add ctx_len at the call site for absolute position ids.
        self.position_ids = self.depths.clone()  # [tree_size]

    # ── Convenience helpers (CPU, not compiled) ──────────────────────────────

    def path_to_root(self, node: int) -> list[int]:
        """Return [node, parent, grandparent, …, root] for ``node``."""
        path = []
        cur = node
        while cur >= 0:
            path.append(cur)
            cur = self.parent_ids[cur].item()
        return path

    def leaves(self) -> list[int]:
        """Nodes with no children."""
        has_child = set(self.parent_ids[self.parent_ids >= 0].tolist())
        return [i for i in range(self.tree_size) if i not in has_child]

    def leftmost_path(self) -> list[int]:
        """The primary path: nodes 0..seq_depth-1."""
        return list(range(self.seq_depth))
