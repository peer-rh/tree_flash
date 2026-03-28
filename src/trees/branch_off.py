from __future__ import annotations

from typing import Sequence

import torch

from data_pipeline.stage2 import DEFAULT_SUB_TREE_PATHS, SubTreeInfo

from .blocked import (
    REL_ANCESTOR,
    REL_CHILD,
    REL_DESCENDANT,
    REL_OTHER,
    REL_PARENT,
    REL_SELF,
    REL_SIBLING,
    TreeInfo,
    _build_layout_vectorization_tensors,
    _build_vectorized_anchor_tensors,
)


class BranchOffTreeProcessor:
    def __init__(
        self,
        tree_seq_depth: int,
        *,
        branching_pattern: Sequence[Sequence[int]] | None = None,
        sub_tree_paths: Sequence[str] | None = None,
    ) -> None:
        self.tree_seq_depth = tree_seq_depth
        self.sub_tree_paths = tuple(sub_tree_paths or DEFAULT_SUB_TREE_PATHS)
        self.subtree = SubTreeInfo(self.sub_tree_paths)
        self.subtree_size = self.subtree.size
        self.branching_pattern = self._normalize_branching_pattern(branching_pattern)
        self.layout = self._build_layout()
        self.block_size = len(self.layout)
        (
            self.parent_idx,
            self.depth,
            self.tree_mask,
            self.relation_template,
            self.tree_position_template,
            self.non_root_mask,
            self.primary_path_mask,
            self.primary_path_indices,
            self.flat_to_depth_vertex,
        ) = self._build_templates()
        (
            self.flat_depth_indices,
            self.flat_vertex_indices,
            self.flat_depth_offsets,
            self.flat_path_indices,
            self.flat_path_mask,
        ) = _build_layout_vectorization_tensors(self.subtree, self.flat_to_depth_vertex)

    def _normalize_branching_pattern(
        self,
        branching_pattern: Sequence[Sequence[int]] | None,
    ) -> tuple[tuple[int, ...], ...]:
        if branching_pattern is None:
            default_vertices = tuple(range(self.subtree_size))
            return tuple(default_vertices for _ in range(self.tree_seq_depth))
        if len(branching_pattern) != self.tree_seq_depth:
            raise ValueError(
                "branching_pattern must contain one vertex list per tree depth: "
                f"expected {self.tree_seq_depth}, got {len(branching_pattern)}"
            )
        normalized: list[tuple[int, ...]] = []
        for depth_idx, vertices in enumerate(branching_pattern):
            selected = {0}
            for vertex in vertices:
                vertex_idx = int(vertex)
                if vertex_idx < 0 or vertex_idx >= self.subtree_size:
                    raise ValueError(
                        f"branching_pattern[{depth_idx}] contains invalid subtree vertex {vertex_idx}"
                    )
                cur = vertex_idx
                while True:
                    selected.add(cur)
                    if cur == 0:
                        break
                    cur = self.subtree.parent_map[cur]
            normalized.append(tuple(sorted(selected)))
        return tuple(normalized)

    def _build_layout(self) -> list[tuple[int, int]]:
        layout: list[tuple[int, int]] = []
        for depth_idx, selected_vertices in enumerate(self.branching_pattern):
            for vertex_idx in selected_vertices:
                layout.append((depth_idx, vertex_idx))
        return layout

    def _build_templates(
        self,
    ) -> tuple[torch.Tensor, ...]:
        block_size = len(self.layout)
        parent_idx = torch.full((block_size,), -1, dtype=torch.long)
        depth = torch.zeros((block_size,), dtype=torch.long)
        tree_position_template = torch.arange(block_size, dtype=torch.long)
        non_root_mask = torch.ones((block_size,), dtype=torch.bool)
        primary_path_mask = torch.zeros((block_size,), dtype=torch.bool)
        primary_path_indices: list[int] = []
        flat_to_depth_vertex = torch.tensor(self.layout, dtype=torch.long)
        lookup = {pair: idx for idx, pair in enumerate(self.layout)}

        for flat_idx, (depth_idx, vertex_idx) in enumerate(self.layout):
            depth[flat_idx] = depth_idx + self.subtree.depth_of[vertex_idx]
            if depth_idx == 0 and vertex_idx == 0:
                non_root_mask[flat_idx] = False
                primary_path_mask[flat_idx] = True
                primary_path_indices.append(flat_idx)
                if depth_idx > 0:
                    parent_idx[flat_idx] = lookup[(depth_idx - 1, 0)]
            elif vertex_idx == 0:
                primary_path_mask[flat_idx] = True
                primary_path_indices.append(flat_idx)
                parent_idx[flat_idx] = lookup[(depth_idx - 1, 0)]
            else:
                parent_idx[flat_idx] = lookup[(depth_idx, self.subtree.parent_map[vertex_idx])]

        tree_mask = torch.zeros((block_size, block_size), dtype=torch.bool)
        for node_idx in range(block_size):
            cur = node_idx
            while cur >= 0:
                tree_mask[node_idx, cur] = True
                cur = int(parent_idx[cur].item())

        relation_template = torch.full((block_size, block_size), REL_OTHER, dtype=torch.long)
        for q_idx in range(block_size):
            for k_idx in range(block_size):
                if q_idx == k_idx:
                    relation_template[q_idx, k_idx] = REL_SELF
                elif parent_idx[q_idx] == k_idx:
                    relation_template[q_idx, k_idx] = REL_PARENT
                elif parent_idx[k_idx] == q_idx:
                    relation_template[q_idx, k_idx] = REL_CHILD
                elif parent_idx[q_idx] >= 0 and parent_idx[q_idx] == parent_idx[k_idx]:
                    relation_template[q_idx, k_idx] = REL_SIBLING
                elif tree_mask[q_idx, k_idx]:
                    relation_template[q_idx, k_idx] = REL_ANCESTOR
                elif tree_mask[k_idx, q_idx]:
                    relation_template[q_idx, k_idx] = REL_DESCENDANT

        return (
            parent_idx,
            depth,
            tree_mask,
            relation_template,
            tree_position_template,
            non_root_mask,
            primary_path_mask,
            torch.tensor(primary_path_indices, dtype=torch.long),
            flat_to_depth_vertex,
        )

    def build_tree_info(
        self,
        batch_size: int,
        num_blocks: int,
        device: torch.device,
    ) -> TreeInfo:
        if num_blocks == 0:
            relation_map = torch.empty((batch_size, 0, self.block_size, self.block_size), dtype=torch.long, device=device)
            tree_position_ids = torch.empty((batch_size, 0, self.block_size), dtype=torch.long, device=device)
        else:
            relation_map = self.relation_template.to(device).view(1, 1, self.block_size, self.block_size)
            relation_map = relation_map.expand(batch_size, num_blocks, -1, -1).contiguous()
            tree_position_ids = self.tree_position_template.to(device).view(1, 1, self.block_size)
            tree_position_ids = tree_position_ids.expand(batch_size, num_blocks, -1).contiguous()
        return TreeInfo(
            tree_mask=self.tree_mask.to(device),
            parent_idx=self.parent_idx.to(device),
            depth=self.depth.to(device),
            relation_map=relation_map,
            tree_position_ids=tree_position_ids,
            non_root_mask=self.non_root_mask.to(device),
            primary_path_mask=self.primary_path_mask.to(device),
            primary_path_indices=self.primary_path_indices.to(device),
            block_size=self.block_size,
            num_blocks=num_blocks,
        )

    def build_anchor_tensors(
        self,
        *,
        response_subtrees: torch.Tensor,
        response_probs: torch.Tensor,
        anchor_local_positions: Sequence[int],
        anchor_positions: Sequence[int],
        mask_token_id: int,
    ) -> dict[str, torch.Tensor]:
        return _build_vectorized_anchor_tensors(
            response_subtrees=response_subtrees,
            response_probs=response_probs,
            anchor_local_positions=anchor_local_positions,
            anchor_positions=anchor_positions,
            mask_token_id=mask_token_id,
            tree_seq_depth=self.tree_seq_depth,
            block_size=self.block_size,
            flat_depth_indices=self.flat_depth_indices,
            flat_vertex_indices=self.flat_vertex_indices,
            flat_depth_offsets=self.flat_depth_offsets,
            flat_path_indices=self.flat_path_indices,
            flat_path_mask=self.flat_path_mask,
        )
