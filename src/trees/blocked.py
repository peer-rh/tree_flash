from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from data_pipeline.stage2 import DEFAULT_SUB_TREE_PATHS, IGNORE_IDX, SubTreeInfo
from .relation_ids import (
    REL_ANCESTOR,
    REL_CHILD,
    REL_DESCENDANT,
    REL_OTHER,
    REL_PARENT,
    REL_SELF,
    REL_SIBLING,
)


def _build_layout_vectorization_tensors(
    subtree: SubTreeInfo,
    flat_to_depth_vertex: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_depth_indices = flat_to_depth_vertex[:, 0].clone()
    flat_vertex_indices = flat_to_depth_vertex[:, 1].clone()
    flat_depth_offsets = torch.tensor(
        [subtree.depth_of[int(vertex_idx)] for vertex_idx in flat_vertex_indices.tolist()],
        dtype=torch.long,
    )

    path_rows: list[list[int]] = []
    max_path_len = 0
    for vertex_idx in flat_vertex_indices.tolist():
        local_path: list[int] = []
        cur = int(vertex_idx)
        while cur != 0:
            local_path.append(cur)
            cur = subtree.parent_map[cur]
        path = list(reversed(local_path))
        path_rows.append(path)
        max_path_len = max(max_path_len, len(path))

    flat_path_indices = torch.zeros((flat_vertex_indices.numel(), max_path_len), dtype=torch.long)
    flat_path_mask = torch.zeros((flat_vertex_indices.numel(), max_path_len), dtype=torch.bool)
    for flat_idx, path in enumerate(path_rows):
        if not path:
            continue
        path_len = len(path)
        flat_path_indices[flat_idx, :path_len] = torch.tensor(path, dtype=torch.long)
        flat_path_mask[flat_idx, :path_len] = True

    return (
        flat_depth_indices,
        flat_vertex_indices,
        flat_depth_offsets,
        flat_path_indices,
        flat_path_mask,
    )


def _build_empty_anchor_tensors(
    *,
    block_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    empty_tokens = torch.empty((0, block_size), dtype=torch.long, device=device)
    return {
        "tree_labels": empty_tokens,
        "tree_noise_ids": empty_tokens.clone(),
        "tree_position_ids": empty_tokens.clone(),
        "tree_cum_probs": torch.empty((0, block_size), dtype=torch.float32, device=device),
        "tree_valid_mask": torch.empty((0, block_size), dtype=torch.bool, device=device),
    }


def _build_vectorized_anchor_tensors(
    *,
    response_subtrees: torch.Tensor,
    response_probs: torch.Tensor,
    anchor_local_positions: Sequence[int],
    anchor_positions: Sequence[int],
    mask_token_id: int,
    tree_seq_depth: int,
    block_size: int,
    flat_depth_indices: torch.Tensor,
    flat_vertex_indices: torch.Tensor,
    flat_depth_offsets: torch.Tensor,
    flat_path_indices: torch.Tensor,
    flat_path_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    device = response_subtrees.device
    n_anchors = len(anchor_local_positions)
    if n_anchors == 0:
        return _build_empty_anchor_tensors(block_size=block_size, device=device)

    anchor_locals = torch.as_tensor(anchor_local_positions, dtype=torch.long, device=device)
    anchor_positions_tensor = torch.as_tensor(anchor_positions, dtype=torch.long, device=device)
    row_offsets = torch.arange(tree_seq_depth, dtype=torch.long, device=device)
    row_indices = anchor_locals.unsqueeze(1) + row_offsets.unsqueeze(0)

    gathered_subtrees = response_subtrees.index_select(0, row_indices.reshape(-1)).view(
        n_anchors,
        tree_seq_depth,
        -1,
    )
    gathered_probs = response_probs.index_select(0, row_indices.reshape(-1)).view(
        n_anchors,
        tree_seq_depth,
        -1,
    ).to(torch.float32)

    flat_depth_indices = flat_depth_indices.to(device=device)
    flat_vertex_indices = flat_vertex_indices.to(device=device)
    flat_depth_offsets = flat_depth_offsets.to(device=device)
    flat_path_indices = flat_path_indices.to(device=device)
    flat_path_mask = flat_path_mask.to(device=device)

    flat_subtrees = gathered_subtrees.index_select(1, flat_depth_indices)
    tree_labels = flat_subtrees.gather(
        -1,
        flat_vertex_indices.view(1, block_size, 1).expand(n_anchors, -1, 1),
    ).squeeze(-1)

    row_probs = gathered_probs.index_select(1, flat_depth_indices)
    if flat_path_indices.shape[1] == 0:
        branch_factors = torch.ones((n_anchors, block_size), dtype=torch.float32, device=device)
    else:
        local_probs = row_probs.gather(
            -1,
            flat_path_indices.view(1, block_size, -1).expand(n_anchors, -1, -1),
        )
        local_probs = local_probs.masked_fill(~flat_path_mask.view(1, block_size, -1), 1.0)
        branch_factors = local_probs.prod(dim=-1)

    root_prefix = gathered_probs[:, :, 0].cumprod(dim=1).index_select(1, flat_depth_indices)
    tree_cum_probs = root_prefix * branch_factors
    tree_position_ids = (
        anchor_positions_tensor.unsqueeze(1)
        + flat_depth_indices.unsqueeze(0)
        + flat_depth_offsets.unsqueeze(0)
    )
    tree_valid_mask = tree_labels != IGNORE_IDX
    tree_noise_ids = torch.full((n_anchors, block_size), mask_token_id, dtype=torch.long, device=device)
    valid_roots = tree_valid_mask[:, 0]
    tree_noise_ids[valid_roots, 0] = tree_labels[valid_roots, 0]

    return {
        "tree_labels": tree_labels,
        "tree_noise_ids": tree_noise_ids,
        "tree_position_ids": tree_position_ids,
        "tree_cum_probs": tree_cum_probs,
        "tree_valid_mask": tree_valid_mask,
    }


@dataclass
class TreeInfo:
    tree_mask: torch.Tensor
    parent_idx: torch.Tensor
    depth: torch.Tensor
    relation_map: torch.Tensor
    tree_position_ids: torch.Tensor
    non_root_mask: torch.Tensor
    primary_path_mask: torch.Tensor
    primary_path_indices: torch.Tensor
    block_size: int
    num_blocks: int


def _build_relation_template_from_parent(
    parent_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_size = int(parent_idx.numel())
    tree_mask = torch.zeros((block_size, block_size), dtype=torch.bool, device=parent_idx.device)
    for node_idx in range(block_size):
        cur = node_idx
        while cur >= 0:
            tree_mask[node_idx, cur] = True
            cur = int(parent_idx[cur].item())

    relation_template = torch.full(
        (block_size, block_size),
        REL_OTHER,
        dtype=torch.long,
        device=parent_idx.device,
    )
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
    return tree_mask, relation_template


def subset_tree_info(
    tree_info: TreeInfo,
    keep_indices: Sequence[int] | torch.Tensor,
) -> TreeInfo:
    keep = torch.as_tensor(keep_indices, dtype=torch.long, device=tree_info.parent_idx.device)
    if keep.ndim != 1 or keep.numel() == 0:
        raise ValueError("keep_indices must be a non-empty 1D sequence.")

    block_size = int(tree_info.block_size)
    if int(keep[0].item()) != 0:
        raise ValueError("keep_indices must include the root node at index 0.")
    if bool((keep[1:] <= keep[:-1]).any().item()):
        raise ValueError("keep_indices must be strictly increasing.")
    if int(keep[-1].item()) >= block_size:
        raise ValueError("keep_indices contains an out-of-range node index.")

    remap = torch.full((block_size,), -1, dtype=torch.long, device=keep.device)
    remap[keep] = torch.arange(keep.numel(), dtype=torch.long, device=keep.device)
    parent_idx = torch.full((keep.numel(),), -1, dtype=torch.long, device=keep.device)

    for new_idx, old_idx in enumerate(keep.tolist()):
        old_parent = int(tree_info.parent_idx[old_idx].item())
        if old_parent >= 0:
            new_parent = int(remap[old_parent].item())
            if new_parent < 0:
                raise ValueError("keep_indices must be ancestor-closed.")
            parent_idx[new_idx] = new_parent

    depth = tree_info.depth.index_select(0, keep)
    non_root_mask = parent_idx >= 0
    primary_path_mask = tree_info.primary_path_mask.index_select(0, keep)
    primary_path_indices = torch.nonzero(primary_path_mask, as_tuple=False).squeeze(-1)
    tree_mask, relation_template = _build_relation_template_from_parent(parent_idx)

    batch_size, num_blocks = tree_info.relation_map.shape[:2]
    relation_map = relation_template.view(1, 1, keep.numel(), keep.numel()).expand(
        batch_size,
        num_blocks,
        -1,
        -1,
    ).contiguous()
    tree_position_ids = tree_info.tree_position_ids.index_select(-1, keep).contiguous()

    return TreeInfo(
        tree_mask=tree_mask,
        parent_idx=parent_idx,
        depth=depth,
        relation_map=relation_map,
        tree_position_ids=tree_position_ids,
        non_root_mask=non_root_mask,
        primary_path_mask=primary_path_mask,
        primary_path_indices=primary_path_indices,
        block_size=int(keep.numel()),
        num_blocks=tree_info.num_blocks,
    )


class BlockTreeProcessor:
    def __init__(
        self,
        tree_seq_depth: int,
        sub_tree_paths: Sequence[str] | None = None,
    ) -> None:
        self.tree_seq_depth = tree_seq_depth
        self.sub_tree_paths = tuple(sub_tree_paths or DEFAULT_SUB_TREE_PATHS)
        self.subtree = SubTreeInfo(self.sub_tree_paths)
        self.subtree_size = self.subtree.size
        self.layout = [
            (depth_idx, vertex_idx)
            for depth_idx in range(self.tree_seq_depth)
            for vertex_idx in range(self.subtree_size)
        ]
        self.flat_to_depth_vertex = torch.tensor(self.layout, dtype=torch.long)
        self.block_size = self.tree_seq_depth * self.subtree_size
        (
            self.flat_depth_indices,
            self.flat_vertex_indices,
            self.flat_depth_offsets,
            self.flat_path_indices,
            self.flat_path_mask,
        ) = _build_layout_vectorization_tensors(self.subtree, self.flat_to_depth_vertex)
        (
            self.parent_idx,
            self.depth,
            self.tree_mask,
            self.relation_template,
            self.tree_position_template,
            self.non_root_mask,
            self.primary_path_mask,
            self.primary_path_indices,
        ) = self._build_templates()

    def _build_templates(
        self,
    ) -> tuple[torch.Tensor, ...]:
        parent_idx = torch.full((self.block_size,), -1, dtype=torch.long)
        depth = torch.zeros((self.block_size,), dtype=torch.long)
        tree_position_template = torch.arange(self.block_size, dtype=torch.long)
        non_root_mask = torch.ones((self.block_size,), dtype=torch.bool)
        primary_path_mask = torch.zeros((self.block_size,), dtype=torch.bool)
        primary_path_indices = []

        for depth_idx in range(self.tree_seq_depth):
            for vertex_idx in range(self.subtree_size):
                flat_idx = depth_idx * self.subtree_size + vertex_idx
                depth[flat_idx] = depth_idx + self.subtree.depth_of[vertex_idx]
                if depth_idx == 0 and vertex_idx == 0:
                    non_root_mask[flat_idx] = False
                    primary_path_mask[flat_idx] = True
                    primary_path_indices.append(flat_idx)
                    if depth_idx > 0:
                        parent_idx[flat_idx] = (depth_idx - 1) * self.subtree_size
                elif vertex_idx == 0:
                    primary_path_mask[flat_idx] = True
                    primary_path_indices.append(flat_idx)
                    parent_idx[flat_idx] = (depth_idx - 1) * self.subtree_size
                else:
                    parent_idx[flat_idx] = depth_idx * self.subtree_size + self.subtree.parent_map[vertex_idx]

        tree_mask, relation_template = _build_relation_template_from_parent(parent_idx)

        return (
            parent_idx,
            depth,
            tree_mask,
            relation_template,
            tree_position_template,
            non_root_mask,
            primary_path_mask,
            torch.tensor(primary_path_indices, dtype=torch.long),
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
