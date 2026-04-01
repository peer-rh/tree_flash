from __future__ import annotations

from typing import Sequence

import torch

from .blocked import (
    REL_ANCESTOR,
    REL_DESCENDANT,
    REL_OTHER,
    REL_SELF,
    REL_SIBLING,
    TreeInfo,
)
from .relation_ids import (
    relation_id_for_child_rank,
    relation_id_for_parent_rank,
    relation_id_for_sibling_ranks,
)


def _build_single_tree_relations(
    *,
    parent_idx: torch.Tensor,
    node_ranks: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ancestor visibility and categorical relations for one padded tree."""
    block_size = int(parent_idx.numel())
    device = parent_idx.device
    tree_mask = torch.zeros((block_size, block_size), dtype=torch.bool, device=device)

    for node_idx in range(block_size):
        if not bool(valid_mask[node_idx]):
            continue
        cur = node_idx
        while cur >= 0:
            if not bool(valid_mask[cur]):
                break
            tree_mask[node_idx, cur] = True
            cur = int(parent_idx[cur].item())

    relation_map = torch.full(
        (block_size, block_size),
        REL_OTHER,
        dtype=torch.long,
        device=device,
    )
    for q_idx in range(block_size):
        for k_idx in range(block_size):
            q_valid = bool(valid_mask[q_idx])
            k_valid = bool(valid_mask[k_idx])
            if not q_valid or not k_valid:
                continue
            if q_idx == k_idx:
                relation_map[q_idx, k_idx] = REL_SELF
            elif int(parent_idx[q_idx].item()) == k_idx:
                relation_map[q_idx, k_idx] = relation_id_for_parent_rank(int(node_ranks[q_idx].item()))
            elif int(parent_idx[k_idx].item()) == q_idx:
                relation_map[q_idx, k_idx] = relation_id_for_child_rank(int(node_ranks[k_idx].item()))
            elif (
                int(parent_idx[q_idx].item()) >= 0
                and int(parent_idx[q_idx].item()) == int(parent_idx[k_idx].item())
            ):
                relation_map[q_idx, k_idx] = relation_id_for_sibling_ranks(
                    int(node_ranks[q_idx].item()),
                    int(node_ranks[k_idx].item()),
                )
            elif bool(tree_mask[q_idx, k_idx]):
                relation_map[q_idx, k_idx] = REL_ANCESTOR
            elif bool(tree_mask[k_idx, q_idx]):
                relation_map[q_idx, k_idx] = REL_DESCENDANT
    return tree_mask, relation_map


def _build_primary_path_indices(primary_path_mask: torch.Tensor) -> torch.Tensor:
    """Pack primary-path node indices into a padded ``(..., block_size)`` tensor."""
    batch_size, num_blocks, block_size = primary_path_mask.shape
    primary_indices = torch.full(
        (batch_size, num_blocks, block_size),
        -1,
        dtype=torch.long,
        device=primary_path_mask.device,
    )
    for batch_idx in range(batch_size):
        for block_idx in range(num_blocks):
            keep = torch.nonzero(primary_path_mask[batch_idx, block_idx], as_tuple=False).squeeze(-1)
            if keep.numel() > 0:
                primary_indices[batch_idx, block_idx, : keep.numel()] = keep
    return primary_indices


class VarTreeProcessor:
    """Dynamic tree processor for Stage 2 v2 sampled subtrees.

    ``VarTree`` does not own a compile-time layout. Instead the loader provides
    padded per-anchor parent/depth/primary-path tensors and this processor turns
    them into a batch-local ``TreeInfo`` instance.
    """

    tree_seq_depth = 0
    block_size = 0
    sub_tree_paths: tuple[str, ...] = ()
    training_only = True

    def build_anchor_tensors(self, **kwargs):
        _ = kwargs
        raise NotImplementedError("VarTree does not build fixed anchor tensors.")

    def build_tree_info(
        self,
        batch_size: int,
        num_blocks: int,
        device: torch.device,
    ) -> TreeInfo:
        _ = batch_size, num_blocks, device
        raise NotImplementedError("VarTree requires batch-provided subtree metadata.")

    def build_tree_info_from_batch(
        self,
        *,
        tree_parent_indices: torch.Tensor,
        tree_depths: torch.Tensor,
        tree_node_ranks: torch.Tensor,
        tree_position_ids: torch.Tensor,
        tree_valid_mask: torch.Tensor,
        tree_primary_path_mask: torch.Tensor,
    ) -> TreeInfo:
        """Build a dynamic ``TreeInfo`` from padded per-anchor subtree tensors."""
        batch_size, num_blocks, block_size = tree_parent_indices.shape
        tree_masks = []
        relation_maps = []
        for batch_idx in range(batch_size):
            batch_masks = []
            batch_relations = []
            for block_idx in range(num_blocks):
                mask, relations = _build_single_tree_relations(
                    parent_idx=tree_parent_indices[batch_idx, block_idx],
                    node_ranks=tree_node_ranks[batch_idx, block_idx],
                    valid_mask=tree_valid_mask[batch_idx, block_idx],
                )
                batch_masks.append(mask)
                batch_relations.append(relations)
            tree_masks.append(torch.stack(batch_masks, dim=0))
            relation_maps.append(torch.stack(batch_relations, dim=0))

        relation_map = torch.stack(relation_maps, dim=0)
        tree_mask = torch.stack(tree_masks, dim=0)
        non_root_mask = (tree_parent_indices >= 0) & tree_valid_mask
        primary_path_mask = tree_primary_path_mask & tree_valid_mask
        primary_path_indices = _build_primary_path_indices(primary_path_mask)

        return TreeInfo(
            tree_mask=tree_mask,
            parent_idx=tree_parent_indices,
            depth=tree_depths,
            relation_map=relation_map,
            tree_position_ids=tree_position_ids,
            non_root_mask=non_root_mask,
            primary_path_mask=primary_path_mask,
            primary_path_indices=primary_path_indices,
            block_size=block_size,
            num_blocks=num_blocks,
        )


__all__: Sequence[str] = ["VarTreeProcessor"]
