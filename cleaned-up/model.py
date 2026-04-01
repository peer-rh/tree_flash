from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from src.models.dflash import DFlashDraftModel
from src.trees.prunable import PrunableTreeProcessor


RANK_RELATION_CAP = 8

REL_SELF = 0
REL_PARENT = 1
REL_CHILD = 2
REL_SIBLING = 3
REL_ANCESTOR = 4
REL_DESCENDANT = 5
REL_OTHER = 6

REL_PARENT_RANK_BASE = 7
REL_CHILD_RANK_BASE = REL_PARENT_RANK_BASE + RANK_RELATION_CAP
REL_SIBLING_RANK_BASE = REL_CHILD_RANK_BASE + RANK_RELATION_CAP
RELATION_VOCAB_SIZE = REL_SIBLING_RANK_BASE + (RANK_RELATION_CAP * RANK_RELATION_CAP)


def clamp_relation_rank(rank: int) -> int:
    """Clamp child-rank ids into the finite relation vocabulary."""
    rank = int(rank)
    if rank <= 0:
        return 0
    return min(rank, RANK_RELATION_CAP)


def relation_id_for_parent_rank(rank: int) -> int:
    bucket = clamp_relation_rank(rank)
    if bucket == 0:
        return REL_PARENT
    return REL_PARENT_RANK_BASE + bucket - 1


def relation_id_for_child_rank(rank: int) -> int:
    bucket = clamp_relation_rank(rank)
    if bucket == 0:
        return REL_CHILD
    return REL_CHILD_RANK_BASE + bucket - 1


def relation_id_for_sibling_ranks(rank_i: int, rank_j: int) -> int:
    bucket_i = clamp_relation_rank(rank_i)
    bucket_j = clamp_relation_rank(rank_j)
    if bucket_i == 0 or bucket_j == 0:
        return REL_SIBLING
    return REL_SIBLING_RANK_BASE + ((bucket_i - 1) * RANK_RELATION_CAP) + (bucket_j - 1)


@dataclass
class TreeInfo:
    """Dynamic tree metadata consumed by the drafter.

    Shapes:
    - `tree_mask`: `(batch_size, num_anchors, tree_size, tree_size)`
    - `parent_idx`: `(batch_size, num_anchors, tree_size)`
    - `depth`: `(batch_size, num_anchors, tree_size)`
    - `relation_map`: `(batch_size, num_anchors, tree_size, tree_size)`
    - `tree_position_ids`: `(batch_size, num_anchors, tree_size)`
    - `non_root_mask`: `(batch_size, num_anchors, tree_size)`
    - `primary_path_mask`: `(batch_size, num_anchors, tree_size)`
    - `primary_path_indices`: `(batch_size, num_anchors, tree_size)` with `-1`
      padding after the valid primary-path indices.
    """

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


def _build_single_tree_relations(
    *,
    parent_idx: torch.Tensor,
    node_ranks: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ancestor visibility and relation ids for one padded tree."""
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

    relation_map = torch.full((block_size, block_size), REL_OTHER, dtype=torch.long, device=device)
    for q_idx in range(block_size):
        for k_idx in range(block_size):
            if not bool(valid_mask[q_idx]) or not bool(valid_mask[k_idx]):
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
    """Pack primary-path indices into a padded tensor with `-1` tail padding."""
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


def build_dynamic_tree_info_from_batch(
    *,
    tree_parent_indices: torch.Tensor,
    tree_depths: torch.Tensor,
    tree_node_ranks: torch.Tensor,
    tree_position_ids: torch.Tensor,
    tree_valid_mask: torch.Tensor,
    tree_primary_path_mask: torch.Tensor,
) -> TreeInfo:
    """Build batch-local tree metadata from padded subtree tensors.

    Args:
        `tree_parent_indices`: `(batch_size, num_anchors, tree_size)`
        `tree_depths`: `(batch_size, num_anchors, tree_size)`
        `tree_node_ranks`: `(batch_size, num_anchors, tree_size)`
        `tree_position_ids`: `(batch_size, num_anchors, tree_size)`
        `tree_valid_mask`: `(batch_size, num_anchors, tree_size)`
        `tree_primary_path_mask`: `(batch_size, num_anchors, tree_size)`
    """
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

    tree_mask = torch.stack(tree_masks, dim=0)
    relation_map = torch.stack(relation_maps, dim=0)
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


def load_drafter_model(source: dict[str, Any] | str, *, torch_dtype: torch.dtype) -> DFlashDraftModel:
    """Load the single supported drafter variant.

    This cleaned-up path requires:
    - q-head enabled
    - no AR head
    """
    if isinstance(source, str):
        model = DFlashDraftModel.from_pretrained(source, torch_dtype=torch_dtype)
    else:
        model = DFlashDraftModel(source)
    raw = model
    if getattr(raw, "q_head", None) is None:
        raise ValueError("The cleaned-up trainer requires a drafter with `use_q_head=True`.")
    if getattr(raw, "ar_block", None) is not None:
        raise ValueError("The cleaned-up trainer does not support the AR head.")
    return model


def build_inference_tree(
    *,
    tree_seq_depth: int,
    branching_pattern: Sequence[Sequence[int]],
    candidate_tree_size: int,
    sub_tree_paths: Sequence[str] | None = None,
):
    """Build the single supported inference tree.

    The cleaned-up inference path always uses:
    - a fixed branch-off layout for drafting
    - q-head pruning to `candidate_tree_size`
    """
    return PrunableTreeProcessor(
        tree_seq_depth=tree_seq_depth,
        base_tree_type="branch_off",
        candidate_tree_size=int(candidate_tree_size),
        branching_pattern=branching_pattern,
        sub_tree_paths=sub_tree_paths,
    )
