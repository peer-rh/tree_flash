from __future__ import annotations

from typing import Sequence

import torch

from .blocked import BlockTreeProcessor, TreeInfo
from .branch_off import BranchOffTreeProcessor


class PrunableTreeProcessor:
    def __init__(
        self,
        tree_seq_depth: int,
        *,
        base_tree_type: str = "block",
        candidate_tree_size: int = 1,
        branching_pattern: Sequence[Sequence[int]] | None = None,
        sub_tree_paths: Sequence[str] | None = None,
    ) -> None:
        if base_tree_type not in {"block", "branch_off"}:
            raise ValueError(
                "prunable tree base_tree_type must be 'block' or 'branch_off', "
                f"got {base_tree_type!r}."
            )
        if candidate_tree_size <= 0:
            raise ValueError(f"candidate_tree_size must be > 0, got {candidate_tree_size}.")

        self.tree_seq_depth = tree_seq_depth
        self.base_tree_type = base_tree_type
        self.candidate_tree_size = int(candidate_tree_size)
        if base_tree_type == "block":
            self.base_tree_processor = BlockTreeProcessor(
                tree_seq_depth=tree_seq_depth,
                sub_tree_paths=sub_tree_paths,
            )
        else:
            self.base_tree_processor = BranchOffTreeProcessor(
                tree_seq_depth=tree_seq_depth,
                sub_tree_paths=sub_tree_paths,
                branching_pattern=branching_pattern,
            )

        self.sub_tree_paths = self.base_tree_processor.sub_tree_paths
        self.subtree = self.base_tree_processor.subtree
        self.subtree_size = self.base_tree_processor.subtree_size
        self.layout = self.base_tree_processor.layout
        self.flat_to_depth_vertex = self.base_tree_processor.flat_to_depth_vertex
        self.block_size = self.base_tree_processor.block_size
        self.parent_idx = self.base_tree_processor.parent_idx
        self.depth = self.base_tree_processor.depth
        self.tree_mask = self.base_tree_processor.tree_mask
        self.relation_template = self.base_tree_processor.relation_template
        self.tree_position_template = self.base_tree_processor.tree_position_template
        self.non_root_mask = self.base_tree_processor.non_root_mask
        self.primary_path_mask = self.base_tree_processor.primary_path_mask
        self.primary_path_indices = self.base_tree_processor.primary_path_indices
        self.prune_topk = self.candidate_tree_size

    def build_tree_info(
        self,
        batch_size: int,
        num_blocks: int,
        device: torch.device,
    ) -> TreeInfo:
        return self.base_tree_processor.build_tree_info(batch_size=batch_size, num_blocks=num_blocks, device=device)

    def build_anchor_tensors(
        self,
        *,
        response_subtrees: torch.Tensor,
        response_probs: torch.Tensor,
        anchor_local_positions: Sequence[int],
        anchor_positions: Sequence[int],
        mask_token_id: int,
    ) -> dict[str, torch.Tensor]:
        return self.base_tree_processor.build_anchor_tensors(
            response_subtrees=response_subtrees,
            response_probs=response_probs,
            anchor_local_positions=anchor_local_positions,
            anchor_positions=anchor_positions,
            mask_token_id=mask_token_id,
        )
