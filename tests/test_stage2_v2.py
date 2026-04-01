from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

from data_pipeline.stage2_v2 import (
    GeneratedAnchorTree,
    GeneratedSequenceTree,
    SequenceTreeNode,
    _select_anchor_positions_for_sequence,
    build_anchor_tree_from_candidate_provider,
    build_stage2_v2_runtime,
    compute_exact_token_rank,
    flush_stage2_v2_hdf5,
    initialize_stage2_v2_hdf5,
    merge_stage2_v2_parts,
)


def test_select_anchor_positions_uses_response_tokens_and_keeps_lowest_probs() -> None:
    is_response = torch.tensor([False, False, True, True, True, True], dtype=torch.bool)
    valid_tokens = torch.tensor([True, True, True, True, True, True], dtype=torch.bool)
    next_token_probs = torch.tensor([1.0, 0.9, 0.8, 0.1, 0.6, 0.2], dtype=torch.float32)

    positions, probs = _select_anchor_positions_for_sequence(
        is_response=is_response,
        valid_tokens=valid_tokens,
        next_token_probs=next_token_probs,
        alpha=0.65,
        max_anchors_per_sequence=2,
    )

    assert positions == [2, 4]
    assert probs == pytest.approx([0.1, 0.2], rel=1e-6)


def test_build_anchor_tree_excludes_forced_root_child_from_alternatives() -> None:
    def provider(node: SequenceTreeNode) -> tuple[list[int], list[float]]:
        if node.token_id == 31:
            return [41, 42], [0.6, 0.4]
        return [99], [1.0]

    tree = build_anchor_tree_from_candidate_provider(
        anchor_main_path_position=2,
        anchor_next_token_prob=0.35,
        main_path_ids=[10, 11, 21, 22, 23],
        response_start_position=2,
        num_attend_tokens_per_anchor=1,
        child_coverage_alpha=0.65,
        root_sorted_token_ids=[31, 22, 32, 40],
        root_sorted_token_probs=[0.4, 0.35, 0.2, 0.05],
        candidate_provider=provider,
    )

    root_children = [tree.nodes[idx].token_id for idx in range(1, 1 + tree.nodes[0].child_indices[1])]
    assert root_children == [22, 31]
    assert root_children.count(22) == 1


def test_build_anchor_tree_forced_main_path_child_and_path_probs() -> None:
    def provider(node: SequenceTreeNode) -> tuple[list[int], list[float]]:
        if node.token_id == 22:
            return [50, 23, 51], [0.4, 0.35, 0.15]
        return [99], [1.0]

    tree = build_anchor_tree_from_candidate_provider(
        anchor_main_path_position=2,
        anchor_next_token_prob=0.45,
        main_path_ids=[10, 11, 21, 22, 23, 24],
        response_start_position=2,
        num_attend_tokens_per_anchor=1,
        child_coverage_alpha=0.5,
        root_sorted_token_ids=[22, 31, 32],
        root_sorted_token_probs=[0.45, 0.3, 0.25],
        candidate_provider=provider,
    )

    root_child_count = tree.nodes[0].child_indices[1]
    assert root_child_count == 2

    expanded_non_root = [node for node in tree.nodes[1:] if node.child_indices[1] > 0]
    assert len(expanded_non_root) == 1
    assert expanded_non_root[0].token_id == 22

    child_tokens = [node.token_id for node in tree.nodes]
    assert 23 in child_tokens
    assert 50 in child_tokens

    node_23 = next(node for node in tree.nodes if node.token_id == 23)
    assert node_23.is_main_path is True
    assert node_23.main_path_position == 4
    assert node_23.path_prob == pytest.approx(0.45 * 0.35, rel=1e-6)


def test_compute_exact_token_rank_breaks_ties_by_token_id() -> None:
    logits = torch.tensor([1.0, 2.0, 2.0, 1.0], dtype=torch.float32)

    assert compute_exact_token_rank(logits, 1) == 1
    assert compute_exact_token_rank(logits, 2) == 2
    assert compute_exact_token_rank(logits, 0) == 3


def _make_anchor(anchor_position: int, next_prob: float, child_token: int) -> GeneratedAnchorTree:
    return GeneratedAnchorTree(
        anchor_main_path_position=anchor_position,
        anchor_next_token_prob=next_prob,
        nodes=[
            SequenceTreeNode(
                token_id=-1,
                parent_index=-1,
                depth=0,
                local_prob=1.0,
                path_prob=1.0,
                rank=0,
                main_path_position=-1,
                is_main_path=False,
                child_indices=[1, 1],
            ),
            SequenceTreeNode(
                token_id=child_token,
                parent_index=0,
                depth=1,
                local_prob=next_prob,
                path_prob=next_prob,
                rank=3,
                main_path_position=anchor_position + 1,
                is_main_path=True,
                child_indices=[-1, 0],
            ),
        ],
    )


def test_stage2_v2_hdf5_flush_and_merge_preserve_offsets(tmp_path: Path) -> None:
    attrs = {
        "format_version": "stage2_v2",
        "attn_implementation": "flex_attention",
        "alpha": 0.2,
        "max_anchors_per_sequence": 4,
        "num_attend_tokens_per_anchor": 2,
        "child_coverage_alpha": 0.8,
        "model_name_or_path": "fake",
        "tokenizer_name_or_path": "fake",
    }

    part1 = tmp_path / "part1.h5"
    part2 = tmp_path / "part2.h5"
    merged = tmp_path / "merged.h5"

    for path, sequence in [
        (
            part1,
            GeneratedSequenceTree(
                record_idx=1,
                main_path_ids=[11, 12, 21],
                response_start_position=2,
                anchors=[_make_anchor(2, 0.3, 21)],
            ),
        ),
        (
            part2,
            GeneratedSequenceTree(
                record_idx=0,
                main_path_ids=[13, 14, 22],
                response_start_position=2,
                anchors=[_make_anchor(2, 0.4, 22)],
            ),
        ),
    ]:
        with h5py.File(path, "w") as hf:
            initialize_stage2_v2_hdf5(hf, prob_dtype=np.float32, attrs=attrs)
            flush_stage2_v2_hdf5(
                hf,
                [sequence],
                n_sequences_written=0,
                n_main_path_ids_written=0,
                n_anchors_written=0,
                n_nodes_written=0,
                prob_dtype=np.float32,
            )

    counts = merge_stage2_v2_parts(
        part_paths=[part1, part2],
        output_path=merged,
        prob_dtype=np.float32,
        attrs=attrs,
        log_fn=lambda _: None,
    )

    assert counts == (2, 6, 2, 4)

    with h5py.File(merged, "r") as hf:
        assert hf["record_idx"][:].tolist() == [0, 1]
        assert hf["main_path_offsets"][:].tolist() == [0, 3, 6]
        assert hf["sequence_anchor_offsets"][:].tolist() == [0, 1, 2]
        assert hf["anchor_node_offsets"][:].tolist() == [0, 2, 4]
        assert hf["main_path_ids"][:].tolist() == [13, 14, 22, 11, 12, 21]
        assert hf["node_token_ids"][:].tolist() == [-1, 22, -1, 21]


def test_build_stage2_v2_runtime_falls_back_when_compiled_forward_raises(monkeypatch) -> None:
    class FakeConfig:
        _attn_implementation = "flex_attention"

    class FakeModel:
        config = FakeConfig()

        @staticmethod
        def base_model(**kwargs):
            return kwargs["input_ids"] + 1

    def fake_compile(fn, **kwargs):
        _ = kwargs

        def compiled(*args, **inner_kwargs):
            _ = args, inner_kwargs
            raise RuntimeError("compile failed")

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)

    runtime = build_stage2_v2_runtime(FakeModel(), compile_enabled=True, log_enabled=False)
    result = runtime.base_model_forward(input_ids=torch.tensor([1, 2, 3], dtype=torch.long))

    assert torch.equal(result, torch.tensor([2, 3, 4], dtype=torch.long))
    assert runtime.base_model_forward.compiled_fn is None
