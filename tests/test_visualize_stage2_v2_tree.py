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
    flush_stage2_v2_hdf5,
    initialize_stage2_v2_hdf5,
)
import data_pipeline.visualize_stage2_v2_tree as vis_mod


def _write_visualizer_fixture(path: Path) -> None:
    attrs = {
        "format_version": "stage2_v2",
        "attn_implementation": "flex_attention",
        "alpha": 0.2,
        "max_anchors_per_sequence": 4,
        "num_attend_tokens_per_anchor": 2,
        "child_coverage_alpha": 0.8,
        "model_name_or_path": "fake-model",
        "tokenizer_name_or_path": "fake-tokenizer",
    }
    sequence = GeneratedSequenceTree(
        record_idx=7,
        main_path_ids=[11, 12, 21, 22, 23],
        response_start_position=2,
        anchors=[
            GeneratedAnchorTree(
                anchor_main_path_position=2,
                anchor_next_token_prob=0.3,
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
                        child_indices=[1, 2],
                    ),
                    SequenceTreeNode(
                        token_id=22,
                        parent_index=0,
                        depth=1,
                        local_prob=0.6,
                        path_prob=0.6,
                        rank=1,
                        main_path_position=3,
                        is_main_path=True,
                        child_indices=[3, 1],
                    ),
                    SequenceTreeNode(
                        token_id=31,
                        parent_index=0,
                        depth=1,
                        local_prob=0.4,
                        path_prob=0.4,
                        rank=2,
                        main_path_position=-1,
                        is_main_path=False,
                        child_indices=[-1, 0],
                    ),
                    SequenceTreeNode(
                        token_id=23,
                        parent_index=1,
                        depth=2,
                        local_prob=0.7,
                        path_prob=0.42,
                        rank=1,
                        main_path_position=4,
                        is_main_path=True,
                        child_indices=[-1, 0],
                    ),
                ],
            ),
            GeneratedAnchorTree(
                anchor_main_path_position=3,
                anchor_next_token_prob=0.5,
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
                        child_indices=[1, 2],
                    ),
                    SequenceTreeNode(
                        token_id=23,
                        parent_index=0,
                        depth=1,
                        local_prob=0.8,
                        path_prob=0.8,
                        rank=1,
                        main_path_position=4,
                        is_main_path=True,
                        child_indices=[-1, 0],
                    ),
                    SequenceTreeNode(
                        token_id=44,
                        parent_index=0,
                        depth=1,
                        local_prob=0.2,
                        path_prob=0.2,
                        rank=2,
                        main_path_position=-1,
                        is_main_path=False,
                        child_indices=[-1, 0],
                    ),
                ],
            ),
        ],
    )
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


def test_load_stage2_v2_tree_includes_backbone_and_all_anchor_nodes(tmp_path: Path) -> None:
    fixture_path = tmp_path / "stage2_v2_visualizer.h5"
    _write_visualizer_fixture(fixture_path)

    tree = vis_mod.load_stage2_v2_tree(fixture_path, sequence_index=0)

    assert tree.record_idx == 7
    assert tree.response_start_position == 2
    assert tree.tokenizer_name_or_path == "fake-tokenizer"
    assert len(tree.nodes) == 10
    assert sum(1 for node in tree.nodes if node.source == "main") == 5
    assert sum(1 for node in tree.nodes if node.source == "anchor") == 5
    assert [node.position_id for node in tree.nodes[:5]] == [0, 1, 2, 3, 4]
    assert [node.token_id for node in tree.nodes[:5]] == [11, 12, 21, 22, 23]
    assert [node.parent_index for node in tree.nodes[:5]] == [-1, 0, 1, 2, 3]
    assert [node.token_id for node in tree.nodes[5:]] == [22, 31, 23, 23, 44]
    assert [node.position_id for node in tree.nodes[5:]] == [3, 3, 4, 4, 4]
    assert [node.rank for node in tree.nodes[5:]] == [1, 2, 1, 1, 2]
    assert [node.path_prob for node in tree.nodes[5:]] == pytest.approx([0.6, 0.4, 0.42, 0.8, 0.2], rel=1e-6)


def test_assign_x_slots_pins_main_path_left_and_packs_anchor_nodes_by_row(tmp_path: Path) -> None:
    fixture_path = tmp_path / "stage2_v2_visualizer.h5"
    _write_visualizer_fixture(fixture_path)

    tree = vis_mod.load_stage2_v2_tree(fixture_path, sequence_index=0)
    x_slots = vis_mod._assign_x_slots(tree.nodes)

    row_slots: dict[int, list[tuple[str, int, int]]] = {}
    for node in tree.nodes:
        row_slots.setdefault(node.position_id, []).append(
            (
                node.source,
                int(x_slots[node.index]),
                node.rank,
            )
        )

    for position_id, entries in row_slots.items():
        main_slots = [slot for source, slot, _ in entries if source == "main"]
        anchor_slots = sorted(slot for source, slot, _ in entries if source == "anchor")
        if main_slots:
            assert main_slots == [0], f"main node for row {position_id} should be in the leftmost slot"
            assert anchor_slots == list(range(1, 1 + len(anchor_slots)))

    row3_anchor_ranks = [node.rank for node in tree.nodes if node.position_id == 3 and node.source == "anchor"]
    row3_anchor_slots = [
        int(x_slots[node.index]) for node in tree.nodes if node.position_id == 3 and node.source == "anchor"
    ]
    assert row3_anchor_ranks == [1, 2]
    assert row3_anchor_slots == [1, 2]

    row4_anchor_tokens = [node.token_id for node in tree.nodes if node.position_id == 4 and node.source == "anchor"]
    row4_anchor_slots = [
        int(x_slots[node.index]) for node in tree.nodes if node.position_id == 4 and node.source == "anchor"
    ]
    assert row4_anchor_tokens == [23, 23, 44]
    assert row4_anchor_slots == [1, 2, 3]


def test_render_stage2_v2_tree_html_contains_main_path_and_branch_tokens(tmp_path: Path, monkeypatch) -> None:
    fixture_path = tmp_path / "stage2_v2_visualizer.h5"
    _write_visualizer_fixture(fixture_path)

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(_name):
            return FakeTokenizer()

        def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
            _ = skip_special_tokens, clean_up_tokenization_spaces
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            return "|".join(f"tok{int(token_id)}" for token_id in token_ids)

        def convert_ids_to_tokens(self, token_ids):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            return [f"T{int(token_id)}" for token_id in token_ids]

    monkeypatch.setattr(vis_mod, "AutoTokenizer", FakeTokenizer)

    output_path = tmp_path / "tree.html"
    written = vis_mod.write_stage2_v2_tree_html(fixture_path, output_path, sequence_index=0)

    assert written == output_path
    html_text = output_path.read_text(encoding="utf-8")
    assert "Stage 2 v2 Sequence Tree" in html_text
    assert str(fixture_path) in html_text
    assert "fake-tokenizer" in html_text
    assert "Main Nodes" in html_text
    assert "Anchor Nodes" in html_text
    assert "Anchors" in html_text
    assert 'data-source="main"' in html_text
    assert 'data-source="anchor"' in html_text
    assert 'data-position-id="4"' in html_text
    assert 'data-prob="0.800000"' in html_text
    assert "tok11" in html_text
    assert "tok21" in html_text
    assert "tok31" in html_text
    assert "tok44" in html_text
