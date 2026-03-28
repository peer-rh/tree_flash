from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

from src.data import PackedBatchCollator, Stage2Dataset
from src.trees import BlockTreeProcessor, BranchOffTreeProcessor


SUB_TREE_PATHS = ["0-1", "0-2", "1-3"]


def _write_stage2_fixture(path: Path) -> None:
    with h5py.File(path, "w") as hf:
        vlen = h5py.vlen_dtype("int64")
        hf.create_dataset("prompt_ids", shape=(2,), dtype=vlen)
        hf.create_dataset("response_ids", shape=(2,), dtype=vlen)
        hf["prompt_ids"][0] = [11, 12]
        hf["response_ids"][0] = [21, 22, 23, 24]
        hf["prompt_ids"][1] = [13]
        hf["response_ids"][1] = [25, 26, 27]

        sub_trees = hf.create_dataset("sub_trees", shape=(7, 4), dtype="int64")
        probs = hf.create_dataset("sub_trees_ar_probs", shape=(7, 4), dtype="float32")
        hf.create_dataset("sequence_offsets", data=[0, 4, 7], dtype="int64")
        hf.attrs["sub_tree_paths"] = SUB_TREE_PATHS

        sub_trees[...] = [
            [21, 31, 32, 33],
            [22, 41, 42, 43],
            [23, 51, 52, 53],
            [24, -1, -1, -1],
            [25, 61, 62, 63],
            [26, 71, 72, 73],
            [27, -1, -1, -1],
        ]
        probs[...] = [
            [0.8, 0.5, 0.25, 0.2],
            [0.6, 0.4, 0.3, 0.1],
            [0.5, 0.3, 0.2, 0.1],
            [0.4, 0.0, 0.0, 0.0],
            [0.9, 0.6, 0.5, 0.4],
            [0.7, 0.5, 0.25, 0.2],
            [0.5, 0.0, 0.0, 0.0],
        ]


def test_stage2_dataset_reads_offsets_and_rows(tmp_path: Path) -> None:
    stage2_path = tmp_path / "stage2.h5"
    _write_stage2_fixture(stage2_path)

    dataset = Stage2Dataset(stage2_path)
    assert len(dataset) == 2
    assert dataset.sub_tree_paths == tuple(SUB_TREE_PATHS)

    sample0 = dataset[0]
    sample1 = dataset[1]

    assert sample0["prompt_ids"].tolist() == [11, 12]
    assert sample0["response_ids"].tolist() == [21, 22, 23, 24]
    assert sample0["sub_trees"].shape == (4, 4)
    assert sample0["sub_trees"][1].tolist() == [22, 41, 42, 43]

    assert sample1["prompt_ids"].tolist() == [13]
    assert sample1["response_ids"].tolist() == [25, 26, 27]
    assert sample1["sub_trees"].shape == (3, 4)
    assert sample1["sub_trees"][0].tolist() == [25, 61, 62, 63]


def test_packed_collator_restarts_positions_and_isolates_documents(tmp_path: Path) -> None:
    stage2_path = tmp_path / "stage2.h5"
    _write_stage2_fixture(stage2_path)

    dataset = Stage2Dataset(stage2_path)
    tree_processor = BlockTreeProcessor(tree_seq_depth=2, sub_tree_paths=SUB_TREE_PATHS)
    collator = PackedBatchCollator(
        tree_processor=tree_processor,
        pack_length=16,
        num_anchors=2,
        mask_token_id=99,
        pad_token_id=0,
        seed=0,
        sample_anchors=False,
    )
    batch = collator([dataset[0], dataset[1]])

    assert batch.input_ids.shape == (1, 16)
    assert batch.input_ids[0, :10].tolist() == [11, 12, 21, 22, 23, 24, 13, 25, 26, 27]
    assert batch.position_ids[0, :10].tolist() == [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]
    assert batch.document_mask[0, :10].tolist() == [1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    assert batch.context_valid_mask[0, :10].tolist() == [True] * 10

    assert batch.anchor_positions.tolist() == [[2, 3, 7, 8]]
    assert batch.anchor_document_ids.tolist() == [[1, 1, 2, 2]]
    assert batch.anchor_valid_mask.tolist() == [[True, True, True, True]]
    assert batch.tree_labels.shape == (1, 4, 8)


def test_block_tree_processor_builds_expected_metadata() -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2, sub_tree_paths=SUB_TREE_PATHS)

    assert tree_processor.parent_idx.tolist() == [-1, 0, 0, 1, 0, 4, 4, 5]
    assert tree_processor.depth.tolist() == [0, 1, 1, 2, 1, 2, 2, 3]
    assert tree_processor.non_root_mask.tolist() == [False, True, True, True, True, True, True, True]
    assert tree_processor.tree_mask[7].tolist() == [True, False, False, False, True, True, False, True]

    response_subtrees = torch.tensor(
        [
            [21, 31, 32, 33],
            [22, 41, 42, 43],
            [23, 51, 52, 53],
        ],
        dtype=torch.long,
    )
    response_probs = torch.tensor(
        [
            [0.8, 0.5, 0.25, 0.2],
            [0.6, 0.4, 0.3, 0.1],
            [0.5, 0.3, 0.2, 0.1],
        ],
        dtype=torch.float32,
    )
    tensors = tree_processor.build_anchor_tensors(
        response_subtrees=response_subtrees,
        response_probs=response_probs,
        anchor_local_positions=[0],
        anchor_positions=[10],
        mask_token_id=99,
    )

    assert tensors["tree_labels"][0].tolist() == [21, 31, 32, 33, 22, 41, 42, 43]
    assert tensors["tree_noise_ids"][0].tolist() == [21, 99, 99, 99, 99, 99, 99, 99]
    assert tensors["tree_position_ids"][0].tolist() == [10, 11, 11, 12, 11, 12, 12, 13]
    assert tensors["tree_cum_probs"][0].tolist() == pytest.approx(
        [0.8, 0.4, 0.2, 0.08, 0.48, 0.192, 0.144, 0.0192],
        rel=1e-5,
    )

    info = tree_processor.build_tree_info(batch_size=2, num_blocks=3, device=torch.device("cpu"))
    assert info.relation_map.shape == (2, 3, 8, 8)
    assert info.tree_position_ids.shape == (2, 3, 8)


def test_branch_off_tree_processor_selects_sparse_layout() -> None:
    tree_processor = BranchOffTreeProcessor(
        tree_seq_depth=3,
        sub_tree_paths=SUB_TREE_PATHS,
        branching_pattern=[[0, 2], [0, 3], [0]],
    )

    assert tree_processor.branching_pattern == ((0, 2), (0, 1, 3), (0,))
    assert tree_processor.layout == [(0, 0), (0, 2), (1, 0), (1, 1), (1, 3), (2, 0)]
    assert tree_processor.parent_idx.tolist() == [-1, 0, 0, 2, 3, 2]
    assert tree_processor.primary_path_indices.tolist() == [0, 2, 5]
    assert tree_processor.non_root_mask.tolist() == [False, True, True, True, True, True]

    response_subtrees = torch.tensor(
        [
            [21, 31, 32, 33],
            [22, 41, 42, 43],
            [23, 51, 52, 53],
        ],
        dtype=torch.long,
    )
    response_probs = torch.tensor(
        [
            [0.8, 0.5, 0.25, 0.2],
            [0.6, 0.4, 0.3, 0.1],
            [0.5, 0.3, 0.2, 0.1],
        ],
        dtype=torch.float32,
    )
    tensors = tree_processor.build_anchor_tensors(
        response_subtrees=response_subtrees,
        response_probs=response_probs,
        anchor_local_positions=[0],
        anchor_positions=[10],
        mask_token_id=99,
    )

    assert tensors["tree_labels"][0].tolist() == [21, 32, 22, 41, 43, 23]
    assert tensors["tree_noise_ids"][0].tolist() == [21, 99, 99, 99, 99, 99]
    assert tensors["tree_position_ids"][0].tolist() == [10, 11, 11, 12, 13, 12]
    assert tensors["tree_cum_probs"][0].tolist() == pytest.approx(
        [0.8, 0.2, 0.48, 0.192, 0.0192, 0.24],
        rel=1e-5,
    )
