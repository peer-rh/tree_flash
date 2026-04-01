from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")
np = pytest.importorskip("numpy")

from data_pipeline.stage2_v2 import (
    GeneratedAnchorTree,
    GeneratedSequenceTree,
    SequenceTreeNode,
    flush_stage2_v2_hdf5,
    initialize_stage2_v2_hdf5,
)
from src.data import (
    DataModuleConfig,
    FixedPackedBatchSampler,
    PackedBatchCollator,
    Stage2Dataset,
    Stage2V2Dataset,
    build_dataloaders,
)
from src.trees import (
    BlockTreeProcessor,
    BranchOffTreeProcessor,
    PrunableTreeProcessor,
    VarTreeProcessor,
    subset_tree_info,
)
from src.trees.relation_ids import (
    REL_ANCESTOR,
    REL_DESCENDANT,
    REL_OTHER,
    REL_SIBLING,
    relation_id_for_child_rank,
    relation_id_for_parent_rank,
    relation_id_for_sibling_ranks,
)


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


def _write_stage2_fixture_many(path: Path) -> None:
    prompts = [
        [11, 12],
        [13],
        [14, 15],
        [16],
        [17, 18],
    ]
    responses = [
        [21, 22, 23, 24],
        [25, 26, 27],
        [28, 29, 30, 31],
        [32, 33, 34],
        [35, 36, 37, 38],
    ]

    with h5py.File(path, "w") as hf:
        vlen = h5py.vlen_dtype("int64")
        hf.create_dataset("prompt_ids", shape=(len(prompts),), dtype=vlen)
        hf.create_dataset("response_ids", shape=(len(responses),), dtype=vlen)
        for idx, prompt in enumerate(prompts):
            hf["prompt_ids"][idx] = prompt
        for idx, response in enumerate(responses):
            hf["response_ids"][idx] = response

        total_rows = sum(len(response) for response in responses)
        sub_trees = hf.create_dataset("sub_trees", shape=(total_rows, 4), dtype="int64")
        probs = hf.create_dataset("sub_trees_ar_probs", shape=(total_rows, 4), dtype="float32")
        offsets = [0]
        cursor = 0
        for response in responses:
            cursor += len(response)
            offsets.append(cursor)
        hf.create_dataset("sequence_offsets", data=offsets, dtype="int64")
        hf.attrs["sub_tree_paths"] = SUB_TREE_PATHS

        row = 0
        for response in responses:
            for token in response:
                sub_trees[row] = [token, token + 10, token + 11, token + 12]
                probs[row] = [0.8, 0.5, 0.25, 0.2]
                row += 1


def _write_stage2_v2_fixture(path: Path) -> None:
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
    sequence = GeneratedSequenceTree(
        record_idx=0,
        main_path_ids=[11, 12, 21, 22, 23, 24],
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

    assert batch.anchor_positions.tolist() == [[2, 3]]
    assert batch.anchor_document_ids.tolist() == [[1, 1]]
    assert batch.anchor_valid_mask.tolist() == [[True, True]]
    assert batch.tree_labels.shape == (1, 2, 8)


def test_build_dataloaders_reads_ahead_to_emit_fixed_packed_row_count(tmp_path: Path) -> None:
    stage2_path = tmp_path / "stage2_many.h5"
    _write_stage2_fixture_many(stage2_path)

    train_loader, _ = build_dataloaders(
        config=DataModuleConfig(
            path=str(stage2_path),
            batch_size=2,
            pack_length=10,
            num_anchors=2,
            tree_seq_depth=2,
            shuffle=False,
            drop_last=False,
        ),
        tree_processor=BlockTreeProcessor(tree_seq_depth=2, sub_tree_paths=SUB_TREE_PATHS),
        mask_token_id=99,
        pad_token_id=0,
    )
    batch = next(iter(train_loader))

    assert batch.batch_size == 2
    assert batch.input_ids.shape == (2, 10)
    assert batch.input_ids[0].tolist()[:10] == [11, 12, 21, 22, 23, 24, 13, 25, 26, 27]
    assert batch.input_ids[1].tolist()[:10] == [14, 15, 28, 29, 30, 31, 16, 32, 33, 34]
    assert batch.context_valid_mask[0].tolist()[:10] == [True] * 10
    assert batch.context_valid_mask[1].tolist()[:10] == [True] * 10


def test_train_loader_wraps_with_real_data_to_finish_last_batch(tmp_path: Path) -> None:
    stage2_path = tmp_path / "stage2_many.h5"
    _write_stage2_fixture_many(stage2_path)

    train_loader, _ = build_dataloaders(
        config=DataModuleConfig(
            path=str(stage2_path),
            batch_size=2,
            pack_length=10,
            num_anchors=2,
            tree_seq_depth=2,
            shuffle=False,
            drop_last=False,
        ),
        tree_processor=BlockTreeProcessor(tree_seq_depth=2, sub_tree_paths=SUB_TREE_PATHS),
        mask_token_id=99,
        pad_token_id=0,
    )
    batches = list(iter(train_loader))

    assert len(batches) == 2
    assert all(batch.batch_size == 2 for batch in batches)
    assert batches[1].context_valid_mask[0].sum().item() > 0
    assert batches[1].context_valid_mask[1].sum().item() > 0


def test_eval_loader_is_deterministic_across_iterations(tmp_path: Path) -> None:
    stage2_path = tmp_path / "stage2_many.h5"
    _write_stage2_fixture_many(stage2_path)

    _, eval_loader = build_dataloaders(
        config=DataModuleConfig(
            path=str(stage2_path),
            eval_path=str(stage2_path),
            batch_size=2,
            pack_length=10,
            num_anchors=2,
            tree_seq_depth=2,
            shuffle=True,
            drop_last=False,
        ),
        tree_processor=BlockTreeProcessor(tree_seq_depth=2, sub_tree_paths=SUB_TREE_PATHS),
        mask_token_id=99,
        pad_token_id=0,
    )
    first_pass = list(iter(eval_loader))
    second_pass = list(iter(eval_loader))

    assert len(first_pass) == len(second_pass)
    assert torch.equal(first_pass[0].input_ids, second_pass[0].input_ids)
    assert torch.equal(first_pass[0].anchor_positions, second_pass[0].anchor_positions)


def test_fixed_packed_batch_sampler_shards_batches_across_replicas() -> None:
    base_kwargs = dict(
        sample_lengths=[6, 4, 6, 4, 6],
        pack_length=10,
        packed_batch_size=2,
        shuffle=False,
        drop_last=False,
        seed=0,
        persistent_state=False,
    )
    global_sampler = FixedPackedBatchSampler(**base_kwargs)
    kwargs = dict(
        **base_kwargs,
        num_replicas=2,
    )
    rank0 = FixedPackedBatchSampler(rank=0, **kwargs)
    rank1 = FixedPackedBatchSampler(rank=1, **kwargs)

    global_batches = list(iter(global_sampler))
    rank0_batches = list(iter(rank0))
    rank1_batches = list(iter(rank1))

    assert global_batches == [[0, 1, 2], [3, 4, 0]]
    assert len(rank0_batches) == len(rank1_batches) == 1
    assert rank0_batches == [global_batches[0]]
    assert rank1_batches == [global_batches[1]]


def test_train_collator_samples_valid_anchors_at_row_level(tmp_path: Path) -> None:
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
        sample_anchors=True,
    )
    batch = collator([dataset[0], dataset[1]])

    assert batch.anchor_positions.shape == (1, 2)
    assert batch.anchor_valid_mask.tolist() == [[True, True]]
    assert batch.anchor_positions[0].tolist() == sorted(batch.anchor_positions[0].tolist())
    assert set(batch.anchor_positions[0].tolist()).issubset({2, 3, 7, 8})


def test_row_level_anchor_sampling_uses_remaining_valid_documents(tmp_path: Path) -> None:
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

    sample0 = dataset[0]
    sample1 = dataset[1]
    sample0 = {
        **sample0,
        "sub_trees": sample0["sub_trees"].clone(),
    }
    sample0["sub_trees"][:, 0] = -1

    batch = collator([sample0, sample1])

    assert batch.anchor_positions.tolist() == [[7, 8]]
    assert batch.anchor_document_ids.tolist() == [[2, 2]]
    assert batch.anchor_valid_mask.tolist() == [[True, True]]


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


def test_block_tree_processor_builds_multi_anchor_tensors() -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2, sub_tree_paths=SUB_TREE_PATHS)

    response_subtrees = torch.tensor(
        [
            [21, 31, 32, 33],
            [22, 41, 42, 43],
            [23, 51, 52, 53],
            [24, -1, -1, -1],
        ],
        dtype=torch.long,
    )
    response_probs = torch.tensor(
        [
            [0.8, 0.5, 0.25, 0.2],
            [0.6, 0.4, 0.3, 0.1],
            [0.5, 0.3, 0.2, 0.1],
            [0.4, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    tensors = tree_processor.build_anchor_tensors(
        response_subtrees=response_subtrees,
        response_probs=response_probs,
        anchor_local_positions=[0, 2],
        anchor_positions=[10, 20],
        mask_token_id=99,
    )

    assert tensors["tree_labels"].tolist() == [
        [21, 31, 32, 33, 22, 41, 42, 43],
        [23, 51, 52, 53, 24, -1, -1, -1],
    ]
    assert tensors["tree_noise_ids"].tolist() == [
        [21, 99, 99, 99, 99, 99, 99, 99],
        [23, 99, 99, 99, 99, 99, 99, 99],
    ]
    assert tensors["tree_position_ids"].tolist() == [
        [10, 11, 11, 12, 11, 12, 12, 13],
        [20, 21, 21, 22, 21, 22, 22, 23],
    ]
    assert tensors["tree_cum_probs"].tolist() == pytest.approx(
        [
            [0.8, 0.4, 0.2, 0.08, 0.48, 0.192, 0.144, 0.0192],
            [0.5, 0.15, 0.1, 0.015, 0.2, 0.0, 0.0, 0.0],
        ],
        rel=1e-5,
    )
    assert tensors["tree_valid_mask"].tolist() == [
        [True, True, True, True, True, True, True, True],
        [True, True, True, True, True, False, False, False],
    ]


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


def test_branch_off_tree_processor_builds_multi_anchor_tensors() -> None:
    tree_processor = BranchOffTreeProcessor(
        tree_seq_depth=2,
        sub_tree_paths=SUB_TREE_PATHS,
        branching_pattern=[[0, 2], [0, 3]],
    )

    response_subtrees = torch.tensor(
        [
            [21, 31, 32, 33],
            [22, 41, 42, 43],
            [23, 51, 52, 53],
            [24, -1, -1, -1],
        ],
        dtype=torch.long,
    )
    response_probs = torch.tensor(
        [
            [0.8, 0.5, 0.25, 0.2],
            [0.6, 0.4, 0.3, 0.1],
            [0.5, 0.3, 0.2, 0.1],
            [0.4, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    tensors = tree_processor.build_anchor_tensors(
        response_subtrees=response_subtrees,
        response_probs=response_probs,
        anchor_local_positions=[0, 2],
        anchor_positions=[10, 20],
        mask_token_id=99,
    )

    assert tensors["tree_labels"].tolist() == [
        [21, 32, 22, 41, 43],
        [23, 52, 24, -1, -1],
    ]
    assert tensors["tree_noise_ids"].tolist() == [
        [21, 99, 99, 99, 99],
        [23, 99, 99, 99, 99],
    ]
    assert tensors["tree_position_ids"].tolist() == [
        [10, 11, 11, 12, 13],
        [20, 21, 21, 22, 23],
    ]
    assert tensors["tree_cum_probs"].tolist() == pytest.approx(
        [
            [0.8, 0.2, 0.48, 0.192, 0.0192],
            [0.5, 0.1, 0.2, 0.0, 0.0],
        ],
        rel=1e-5,
    )
    assert tensors["tree_valid_mask"].tolist() == [
        [True, True, True, True, True],
        [True, True, True, False, False],
    ]


def test_subset_tree_info_rebuilds_parent_and_relations() -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2, sub_tree_paths=SUB_TREE_PATHS)
    tree_info = tree_processor.build_tree_info(batch_size=1, num_blocks=1, device=torch.device("cpu"))

    subset = subset_tree_info(tree_info, [0, 4, 5, 7])

    assert subset.parent_idx.tolist() == [-1, 0, 1, 2]
    assert subset.depth.tolist() == [0, 1, 2, 3]
    assert subset.tree_position_ids.tolist() == [[[0, 4, 5, 7]]]
    assert subset.non_root_mask.tolist() == [False, True, True, True]
    assert subset.primary_path_indices.tolist() == [0, 1, 2, 3]
    assert subset.tree_mask[3].tolist() == [True, True, True, True]


def test_prunable_tree_processor_wraps_branch_off_base_tree() -> None:
    tree_processor = PrunableTreeProcessor(
        tree_seq_depth=3,
        base_tree_type="branch_off",
        candidate_tree_size=2,
        sub_tree_paths=SUB_TREE_PATHS,
        branching_pattern=[[0, 2], [0, 3], [0]],
    )

    assert tree_processor.candidate_tree_size == 2
    assert tree_processor.base_tree_type == "branch_off"
    assert tree_processor.block_size == 6
    assert tree_processor.primary_path_indices.tolist() == [0, 2, 5]


def test_vartree_processor_builds_dynamic_relations() -> None:
    tree_processor = VarTreeProcessor()
    tree_info = tree_processor.build_tree_info_from_batch(
        tree_parent_indices=torch.tensor([[[-1, 0, 0, 1, -1]]], dtype=torch.long),
        tree_depths=torch.tensor([[[0, 1, 1, 2, 0]]], dtype=torch.long),
        tree_node_ranks=torch.tensor([[[0, 1, 2, 9, 0]]], dtype=torch.long),
        tree_position_ids=torch.tensor([[[2, 3, 3, 4, 0]]], dtype=torch.long),
        tree_valid_mask=torch.tensor([[[True, True, True, True, False]]], dtype=torch.bool),
        tree_primary_path_mask=torch.tensor([[[True, True, False, True, False]]], dtype=torch.bool),
    )

    assert tree_info.parent_idx.shape == (1, 1, 5)
    assert tree_info.tree_mask.shape == (1, 1, 5, 5)
    assert tree_info.non_root_mask.tolist() == [[[False, True, True, True, False]]]
    assert tree_info.primary_path_indices.tolist() == [[[0, 1, 3, -1, -1]]]
    assert tree_info.tree_mask[0, 0, 3].tolist() == [True, True, False, True, False]
    assert int(tree_info.relation_map[0, 0, 1, 2].item()) == relation_id_for_sibling_ranks(1, 2)
    assert int(tree_info.relation_map[0, 0, 2, 1].item()) == relation_id_for_sibling_ranks(2, 1)
    assert int(tree_info.relation_map[0, 0, 3, 1].item()) == relation_id_for_parent_rank(8)
    assert int(tree_info.relation_map[0, 0, 1, 3].item()) == relation_id_for_child_rank(8)
    assert int(tree_info.relation_map[0, 0, 3, 0].item()) == REL_ANCESTOR
    assert int(tree_info.relation_map[0, 0, 0, 3].item()) == REL_DESCENDANT
    assert int(tree_info.relation_map[0, 0, 4, 0].item()) == REL_OTHER


def test_vartree_processor_encodes_direct_edge_ranks() -> None:
    tree_processor = VarTreeProcessor()
    tree_info = tree_processor.build_tree_info_from_batch(
        tree_parent_indices=torch.tensor([[[-1, 0, 0]]], dtype=torch.long),
        tree_depths=torch.tensor([[[0, 1, 1]]], dtype=torch.long),
        tree_node_ranks=torch.tensor([[[0, 1, 2]]], dtype=torch.long),
        tree_position_ids=torch.tensor([[[2, 3, 3]]], dtype=torch.long),
        tree_valid_mask=torch.tensor([[[True, True, True]]], dtype=torch.bool),
        tree_primary_path_mask=torch.tensor([[[True, True, False]]], dtype=torch.bool),
    )

    assert int(tree_info.relation_map[0, 0, 1, 0].item()) == relation_id_for_parent_rank(1)
    assert int(tree_info.relation_map[0, 0, 0, 1].item()) == relation_id_for_child_rank(1)
    assert int(tree_info.relation_map[0, 0, 2, 0].item()) == relation_id_for_parent_rank(2)
    assert int(tree_info.relation_map[0, 0, 0, 2].item()) == relation_id_for_child_rank(2)


def test_vartree_processor_falls_back_to_generic_sibling_for_unknown_rank() -> None:
    tree_processor = VarTreeProcessor()
    tree_info = tree_processor.build_tree_info_from_batch(
        tree_parent_indices=torch.tensor([[[-1, 0, 0]]], dtype=torch.long),
        tree_depths=torch.tensor([[[0, 1, 1]]], dtype=torch.long),
        tree_node_ranks=torch.tensor([[[0, 0, 2]]], dtype=torch.long),
        tree_position_ids=torch.tensor([[[2, 3, 3]]], dtype=torch.long),
        tree_valid_mask=torch.tensor([[[True, True, True]]], dtype=torch.bool),
        tree_primary_path_mask=torch.tensor([[[True, True, False]]], dtype=torch.bool),
    )

    assert int(tree_info.relation_map[0, 0, 1, 2].item()) == REL_SIBLING


def test_stage2_v2_dataset_and_collator_emit_dynamic_tree_metadata(tmp_path: Path) -> None:
    stage2_v2_path = tmp_path / "stage2_v2.h5"
    _write_stage2_v2_fixture(stage2_v2_path)

    dataset = Stage2V2Dataset(stage2_v2_path)
    assert len(dataset) == 1

    _, eval_loader = build_dataloaders(
        config=DataModuleConfig(
            path=str(stage2_v2_path),
            eval_path=str(stage2_v2_path),
            batch_size=1,
            pack_length=16,
            num_anchors=2,
            training_tree_size=4,
            shuffle=False,
            drop_last=False,
        ),
        tree_processor=VarTreeProcessor(),
        mask_token_id=99,
        pad_token_id=0,
    )
    batch = next(iter(eval_loader))

    assert batch.anchor_positions.tolist() == [[2, 3]]
    assert batch.tree_parent_indices is not None
    assert batch.tree_depths is not None
    assert batch.tree_node_ranks is not None
    assert batch.tree_primary_path_mask is not None
    assert batch.tree_labels.shape == (1, 2, 4)
    assert batch.tree_labels[0, 0].tolist() == [21, 22, 23, 24]
    assert batch.tree_parent_indices[0, 0].tolist() == [-1, 0, 1, 2]
    assert batch.tree_node_ranks[0, 0].tolist() == [0, 1, 0, 0]
    assert batch.tree_primary_path_mask[0, 0].tolist() == [True, True, True, True]
    assert batch.tree_parent_indices[0, 1].tolist() == [-1, 0, 1, -1]
    assert batch.tree_node_ranks[0, 1].tolist() == [0, 0, 0, 0]
    assert batch.tree_valid_mask[0, 1].tolist() == [True, True, True, False]
