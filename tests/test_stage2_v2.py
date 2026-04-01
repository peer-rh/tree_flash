from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

import data_pipeline.stage2_v2 as stage2_v2_mod
from data_pipeline.stage2_v2 import (
    GeneratedAnchorTree,
    GeneratedSequenceTree,
    SequenceTreeNode,
    _compute_next_token_stats,
    _generate_sequence_trees_with_verifier_batch,
    _generate_sequence_trees_with_verifier,
    _generate_sequences_for_pending_batch,
    _select_children_for_storage,
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


class _FakeStage2V2LMHead(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.weight = torch.eye(vocab_size, dtype=torch.float32)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.float() @ self.weight.t()


class _FakeStage2V2Model:
    def __init__(self, vocab_size: int = 8) -> None:
        self.vocab_size = vocab_size
        self.config = type("Cfg", (), {"_attn_implementation": "sdpa"})()
        self._lm_head = _FakeStage2V2LMHead(vocab_size)

    def get_output_embeddings(self) -> torch.nn.Module:
        return self._lm_head

    def base_model(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        **kwargs,
    ):
        _ = attention_mask, position_ids, past_key_values, use_cache, kwargs
        hidden = torch.nn.functional.one_hot(input_ids.clamp(min=0), num_classes=self.vocab_size).to(torch.float32)
        return type("Out", (), {"last_hidden_state": hidden, "past_key_values": None})()


def _build_fake_stage2_v2_batch(examples: list[tuple[list[int], list[int]]]) -> dict[str, torch.Tensor]:
    from data_pipeline.stage2 import build_batch

    return build_batch(examples, pad_token_id=0, device=torch.device("cpu"))


def _batched_generation_inputs():
    pending = [
        (10, [1], [2, 2, 3]),
        (11, [1], [4, 5, 6, 7]),
    ]
    batch = _build_fake_stage2_v2_batch([(prompt_ids, response_ids) for _, prompt_ids, response_ids in pending])
    input_ids = batch["input_ids"]
    valid_tokens = batch["document_mask"] >= 0
    is_response = batch["is_response"]
    main_path_ids_per_row = [[int(token_id) for token_id in (prompt_ids + response_ids)] for _, prompt_ids, response_ids in pending]
    response_start_positions = [len(prompt_ids) for _, prompt_ids, _ in pending]
    return pending, input_ids, valid_tokens, is_response, batch["document_mask"], main_path_ids_per_row, response_start_positions


def test_select_children_for_storage_stops_at_child_cap_before_alpha() -> None:
    selected = _select_children_for_storage(
        sorted_token_ids=torch.tensor([10, 11, 12], dtype=torch.long),
        sorted_token_probs=torch.tensor([0.45, 0.30, 0.25], dtype=torch.float32),
        coverage_alpha=0.95,
        forced_child=None,
        max_children_per_parent=2,
    )

    assert [token_id for token_id, *_ in selected] == [10, 11]


def test_select_children_for_storage_forced_child_counts_toward_cap() -> None:
    selected = _select_children_for_storage(
        sorted_token_ids=torch.tensor([10, 11, 12], dtype=torch.long),
        sorted_token_probs=torch.tensor([0.40, 0.35, 0.25], dtype=torch.float32),
        coverage_alpha=0.95,
        forced_child=(11, 7),
        max_children_per_parent=1,
    )

    assert selected == [(11, pytest.approx(0.35, rel=1e-6), 2, True, 7)]


def test_build_anchor_tree_default_child_cap_limits_wide_roots() -> None:
    tree = build_anchor_tree_from_candidate_provider(
        anchor_main_path_position=2,
        anchor_next_token_prob=0.45,
        main_path_ids=[10, 11, 21, 22, 23, 24],
        response_start_position=2,
        num_attend_tokens_per_anchor=0,
        child_coverage_alpha=2.0,
        root_sorted_token_ids=[31, 32, 33, 34, 35, 36, 37, 22, 38, 39, 40],
        root_sorted_token_probs=[0.2, 0.12, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.045, 0.04, 0.035],
        candidate_provider=lambda _node: ([], []),
    )

    assert tree.nodes[0].child_indices[1] == 8


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


def test_batched_generation_matches_single_sequence_results_with_variable_anchor_counts() -> None:
    (
        _pending,
        input_ids,
        valid_tokens,
        is_response,
        document_mask,
        main_path_ids_per_row,
        response_start_positions,
    ) = _batched_generation_inputs()
    model = _FakeStage2V2Model()

    hidden_states = model.base_model(input_ids=input_ids).last_hidden_state
    lm_head = model.get_output_embeddings()
    next_token_probs = _compute_next_token_stats(
        hidden_states=hidden_states,
        input_ids=input_ids,
        valid_tokens=valid_tokens,
        lm_head=lm_head,
        logit_chunk_size=0,
    )

    anchor_positions_per_row: list[list[int]] = []
    anchor_probs_per_row: list[list[float]] = []
    for row_idx in range(input_ids.shape[0]):
        positions, probs = _select_anchor_positions_for_sequence(
            is_response=is_response[row_idx],
            valid_tokens=valid_tokens[row_idx],
            next_token_probs=next_token_probs[row_idx],
            alpha=0.2,
            max_anchors_per_sequence=8,
        )
        anchor_positions_per_row.append(positions)
        anchor_probs_per_row.append(probs)

    batched = _generate_sequence_trees_with_verifier_batch(
        model=model,
        runtime=None,
        input_ids=input_ids,
        valid_tokens=valid_tokens,
        document_mask=document_mask,
        main_path_ids_per_row=main_path_ids_per_row,
        response_start_positions=response_start_positions,
        anchor_positions_per_row=anchor_positions_per_row,
        anchor_next_token_probs_per_row=anchor_probs_per_row,
        hidden_states=hidden_states,
        kv_cache=None,
        logit_chunk_size=0,
        num_attend_tokens_per_anchor=2,
        child_coverage_alpha=0.8,
        max_children_per_node=8,
    )
    single = [
        _generate_sequence_trees_with_verifier(
            model=model,
            runtime=None,
            input_ids=input_ids[row_idx : row_idx + 1],
            valid_tokens=valid_tokens[row_idx : row_idx + 1],
            document_mask=document_mask[row_idx : row_idx + 1],
            main_path_ids=main_path_ids_per_row[row_idx],
            response_start_position=response_start_positions[row_idx],
            anchor_positions=anchor_positions_per_row[row_idx],
            anchor_next_token_probs=anchor_probs_per_row[row_idx],
            hidden_states=hidden_states[row_idx : row_idx + 1],
            kv_cache=None,
            logit_chunk_size=0,
            num_attend_tokens_per_anchor=2,
            child_coverage_alpha=0.8,
            max_children_per_node=8,
        )
        for row_idx in range(input_ids.shape[0])
    ]

    assert [len(row) for row in batched] == [1, 3]
    assert batched == single


def test_batched_pending_generation_keeps_variable_real_anchor_counts_on_disk(tmp_path: Path) -> None:
    pending = [
        (10, [1], [2, 2, 3]),
        (11, [1], [4, 5, 6, 7]),
    ]
    model = _FakeStage2V2Model()

    sequences = _generate_sequences_for_pending_batch(
        pending=pending,
        model=model,
        runtime=None,
        pad_token_id=0,
        device=torch.device("cpu"),
        alpha=0.2,
        max_anchors_per_sequence=8,
        logit_chunk_size=0,
        num_attend_tokens_per_anchor=2,
        child_coverage_alpha=0.8,
        max_children_per_node=8,
    )

    assert [len(sequence.anchors) for sequence in sequences] == [1, 3]

    attrs = {
        "format_version": "stage2_v2",
        "attn_implementation": "sdpa",
        "alpha": 0.1,
        "max_anchors_per_sequence": 8,
        "num_attend_tokens_per_anchor": 2,
        "child_coverage_alpha": 0.8,
        "max_children_per_node": 8,
        "model_name_or_path": "fake",
        "tokenizer_name_or_path": "fake",
    }
    output_path = tmp_path / "stage2_v2_batch.h5"
    with h5py.File(output_path, "w") as hf:
        initialize_stage2_v2_hdf5(hf, prob_dtype=np.float32, attrs=attrs)
        flush_stage2_v2_hdf5(
            hf,
            sequences,
            n_sequences_written=0,
            n_main_path_ids_written=0,
            n_anchors_written=0,
            n_nodes_written=0,
            prob_dtype=np.float32,
        )
        assert hf["sequence_anchor_offsets"][:].tolist() == [0, 1, 4]


def test_batched_pending_generation_allows_zero_anchor_rows() -> None:
    pending = [
        (10, [1], [2, 2, 2]),
        (11, [1], [4, 5, 6]),
    ]
    model = _FakeStage2V2Model()

    sequences = _generate_sequences_for_pending_batch(
        pending=pending,
        model=model,
        runtime=None,
        pad_token_id=0,
        device=torch.device("cpu"),
        alpha=0.2,
        max_anchors_per_sequence=8,
        logit_chunk_size=0,
        num_attend_tokens_per_anchor=2,
        child_coverage_alpha=0.8,
        max_children_per_node=8,
    )

    assert len(sequences[0].anchors) == 0
    assert len(sequences[1].anchors) == 2


def test_generate_sequences_for_pending_batch_uses_one_build_batch_and_initial_forward(monkeypatch) -> None:
    pending = [
        (10, [1], [2, 2, 3]),
        (11, [1], [4, 5, 6, 7]),
    ]
    model = _FakeStage2V2Model()

    build_batch_calls = 0
    initial_forward_calls = 0
    original_build_batch = stage2_v2_mod.build_batch
    original_run_forward = stage2_v2_mod._run_base_model_forward

    def counting_build_batch(*args, **kwargs):
        nonlocal build_batch_calls
        build_batch_calls += 1
        return original_build_batch(*args, **kwargs)

    def counting_run_forward(*args, **kwargs):
        nonlocal initial_forward_calls
        if kwargs.get("profile_key") == "initial_forward_s":
            initial_forward_calls += 1
        return original_run_forward(*args, **kwargs)

    monkeypatch.setattr(stage2_v2_mod, "build_batch", counting_build_batch)
    monkeypatch.setattr(stage2_v2_mod, "_run_base_model_forward", counting_run_forward)

    sequences = _generate_sequences_for_pending_batch(
        pending=pending,
        model=model,
        runtime=None,
        pad_token_id=0,
        device=torch.device("cpu"),
        alpha=0.2,
        max_anchors_per_sequence=8,
        logit_chunk_size=0,
        num_attend_tokens_per_anchor=2,
        child_coverage_alpha=0.8,
        max_children_per_node=8,
    )

    assert len(sequences) == 2
    assert build_batch_calls == 1
    assert initial_forward_calls == 1


def test_stage2_v2_hdf5_flush_and_merge_preserve_offsets(tmp_path: Path) -> None:
    attrs = {
        "format_version": "stage2_v2",
        "attn_implementation": "flex_attention",
        "alpha": 0.2,
        "max_anchors_per_sequence": 4,
        "num_attend_tokens_per_anchor": 2,
        "child_coverage_alpha": 0.8,
        "max_children_per_node": 8,
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
