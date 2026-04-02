from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")
pytest.importorskip("transformers")


ROOT = Path(__file__).resolve().parents[1]
CLEANED_UP_DIR = ROOT / "cleaned-up"
if str(CLEANED_UP_DIR) not in sys.path:
    sys.path.insert(0, str(CLEANED_UP_DIR))

stage2_spec = importlib.util.spec_from_file_location("cleaned_up_stage2_v2", CLEANED_UP_DIR / "stage2_v2.py")
assert stage2_spec is not None and stage2_spec.loader is not None
cleaned_up_stage2_v2 = importlib.util.module_from_spec(stage2_spec)
sys.modules[stage2_spec.name] = cleaned_up_stage2_v2
stage2_spec.loader.exec_module(cleaned_up_stage2_v2)

data_spec = importlib.util.spec_from_file_location("cleaned_up_data", CLEANED_UP_DIR / "data.py")
assert data_spec is not None and data_spec.loader is not None
cleaned_up_data = importlib.util.module_from_spec(data_spec)
sys.modules[data_spec.name] = cleaned_up_data
data_spec.loader.exec_module(cleaned_up_data)


class FakeDataset:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = [dict(row) for row in rows]
        self.column_names = list(self.rows[0].keys()) if self.rows else []

    def map(
        self,
        fn,
        *,
        batched: bool,
        remove_columns=None,
        with_indices: bool = False,
        **kwargs,
    ):
        del kwargs
        if not batched:
            raise NotImplementedError("FakeDataset only implements batched map.")
        batch = {column: [row[column] for row in self.rows] for column in self.column_names}
        if with_indices:
            mapped = fn(batch, list(range(len(self.rows))))
        else:
            mapped = fn(batch)
        if self.rows:
            new_len = len(next(iter(mapped.values())))
        else:
            new_len = 0
        new_rows = []
        for idx in range(new_len):
            row = {} if remove_columns is not None else dict(self.rows[idx])
            for key, values in mapped.items():
                row[key] = values[idx]
            new_rows.append(row)
        return FakeDataset(new_rows)

    def filter(self, fn, *, batched: bool, input_columns: list[str], **kwargs):
        del kwargs
        if not batched:
            raise NotImplementedError("FakeDataset only implements batched filter.")
        keep = fn(*[[row[column] for row in self.rows] for column in input_columns])
        return FakeDataset([row for row, keep_flag in zip(self.rows, keep, strict=True) if keep_flag])

    def sort(self, column: str, *, reverse: bool):
        return FakeDataset(sorted(self.rows, key=lambda row: row[column], reverse=reverse))

    def select(self, indices: range):
        return FakeDataset([self.rows[idx] for idx in indices])

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.rows[idx]


class FakeTokenizer:
    def __call__(self, texts, *, add_special_tokens: bool):
        del add_special_tokens
        if isinstance(texts, str):
            texts = [texts]
        return {
            "input_ids": [[ord(char) for char in text] for text in texts],
        }


class _FakeStage2V2LMHead(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.weight = torch.eye(vocab_size, dtype=torch.float32)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.float() @ self.weight.t()


class _FakeStage2V2Model:
    def __init__(self, vocab_size: int = 8) -> None:
        self.vocab_size = vocab_size
        self.config = type("Cfg", (), {"_attn_implementation": "flex_attention"})()
        self._lm_head = _FakeStage2V2LMHead(vocab_size)
        self.cache_positions: list[torch.Tensor] = []
        self.cache_objects: list[object | None] = []

    def parameters(self):
        return iter([self._lm_head.weight])

    @property
    def dtype(self):
        return self._lm_head.weight.dtype

    def get_output_embeddings(self) -> torch.nn.Module:
        return self._lm_head

    def base_model(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        cache_position=None,
        use_cache=False,
        **kwargs,
    ):
        del attention_mask, position_ids, use_cache, kwargs
        self.cache_positions.append(None if cache_position is None else cache_position.detach().clone())
        self.cache_objects.append(past_key_values)
        hidden = torch.nn.functional.one_hot(input_ids.clamp(min=0), num_classes=self.vocab_size).to(torch.float32)
        return type("Out", (), {"last_hidden_state": hidden, "past_key_values": past_key_values})()


def _make_anchor(anchor_position: int, next_prob: float, child_token: int):
    return cleaned_up_stage2_v2.GeneratedAnchorTree(
        anchor_main_path_position=anchor_position,
        anchor_next_token_prob=next_prob,
        nodes=[
            cleaned_up_stage2_v2.SequenceTreeNode(
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
            cleaned_up_stage2_v2.SequenceTreeNode(
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


def test_load_tokenized_records_supports_jsonl_and_hf_dataset(tmp_path: Path, monkeypatch) -> None:
    rows = [
        {"prompt": "ab", "response": "c"},
        {"prompt": "d", "response": "ef"},
    ]
    jsonl_path = tmp_path / "part.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    def fake_load_dataset(dataset_id, *args, **kwargs):
        if dataset_id == "json":
            data_files = kwargs["data_files"]
            loaded_rows = []
            for path in data_files:
                with Path(path).open("r", encoding="utf-8") as handle:
                    for line in handle:
                        loaded_rows.append(json.loads(line))
            return FakeDataset(loaded_rows)
        assert dataset_id == "fake/dataset"
        return FakeDataset(rows)

    monkeypatch.setattr(cleaned_up_stage2_v2, "load_dataset", fake_load_dataset)
    tokenizer = FakeTokenizer()

    json_records = cleaned_up_stage2_v2.load_tokenized_records(
        tokenizer,
        None,
        data_dir=tmp_path,
        sort_descending=True,
    )
    hf_records = cleaned_up_stage2_v2.load_tokenized_records(
        tokenizer,
        None,
        hf_dataset="fake/dataset",
        sort_descending=True,
    )

    assert [row["prompt_ids"] for row in json_records] == [row["prompt_ids"] for row in hf_records]
    assert [row["response_ids"] for row in json_records] == [row["response_ids"] for row in hf_records]
    assert [row["total_len"] for row in json_records] == [3, 3]


def test_anchor_selection_and_child_storage_match_cleaned_behavior() -> None:
    is_response = torch.tensor([False, False, True, True, True, True], dtype=torch.bool)
    valid_tokens = torch.tensor([True, True, True, True, True, True], dtype=torch.bool)
    next_token_probs = torch.tensor([1.0, 0.9, 0.8, 0.1, 0.6, 0.2], dtype=torch.float32)

    positions, probs = cleaned_up_stage2_v2._select_anchor_positions_for_sequence(
        is_response=is_response,
        valid_tokens=valid_tokens,
        next_token_probs=next_token_probs,
        alpha=0.65,
        max_anchors_per_sequence=2,
    )

    assert positions == [2, 4]
    assert probs == pytest.approx([0.1, 0.2], rel=1e-6)

    selected = cleaned_up_stage2_v2._select_children_for_storage(
        sorted_token_ids=torch.tensor([10, 11, 12], dtype=torch.long),
        sorted_token_probs=torch.tensor([0.40, 0.35, 0.25], dtype=torch.float32),
        coverage_alpha=0.95,
        forced_child=(11, 7),
        max_children_per_parent=1,
    )
    assert selected == [(11, pytest.approx(0.35, rel=1e-6), 2, True, 7)]

    fallback = cleaned_up_stage2_v2._select_children_for_storage(
        sorted_token_ids=torch.tensor([20, 21, 22], dtype=torch.long),
        sorted_token_probs=torch.tensor([0.40, 0.35, 0.25], dtype=torch.float32),
        coverage_alpha=0.0,
        forced_child=None,
        max_children_per_parent=8,
    )
    assert fallback == [(20, pytest.approx(0.40, rel=1e-6), 1, False, -1)]


def test_local_hdf5_writer_preserves_schema_and_data_reader_contract(tmp_path: Path) -> None:
    output_path = tmp_path / "stage2_v2.h5"
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
    sequence = cleaned_up_stage2_v2.GeneratedSequenceTree(
        record_idx=3,
        main_path_ids=[11, 12, 21],
        response_start_position=2,
        anchors=[_make_anchor(2, 0.3, 21)],
    )

    cleaned_up_data.write_sequence_tree_hdf5(
        output_path=output_path,
        sequences=[sequence],
        attrs=attrs,
        prob_dtype=np.float32,
    )

    with h5py.File(output_path, "r") as hf:
        assert hf["main_path_offsets"][:].tolist() == [0, 3]
        assert hf["sequence_anchor_offsets"][:].tolist() == [0, 1]
        assert hf["anchor_node_offsets"][:].tolist() == [0, 2]
        assert hf["node_first_child"][:].tolist() == [1, -1]
        assert hf["node_child_count"][:].tolist() == [1, 0]

    dataset = cleaned_up_data.SequenceTreeDataset(output_path)
    sample = dataset[0]
    assert sample["record_idx"] == 3
    assert sample["main_path_ids"].tolist() == [11, 12, 21]
    assert sample["anchors"][0]["node_token_ids"].tolist() == [-1, 21]


def test_static_cache_backed_generation_keeps_expected_tree_counts(monkeypatch) -> None:
    pending = [
        (10, [1], [2, 2, 3]),
        (11, [1], [4, 5, 6, 7]),
    ]
    model = _FakeStage2V2Model()
    created: dict[str, int] = {}

    class FakeStaticCache:
        pass

    def fake_build_static_cache(model_arg, *, batch_size: int, max_cache_len: int):
        assert model_arg is model
        created["batch_size"] = batch_size
        created["max_cache_len"] = max_cache_len
        return FakeStaticCache()

    monkeypatch.setattr(cleaned_up_stage2_v2, "_build_static_cache", fake_build_static_cache)

    sequences = cleaned_up_stage2_v2._generate_sequences_for_pending_batch(
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

    batch = cleaned_up_stage2_v2.build_batch([(prompt_ids, response_ids) for _, prompt_ids, response_ids in pending], 0, torch.device("cpu"))
    assert created == {
        "batch_size": 2,
        "max_cache_len": int(batch["input_ids"].shape[1]) + (8 * 2),
    }
    assert [len(sequence.anchors) for sequence in sequences] == [1, 3]
    assert isinstance(model.cache_objects[0], FakeStaticCache)
    assert model.cache_positions[0].tolist() == list(range(batch["input_ids"].shape[1]))


def test_batched_generation_matches_single_sequence_generation(monkeypatch) -> None:
    pending = [
        (10, [1], [2, 2, 3]),
        (11, [1], [4, 5, 6, 7]),
    ]
    model = _FakeStage2V2Model()

    monkeypatch.setattr(cleaned_up_stage2_v2, "_build_static_cache", lambda *args, **kwargs: object())

    batch = cleaned_up_stage2_v2.build_batch(
        [(prompt_ids, response_ids) for _, prompt_ids, response_ids in pending],
        pad_token_id=0,
        device=torch.device("cpu"),
    )
    input_ids = batch["input_ids"]
    valid_tokens = batch["document_mask"] >= 0
    is_response = batch["is_response"]
    document_mask = batch["document_mask"]
    hidden_states = model.base_model(input_ids=input_ids).last_hidden_state
    main_path_ids_per_row = [
        [int(token_id) for token_id in (prompt_ids + response_ids)]
        for _, prompt_ids, response_ids in pending
    ]
    response_start_positions = [len(prompt_ids) for _, prompt_ids, _ in pending]
    next_token_probs = cleaned_up_stage2_v2._compute_next_token_stats(
        hidden_states=hidden_states,
        input_ids=input_ids,
        valid_tokens=valid_tokens,
        lm_head=model.get_output_embeddings(),
        logit_chunk_size=0,
    )

    anchor_positions_per_row: list[list[int]] = []
    anchor_probs_per_row: list[list[float]] = []
    for row_idx in range(input_ids.shape[0]):
        positions, probs = cleaned_up_stage2_v2._select_anchor_positions_for_sequence(
            is_response=is_response[row_idx],
            valid_tokens=valid_tokens[row_idx],
            next_token_probs=next_token_probs[row_idx],
            alpha=0.2,
            max_anchors_per_sequence=8,
        )
        anchor_positions_per_row.append(positions)
        anchor_probs_per_row.append(probs)

    batched = cleaned_up_stage2_v2._generate_sequence_trees_with_verifier_batch(
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
        kv_cache=object(),
        logit_chunk_size=0,
        num_attend_tokens_per_anchor=2,
        child_coverage_alpha=0.8,
        max_children_per_node=8,
        max_trees_per_row=8,
    )
    single = [
        cleaned_up_stage2_v2._generate_sequence_trees_with_verifier(
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
            kv_cache=object(),
            logit_chunk_size=0,
            num_attend_tokens_per_anchor=2,
            child_coverage_alpha=0.8,
            max_children_per_node=8,
            max_trees_per_row=8,
        )
        for row_idx in range(input_ids.shape[0])
    ]

    assert [len(row) for row in batched] == [1, 3]
    assert batched == single


def test_build_stage2_v2_runtime_falls_back_when_compiled_forward_raises(monkeypatch) -> None:
    class FakeConfig:
        _attn_implementation = "flex_attention"

    class FakeModel:
        config = FakeConfig()

        @staticmethod
        def base_model(**kwargs):
            return kwargs["input_ids"] + 1

    def fake_compile(fn, **kwargs):
        del fn, kwargs

        def compiled(*args, **inner_kwargs):
            del args, inner_kwargs
            raise RuntimeError("compile failed")

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)

    runtime = cleaned_up_stage2_v2.build_stage2_v2_runtime(FakeModel(), compile_enabled=True, log_enabled=False)
    result = runtime.base_model_forward(input_ids=torch.tensor([1, 2, 3], dtype=torch.long))

    assert torch.equal(result, torch.tensor([2, 3, 4], dtype=torch.long))
    assert runtime.base_model_forward.compiled_fn is None


def test_cleaned_up_data_uses_local_stage2_v2_module() -> None:
    assert cleaned_up_data.GeneratedSequenceTree is cleaned_up_stage2_v2.GeneratedSequenceTree
    assert cleaned_up_data.initialize_stage2_v2_hdf5 is cleaned_up_stage2_v2.initialize_stage2_v2_hdf5
