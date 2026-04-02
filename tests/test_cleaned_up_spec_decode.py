from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.cache_utils")
pytest.importorskip("transformers.models.qwen3.modeling_qwen3")
from torch import nn

from src.trees import BlockTreeProcessor, PrunableTreeProcessor


ROOT = Path(__file__).resolve().parents[1]
CLEANED_UP_DIR = ROOT / "cleaned-up"
if str(CLEANED_UP_DIR) not in sys.path:
    sys.path.insert(0, str(CLEANED_UP_DIR))

spec = importlib.util.spec_from_file_location("cleaned_up_spec_decode", CLEANED_UP_DIR / "spec_decode.py")
assert spec is not None and spec.loader is not None
cleaned_up_spec_decode = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cleaned_up_spec_decode
spec.loader.exec_module(cleaned_up_spec_decode)

infer_spec = importlib.util.spec_from_file_location("cleaned_up_infer", CLEANED_UP_DIR / "infer.py")
assert infer_spec is not None and infer_spec.loader is not None
cleaned_up_infer = importlib.util.module_from_spec(infer_spec)
sys.modules[infer_spec.name] = cleaned_up_infer
infer_spec.loader.exec_module(cleaned_up_infer)


class FakeLayerCache:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.is_initialized = True
        self.keys = keys
        self.values = values
        self.cumulative_length = int(keys.shape[-2])


class FakeCache:
    def __init__(self, seq_length: int = 0, layers: list[FakeLayerCache] | None = None) -> None:
        self.seq_length = seq_length
        self.layers = layers or []
        self.crop_calls: list[int] = []

    def get_seq_length(self) -> int:
        return self.seq_length

    def crop(self, new_length: int) -> None:
        self.crop_calls.append(int(new_length))
        self.seq_length = int(new_length)


class FakeEmbedding(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.last_ids: torch.Tensor | None = None

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.last_ids = token_ids.detach().clone()
        return token_ids.unsqueeze(-1).expand(*token_ids.shape, self.hidden_size).to(torch.float32)


class FakeLmHead(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((*hidden_states.shape[:-1], self.vocab_size), dtype=hidden_states.dtype)
        token_ids = hidden_states[..., 0].round().long().clamp(min=0, max=self.vocab_size - 1)
        logits.scatter_(-1, token_ids.unsqueeze(-1), 5.0)
        return logits


class FakeDrafter(nn.Module):
    def __init__(self, q_logits: torch.Tensor, *, mask_token_id: int = 99) -> None:
        super().__init__()
        self.mask_token_id = mask_token_id
        self.q_head = nn.Linear(1, 1, bias=False)
        self.q_logits = q_logits
        self.calls: list[dict[str, torch.Tensor | bool | object]] = []
        self.ctx_extracts = 0

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        tree_info,
        target_ctx_features: torch.Tensor,
        past_key_values,
        use_cache: bool,
        return_aux: bool,
    ):
        self.calls.append(
            {
                "hidden_states": hidden_states.detach().clone(),
                "position_ids": position_ids.detach().clone(),
                "target_ctx_features": target_ctx_features.detach().clone(),
                "tree_info": tree_info,
                "use_cache": use_cache,
                "return_aux": return_aux,
            }
        )
        q_logits = self.q_logits.to(hidden_states.device, hidden_states.dtype).view(1, -1)
        return hidden_states + 1.0, hidden_states + 2.0, q_logits

    def extract_ctx_features(self, hidden_states):
        self.ctx_extracts += 1
        if isinstance(hidden_states, (list, tuple)):
            return torch.cat(list(hidden_states), dim=-1)
        return hidden_states


def test_trim_dynamic_cache_keeps_prefix_and_selected_tree_nodes() -> None:
    keys = torch.arange(1 * 1 * 6 * 2, dtype=torch.float32).view(1, 1, 6, 2)
    values = (100 + torch.arange(1 * 1 * 6 * 2, dtype=torch.float32)).view(1, 1, 6, 2)
    cache = FakeCache(
        seq_length=6,
        layers=[FakeLayerCache(keys=keys.clone(), values=values.clone())],
    )

    trimmed = cleaned_up_spec_decode.trim_dynamic_cache(
        cache,
        prefix_len=2,
        keep_tree_indices=[0, 2],
    )

    assert trimmed.layers[0].keys.shape == (1, 1, 4, 2)
    assert trimmed.layers[0].keys[0, 0, :, 0].tolist() == [0.0, 2.0, 4.0, 8.0]
    assert trimmed.layers[0].values[0, 0, :, 0].tolist() == [100.0, 102.0, 104.0, 108.0]
    assert trimmed.layers[0].cumulative_length == 4


def test_draft_tree_uses_q_head_outputs_and_crops_drafter_cache() -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2, sub_tree_paths=[])
    target_embeddings = FakeEmbedding(hidden_size=3)
    target_lm_head = FakeLmHead(vocab_size=16)
    drafter_cache = FakeCache(seq_length=4)
    q_logits = torch.arange(tree_processor.block_size, dtype=torch.float32)
    drafter = FakeDrafter(q_logits, mask_token_id=77)
    target_ctx_features = torch.ones((1, 3, 2), dtype=torch.float32)

    tree_token_ids, draft_logits, draft_token_probs, tree_info, pruning_scores = cleaned_up_spec_decode.draft_tree(
        drafter_model=drafter,
        raw_drafter=drafter,
        target_embeddings=target_embeddings,
        target_lm_head=target_lm_head,
        tree_processor=tree_processor,
        target_ctx_features=target_ctx_features,
        drafter_cache=drafter_cache,
        current_root_token=5,
        root_position=10,
        temperature=0.0,
        device=torch.device("cpu"),
    )

    assert target_embeddings.last_ids is not None
    assert target_embeddings.last_ids.tolist() == [[5] + [77] * (tree_info.block_size - 1)]
    assert drafter.calls[0]["use_cache"] is True
    assert drafter.calls[0]["return_aux"] is True
    expected_position_ids = [[4, 5, 6] + (tree_info.depth + 10).tolist()]
    assert drafter.calls[0]["position_ids"].tolist() == expected_position_ids
    assert drafter_cache.crop_calls == [7]
    assert tree_token_ids.tolist()[0] == 5
    assert draft_logits.shape == (tree_info.block_size, 16)
    assert draft_token_probs.shape == (tree_info.block_size,)
    expected_pruning_scores = torch.sigmoid(q_logits)
    expected_pruning_scores[0] = 1.0
    assert torch.allclose(pruning_scores, expected_pruning_scores)


def test_prune_drafted_tree_preserves_ancestor_closure() -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2, sub_tree_paths=["0-1", "0-2", "1-3"])
    tree_info = tree_processor.build_tree_info(batch_size=1, num_blocks=1, device=torch.device("cpu"))
    tree_token_ids = torch.arange(tree_info.block_size, dtype=torch.long)
    draft_logits = torch.randn(tree_info.block_size, 5)
    draft_token_probs = torch.linspace(1.0, 0.1, steps=tree_info.block_size, dtype=torch.float32)
    node_correctness_probs = torch.tensor([1.0, 0.2, 0.4, 0.3, 0.9, 0.8, 0.1, 0.7], dtype=torch.float32)

    pruned_tokens, _, _, pruned_scores, pruned_info = cleaned_up_spec_decode.prune_drafted_tree(
        tree_token_ids=tree_token_ids,
        draft_logits=draft_logits,
        draft_token_probs=draft_token_probs,
        tree_info=tree_info,
        node_correctness_probs=node_correctness_probs,
        candidate_tree_size=4,
    )

    assert pruned_tokens.tolist() == [0, 4, 5, 7]
    assert pruned_scores.tolist() == pytest.approx([1.0, 0.9, 0.8, 0.7])
    assert pruned_info.parent_idx.tolist() == [-1, 0, 1, 2]


def test_build_acceptance_mask_handles_greedy_and_temperature_modes(monkeypatch) -> None:
    tree_info = SimpleNamespace(
        block_size=2,
        parent_idx=torch.tensor([-1, 0], dtype=torch.long),
    )
    tree_token_ids = torch.tensor([3, 4], dtype=torch.long)
    verifier_logits = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 5.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    q_scores = torch.tensor([1.0, 0.4], dtype=torch.float32)

    greedy = cleaned_up_spec_decode.build_acceptance_mask(
        q_scores=q_scores,
        verifier_logits=verifier_logits,
        tree_token_ids=tree_token_ids,
        tree_info=tree_info,
        temperature=0.0,
    )
    assert greedy.tolist() == [True, True]

    monkeypatch.setattr(cleaned_up_spec_decode.torch, "rand", lambda *args, **kwargs: torch.tensor(0.2))
    sampled = cleaned_up_spec_decode.build_acceptance_mask(
        q_scores=q_scores,
        verifier_logits=verifier_logits,
        tree_token_ids=tree_token_ids,
        tree_info=tree_info,
        temperature=1.0,
    )
    assert sampled.tolist() == [True, True]


def test_build_acceptance_mask_normalizes_q_scores_per_parent(monkeypatch) -> None:
    tree_info = SimpleNamespace(
        block_size=3,
        parent_idx=torch.tensor([-1, 0, 0], dtype=torch.long),
    )
    tree_token_ids = torch.tensor([3, 0, 1], dtype=torch.long)
    verifier_logits = torch.tensor(
        [
            [0.0, 0.0, -10.0, -10.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    q_scores = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32)

    monkeypatch.setattr(cleaned_up_spec_decode.torch, "rand", lambda *args, **kwargs: torch.tensor(0.8))
    accepted = cleaned_up_spec_decode.build_acceptance_mask(
        q_scores=q_scores,
        verifier_logits=verifier_logits,
        tree_token_ids=tree_token_ids,
        tree_info=tree_info,
        temperature=1.0,
    )

    assert accepted.tolist() == [True, True, False]


def test_speculative_generate_from_ids_preserves_caches_across_steps(monkeypatch) -> None:
    class LocalDynamicCache(FakeCache):
        def __init__(self) -> None:
            super().__init__(seq_length=0, layers=[])

    monkeypatch.setattr(cleaned_up_spec_decode, "DynamicCache", LocalDynamicCache)

    trim_calls: list[tuple[int, list[int]]] = []

    def fake_trim_dynamic_cache(past_key_values, *, prefix_len: int, keep_tree_indices: list[int]):
        trim_calls.append((prefix_len, list(keep_tree_indices)))
        past_key_values.seq_length = prefix_len + len(keep_tree_indices)
        return past_key_values

    monkeypatch.setattr(cleaned_up_spec_decode, "trim_dynamic_cache", fake_trim_dynamic_cache)
    tree_info = SimpleNamespace(
        block_size=2,
        parent_idx=torch.tensor([-1, 0], dtype=torch.long),
        depth=torch.tensor([0, 1], dtype=torch.long),
        tree_mask=torch.tensor(
            [
                [True, False],
                [True, True],
            ],
            dtype=torch.bool,
        ),
        primary_path_mask=torch.tensor([True, True], dtype=torch.bool),
    )
    call_state = {"draft": 0}

    def fake_draft_tree(**kwargs):
        call_state["draft"] += 1
        drafter_cache = kwargs["drafter_cache"]
        if call_state["draft"] == 1:
            assert drafter_cache.get_seq_length() == 0
        else:
            assert drafter_cache.get_seq_length() == 2
        drafter_cache.crop(kwargs["target_ctx_features"].shape[1])
        tree_token_ids = torch.tensor([kwargs["current_root_token"], 4], dtype=torch.long)
        draft_logits = torch.tensor([[0.0, 5.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 5.0]], dtype=torch.float32)
        draft_token_probs = torch.tensor([1.0, 1.0], dtype=torch.float32)
        pruning_scores = torch.tensor([1.0, 0.9], dtype=torch.float32)
        return tree_token_ids, draft_logits, draft_token_probs, tree_info, pruning_scores

    def fake_verify_tree(**kwargs):
        target_cache = kwargs["target_cache"]
        updated_cache = FakeCache(seq_length=target_cache.get_seq_length() + kwargs["tree_info"].block_size, layers=[])
        verifier_logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 5.0], [0.0, 0.0, 0.0, 0.0, 6.0]], dtype=torch.float32)
        next_ctx = torch.ones((1, 2, 3), dtype=torch.float32) * call_state["draft"]
        return updated_cache, verifier_logits, next_ctx

    monkeypatch.setattr(cleaned_up_spec_decode, "draft_tree", fake_draft_tree)
    monkeypatch.setattr(cleaned_up_spec_decode, "verify_tree", fake_verify_tree)

    class FakeTargetModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.param = nn.Parameter(torch.zeros(()))
            self.embedding = FakeEmbedding(hidden_size=3)
            self.lm_head = FakeLmHead(vocab_size=5)

        def get_input_embeddings(self):
            return self.embedding

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, *, input_ids: torch.Tensor, output_hidden_states: bool, use_cache: bool, past_key_values, **kwargs):
            del kwargs, use_cache
            past_key_values.seq_length = input_ids.shape[1]
            hidden = input_ids.unsqueeze(-1).to(torch.float32)
            logits = torch.tensor(
                [[[0.0, 5.0, 0.0, 0.0, 0.0] for _ in range(input_ids.shape[1])]],
                dtype=torch.float32,
            )
            return SimpleNamespace(
                hidden_states=(hidden, hidden + 1.0) if output_hidden_states else None,
                logits=logits,
                past_key_values=past_key_values,
            )

    target_model = FakeTargetModel()
    drafter = FakeDrafter(torch.tensor([0.0, 2.0]))
    tokenizer = SimpleNamespace(
        eos_token_id=None,
        decode=lambda token_ids, skip_special_tokens=True: ",".join(str(int(token)) for token in token_ids),
    )
    tree_processor = PrunableTreeProcessor(tree_seq_depth=2, candidate_tree_size=2, sub_tree_paths=[])

    result = cleaned_up_spec_decode.speculative_generate_from_ids(
        target_model=target_model,
        drafter_model=drafter,
        tokenizer=tokenizer,
        prompt_ids=torch.tensor([9, 8], dtype=torch.long),
        tree_processor=tree_processor,
        max_new_tokens=5,
        temperature=0.0,
    )

    assert call_state["draft"] == 2
    assert trim_calls == [(2, [0, 1]), (4, [0, 1])]
    assert result.continuation_ids.tolist() == [4, 4, 4, 4]
    assert result.acceptance_lengths == [1, 1]
    assert result.drafted_tokens == 2
    assert result.committed_tokens == 5


def test_cleaned_up_infer_uses_local_spec_decode() -> None:
    assert cleaned_up_infer.speculative_generate_from_ids is cleaned_up_spec_decode.speculative_generate_from_ids
