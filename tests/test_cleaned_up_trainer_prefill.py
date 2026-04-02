from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
CLEANED_UP_DIR = ROOT / "cleaned-up"
if str(CLEANED_UP_DIR) not in sys.path:
    sys.path.insert(0, str(CLEANED_UP_DIR))

spec = importlib.util.spec_from_file_location("cleaned_up_trainer", CLEANED_UP_DIR / "trainer.py")
assert spec is not None and spec.loader is not None
cleaned_up_trainer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cleaned_up_trainer
spec.loader.exec_module(cleaned_up_trainer)

TreeFlashTrainer = cleaned_up_trainer.TreeFlashTrainer


class OffsetLayer(nn.Module):
    def __init__(self, delta: float) -> None:
        super().__init__()
        self.register_buffer("delta", torch.tensor(delta, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.delta


class HookableFakeTargetModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([OffsetLayer(1.0), OffsetLayer(10.0), OffsetLayer(100.0)])
        self.output_hidden_states_flags: list[bool] = []

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask,
        position_ids: torch.Tensor,
        output_hidden_states: bool,
        use_cache: bool,
    ):
        del attention_mask, position_ids, use_cache
        self.output_hidden_states_flags.append(bool(output_hidden_states))
        hidden_states = input_ids.unsqueeze(-1).to(torch.float32)
        all_hidden_states = [hidden_states]
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
            all_hidden_states.append(hidden_states)
        if output_hidden_states:
            return SimpleNamespace(hidden_states=tuple(all_hidden_states))
        return SimpleNamespace(last_hidden_state=hidden_states)


class FallbackFakeTargetModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Identity()
        self.output_hidden_states_flags: list[bool] = []

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask,
        position_ids: torch.Tensor,
        output_hidden_states: bool,
        use_cache: bool,
    ):
        del attention_mask, position_ids, use_cache
        self.output_hidden_states_flags.append(bool(output_hidden_states))
        hidden_states = input_ids.unsqueeze(-1).to(torch.float32)
        return SimpleNamespace(hidden_states=(hidden_states, hidden_states + 1.0))


class FakeDrafter:
    def __init__(self, target_layer_ids: list[int]) -> None:
        self.target_layer_ids = target_layer_ids
        self.extract_calls = 0

    def extract_ctx_features(self, hidden_states):
        self.extract_calls += 1
        return torch.cat(list(hidden_states), dim=-1)


def _make_batch() -> SimpleNamespace:
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    return SimpleNamespace(
        input_ids=input_ids,
        position_ids=torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0),
        document_mask=torch.ones_like(input_ids, dtype=torch.long),
        context_valid_mask=torch.ones_like(input_ids, dtype=torch.bool),
    )


def test_cleaned_up_prefill_captures_only_requested_target_layers() -> None:
    trainer = TreeFlashTrainer.__new__(TreeFlashTrainer)
    trainer.target_model = HookableFakeTargetModel()
    trainer.raw_drafter = FakeDrafter(target_layer_ids=[0, 2])
    trainer._build_prefill_attention_mask = lambda document_mask, valid_mask: "prefill-mask"

    features = trainer._prefill_target_context(_make_batch())

    expected = torch.tensor(
        [[[2.0, 112.0], [3.0, 113.0], [4.0, 114.0]]],
        dtype=torch.float32,
    )
    assert torch.equal(features, expected)
    assert trainer.target_model.output_hidden_states_flags == [False]
    assert trainer.raw_drafter.extract_calls == 0


def test_cleaned_up_prefill_falls_back_to_hidden_states_when_layer_hooks_are_unavailable() -> None:
    trainer = TreeFlashTrainer.__new__(TreeFlashTrainer)
    trainer.target_model = FallbackFakeTargetModel()
    trainer.raw_drafter = FakeDrafter(target_layer_ids=[0])
    trainer._build_prefill_attention_mask = lambda document_mask, valid_mask: "prefill-mask"

    features = trainer._prefill_target_context(_make_batch())

    expected = torch.tensor(
        [[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]],
        dtype=torch.float32,
    )
    assert torch.equal(features, expected)
    assert trainer.target_model.output_hidden_states_flags == [True]
    assert trainer.raw_drafter.extract_calls == 1
