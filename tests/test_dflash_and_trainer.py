from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")
pytest.importorskip("h5py")
transformers = pytest.importorskip("transformers")
pytest.importorskip("lightning")

from src.data import DataModuleConfig, PackedBatch
from src.models.dflash import DFlashDraftModel
from src.spec_decode import (
    build_parser as build_spec_decode_parser,
    build_tree_processor,
    choose_deepest_valid_node,
    gather_path_indices,
    prune_drafted_tree,
    select_pruned_keep_indices,
)
from src.trainer import Trainer, TrainerConfig, build_parser as build_trainer_parser
from src.trees import BlockTreeProcessor, PrunableTreeProcessor


def _build_fake_trainer_components(block_size: int, *, with_q_head: bool = False):
    class FakeFabric:
        instances: list["FakeFabric"] = []

        def __init__(self, *args, **kwargs):
            self.device = torch.device("cpu")
            self.is_global_zero = True
            self.backward_calls: list[float] = []
            type(self).instances.append(self)

        def launch(self):
            return None

        def seed_everything(self, seed):
            torch.manual_seed(seed)

        def to_device(self, module):
            return module

        def setup(self, model, optimizer):
            return model, optimizer

        def setup_dataloaders(self, *loaders, **kwargs):
            return loaders

        def no_backward_sync(self, module, enabled):
            return nullcontext()

        def backward(self, loss):
            self.backward_calls.append(float(loss.detach().item()))
            loss.backward()

        def clip_gradients(self, model, optimizer, max_norm):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        def save(self, path, state):
            torch.save({"global_step": state["global_step"]}, path)

        def load(self, path, state):
            payload = torch.load(path)
            return payload

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0
        pad_token = "<pad>"
        eos_token = "<pad>"

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class FakeTargetModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(256, 16)
            self.lm_head = nn.Linear(16, 256, bias=False)
            self.config = type("Cfg", (), {"pad_token_id": 0})()

        def get_input_embeddings(self):
            return self.embed

        def get_output_embeddings(self):
            return self.lm_head

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def forward(
            self,
            input_ids,
            attention_mask=None,
            position_ids=None,
            output_hidden_states=False,
            use_cache=False,
        ):
            hidden = self.embed(input_ids)
            hidden_states = (hidden, hidden + 0.1, hidden + 0.2)
            return type("Out", (), {"hidden_states": hidden_states})()

    class FakeDrafter(nn.Module):
        mask_token_id = 255

        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(16, 16)
            self.q_head = nn.Linear(16, 1, bias=False) if with_q_head else None

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def extract_ctx_features(self, hidden_states):
            return torch.cat(hidden_states[1:], dim=-1)

        def forward(self, *, hidden_states, position_ids, tree_info, attention_mask, target_ctx_features):
            return self.proj(hidden_states), hidden_states

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), Path(path) / "pytorch_model.bin")

    packed_batch = PackedBatch(
        input_ids=torch.tensor([[1, 2, 3, 4, 5, 6, 0, 0]], dtype=torch.long),
        position_ids=torch.tensor([[0, 1, 2, 3, 4, 5, 0, 0]], dtype=torch.long),
        document_mask=torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.long),
        context_valid_mask=torch.tensor([[True, True, True, True, True, True, False, False]]),
        anchor_positions=torch.tensor([[2, 3]], dtype=torch.long),
        anchor_document_ids=torch.tensor([[1, 1]], dtype=torch.long),
        anchor_valid_mask=torch.tensor([[True, True]]),
        tree_labels=(torch.arange(2 * block_size, dtype=torch.long).view(1, 2, block_size) % 32) + 10,
        tree_noise_ids=torch.full((1, 2, block_size), 255, dtype=torch.long),
        tree_position_ids=torch.stack(
            [
                torch.arange(block_size, dtype=torch.long) + 2,
                torch.arange(block_size, dtype=torch.long) + 3,
            ],
            dim=0,
        ).unsqueeze(0),
        tree_cum_probs=torch.linspace(0.25, 1.0, steps=2 * block_size, dtype=torch.float32).view(1, 2, block_size),
        tree_valid_mask=torch.ones((1, 2, block_size), dtype=torch.bool),
    )
    packed_batch.tree_noise_ids[:, :, 0] = packed_batch.tree_labels[:, :, 0]
    return FakeFabric, FakeTokenizer, FakeTargetModel, FakeDrafter, packed_batch


def _make_trainer(
    monkeypatch,
    tmp_path: Path,
    *,
    anchor_chunk_size: int | None,
    batch: PackedBatch,
    with_q_head: bool = False,
    tree_type: str = "block",
    tree_args: dict | None = None,
    q_loss_lambda: float = 1.0,
) -> Trainer:
    import src.trainer as trainer_mod

    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    FakeFabric, FakeTokenizer, FakeTargetModel, FakeDrafter, _ = _build_fake_trainer_components(
        tree_processor.block_size,
        with_q_head=with_q_head,
    )

    monkeypatch.setattr(trainer_mod, "Fabric", FakeFabric)
    monkeypatch.setattr(trainer_mod, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(trainer_mod, "AutoModelForCausalLM", FakeTargetModel)
    monkeypatch.setattr(trainer_mod, "DFlashDraftModel", FakeDrafter)
    monkeypatch.setattr(trainer_mod, "build_dataloaders", lambda **kwargs: ([batch], [batch]))
    monkeypatch.setattr(trainer_mod, "build_drafter_block_mask", lambda **kwargs: None)

    torch.manual_seed(0)
    return Trainer(
        config=TrainerConfig(
            num_epochs=1,
            eval_every=1,
            log_every=1,
            save_every=1,
            checkpoint_path=str(tmp_path / "ckpts"),
            no_wandb=True,
            dev_run=True,
            precision="32-true",
            anchor_chunk_size=anchor_chunk_size,
            q_loss_lambda=q_loss_lambda,
        ),
        target="fake-target",
        data=DataModuleConfig(
            path="unused.h5",
            batch_size=1,
            tree_seq_depth=2,
        ),
        drafter="fake-drafter",
        tree_type=tree_type,
        tree_args=tree_args,
    )


def test_dflash_forward_shapes() -> None:
    Qwen3Config = pytest.importorskip("transformers.models.qwen3.modeling_qwen3").Qwen3Config
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    config.num_target_layers = 2
    config.block_size = 8
    config.max_tree_size = 8
    config.dflash_config = {"mask_token_id": 1, "target_layer_ids": [0, 1]}
    config._attn_implementation = "eager"

    model = DFlashDraftModel(config)
    model.config._attn_implementation = "eager"
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    tree_info = tree_processor.build_tree_info(batch_size=1, num_blocks=1, device=torch.device("cpu"))

    hidden_states = torch.randn(1, tree_processor.block_size, config.hidden_size)
    position_ids = torch.arange(tree_processor.block_size).unsqueeze(0)
    target_ctx_features = torch.randn(1, 6, len(model.target_layer_ids) * config.hidden_size)

    final_hidden, backbone_hidden = model(
        hidden_states=hidden_states,
        position_ids=position_ids,
        tree_info=tree_info,
        target_ctx_features=target_ctx_features,
        attention_mask=None,
    )
    assert final_hidden.shape == hidden_states.shape
    assert backbone_hidden.shape == hidden_states.shape


def test_dflash_forward_exposes_backbone_for_q_head() -> None:
    Qwen3Config = pytest.importorskip("transformers.models.qwen3.modeling_qwen3").Qwen3Config
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    config.num_target_layers = 2
    config.block_size = 8
    config.max_tree_size = 8
    config.dflash_config = {"mask_token_id": 1, "target_layer_ids": [0, 1]}
    config._attn_implementation = "eager"
    config.use_q_head = True

    model = DFlashDraftModel(config)
    model.config._attn_implementation = "eager"
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    tree_info = tree_processor.build_tree_info(batch_size=1, num_blocks=1, device=torch.device("cpu"))

    hidden_states = torch.randn(1, tree_processor.block_size, config.hidden_size)
    position_ids = torch.arange(tree_processor.block_size).unsqueeze(0)
    target_ctx_features = torch.randn(1, 6, len(model.target_layer_ids) * config.hidden_size)

    _, backbone_hidden = model(
        hidden_states=hidden_states,
        position_ids=position_ids,
        tree_info=tree_info,
        target_ctx_features=target_ctx_features,
        attention_mask=None,
    )
    q_logits = model.q_head(backbone_hidden)
    assert q_logits.shape == (1, tree_processor.block_size, 1)


def test_trainer_smoke_with_fakes(monkeypatch, tmp_path: Path) -> None:
    import src.trainer as trainer_mod

    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    FakeFabric, FakeTokenizer, FakeTargetModel, FakeDrafter, packed_batch = _build_fake_trainer_components(
        tree_processor.block_size
    )

    monkeypatch.setattr(trainer_mod, "Fabric", FakeFabric)
    monkeypatch.setattr(trainer_mod, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(trainer_mod, "AutoModelForCausalLM", FakeTargetModel)
    monkeypatch.setattr(trainer_mod, "DFlashDraftModel", FakeDrafter)
    monkeypatch.setattr(
        trainer_mod,
        "build_dataloaders",
        lambda **kwargs: ([packed_batch], [packed_batch]),
    )
    monkeypatch.setattr(
        trainer_mod,
        "build_drafter_block_mask",
        lambda **kwargs: None,
    )

    trainer = Trainer(
        config=TrainerConfig(
            num_epochs=1,
            eval_every=1,
            log_every=1,
            save_every=1,
            checkpoint_path=str(tmp_path / "ckpts"),
            no_wandb=True,
            dev_run=True,
            precision="32-true",
        ),
        target="fake-target",
        data=DataModuleConfig(
            path="unused.h5",
            batch_size=1,
            tree_seq_depth=2,
        ),
        drafter="fake-drafter",
        tree_type="block",
    )
    trainer.fit()

    assert (tmp_path / "ckpts" / "final" / "fabric_ckpt.pt").exists()
    assert (tmp_path / "ckpts" / "final" / "hf_draft" / "pytorch_model.bin").exists()


def test_train_batch_chunking_preserves_loss_and_gradients(monkeypatch, tmp_path: Path) -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    _, _, _, _, packed_batch = _build_fake_trainer_components(tree_processor.block_size)

    trainer_full = _make_trainer(
        monkeypatch,
        tmp_path / "full",
        anchor_chunk_size=None,
        batch=packed_batch,
    )
    trainer_chunked = _make_trainer(
        monkeypatch,
        tmp_path / "chunked",
        anchor_chunk_size=1,
        batch=packed_batch,
    )

    trainer_full.optimizer.zero_grad()
    trainer_chunked.optimizer.zero_grad()
    full_result = trainer_full._train_batch(packed_batch)
    chunked_result = trainer_chunked._train_batch(packed_batch)

    assert torch.isclose(full_result["loss"], chunked_result["loss"])
    assert full_result["valid_count"] == chunked_result["valid_count"]

    full_params = dict(trainer_full.drafter_model.named_parameters())
    chunked_params = dict(trainer_chunked.drafter_model.named_parameters())
    for name in full_params:
        assert torch.allclose(full_params[name].grad, chunked_params[name].grad)


def test_train_batch_backwards_once_per_anchor_chunk(monkeypatch, tmp_path: Path) -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    _, _, _, _, packed_batch = _build_fake_trainer_components(tree_processor.block_size)
    trainer = _make_trainer(
        monkeypatch,
        tmp_path,
        anchor_chunk_size=1,
        batch=packed_batch,
    )

    result = trainer._train_batch(packed_batch)

    assert result["valid_count"] > 0
    assert len(trainer.fabric.backward_calls) == 2


def test_train_batch_adds_q_loss_when_q_head_is_present(monkeypatch, tmp_path: Path) -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    _, _, _, _, packed_batch = _build_fake_trainer_components(tree_processor.block_size, with_q_head=True)
    trainer = _make_trainer(
        monkeypatch,
        tmp_path,
        anchor_chunk_size=None,
        batch=packed_batch,
        with_q_head=True,
        q_loss_lambda=0.5,
    )

    result = trainer._train_batch(packed_batch)

    assert result["q_loss"].item() > 0
    assert torch.isclose(result["loss"], result["ce_loss"] + 0.5 * result["q_loss"])


def test_q_loss_respects_non_root_valid_mask(monkeypatch, tmp_path: Path) -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    _, _, _, _, packed_batch = _build_fake_trainer_components(tree_processor.block_size, with_q_head=True)
    masked_batch = PackedBatch(
        input_ids=packed_batch.input_ids,
        position_ids=packed_batch.position_ids,
        document_mask=packed_batch.document_mask,
        context_valid_mask=packed_batch.context_valid_mask,
        anchor_positions=packed_batch.anchor_positions,
        anchor_document_ids=packed_batch.anchor_document_ids,
        anchor_valid_mask=packed_batch.anchor_valid_mask,
        tree_labels=packed_batch.tree_labels,
        tree_noise_ids=packed_batch.tree_noise_ids,
        tree_position_ids=packed_batch.tree_position_ids,
        tree_cum_probs=packed_batch.tree_cum_probs,
        tree_valid_mask=torch.zeros_like(packed_batch.tree_valid_mask),
    )
    masked_batch.tree_valid_mask[:, :, 0] = True
    trainer = _make_trainer(
        monkeypatch,
        tmp_path,
        anchor_chunk_size=None,
        batch=masked_batch,
        with_q_head=True,
    )

    chunk_result = trainer._forward_anchor_chunk(
        masked_batch,
        trainer._prefill_target_context(masked_batch),
        slice(0, masked_batch.num_anchors),
        compute_predictions=False,
    )

    assert chunk_result["valid_count"] == 0
    assert float(chunk_result["q_loss_sum"].item()) == 0.0


def test_train_batch_skips_backward_when_no_valid_targets(monkeypatch, tmp_path: Path) -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    _, _, _, _, packed_batch = _build_fake_trainer_components(tree_processor.block_size)
    packed_batch = PackedBatch(
        input_ids=packed_batch.input_ids,
        position_ids=packed_batch.position_ids,
        document_mask=packed_batch.document_mask,
        context_valid_mask=packed_batch.context_valid_mask,
        anchor_positions=packed_batch.anchor_positions,
        anchor_document_ids=packed_batch.anchor_document_ids,
        anchor_valid_mask=packed_batch.anchor_valid_mask,
        tree_labels=packed_batch.tree_labels,
        tree_noise_ids=packed_batch.tree_noise_ids,
        tree_position_ids=packed_batch.tree_position_ids,
        tree_cum_probs=packed_batch.tree_cum_probs,
        tree_valid_mask=torch.zeros_like(packed_batch.tree_valid_mask),
    )
    trainer = _make_trainer(
        monkeypatch,
        tmp_path,
        anchor_chunk_size=1,
        batch=packed_batch,
    )

    result = trainer._train_batch(packed_batch)

    assert result["valid_count"] == 0
    assert float(result["loss"].item()) == 0.0
    assert trainer.fabric.backward_calls == []


def test_trainer_accepts_branch_off_tree_type(monkeypatch, tmp_path: Path) -> None:
    import src.trainer as trainer_mod

    class FakeFabric:
        def __init__(self, *args, **kwargs):
            self.device = torch.device("cpu")
            self.is_global_zero = True

        def launch(self):
            return None

        def seed_everything(self, seed):
            torch.manual_seed(seed)

        def to_device(self, module):
            return module

        def setup(self, model, optimizer):
            return model, optimizer

        def setup_dataloaders(self, *loaders, **kwargs):
            return loaders

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0
        pad_token = "<pad>"
        eos_token = "<pad>"

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class FakeTargetModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(32, 8)
            self.lm_head = nn.Linear(8, 32, bias=False)
            self.config = type("Cfg", (), {"pad_token_id": 0})()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def get_input_embeddings(self):
            return self.embed

        def get_output_embeddings(self):
            return self.lm_head

    class FakeDrafter(nn.Module):
        mask_token_id = 1

        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 8)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    monkeypatch.setattr(trainer_mod, "Fabric", FakeFabric)
    monkeypatch.setattr(trainer_mod, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(trainer_mod, "AutoModelForCausalLM", FakeTargetModel)
    monkeypatch.setattr(trainer_mod, "DFlashDraftModel", FakeDrafter)
    monkeypatch.setattr(trainer_mod, "build_dataloaders", lambda **kwargs: ([], []))

    trainer = Trainer(
        config=TrainerConfig(
            checkpoint_path=str(tmp_path / "ckpts"),
            no_wandb=True,
            precision="32-true",
        ),
        target="fake-target",
        data=DataModuleConfig(
            path="unused.h5",
            batch_size=1,
            tree_seq_depth=3,
        ),
        drafter="fake-drafter",
        tree_type="branch_off",
        tree_args={"branching_pattern": [[0, 2], [0, 3], [0]]},
    )
    assert trainer.tree_processor.block_size == 6
    assert trainer.tree_processor.primary_path_indices.tolist() == [0, 2, 5]


def test_trainer_accepts_prunable_tree_type(monkeypatch, tmp_path: Path) -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    _, _, _, _, packed_batch = _build_fake_trainer_components(tree_processor.block_size, with_q_head=True)
    trainer = _make_trainer(
        monkeypatch,
        tmp_path,
        anchor_chunk_size=None,
        batch=packed_batch,
        with_q_head=True,
        tree_type="prunable",
        tree_args={"base_tree_type": "branch_off", "branching_pattern": [[0, 2], [0, 3]], "prune_topk": 2},
    )

    assert isinstance(trainer.tree_processor, PrunableTreeProcessor)
    assert trainer.tree_processor.base_tree_type == "branch_off"
    assert trainer.tree_processor.prune_topk == 2


def test_trainer_prunable_requires_q_head(monkeypatch, tmp_path: Path) -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    _, _, _, _, packed_batch = _build_fake_trainer_components(tree_processor.block_size)

    with pytest.raises(ValueError, match="use_q_head"):
        _make_trainer(
            monkeypatch,
            tmp_path,
            anchor_chunk_size=None,
            batch=packed_batch,
            tree_type="prunable",
            tree_args={"base_tree_type": "block", "prune_topk": 1},
        )


def test_prunable_tree_args_validation(monkeypatch, tmp_path: Path) -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    _, _, _, _, packed_batch = _build_fake_trainer_components(tree_processor.block_size, with_q_head=True)

    with pytest.raises(ValueError, match="base_tree_type"):
        _make_trainer(
            monkeypatch,
            tmp_path / "bad-base",
            anchor_chunk_size=None,
            batch=packed_batch,
            with_q_head=True,
            tree_type="prunable",
            tree_args={"base_tree_type": "bad", "prune_topk": 1},
        )
    with pytest.raises(ValueError, match="prune_topk"):
        _make_trainer(
            monkeypatch,
            tmp_path / "bad-topk",
            anchor_chunk_size=None,
            batch=packed_batch,
            with_q_head=True,
            tree_type="prunable",
            tree_args={"base_tree_type": "block", "prune_topk": -1},
        )


def test_spec_decode_path_helpers_use_tree_structure() -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    tree_info = tree_processor.build_tree_info(batch_size=1, num_blocks=1, device=torch.device("cpu"))
    accepted_mask = torch.tensor([True, True, False, True, True, True, False, True], dtype=torch.bool)

    deepest_idx = choose_deepest_valid_node(accepted_mask, tree_info)
    assert deepest_idx == 7
    assert gather_path_indices(deepest_idx, tree_info) == [0, 4, 5, 7]


def test_prune_selection_keeps_ancestor_closure() -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    tree_info = tree_processor.build_tree_info(batch_size=1, num_blocks=1, device=torch.device("cpu"))
    q_scores = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9], dtype=torch.float32)

    keep_indices = select_pruned_keep_indices(tree_info=tree_info, q_scores=q_scores, prune_topk=1)

    assert keep_indices == [0, 4, 5, 7]


def test_prune_drafted_tree_supports_root_only_and_branch_off_base() -> None:
    base_tree = build_tree_processor(
        tree_type="prunable",
        tree_seq_depth=3,
        tree_args={"base_tree_type": "branch_off", "branching_pattern": [[0, 2], [0, 3], [0]], "prune_topk": 0},
    )
    assert isinstance(base_tree, PrunableTreeProcessor)
    tree_info = base_tree.build_tree_info(batch_size=1, num_blocks=1, device=torch.device("cpu"))
    tree_token_ids = torch.arange(tree_info.block_size, dtype=torch.long)
    draft_logits = torch.randn(tree_info.block_size, 5)
    draft_token_probs = torch.linspace(1.0, 0.1, steps=tree_info.block_size, dtype=torch.float32)
    q_scores = torch.linspace(0.0, 1.0, steps=tree_info.block_size, dtype=torch.float32)

    pruned_tokens, pruned_logits, pruned_probs, pruned_info = prune_drafted_tree(
        tree_token_ids=tree_token_ids,
        draft_logits=draft_logits,
        draft_token_probs=draft_token_probs,
        tree_info=tree_info,
        q_scores=q_scores,
        prune_topk=0,
    )

    assert pruned_info.block_size == 1
    assert pruned_info.parent_idx.tolist() == [-1]
    assert pruned_tokens.tolist() == [0]
    assert pruned_logits.shape[0] == 1
    assert pruned_probs.tolist() == [1.0]


def test_parsers_accept_prunable_tree_type() -> None:
    trainer_tree_action = next(action for action in build_trainer_parser()._actions if action.dest == "tree_type")
    spec_tree_action = next(action for action in build_spec_decode_parser()._actions if action.dest == "tree_type")

    assert "prunable" in trainer_tree_action.choices
    assert "prunable" in spec_tree_action.choices
