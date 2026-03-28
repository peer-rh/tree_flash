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
from src.spec_decode import choose_deepest_valid_node, gather_path_indices
from src.trainer import Trainer, TrainerConfig
from src.trees import BlockTreeProcessor


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


def test_trainer_smoke_with_fakes(monkeypatch, tmp_path: Path) -> None:
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

        def no_backward_sync(self, module, enabled):
            return nullcontext()

        def backward(self, loss):
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

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def extract_ctx_features(self, hidden_states):
            return torch.cat(hidden_states[1:], dim=-1)

        def forward(self, *, hidden_states, position_ids, tree_info, attention_mask, target_ctx_features):
            return self.proj(hidden_states), self.proj(hidden_states)

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), Path(path) / "pytorch_model.bin")

    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    block_size = tree_processor.block_size
    packed_batch = PackedBatch(
        input_ids=torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]], dtype=torch.long),
        position_ids=torch.tensor([[0, 1, 2, 3, 4, 0, 0, 0]], dtype=torch.long),
        document_mask=torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.long),
        context_valid_mask=torch.tensor([[True, True, True, True, True, False, False, False]]),
        anchor_positions=torch.tensor([[2]], dtype=torch.long),
        anchor_document_ids=torch.tensor([[1]], dtype=torch.long),
        anchor_valid_mask=torch.tensor([[True]]),
        tree_labels=torch.arange(block_size, dtype=torch.long).view(1, 1, block_size) + 10,
        tree_noise_ids=torch.full((1, 1, block_size), 255, dtype=torch.long),
        tree_position_ids=torch.arange(block_size, dtype=torch.long).view(1, 1, block_size) + 2,
        tree_cum_probs=torch.ones((1, 1, block_size), dtype=torch.float32),
        tree_valid_mask=torch.ones((1, 1, block_size), dtype=torch.bool),
    )
    packed_batch.tree_noise_ids[:, :, 0] = packed_batch.tree_labels[:, :, 0]

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


def test_spec_decode_path_helpers_use_tree_structure() -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    tree_info = tree_processor.build_tree_info(batch_size=1, num_blocks=1, device=torch.device("cpu"))
    accepted_mask = torch.tensor([True, True, False, True, True, True, False, True], dtype=torch.bool)

    deepest_idx = choose_deepest_valid_node(accepted_mask, tree_info)
    assert deepest_idx == 7
    assert gather_path_indices(deepest_idx, tree_info) == [0, 4, 5, 7]
