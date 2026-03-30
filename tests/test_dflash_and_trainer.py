from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")
F = pytest.importorskip("torch.nn.functional")
pytest.importorskip("h5py")
transformers = pytest.importorskip("transformers")
pytest.importorskip("lightning")

from data_pipeline.stage2 import IGNORE_IDX
from src.data import DataModuleConfig, PackedBatch
from src.models.dflash import DFlashDraftModel
from src.spec_decode import (
    build_parser as build_spec_decode_parser,
    build_pruning_scores,
    build_verifier_score_mod,
    build_tree_processor,
    compare_generation_to_reference,
    choose_deepest_valid_node,
    evaluate_prompt_suite,
    gather_path_indices,
    prune_drafted_tree,
    render_eval_suite_html,
    resolve_official_dflash_model,
    infer_target_model_for_draft_model,
    select_pruned_keep_indices,
    verify_tree,
)
from src.trainer import (
    Trainer,
    TrainerConfig,
    build_ar_block_mask,
    build_parser as build_trainer_parser,
    build_prefill_attention_mask,
)
from src.trees import BlockTreeProcessor, PrunableTreeProcessor


def _require_flex_attention() -> None:
    try:
        import torch.nn.attention.flex_attention  # noqa: F401
    except ImportError:
        pytest.skip("flex attention is unavailable in this test environment")


def _write_stage2_fixture(path: Path) -> None:
    import h5py

    with h5py.File(path, "w") as hf:
        vlen = h5py.vlen_dtype("int64")
        hf.create_dataset("prompt_ids", shape=(2,), dtype=vlen)
        hf.create_dataset("response_ids", shape=(2,), dtype=vlen)
        hf["prompt_ids"][0] = [11, 12]
        hf["response_ids"][0] = [21, 22, 23]
        hf["prompt_ids"][1] = [13]
        hf["response_ids"][1] = [24, 25]

        hf.create_dataset("sequence_offsets", data=[0, 3, 5], dtype="int64")
        hf.attrs["sub_tree_paths"] = ["0-1", "0-2", "1-3"]
        hf.create_dataset(
            "sub_trees",
            data=[
                [21, 31, 32, 33],
                [22, 41, 42, 43],
                [23, 51, 52, 53],
                [24, 61, 62, 63],
                [25, 71, 72, 73],
            ],
            dtype="int64",
        )
        hf.create_dataset(
            "sub_trees_ar_probs",
            data=[
                [0.8, 0.5, 0.25, 0.2],
                [0.7, 0.4, 0.3, 0.1],
                [0.6, 0.3, 0.2, 0.1],
                [0.9, 0.6, 0.5, 0.4],
                [0.8, 0.5, 0.25, 0.2],
            ],
            dtype="float32",
        )


def _write_stage2_fixture_many(path: Path) -> None:
    import h5py

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
        hf.attrs["sub_tree_paths"] = ["0-1", "0-2", "1-3"]

        row = 0
        for response in responses:
            for token in response:
                sub_trees[row] = [token, token + 10, token + 11, token + 12]
                probs[row] = [0.8, 0.5, 0.25, 0.2]
                row += 1


class _TinyTokenizer:
    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(chr(64 + int(token_id)) for token_id in token_ids)

    def convert_ids_to_tokens(self, token_ids):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return [f"T{int(token_id)}" for token_id in token_ids]


def _build_fake_trainer_components(
    block_size: int,
    *,
    with_q_head: bool = False,
    with_ar_head: bool = False,
):
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
            self.ar_input_proj = nn.Linear(32, 16, bias=False) if with_ar_head else None
            self.ar_block = nn.Linear(16, 16, bias=False) if with_ar_head else None

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def extract_ctx_features(self, hidden_states):
            return torch.cat(hidden_states[1:], dim=-1)

        def compute_q_logits(self, backbone_hidden_states):
            if self.q_head is None:
                return backbone_hidden_states.new_zeros(backbone_hidden_states.shape[:2])
            return self.q_head(backbone_hidden_states).squeeze(-1)

        def build_ar_hidden_states(
            self,
            backbone_hidden_states,
            parent_embeddings,
            *,
            target_ctx_features,
            tree_info,
            position_ids,
            attention_mask=None,
        ):
            _ = target_ctx_features, tree_info, position_ids, attention_mask
            if self.ar_input_proj is None or self.ar_block is None:
                raise ValueError("AR head is not enabled")
            return self.ar_block(self.ar_input_proj(torch.cat([backbone_hidden_states, parent_embeddings], dim=-1)))

        def forward(
            self,
            *,
            hidden_states,
            position_ids,
            tree_info,
            attention_mask,
            target_ctx_features,
            return_aux=False,
            parent_embeddings=None,
            ar_position_ids=None,
            ar_attention_mask=None,
        ):
            _ = position_ids, tree_info, attention_mask
            draft_hidden = self.proj(hidden_states)
            backbone_hidden = hidden_states
            if not return_aux:
                return draft_hidden, backbone_hidden
            q_logits = self.compute_q_logits(backbone_hidden)
            if self.ar_input_proj is not None and self.ar_block is not None:
                if parent_embeddings is None or ar_position_ids is None:
                    raise ValueError("AR inputs are required when AR head is enabled")
                ar_hidden = self.build_ar_hidden_states(
                    backbone_hidden,
                    parent_embeddings,
                    target_ctx_features=target_ctx_features,
                    tree_info=tree_info,
                    position_ids=ar_position_ids,
                    attention_mask=ar_attention_mask,
                )
            else:
                ar_hidden = backbone_hidden.new_zeros(backbone_hidden.shape)
            return draft_hidden, backbone_hidden, q_logits, ar_hidden

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
    with_ar_head: bool = False,
    tree_type: str = "block",
    tree_args: dict | None = None,
    q_loss_lambda: float = 1.0,
    ar_loss_lambda: float = 0.1,
    compile: bool = False,
) -> Trainer:
    import src.trainer as trainer_mod

    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    FakeFabric, FakeTokenizer, FakeTargetModel, FakeDrafter, _ = _build_fake_trainer_components(
        tree_processor.block_size,
        with_q_head=with_q_head,
        with_ar_head=with_ar_head,
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
            ar_loss_lambda=ar_loss_lambda,
            compile=compile,
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
    _require_flex_attention()
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
    config._attn_implementation = "flex_attention"

    model = DFlashDraftModel(config)
    model.config._attn_implementation = "flex_attention"
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
    _require_flex_attention()
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
    config._attn_implementation = "flex_attention"
    config.use_q_head = True

    model = DFlashDraftModel(config)
    model.config._attn_implementation = "flex_attention"
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


def test_dflash_ar_head_reuses_shared_lm_head() -> None:
    _require_flex_attention()
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
    config._attn_implementation = "flex_attention"
    config.use_ar_head = True

    model = DFlashDraftModel(config)
    model.config._attn_implementation = "flex_attention"
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    tree_info = tree_processor.build_tree_info(batch_size=1, num_blocks=1, device=torch.device("cpu"))

    hidden_states = torch.randn(1, tree_processor.block_size, config.hidden_size)
    parent_embeddings = torch.randn_like(hidden_states)
    position_ids = torch.arange(tree_processor.block_size).unsqueeze(0)
    target_ctx_features = torch.randn(1, 6, len(model.target_layer_ids) * config.hidden_size)
    shared_lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    _, backbone_hidden = model(
        hidden_states=hidden_states,
        position_ids=position_ids,
        tree_info=tree_info,
        target_ctx_features=target_ctx_features,
        attention_mask=None,
    )
    ar_hidden = model.build_ar_hidden_states(
        backbone_hidden,
        parent_embeddings,
        target_ctx_features=target_ctx_features,
        tree_info=tree_info,
        position_ids=position_ids,
    )
    ar_logits = shared_lm_head(ar_hidden)

    assert ar_hidden.shape == hidden_states.shape
    assert ar_logits.shape == (1, tree_processor.block_size, config.vocab_size)


def test_dflash_forward_return_aux_includes_q_logits_and_ar_hidden() -> None:
    _require_flex_attention()
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
    config._attn_implementation = "flex_attention"
    config.use_q_head = True
    config.use_ar_head = True

    model = DFlashDraftModel(config)
    model.config._attn_implementation = "flex_attention"
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    tree_info = tree_processor.build_tree_info(batch_size=1, num_blocks=1, device=torch.device("cpu"))

    hidden_states = torch.randn(1, tree_processor.block_size, config.hidden_size)
    parent_embeddings = torch.randn_like(hidden_states)
    tree_position_ids = torch.arange(tree_processor.block_size).unsqueeze(0)
    position_ids = torch.cat([torch.arange(6).unsqueeze(0), tree_position_ids], dim=1)
    target_ctx_features = torch.randn(1, 6, len(model.target_layer_ids) * config.hidden_size)

    final_hidden, backbone_hidden, q_logits, ar_hidden = model(
        hidden_states=hidden_states,
        position_ids=position_ids,
        tree_info=tree_info,
        target_ctx_features=target_ctx_features,
        attention_mask=None,
        return_aux=True,
        parent_embeddings=parent_embeddings,
        ar_position_ids=tree_position_ids,
        ar_attention_mask=None,
    )

    assert final_hidden.shape == hidden_states.shape
    assert backbone_hidden.shape == hidden_states.shape
    assert q_logits.shape == hidden_states.shape[:2]
    assert ar_hidden.shape == hidden_states.shape


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


def test_trainer_compile_wraps_internal_compute_helpers(monkeypatch, tmp_path: Path) -> None:
    import src.trainer as trainer_mod

    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    _, _, _, _, packed_batch = _build_fake_trainer_components(tree_processor.block_size)
    compile_calls: list[tuple[str, dict]] = []

    def fake_compile(module, **kwargs):
        module_name = getattr(module, "__name__", type(module).__name__)
        compile_calls.append((module_name, kwargs))
        return module

    monkeypatch.setattr(trainer_mod.torch, "compile", fake_compile)

    _make_trainer(
        monkeypatch,
        tmp_path,
        anchor_chunk_size=None,
        batch=packed_batch,
        compile=True,
    )

    compiled_names = [name for name, _ in compile_calls]
    assert compiled_names.count("TrainerValidTargetCounter") == 1
    assert compiled_names.count("TrainerAcceptanceProxy") == 1
    assert compiled_names.count("TrainerPrefillTargetContext") == 1
    assert compiled_names.count("TrainerAnchorChunkForward") == 1
    assert compiled_names.count("TrainerLossAndPredictions") == 1


def test_trainer_real_loader_emits_fixed_packed_row_batches(monkeypatch, tmp_path: Path) -> None:
    import src.trainer as trainer_mod

    stage2_path = tmp_path / "stage2_many.h5"
    _write_stage2_fixture_many(stage2_path)
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    FakeFabric, FakeTokenizer, FakeTargetModel, FakeDrafter, _ = _build_fake_trainer_components(
        tree_processor.block_size
    )

    monkeypatch.setattr(trainer_mod, "Fabric", FakeFabric)
    monkeypatch.setattr(trainer_mod, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(trainer_mod, "AutoModelForCausalLM", FakeTargetModel)
    monkeypatch.setattr(trainer_mod, "DFlashDraftModel", FakeDrafter)

    trainer = Trainer(
        config=TrainerConfig(
            checkpoint_path=str(tmp_path / "ckpts"),
            no_wandb=True,
            precision="32-true",
            dev_run=True,
        ),
        target="fake-target",
        data=DataModuleConfig(
            path=str(stage2_path),
            eval_path=str(stage2_path),
            batch_size=2,
            pack_length=10,
            tree_seq_depth=2,
            num_workers=0,
            shuffle=False,
            drop_last=False,
        ),
        drafter="fake-drafter",
        tree_type="block",
    )

    batch = next(iter(trainer.train_loader))

    assert batch.batch_size == 2
    assert batch.input_ids.shape == (2, 10)


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


def test_train_batch_caches_tree_info_across_anchor_chunks(monkeypatch, tmp_path: Path) -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    _, _, _, _, packed_batch = _build_fake_trainer_components(tree_processor.block_size)
    trainer = _make_trainer(
        monkeypatch,
        tmp_path,
        anchor_chunk_size=1,
        batch=packed_batch,
    )

    call_count = 0
    original_build_tree_info = trainer.tree_processor.build_tree_info

    def counting_build_tree_info(batch_size, num_blocks, device):
        nonlocal call_count
        call_count += 1
        return original_build_tree_info(batch_size, num_blocks, device)

    trainer.tree_processor.build_tree_info = counting_build_tree_info

    trainer._train_batch(packed_batch)

    assert call_count == 1


def test_train_batch_reports_acceptance_proxy(monkeypatch, tmp_path: Path) -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    _, _, _, _, packed_batch = _build_fake_trainer_components(tree_processor.block_size)
    trainer = _make_trainer(
        monkeypatch,
        tmp_path,
        anchor_chunk_size=None,
        batch=packed_batch,
    )

    result = trainer._train_batch(packed_batch)

    assert result["acceptance_count"] > 0

    target_ctx_features = trainer._prefill_target_context(packed_batch)
    chunk_result = trainer._forward_anchor_chunk(
        packed_batch,
        target_ctx_features,
        slice(0, packed_batch.num_anchors),
        compute_predictions=True,
    )
    expected_total, expected_count = trainer._acceptance_proxy(
        predictions=chunk_result["predictions"],
        labels=packed_batch.tree_labels,
        anchor_valid_mask=packed_batch.anchor_valid_mask,
        tree_valid_mask=chunk_result["tree_valid_mask"],
    )

    assert result["acceptance_total"] == expected_total
    assert result["acceptance_count"] == expected_count


def test_train_batch_acceptance_proxy_is_chunk_invariant(monkeypatch, tmp_path: Path) -> None:
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

    full_result = trainer_full._train_batch(packed_batch)
    chunked_result = trainer_chunked._train_batch(packed_batch)

    assert full_result["acceptance_total"] == chunked_result["acceptance_total"]
    assert full_result["acceptance_count"] == chunked_result["acceptance_count"]


def test_acceptance_proxy_matches_loop_reference(monkeypatch, tmp_path: Path) -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    _, _, _, _, packed_batch = _build_fake_trainer_components(tree_processor.block_size)
    trainer = _make_trainer(
        monkeypatch,
        tmp_path,
        anchor_chunk_size=None,
        batch=packed_batch,
    )

    batch_size = 2
    num_anchors = 3
    block_size = trainer.tree_processor.block_size
    predictions = torch.randint(0, 20, (batch_size, num_anchors, block_size), dtype=torch.long)
    labels = torch.randint(0, 20, (batch_size, num_anchors, block_size), dtype=torch.long)
    anchor_valid_mask = torch.tensor(
        [[True, False, True], [True, True, False]],
        dtype=torch.bool,
    )
    tree_valid_mask = torch.ones((batch_size, num_anchors, block_size), dtype=torch.bool)

    primary_indices = trainer.tree_processor.primary_path_indices
    future_primary_indices = primary_indices[1:]
    if future_primary_indices.numel() > 0:
        target_prefix = labels.index_select(-1, primary_indices)[..., 1:]
        predictions[0, 0, future_primary_indices[0]] = target_prefix[0, 0, 0]
        predictions[1, 1, future_primary_indices] = target_prefix[1, 1]

    tree_info = trainer.tree_processor.build_tree_info(
        batch_size=batch_size,
        num_blocks=num_anchors,
        device=torch.device("cpu"),
    )
    non_primary_nodes = [
        node_idx
        for node_idx in range(block_size)
        if node_idx not in set(primary_indices.tolist()) and len(gather_path_indices(node_idx, tree_info)) > 1
    ]
    assert non_primary_nodes
    best_branch = max(non_primary_nodes, key=lambda idx: len(gather_path_indices(idx, tree_info)))
    best_branch_path = gather_path_indices(best_branch, tree_info)[1:]
    target_prefix = labels.index_select(-1, primary_indices)[..., 1:]
    for depth_idx, node_idx in enumerate(best_branch_path):
        predictions[0, 2, node_idx] = target_prefix[0, 2, depth_idx]
    if future_primary_indices.numel() > 0:
        predictions[0, 2, future_primary_indices[0]] = target_prefix[0, 2, 0] + 1

    total, count = trainer._acceptance_proxy(
        predictions=predictions,
        labels=labels,
        anchor_valid_mask=anchor_valid_mask,
        tree_valid_mask=tree_valid_mask,
    )

    expected_total = 0.0
    expected_count = 0
    primary_list = primary_indices.tolist()
    for batch_idx in range(batch_size):
        for anchor_idx in range(num_anchors):
            if not bool(anchor_valid_mask[batch_idx, anchor_idx]):
                continue
            target_main = labels[batch_idx, anchor_idx, primary_list][1:].tolist()
            accepted = 0
            for node_idx in range(block_size):
                if not bool(tree_valid_mask[batch_idx, anchor_idx, node_idx]):
                    continue
                path = gather_path_indices(node_idx, tree_info)[1:]
                if len(path) > len(target_main):
                    continue
                if all(
                    int(predictions[batch_idx, anchor_idx, tree_idx].item()) == int(target_main[depth_idx])
                    for depth_idx, tree_idx in enumerate(path)
                ):
                    accepted = max(accepted, len(path))
            expected_total += float(accepted)
            expected_count += 1

    assert total == expected_total
    assert count == expected_count


def test_acceptance_proxy_prefers_best_matching_non_primary_branch(monkeypatch, tmp_path: Path) -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    _, _, _, _, packed_batch = _build_fake_trainer_components(tree_processor.block_size)
    trainer = _make_trainer(
        monkeypatch,
        tmp_path,
        anchor_chunk_size=None,
        batch=packed_batch,
    )

    batch_size = 1
    num_anchors = 1
    block_size = trainer.tree_processor.block_size
    predictions = torch.full((batch_size, num_anchors, block_size), -999, dtype=torch.long)
    labels = torch.randint(100, 200, (batch_size, num_anchors, block_size), dtype=torch.long)
    anchor_valid_mask = torch.tensor([[True]], dtype=torch.bool)
    tree_valid_mask = torch.ones((batch_size, num_anchors, block_size), dtype=torch.bool)
    tree_info = trainer.tree_processor.build_tree_info(
        batch_size=batch_size,
        num_blocks=num_anchors,
        device=torch.device("cpu"),
    )

    primary_indices = trainer.tree_processor.primary_path_indices
    target_main = labels.index_select(-1, primary_indices)[0, 0, 1:]
    non_primary_nodes = [
        node_idx
        for node_idx in range(block_size)
        if node_idx not in set(primary_indices.tolist()) and len(gather_path_indices(node_idx, tree_info)) > 1
    ]
    assert non_primary_nodes
    branch_node = max(non_primary_nodes, key=lambda idx: len(gather_path_indices(idx, tree_info)))
    branch_path = gather_path_indices(branch_node, tree_info)[1:]
    assert len(branch_path) >= 1

    for depth_idx, node_idx in enumerate(branch_path):
        predictions[0, 0, node_idx] = target_main[depth_idx]
    if primary_indices.numel() > 1:
        predictions[0, 0, primary_indices[1]] = target_main[0] + 1

    total, count = trainer._acceptance_proxy(
        predictions=predictions,
        labels=labels,
        anchor_valid_mask=anchor_valid_mask,
        tree_valid_mask=tree_valid_mask,
    )

    assert total == float(len(branch_path))
    assert count == 1


def test_acceptance_proxy_ignores_invalid_nodes_in_best_branch(monkeypatch, tmp_path: Path) -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    _, _, _, _, packed_batch = _build_fake_trainer_components(tree_processor.block_size)
    trainer = _make_trainer(
        monkeypatch,
        tmp_path,
        anchor_chunk_size=None,
        batch=packed_batch,
    )

    batch_size = 1
    num_anchors = 1
    block_size = trainer.tree_processor.block_size
    predictions = torch.full((batch_size, num_anchors, block_size), -999, dtype=torch.long)
    labels = torch.randint(100, 200, (batch_size, num_anchors, block_size), dtype=torch.long)
    anchor_valid_mask = torch.tensor([[True]], dtype=torch.bool)
    tree_valid_mask = torch.ones((batch_size, num_anchors, block_size), dtype=torch.bool)
    tree_info = trainer.tree_processor.build_tree_info(
        batch_size=batch_size,
        num_blocks=num_anchors,
        device=torch.device("cpu"),
    )

    primary_indices = trainer.tree_processor.primary_path_indices
    target_main = labels.index_select(-1, primary_indices)[0, 0, 1:]
    non_primary_nodes = [
        node_idx
        for node_idx in range(block_size)
        if node_idx not in set(primary_indices.tolist()) and len(gather_path_indices(node_idx, tree_info)) > 2
    ]
    assert non_primary_nodes
    branch_node = max(non_primary_nodes, key=lambda idx: len(gather_path_indices(idx, tree_info)))
    branch_path = gather_path_indices(branch_node, tree_info)[1:]
    for depth_idx, node_idx in enumerate(branch_path):
        predictions[0, 0, node_idx] = target_main[depth_idx]

    tree_valid_mask[0, 0, branch_path[-1]] = False

    total, count = trainer._acceptance_proxy(
        predictions=predictions,
        labels=labels,
        anchor_valid_mask=anchor_valid_mask,
        tree_valid_mask=tree_valid_mask,
    )

    assert total == float(len(branch_path) - 1)
    assert count == 1


def test_acceptance_proxy_returns_zero_for_empty_predictions(monkeypatch, tmp_path: Path) -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    _, _, _, _, packed_batch = _build_fake_trainer_components(tree_processor.block_size)
    trainer = _make_trainer(
        monkeypatch,
        tmp_path,
        anchor_chunk_size=None,
        batch=packed_batch,
    )

    predictions = torch.empty((0, 0, 0), dtype=torch.long)
    labels = torch.empty((0, 0, 0), dtype=torch.long)
    anchor_valid_mask = torch.empty((0, 0), dtype=torch.bool)
    total, count = trainer._acceptance_proxy(
        predictions=predictions,
        labels=labels,
        anchor_valid_mask=anchor_valid_mask,
        tree_valid_mask=torch.empty((0, 0, 0), dtype=torch.bool),
    )

    assert total == 0.0
    assert count == 0


def test_acceptance_proxy_returns_zero_for_degenerate_primary_path(monkeypatch, tmp_path: Path) -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    _, _, _, _, packed_batch = _build_fake_trainer_components(tree_processor.block_size)
    trainer = _make_trainer(
        monkeypatch,
        tmp_path,
        anchor_chunk_size=None,
        batch=packed_batch,
    )

    trainer.tree_processor.primary_path_indices = torch.tensor([0], dtype=torch.long)
    predictions = torch.zeros((1, 1, trainer.tree_processor.block_size), dtype=torch.long)
    labels = torch.zeros_like(predictions)
    anchor_valid_mask = torch.tensor([[True]], dtype=torch.bool)
    total, count = trainer._acceptance_proxy(
        predictions=predictions,
        labels=labels,
        anchor_valid_mask=anchor_valid_mask,
        tree_valid_mask=torch.ones_like(predictions, dtype=torch.bool),
    )

    assert total == 0.0
    assert count == 0


def test_loss_and_predictions_match_dense_reference_with_valid_row_compaction(monkeypatch, tmp_path: Path) -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    _, _, _, _, packed_batch = _build_fake_trainer_components(tree_processor.block_size)
    trainer = _make_trainer(
        monkeypatch,
        tmp_path,
        anchor_chunk_size=None,
        batch=packed_batch,
    )

    hidden_states = torch.randn(1, 6, 16, requires_grad=True)
    labels = torch.tensor([[3, 4, 5, 6, 7, 8]], dtype=torch.long)
    weights = torch.tensor([[1.0, 0.5, 0.25, 0.75, 0.0, 1.25]], dtype=torch.float32)
    valid_mask = torch.tensor([[True, False, True, True, False, True]], dtype=torch.bool)
    prediction_mask = torch.tensor([[True, True, True, True, False, False]], dtype=torch.bool)

    loss_sum, valid_count, predictions, _ = trainer._chunked_loss_and_predictions(
        hidden_states=hidden_states,
        labels=labels,
        weights=weights,
        valid_mask=valid_mask,
        prediction_mask=prediction_mask,
        compute_predictions=True,
    )

    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_labels = labels.reshape(-1)
    flat_weights = weights.reshape(-1)
    flat_valid = valid_mask.reshape(-1)
    flat_prediction = prediction_mask.reshape(-1)
    logits = trainer.target_lm_head(flat_hidden.to(trainer.target_lm_head.weight.dtype))
    reference_loss = (
        F.cross_entropy(logits[flat_valid].float(), flat_labels[flat_valid], reduction="none")
        * flat_weights[flat_valid]
    ).sum()
    reference_predictions = torch.full_like(flat_labels, IGNORE_IDX)
    reference_predictions[flat_prediction] = logits[flat_prediction].argmax(dim=-1)

    assert valid_count == int(flat_valid.sum().item())
    assert torch.isclose(loss_sum, reference_loss)
    assert torch.equal(predictions.reshape(-1), reference_predictions)


def test_loss_uses_cce_backend_when_available(monkeypatch, tmp_path: Path) -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    _, _, _, _, packed_batch = _build_fake_trainer_components(tree_processor.block_size)
    trainer = _make_trainer(
        monkeypatch,
        tmp_path,
        anchor_chunk_size=None,
        batch=packed_batch,
    )
    calls: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    def fake_cce(hidden_states, weight, labels, *, bias=None, reduction="none", impl="torch_compile"):
        calls.append((impl, tuple(hidden_states.shape), tuple(labels.shape)))
        logits = F.linear(hidden_states, weight, bias)
        return F.cross_entropy(logits.float(), labels, reduction=reduction)

    import src.trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "cce_linear_cross_entropy", fake_cce)
    hidden_states = torch.randn(1, 5, 16)
    labels = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    weights = torch.ones((1, 5), dtype=torch.float32)
    valid_mask = torch.tensor([[True, False, True, True, False]], dtype=torch.bool)

    loss_sum, valid_count, _, _ = trainer._chunked_loss_and_predictions(
        hidden_states=hidden_states,
        labels=labels,
        weights=weights,
        valid_mask=valid_mask,
        prediction_mask=None,
        compute_predictions=False,
    )

    assert valid_count == 3
    assert calls == [("torch_compile", (3, 16), (3,))]
    assert torch.isfinite(loss_sum)


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


def test_train_batch_adds_ar_loss_when_ar_head_is_present(monkeypatch, tmp_path: Path) -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    _, _, _, _, packed_batch = _build_fake_trainer_components(tree_processor.block_size, with_ar_head=True)
    trainer = _make_trainer(
        monkeypatch,
        tmp_path,
        anchor_chunk_size=None,
        batch=packed_batch,
        with_ar_head=True,
        ar_loss_lambda=0.25,
    )

    result = trainer._train_batch(packed_batch)

    assert result["ar_loss"].item() > 0
    assert torch.isclose(result["loss"], result["ce_loss"] + 0.25 * result["ar_loss"])


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


def test_ar_loss_respects_non_root_valid_mask(monkeypatch, tmp_path: Path) -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    _, _, _, _, packed_batch = _build_fake_trainer_components(tree_processor.block_size, with_ar_head=True)
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
        with_ar_head=True,
    )

    chunk_result = trainer._forward_anchor_chunk(
        masked_batch,
        trainer._prefill_target_context(masked_batch),
        slice(0, masked_batch.num_anchors),
        compute_predictions=False,
    )

    assert chunk_result["valid_count"] == 0
    assert float(chunk_result["ar_loss_sum"].item()) == 0.0


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


def test_trainer_accepts_prunable_tree_type_with_ar_head(monkeypatch, tmp_path: Path) -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    _, _, _, _, packed_batch = _build_fake_trainer_components(tree_processor.block_size, with_ar_head=True)
    trainer = _make_trainer(
        monkeypatch,
        tmp_path,
        anchor_chunk_size=None,
        batch=packed_batch,
        with_ar_head=True,
        tree_type="prunable",
        tree_args={"base_tree_type": "block", "prune_topk": 2},
    )

    assert isinstance(trainer.tree_processor, PrunableTreeProcessor)
    assert trainer.tree_processor.prune_topk == 2


def test_trainer_prunable_requires_q_head(monkeypatch, tmp_path: Path) -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    _, _, _, _, packed_batch = _build_fake_trainer_components(tree_processor.block_size)

    with pytest.raises(ValueError, match="use_q_head=True or use_ar_head=True"):
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


def test_build_pruning_scores_prefers_ar_head_and_uses_shared_lm_head() -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    tree_info = tree_processor.build_tree_info(batch_size=1, num_blocks=1, device=torch.device("cpu"))
    _, _, fake_target_model_cls, fake_drafter_cls, _ = _build_fake_trainer_components(
        tree_processor.block_size,
        with_ar_head=True,
    )
    raw_drafter = fake_drafter_cls()
    target_model = fake_target_model_cls()
    backbone_hidden = torch.randn(1, tree_info.block_size, 16)
    tree_token_ids = torch.arange(tree_info.block_size, dtype=torch.long) % 32

    pruning_scores = build_pruning_scores(
        raw_drafter=raw_drafter,
        backbone_hidden=backbone_hidden,
        tree_token_ids=tree_token_ids,
        tree_info=tree_info,
        target_ctx_features=torch.randn(1, 6, 32),
        position_ids=torch.arange(tree_info.block_size, dtype=torch.long).unsqueeze(0),
        target_embeddings=target_model.get_input_embeddings(),
        target_lm_head=target_model.get_output_embeddings(),
        temperature=0.0,
    )

    assert pruning_scores is not None
    assert pruning_scores.shape == (tree_info.block_size,)
    assert pruning_scores[0].item() == pytest.approx(1.0)


def test_build_prefill_attention_mask_uses_flex_block_mask() -> None:
    _require_flex_attention()
    document_mask = torch.tensor([[1, 1, 1, 2, 2, 0]], dtype=torch.long)
    valid_mask = torch.tensor([[True, True, True, True, True, False]])

    mask = build_prefill_attention_mask(document_mask, valid_mask)

    assert mask is not None
    assert type(mask).__name__ == "BlockMask"


def test_build_ar_block_mask_allows_ctx_and_ancestor_self_only(monkeypatch) -> None:
    _require_flex_attention()
    flex_attention = pytest.importorskip("torch.nn.attention.flex_attention")
    captured = {}

    def fake_create_block_mask(mask_mod, **kwargs):
        captured["mask_mod"] = mask_mod
        return "mask"

    monkeypatch.setattr(flex_attention, "create_block_mask", fake_create_block_mask)
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    tree_info = tree_processor.build_tree_info(batch_size=1, num_blocks=1, device=torch.device("cpu"))

    mask = build_ar_block_mask(
        anchor_positions=torch.tensor([[4]], dtype=torch.long),
        document_mask=torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.long),
        context_valid_mask=torch.tensor([[True, True, True, True, True, False]]),
        tree_valid_mask=torch.ones((1, 1, tree_processor.block_size), dtype=torch.bool),
        tree_info=tree_info,
        block_size=tree_processor.block_size,
    )

    mask_mod = captured["mask_mod"]

    assert mask == "mask"
    assert bool(mask_mod(0, 0, torch.tensor(7), torch.tensor(1)).item())
    assert bool(mask_mod(0, 0, torch.tensor(7), torch.tensor(6 + 7)).item())
    assert bool(mask_mod(0, 0, torch.tensor(7), torch.tensor(6 + 5)).item())
    assert not bool(mask_mod(0, 0, torch.tensor(7), torch.tensor(4)).item())
    assert not bool(mask_mod(0, 0, torch.tensor(7), torch.tensor(6 + 1)).item())
    assert not bool(mask_mod(0, 0, torch.tensor(7), torch.tensor(6 + 6)).item())


def test_build_verifier_score_mod_allows_prefix_and_blocks_non_ancestors() -> None:
    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    tree_info = tree_processor.build_tree_info(batch_size=1, num_blocks=1, device=torch.device("cpu"))
    score_mod = build_verifier_score_mod(tree_info=tree_info, prefix_len=3)
    score = torch.tensor(0.0)

    prefix_score = score_mod(score, 0, 0, torch.tensor(7), torch.tensor(1))
    ancestor_tree_score = score_mod(score, 0, 0, torch.tensor(7), torch.tensor(3 + 5))
    sibling_tree_score = score_mod(score, 0, 0, torch.tensor(7), torch.tensor(3 + 1))

    assert torch.isfinite(prefix_score)
    assert torch.isfinite(ancestor_tree_score)
    assert torch.isneginf(sibling_tree_score)


def test_verify_tree_uses_score_mod_without_dense_attention_mask() -> None:
    class DummyCache:
        def __init__(self, seq_len: int):
            self.seq_len = seq_len

        def get_seq_length(self):
            return self.seq_len

    class FakeRawDrafter:
        def extract_ctx_features(self, hidden_states):
            return hidden_states[0]

    class FakeTargetModel:
        def __call__(self, **kwargs):
            self.kwargs = kwargs
            hidden = torch.zeros((1, 8, 4), dtype=torch.float32)
            logits = torch.zeros((1, 8, 16), dtype=torch.float32)
            return type(
                "Out",
                (),
                {
                    "past_key_values": kwargs["past_key_values"],
                    "logits": logits,
                    "hidden_states": (hidden,),
                },
            )()

    tree_processor = BlockTreeProcessor(tree_seq_depth=2)
    tree_info = tree_processor.build_tree_info(batch_size=1, num_blocks=1, device=torch.device("cpu"))
    target_model = FakeTargetModel()
    target_cache = DummyCache(seq_len=5)

    verify_tree(
        target_model=target_model,
        raw_drafter=FakeRawDrafter(),
        tree_token_ids=torch.arange(tree_info.block_size, dtype=torch.long),
        tree_info=tree_info,
        root_position=4,
        target_cache=target_cache,
        temperature=0.0,
    )

    assert target_model.kwargs["attention_mask"] is None
    assert callable(target_model.kwargs["score_mod"])


def test_compare_generation_to_reference_marks_accepted_rejected_extra_and_missing() -> None:
    tokenizer = _TinyTokenizer()

    comparisons, missing_tokens, metrics = compare_generation_to_reference(
        tokenizer=tokenizer,
        reference_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        generated_ids=torch.tensor([1, 9, 8, 7], dtype=torch.long),
    )

    assert [token.status for token in comparisons] == ["accepted", "rejected", "rejected", "extra"]
    assert [token.status for token in missing_tokens] == []
    assert metrics == {
        "accepted_count": 1,
        "rejected_count": 2,
        "extra_count": 1,
        "missing_count": 0,
        "exact_match": False,
    }


def test_evaluate_prompt_suite_and_render_html(monkeypatch) -> None:
    import src.spec_decode as spec_decode_mod

    tokenizer = _TinyTokenizer()
    monkeypatch.setattr(
        spec_decode_mod,
        "load_and_process_eval_dataset",
        lambda data_name: [
            {"turns": ["Prompt one"]},
            {"turns": ["Prompt two", "Follow up"]},
        ],
    )

    results = [
        type(
            "FakeResult",
            (),
            {
                "continuation_ids": torch.tensor([21, 22, 23], dtype=torch.long),
                "drafted_tokens": 6,
                "committed_tokens": 3,
                "acceptance_lengths": [2, 1],
            },
        )(),
        type(
            "FakeResult",
            (),
            {
                "continuation_ids": torch.tensor([24, 99], dtype=torch.long),
                "drafted_tokens": 5,
                "committed_tokens": 2,
                "acceptance_lengths": [1],
            },
        )(),
    ]

    def generate_fn(*, prompt: str, max_new_tokens: int):
        _ = prompt, max_new_tokens
        return results.pop(0)

    report = evaluate_prompt_suite(
        data_name="alpaca",
        tokenizer=tokenizer,
        generate_fn=generate_fn,
        max_examples=2,
        max_new_tokens=3,
    )

    assert report.total_examples == 2
    assert report.exact_matches == 0
    assert report.accepted_count == 0
    assert report.rejected_count == 0
    assert report.extra_count == 5
    assert report.missing_count == 0

    html = render_eval_suite_html(report)

    assert "Spec Decode Validation Report" in html
    assert "token extra" in html
    assert "match=mismatch" in html
    assert "accepted=0" in html
    assert "Prompt one" in html
    assert "Prompt two" in html


def test_official_dflash_aliases_infer_target_pairing() -> None:
    draft_model = resolve_official_dflash_model("qwen3-8b")

    assert draft_model == "z-lab/Qwen3-8B-DFlash-b16"
    assert infer_target_model_for_draft_model(draft_model) == "Qwen/Qwen3-8B"


def test_parsers_accept_prunable_tree_type() -> None:
    trainer_tree_action = next(action for action in build_trainer_parser()._actions if action.dest == "tree_type")
    spec_tree_action = next(action for action in build_spec_decode_parser()._actions if action.dest == "tree_type")

    assert "prunable" in trainer_tree_action.choices
    assert "prunable" in spec_tree_action.choices
    assert any(action.dest == "official_dflash_model" for action in build_spec_decode_parser()._actions)
    assert any(action.dest == "eval_data" for action in build_spec_decode_parser()._actions)
    assert all(action.dest != "trainer.target_attn_implementation" for action in build_trainer_parser()._actions)
    assert all(action.dest != "attn_implementation" for action in build_spec_decode_parser()._actions)
