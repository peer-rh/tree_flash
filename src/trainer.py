from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import math
import json
import time

from jsonargparse import ArgumentParser, namespace_to_dict
from lightning.fabric import Fabric
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import DataModuleConfig, PackedBatch, build_dataloaders
from .models import DFlashDraftModel
from .trees import BlockTreeProcessor, BranchOffTreeProcessor, PrunableTreeProcessor

try:
    from cut_cross_entropy import linear_cross_entropy as cce_linear_cross_entropy
except ImportError:
    cce_linear_cross_entropy = None


@dataclass
class TrainerConfig:
    num_epochs: int = 10
    eval_every: int = 1024
    log_every: int = 10
    save_every: int = 1024
    target_temperature: float = 1.0
    precision: str = "bf16-mixed"
    ddp: bool = False
    lr: float = 6e-4
    warmup_steps: int = 128
    grad_accum_steps: int = 1
    dev_run: bool = False
    verbose: bool = False
    checkpoint_path: str = "checkpoints"
    compile: bool = False
    devices: int = 1
    anchor_chunk_size: int | None = None
    ce_chunk_size: int | None = None
    q_loss_lambda: float = 1.0
    ar_loss_lambda: float = 0.1
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    min_lr: float = 1e-6
    seed: int = 42
    eval_batches: int = 32
    profile_steps: int = 0
    wandb_project: str = "tree-flash"
    wandb_run_name: str | None = None
    no_wandb: bool = False
    resume_from: str | None = None
    spec_eval_examples: int = 16
    spec_eval_datasets: tuple[str, ...] = ("humaneval", "gsm8k", "alpaca")
    spec_eval_max_new_tokens: int = 1024


def unwrap_model(module):
    """Return the underlying model after removing common wrapper modules.

    Args:
        module: A model or wrapped model instance.

    Returns:
        The unwrapped module. Wrapper attributes such as ``_forward_module`` and
        ``_orig_mod`` are peeled off when present.
    """
    raw = module
    if hasattr(raw, "_forward_module"):
        raw = raw._forward_module
    if hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod
    return raw


def has_pruning_head(module) -> bool:
    """Check whether a drafter exposes any pruning-specific auxiliary head.

    Args:
        module: Drafter module to inspect.

    Returns:
        ``True`` when the module exposes either ``q_head`` or ``ar_block``,
        otherwise ``False``.
    """
    return getattr(module, "q_head", None) is not None or getattr(module, "ar_block", None) is not None


def build_prefill_attention_mask(
    document_mask: torch.Tensor,
    valid_mask: torch.Tensor,
):
    """Build the causal same-document mask for target-model prefill.

    Args:
        document_mask: Tensor of shape ``(batch_size, seq_len)`` whose entries
            identify which packed-document each context token belongs to.
        valid_mask: Boolean tensor of shape ``(batch_size, seq_len)`` marking
            which context tokens are real tokens rather than padding.

    Returns:
        A FlexAttention block mask for a ``(batch_size, seq_len)`` prefill pass.
        Queries can attend only to valid keys in the same packed document and at
        causal positions, while invalid query positions are allowed to self-attend.
    """
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError as exc:
        raise RuntimeError(
            "Flex attention is required for the target prefill path. "
            "Install a PyTorch build that provides torch.nn.attention.flex_attention."
        ) from exc

    batch_size, seq_len = document_mask.shape

    def mask_mod(b, h, q_idx, kv_idx):
        """Return whether a prefill query/key pair is visible.

        Args:
            b: Batch index.
            h: Head index supplied by FlexAttention and unused here.
            q_idx: Query token index in ``[0, seq_len)``.
            kv_idx: Key/value token index in ``[0, seq_len)``.

        Returns:
            Boolean visibility decision for the given query/key pair.
        """
        q_valid = valid_mask[b, q_idx]
        invalid_self = (~q_valid) & (kv_idx == q_idx)
        attend = (
            q_valid
            & valid_mask[b, kv_idx]
            & (document_mask[b, kv_idx] == document_mask[b, q_idx])
            & (kv_idx <= q_idx)
        )
        return invalid_self | attend

    return create_block_mask(
        mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=document_mask.device,
        BLOCK_SIZE=128,
    )


def build_drafter_block_mask(
    *,
    anchor_positions: torch.Tensor,
    document_mask: torch.Tensor,
    context_valid_mask: torch.Tensor,
    tree_valid_mask: torch.Tensor,
    block_size: int,
):
    """Build the drafter attention mask over context tokens and tree blocks.

    Args:
        anchor_positions: Tensor of shape ``(batch_size, num_blocks)`` with the
            context index of each anchor token.
        document_mask: Tensor of shape ``(batch_size, ctx_len)`` containing the
            packed-document id for each context token.
        context_valid_mask: Boolean tensor of shape ``(batch_size, ctx_len)``
            marking valid context tokens.
        tree_valid_mask: Boolean tensor of shape
            ``(batch_size, num_blocks, block_size)`` marking valid tree nodes in
            each speculative block.
        block_size: Number of tree nodes per anchor block.

    Returns:
        A FlexAttention block mask whose query length is
        ``num_blocks * block_size`` and whose key/value length is
        ``ctx_len + num_blocks * block_size``. Each tree query can attend to
        earlier valid context tokens from the same document and to nodes within
        its own tree block.
    """
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError as exc:
        raise RuntimeError(
            "Flex attention is required for the drafter. "
            "Install a PyTorch build that provides torch.nn.attention.flex_attention."
        ) from exc

    batch_size, num_blocks = anchor_positions.shape
    flat_anchor_positions = anchor_positions.unsqueeze(-1).expand(-1, -1, block_size).reshape(batch_size, -1)
    flat_block_valid = tree_valid_mask.reshape(batch_size, -1)
    ctx_len = document_mask.shape[1]
    total_tree_len = flat_anchor_positions.shape[1]
    ctx_clamp_max = max(ctx_len - 1, 0)
    tree_clamp_max = max(total_tree_len - 1, 0)

    def mask_mod(b, h, q_idx, kv_idx):
        """Return whether a drafter tree query may attend to a given key.

        Args:
            b: Batch index.
            h: Head index supplied by FlexAttention and unused here.
            q_idx: Flattened tree-query index in ``[0, num_blocks * block_size)``.
            kv_idx: Context-or-tree key index in
                ``[0, ctx_len + num_blocks * block_size)``.

        Returns:
            Boolean visibility decision for the given query/key pair.
        """
        q_valid = flat_block_valid[b, q_idx]
        invalid_self = (~q_valid) & (kv_idx == (ctx_len + q_idx))

        in_ctx = kv_idx < ctx_len
        ctx_idx = kv_idx.clamp(0, ctx_clamp_max)
        q_anchor = flat_anchor_positions[b, q_idx]
        same_doc = document_mask[b, ctx_idx] == document_mask[b, q_anchor]
        causal_ctx = ctx_idx < q_anchor
        ctx_ok = in_ctx & q_valid & same_doc & causal_ctx & context_valid_mask[b, ctx_idx]

        tree_idx = (kv_idx - ctx_len).clamp(0, tree_clamp_max)
        q_block = q_idx // block_size
        kv_block = tree_idx // block_size
        same_tree = q_block == kv_block
        tree_ok = (~in_ctx) & q_valid & same_tree & flat_block_valid[b, tree_idx]
        return invalid_self | ctx_ok | tree_ok

    return create_block_mask(
        mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=total_tree_len,
        KV_LEN=ctx_len + total_tree_len,
        device=document_mask.device,
        BLOCK_SIZE=128,
    )


def build_ar_block_mask(
    *,
    anchor_positions: torch.Tensor,
    document_mask: torch.Tensor,
    context_valid_mask: torch.Tensor,
    tree_valid_mask: torch.Tensor,
    tree_info,
    block_size: int,
):
    """Build the autoregressive auxiliary-head attention mask for tree blocks.

    Args:
        anchor_positions: Tensor of shape ``(batch_size, num_blocks)`` with the
            context index of each anchor token.
        document_mask: Tensor of shape ``(batch_size, ctx_len)`` containing the
            packed-document id for each context token.
        context_valid_mask: Boolean tensor of shape ``(batch_size, ctx_len)``
            marking valid context tokens.
        tree_valid_mask: Boolean tensor of shape
            ``(batch_size, num_blocks, block_size)`` marking valid tree nodes.
        tree_info: Tree metadata whose ``tree_mask`` field has shape
            ``(block_size, block_size)`` and encodes allowable ancestor links
            inside one block.
        block_size: Number of tree nodes per anchor block.

    Returns:
        A FlexAttention block mask with query length ``num_blocks * block_size``
        and key/value length ``ctx_len + num_blocks * block_size``. Relative to
        :func:`build_drafter_block_mask`, tree-to-tree visibility is further
        restricted by ``tree_info.tree_mask``.
    """
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError as exc:
        raise RuntimeError(
            "Flex attention is required for the AR head. "
            "Install a PyTorch build that provides torch.nn.attention.flex_attention."
        ) from exc

    batch_size, num_blocks = anchor_positions.shape
    flat_anchor_positions = anchor_positions.unsqueeze(-1).expand(-1, -1, block_size).reshape(batch_size, -1)
    flat_block_valid = tree_valid_mask.reshape(batch_size, -1)
    ctx_len = document_mask.shape[1]
    total_tree_len = flat_anchor_positions.shape[1]
    ctx_clamp_max = max(ctx_len - 1, 0)
    tree_clamp_max = max(total_tree_len - 1, 0)
    tree_mask = tree_info.tree_mask

    def mask_mod(b, h, q_idx, kv_idx):
        """Return whether an AR-head tree query may attend to a given key.

        Args:
            b: Batch index.
            h: Head index supplied by FlexAttention and unused here.
            q_idx: Flattened tree-query index in ``[0, num_blocks * block_size)``.
            kv_idx: Context-or-tree key index in
                ``[0, ctx_len + num_blocks * block_size)``.

        Returns:
            Boolean visibility decision for the given query/key pair.
        """
        q_valid = flat_block_valid[b, q_idx]
        invalid_self = (~q_valid) & (kv_idx == (ctx_len + q_idx))

        in_ctx = kv_idx < ctx_len
        ctx_idx = kv_idx.clamp(0, ctx_clamp_max)
        q_anchor = flat_anchor_positions[b, q_idx]
        same_doc = document_mask[b, ctx_idx] == document_mask[b, q_anchor]
        causal_ctx = ctx_idx < q_anchor
        ctx_ok = in_ctx & q_valid & same_doc & causal_ctx & context_valid_mask[b, ctx_idx]

        tree_idx = (kv_idx - ctx_len).clamp(0, tree_clamp_max)
        q_block = q_idx // block_size
        kv_block = tree_idx // block_size
        same_tree = q_block == kv_block
        q_tree_idx = q_idx % block_size
        kv_tree_idx = tree_idx % block_size
        tree_ok = (
            (~in_ctx)
            & q_valid
            & same_tree
            & flat_block_valid[b, tree_idx]
            & tree_mask[q_tree_idx, kv_tree_idx]
        )
        return invalid_self | ctx_ok | tree_ok

    return create_block_mask(
        mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=total_tree_len,
        KV_LEN=ctx_len + total_tree_len,
        device=document_mask.device,
        BLOCK_SIZE=128,
    )


def get_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    lr: float,
    min_lr: float,
) -> float:
    """Compute the learning rate for a warmup-plus-cosine schedule.

    Args:
        step: Current optimizer step.
        warmup_steps: Number of linear-warmup steps.
        total_steps: Total number of optimizer steps in the run.
        lr: Peak learning rate reached after warmup.
        min_lr: Final floor learning rate used at the end of cosine decay.

    Returns:
        The scalar learning rate for ``step``.
    """
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (lr - min_lr) * cosine


class TrainerLossAndPredictions:
    def __init__(self, target_lm_head) -> None:
        """Store the frozen target LM head used for CE loss and argmax decoding.

        Args:
            target_lm_head: Projection module mapping hidden states of shape
                ``(..., hidden_size)`` to vocabulary logits of shape
                ``(..., vocab_size)``.
        """
        self.target_lm_head = target_lm_head

    def _compute_linear_cross_entropy(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute unreduced per-token CE loss against the target LM head.

        Args:
            hidden_states: Tensor of shape ``(num_valid_tokens, hidden_size)``.
            labels: Tensor of shape ``(num_valid_tokens,)`` with vocabulary
                indices for the next-token target.

        Returns:
            Tensor of shape ``(num_valid_tokens,)`` containing one CE loss value
            per valid token.
        """
        weight = self.target_lm_head.weight
        bias = getattr(self.target_lm_head, "bias", None)
        hidden_states = hidden_states.to(weight.dtype)
        if cce_linear_cross_entropy is not None:
            preferred_impl = "cce" if hidden_states.is_cuda else "torch_compile"
            try:
                return cce_linear_cross_entropy(
                    hidden_states,
                    weight,
                    labels,
                    bias=bias,
                    reduction="none",
                    impl=preferred_impl,
                )
            except Exception:
                if preferred_impl != "torch_compile":
                    try:
                        return cce_linear_cross_entropy(
                            hidden_states,
                            weight,
                            labels,
                            bias=bias,
                            reduction="none",
                            impl="torch_compile",
                        )
                    except Exception:
                        pass

        logits = self.target_lm_head(hidden_states)
        return F.cross_entropy(
            logits.float(),
            labels,
            reduction="none",
        )

    def __call__(
        self,
        *,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
        valid_mask: torch.Tensor,
        compute_predictions: bool,
        prediction_chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aggregate weighted CE loss and optional token predictions.

        Args:
            hidden_states: Tensor of shape ``(batch_size, total_tokens, hidden_size)``.
            labels: Tensor of shape ``(batch_size, total_tokens)`` with token ids.
            weights: Tensor of shape ``(batch_size, total_tokens)`` with per-token
                loss weights such as cumulative tree probabilities.
            valid_mask: Boolean tensor of shape ``(batch_size, total_tokens)``
                selecting the tokens that should contribute to loss.
            compute_predictions: Whether to also materialize argmax predictions.
            prediction_chunk_size: Number of flattened tokens processed per LM-head
                chunk when generating predictions.

        Returns:
            A tuple ``(total_loss, valid_count, predictions)`` where
            ``total_loss`` is a scalar weighted sum, ``valid_count`` is a scalar
            tensor counting valid tokens, and ``predictions`` has shape
            ``(batch_size, total_tokens)`` when requested or is an empty tensor
            otherwise.
        """
        batch_size, total_tokens, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.reshape(batch_size * total_tokens, hidden_size)
        flat_labels = labels.reshape(batch_size * total_tokens)
        flat_weights = weights.reshape(batch_size * total_tokens)
        flat_valid = valid_mask.reshape(batch_size * total_tokens)

        valid_hidden = flat_hidden[flat_valid]
        valid_labels = flat_labels[flat_valid]
        valid_weights = flat_weights[flat_valid]
        per_token_loss = self._compute_linear_cross_entropy(valid_hidden, valid_labels)
        total_loss = (per_token_loss * valid_weights).sum()
        valid_count = flat_valid.sum()

        predictions = torch.empty(0, dtype=torch.long, device=flat_hidden.device)
        if compute_predictions:
            predictions = torch.empty(
                (batch_size * total_tokens,),
                dtype=torch.long,
                device=flat_hidden.device,
            )
            for start in range(0, flat_hidden.shape[0], prediction_chunk_size):
                end = min(start + prediction_chunk_size, flat_hidden.shape[0])
                prediction_logits = self.target_lm_head(
                    flat_hidden[start:end].to(self.target_lm_head.weight.dtype)
                )
                predictions[start:end] = prediction_logits.argmax(dim=-1)
            predictions = predictions.view(batch_size, total_tokens)

        return total_loss, valid_count, predictions


class TrainerAcceptanceProxy:
    def __call__(
        self,
        *,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        anchor_valid_mask: torch.Tensor,
        tree_valid_mask: torch.Tensor,
        primary_indices: torch.Tensor,
        ancestor_mask: torch.Tensor,
        node_depths: torch.Tensor,
        node_target_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Approximate accepted speculative depth from drafter predictions.

        Args:
            predictions: Tensor of shape ``(batch_size, num_blocks, block_size)``
                containing drafter argmax predictions for each tree node.
            labels: Tensor of shape ``(batch_size, num_blocks, block_size)`` with
                tree target token ids.
            anchor_valid_mask: Boolean tensor of shape ``(batch_size, num_blocks)``
                marking anchors that correspond to real context positions.
            tree_valid_mask: Boolean tensor of shape
                ``(batch_size, num_blocks, block_size)`` marking valid nodes.
            primary_indices: Tensor of shape ``(primary_path_len,)`` indexing the
                tree nodes that lie on the main path.
            ancestor_mask: Boolean tensor of shape ``(block_size, block_size)``
                where ``ancestor_mask[i, j]`` is true when ``j`` is an ancestor of
                ``i`` within the tree layout.
            node_depths: Tensor of shape ``(block_size,)`` with each node depth.
            node_target_indices: Tensor of shape ``(block_size,)`` mapping each
                node to the comparable depth index on the primary path, or ``-1``
                when no comparison target exists.

        Returns:
            A tuple ``(accepted_depth_sum, valid_anchor_count)`` of scalar tensors.
            The first sums the deepest accepted comparable node over all valid
            anchors; the second counts valid anchors considered in the proxy.
        """
        comparable_mask = node_target_indices >= 0
        target_main_path = labels.index_select(-1, primary_indices)
        target_tree_sd = target_main_path.gather(
            -1,
            node_target_indices.clamp(min=0).view(1, 1, -1).expand(
                predictions.shape[0],
                predictions.shape[1],
                -1,
            ),
        )
        local_matches = predictions.eq(target_tree_sd)
        local_matches = local_matches & tree_valid_mask
        local_matches = local_matches & comparable_mask.view(1, 1, -1)

        accepted_nodes = (
            local_matches.unsqueeze(-2)
            | ~ancestor_mask.view(1, 1, ancestor_mask.shape[0], ancestor_mask.shape[1])
        ).all(dim=-1)
        accepted_nodes = accepted_nodes & comparable_mask.view(1, 1, -1)

        accepted_depth = (
            accepted_nodes.to(node_depths.dtype) * node_depths.view(1, 1, -1)
        ).amax(dim=-1)
        valid_mask = anchor_valid_mask.to(torch.bool)
        accepted_valid = accepted_depth.masked_select(valid_mask)
        return accepted_valid.sum(), valid_mask.sum()


class Trainer:
    tree_processor: Any

    def __init__(
        self,
        config: TrainerConfig,
        target: str,
        data: DataModuleConfig,
        drafter: dict[str, Any] | str,
        tree_type: Literal["fixed", "prunable", "block", "branch_off", "every_branch", "loaded"] = "fixed",
        tree_args: dict[str, Any] | None = None,
    ):
        """Initialize the trainer, models, loaders, caches, and optimizer.

        Args:
            config: Training hyperparameters and runtime configuration.
            target: Hugging Face identifier for the frozen target LM.
            data: Data-module configuration used to build packed train/eval
                loaders. Packed batches expose context tensors of shape
                ``(batch_size, ctx_len)`` and tree tensors of shape
                ``(batch_size, num_anchors, block_size)``.
            drafter: Either a Hugging Face checkpoint id/path for the drafter or
                a config dictionary used to instantiate it from scratch.
            tree_type: Tree layout implementation used to interpret packed tree
                tensors.
            tree_args: Optional tree-processor arguments such as custom subtree
                layouts or pruning parameters.
        """
        self.config = config
        self.target = target
        self.data_config = data
        self.drafter_source = drafter
        self.tree_type = tree_type
        self.tree_args = tree_args or {}
        self.global_step = 0
        self.wandb_run = None
        self.wandb_run_id: str | None = None
        self._tree_info_cache: dict[tuple[int, int, str, int], Any] = {}
        self._spec_eval_dataset_cache: dict[str, Any] = {}
        self._acceptance_path_cache: dict[
            tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], str, int],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self._build_prefill_attention_mask = build_prefill_attention_mask
        self._build_drafter_block_mask = build_drafter_block_mask
        self._build_ar_block_mask = build_ar_block_mask
        self._loss_and_predictions_impl = None
        self._acceptance_proxy_impl = TrainerAcceptanceProxy()

        strategy = "ddp" if config.devices > 1 or config.ddp else "auto"
        self.fabric = Fabric(
            accelerator="auto",
            devices=config.devices,
            strategy=strategy,
            precision=config.precision,
        )
        self.fabric.launch()
        self.fabric.seed_everything(config.seed)

        self.tree_processor = self._build_tree_processor(
            tree_type=tree_type,
            tree_seq_depth=data.tree_seq_depth,
            tree_args=self.tree_args,
        )

        self.output_dir = Path(config.checkpoint_path)
        if self.fabric.is_global_zero:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        dtype_map = {
            "bf16-mixed": torch.bfloat16,
            "16-mixed": torch.float16,
            "32-true": torch.float32,
        }
        load_dtype = dtype_map.get(config.precision, torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(target)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        self.pad_token_id = int(self.tokenizer.pad_token_id)

        self.target_model = AutoModelForCausalLM.from_pretrained(
            target,
            torch_dtype=load_dtype,
            attn_implementation="flex_attention",
        )
        if getattr(self.target_model.config, "pad_token_id", None) is None:
            self.target_model.config.pad_token_id = self.pad_token_id
        self.target_model.eval()
        for param in self.target_model.parameters():
            param.requires_grad_(False)

        if isinstance(drafter, str):
            self.drafter_model = DFlashDraftModel.from_pretrained(
                drafter,
                torch_dtype=load_dtype,
            )
        else:
            self.drafter_model = DFlashDraftModel(drafter)
        self.drafter_model.train()
        self.raw_drafter = unwrap_model(self.drafter_model)

        self.mask_token_id = self.raw_drafter.mask_token_id
        if self.mask_token_id is None:
            raise ValueError("The drafter config must define dflash_config.mask_token_id.")
        self.has_q_head = getattr(self.raw_drafter, "q_head", None) is not None
        self.has_ar_head = getattr(self.raw_drafter, "ar_block", None) is not None
        if tree_type == "prunable" and not has_pruning_head(self.raw_drafter):
            raise ValueError(
                "tree_type='prunable' requires a drafter checkpoint/config with use_q_head=True or use_ar_head=True."
            )

        if config.compile and hasattr(torch, "compile"):
            self.target_model = torch.compile(self.target_model)
            self.drafter_model = torch.compile(self.drafter_model, dynamic=True)
            self._build_prefill_attention_mask = torch.compile(self._build_prefill_attention_mask)
            self._build_drafter_block_mask = torch.compile(self._build_drafter_block_mask)
            self._build_ar_block_mask = torch.compile(self._build_ar_block_mask)

        self.optimizer = torch.optim.AdamW(
            self.drafter_model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        self.target_model = self.fabric.to_device(self.target_model)
        self.drafter_model, self.optimizer = self.fabric.setup(self.drafter_model, self.optimizer)
        self.raw_drafter = unwrap_model(self.drafter_model)
        self.target_embeddings = self.target_model.get_input_embeddings()
        self.target_lm_head = self.target_model.get_output_embeddings()
        if self.target_lm_head is None:
            raise ValueError("Target model must expose an output embedding / LM head.")
        self._loss_and_predictions_impl = TrainerLossAndPredictions(self.target_lm_head)
        if config.compile and hasattr(torch, "compile"):
            self._loss_and_predictions_impl = torch.compile(self._loss_and_predictions_impl, dynamic=True)
            self._acceptance_proxy_impl = torch.compile(self._acceptance_proxy_impl, dynamic=True)

        self.train_loader, self.eval_loader = build_dataloaders(
            config=data,
            tree_processor=self.tree_processor,
            mask_token_id=self.mask_token_id,
            pad_token_id=self.pad_token_id,
        )
        self.train_loader, self.eval_loader = self.fabric.setup_dataloaders(
            self.train_loader,
            self.eval_loader,
            move_to_device=False,
        )

        if config.resume_from:
            self.load_checkpoint(config.resume_from)
        if self.fabric.is_global_zero:
            self._init_wandb()

    def _build_tree_processor(
        self,
        *,
        tree_type: str,
        tree_seq_depth: int,
        tree_args: dict[str, Any],
    ):
        """Construct the tree processor that defines block layout semantics.

        Args:
            tree_type: Tree processor kind, such as ``"block"``,
                ``"branch_off"``, or ``"prunable"``.
            tree_seq_depth: Number of sequential drafting steps represented by
                each anchor block.
            tree_args: Processor-specific keyword arguments.

        Returns:
            A tree processor instance exposing layout tensors such as
            ``parent_idx``, ``depth``, and ``primary_path_indices``.
        """
        if tree_type in {"fixed", "block"}:
            return BlockTreeProcessor(
                tree_seq_depth=tree_seq_depth,
                sub_tree_paths=tree_args.get("sub_tree_paths"),
            )
        if tree_type == "branch_off":
            return BranchOffTreeProcessor(
                tree_seq_depth=tree_seq_depth,
                sub_tree_paths=tree_args.get("sub_tree_paths"),
                branching_pattern=tree_args.get("branching_pattern"),
            )
        if tree_type == "prunable":
            legacy_prune_topk = tree_args.get("prune_topk")
            candidate_tree_size = tree_args.get("candidate_tree_size")
            if candidate_tree_size is None:
                candidate_tree_size = 1 if legacy_prune_topk in {None, 0} else int(legacy_prune_topk)
            return PrunableTreeProcessor(
                tree_seq_depth=tree_seq_depth,
                base_tree_type=tree_args.get("base_tree_type", "block"),
                candidate_tree_size=int(candidate_tree_size),
                sub_tree_paths=tree_args.get("sub_tree_paths"),
                branching_pattern=tree_args.get("branching_pattern"),
            )
        raise NotImplementedError(
            f"tree_type={tree_type!r} is not implemented. Use 'block', 'branch_off', or 'prunable'."
        )

    def _init_wandb(self) -> None:
        """Initialize the W&B run on rank zero unless logging is disabled."""
        if self.config.no_wandb:
            return
        import wandb

        self.wandb_run = wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            resume="allow",
            id=self.wandb_run_id,
            config={
                "trainer": self.config.__dict__,
                "data": self.data_config.__dict__,
                "target": self.target,
                "tree_type": self.tree_type,
            },
        )
        self.wandb_run_id = self.wandb_run.id

    def _log(self, metrics: dict[str, Any], step: int) -> None:
        """Log a metric dictionary to W&B on rank zero.

        Args:
            metrics: Flat mapping of metric names to scalar values or W&B media.
            step: Global optimizer step attached to the log event.
        """
        if self.fabric.is_global_zero and self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)

    def _all_reduce_tensor(
        self,
        tensor: torch.Tensor,
        *,
        reduce_op: str = "sum",
    ) -> torch.Tensor:
        """All-reduce a tensor across ranks when distributed training is active.

        Args:
            tensor: Tensor of any shape to reduce across workers.
            reduce_op: Reduction operation passed through to Lightning Fabric.

        Returns:
            The reduced tensor, or the input tensor unchanged in single-process
            execution.
        """
        all_reduce = getattr(self.fabric, "all_reduce", None)
        if all_reduce is None or getattr(self.fabric, "world_size", 1) <= 1:
            return tensor
        return all_reduce(tensor, reduce_op=reduce_op)

    def _barrier(self, name: str | None = None) -> None:
        """Synchronize all ranks when a Fabric barrier implementation is available.

        Args:
            name: Optional barrier name for backends that support named barriers.
        """
        barrier = getattr(self.fabric, "barrier", None)
        if barrier is None:
            return
        try:
            barrier(name=name)
        except TypeError:
            barrier()

    def _rank_zero_first_context(self):
        """Return a context manager that lets rank zero perform work first.

        Returns:
            A Fabric ``rank_zero_first`` context when available, otherwise a
            ``nullcontext``.
        """
        rank_zero_first = getattr(self.fabric, "rank_zero_first", None)
        if rank_zero_first is None:
            return nullcontext()
        return rank_zero_first(local=True)

    def _spec_eval_acceptance_length_max(self) -> int:
        """Return the maximum acceptance length bucket used in speculative eval.

        Returns:
            Maximum node depth in the active tree layout, or
            ``data_config.tree_seq_depth`` when no explicit depth tensor exists.
        """
        depth = getattr(self.tree_processor, "depth", None)
        if depth is None or depth.numel() == 0:
            return max(self.data_config.tree_seq_depth, 0)
        return int(depth.max().item())

    def _build_wandb_histogram(self, counts: list[int]):
        """Convert histogram counts into a W&B histogram object.

        Args:
            counts: ``counts[i]`` is the number of examples with acceptance length
                ``i``.

        Returns:
            A ``wandb.Histogram`` on rank zero when logging is enabled, otherwise
            ``None``.
        """
        if self.wandb_run is None or not self.fabric.is_global_zero:
            return None
        total = sum(int(count) for count in counts)
        if total <= 0:
            return None
        import wandb

        values: list[int] = []
        for accept_length, count in enumerate(counts):
            values.extend([accept_length] * int(count))
        return wandb.Histogram(values)

    def _build_wandb_html(self, html: str):
        """Wrap a raw HTML string in a W&B HTML artifact on rank zero.

        Args:
            html: HTML payload to display in W&B.

        Returns:
            ``wandb.Html`` when W&B logging is enabled on rank zero, otherwise
            ``None``.
        """
        if self.wandb_run is None or not self.fabric.is_global_zero:
            return None
        import wandb

        return wandb.Html(html, inject=False)

    def _load_spec_eval_dataset(self, data_name: str):
        """Load and cache a processed speculative-evaluation dataset.

        Args:
            data_name: Dataset identifier understood by
                ``load_and_process_eval_dataset``.

        Returns:
            The processed dataset object cached under ``data_name``.
        """
        if data_name not in self._spec_eval_dataset_cache:
            from .data import load_and_process_eval_dataset

            with self._rank_zero_first_context():
                self._spec_eval_dataset_cache[data_name] = load_and_process_eval_dataset(data_name)
        return self._spec_eval_dataset_cache[data_name]

    def _get_spec_eval_sample_indices(self, dataset_size: int) -> list[int]:
        """Shard speculative-eval example indices across distributed ranks.

        Args:
            dataset_size: Total number of examples available in the dataset.

        Returns:
            A rank-specific list of example indices drawn from
            ``range(spec_eval_examples)``.
        """
        sample_count = min(max(self.config.spec_eval_examples, 0), dataset_size)
        rank = int(getattr(self.fabric, "global_rank", 0))
        world_size = max(int(getattr(self.fabric, "world_size", 1)), 1)
        return list(range(sample_count))[rank::world_size]

    @torch.inference_mode()
    def _build_speculative_eval_html_report(
        self,
        *,
        data_name: str,
        generate_fn,
    ):
        """Render a prompt-suite speculative-decoding report as W&B HTML.

        Args:
            data_name: Evaluation dataset name.
            generate_fn: Callable with signature
                ``generate_fn(prompt: str, max_new_tokens: int)`` used to run
                speculative generation.

        Returns:
            ``wandb.Html`` containing the rendered report on rank zero when W&B is
            enabled, otherwise ``None``.
        """
        if self.wandb_run is None or not self.fabric.is_global_zero:
            return None

        from .spec_decode import evaluate_prompt_suite, render_eval_suite_html

        report = evaluate_prompt_suite(
            data_name=data_name,
            tokenizer=self.tokenizer,
            generate_fn=generate_fn,
            max_examples=self.config.spec_eval_examples,
            max_new_tokens=self.config.spec_eval_max_new_tokens,
        )
        return self._build_wandb_html(render_eval_suite_html(report))

    @torch.inference_mode()
    def _run_speculative_eval_suite(self) -> dict[str, Any]:
        """Run the configured speculative-decoding eval suite and aggregate metrics.

        Returns:
            A flat metric dictionary keyed by ``spec_eval/<dataset>/...``. Metrics
            include throughput, average acceptance length, histogram counts, and
            optional W&B media objects.
        """
        if not self.config.spec_eval_datasets or self.config.spec_eval_examples <= 0:
            return {}

        from .spec_decode import aggregate_speculative_eval, speculative_generate

        metrics: dict[str, Any] = {}
        acceptance_length_max = self._spec_eval_acceptance_length_max()

        for data_name in self.config.spec_eval_datasets:
            dataset = self._load_spec_eval_dataset(data_name)
            sample_indices = self._get_spec_eval_sample_indices(len(dataset))

            def generate_fn(*, prompt: str, max_new_tokens: int):
                """Run deterministic speculative decoding for one prompt.

                Args:
                    prompt: Prompt string to decode from.
                    max_new_tokens: Maximum number of tokens to generate.

                Returns:
                    The result object returned by ``speculative_generate`` for the
                    configured target model, drafter, tokenizer, and tree layout.
                """
                return speculative_generate(
                    target_model=self.target_model,
                    drafter_model=self.drafter_model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    tree_processor=self.tree_processor,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                )

            self._barrier(name=f"spec_eval_{data_name}_start")
            local_result = aggregate_speculative_eval(
                dataset=dataset,
                sample_indices=sample_indices,
                generate_fn=generate_fn,
                max_new_tokens=self.config.spec_eval_max_new_tokens,
                acceptance_length_max=acceptance_length_max,
            )
            self._barrier(name=f"spec_eval_{data_name}_end")

            local_counts = torch.tensor(
                [
                    float(local_result.total_examples),
                    float(local_result.total_committed_tokens),
                    float(local_result.total_steps),
                    float(local_result.total_acceptance_length),
                    float(local_result.off_main_path_last_accept_count),
                ],
                dtype=torch.float64,
                device=self.fabric.device,
            )
            global_counts = self._all_reduce_tensor(local_counts, reduce_op="sum")
            local_hist = torch.tensor(
                local_result.acceptance_length_histogram,
                dtype=torch.float64,
                device=self.fabric.device,
            )
            global_hist = self._all_reduce_tensor(local_hist, reduce_op="sum")
            local_elapsed = torch.tensor(
                float(local_result.elapsed_time_sec),
                dtype=torch.float64,
                device=self.fabric.device,
            )
            global_elapsed = self._all_reduce_tensor(local_elapsed, reduce_op="max")

            total_examples = int(round(float(global_counts[0].item())))
            total_committed_tokens = float(global_counts[1].item())
            total_steps = int(round(float(global_counts[2].item())))
            total_acceptance_length = float(global_counts[3].item())
            off_main_path_last_accept_count = float(global_counts[4].item())
            elapsed_time = max(float(global_elapsed.item()), 1e-8)
            hist_counts = [int(round(value)) for value in global_hist.detach().cpu().tolist()]

            prefix = f"spec_eval/{data_name}"
            metrics[f"{prefix}/examples"] = total_examples
            metrics[f"{prefix}/throughput_tok_s"] = total_committed_tokens / elapsed_time
            metrics[f"{prefix}/avg_acceptance_length"] = total_acceptance_length / max(total_steps, 1)
            metrics[f"{prefix}/off_main_path_last_accept_pct"] = (
                100.0 * off_main_path_last_accept_count / max(total_steps, 1)
            )
            for accept_length, count in enumerate(hist_counts):
                metrics[f"{prefix}/accept_len_count_{accept_length}"] = count
            histogram = self._build_wandb_histogram(hist_counts)
            if histogram is not None:
                metrics[f"{prefix}/accept_len_histogram"] = histogram
            html_report = self._build_speculative_eval_html_report(
                data_name=data_name,
                generate_fn=generate_fn,
            )
            if html_report is not None:
                metrics[f"{prefix}/accepted_extra_tokens_html"] = html_report

        return metrics

    def _get_tree_info(
        self,
        *,
        batch_size: int,
        num_blocks: int,
        device: torch.device,
    ):
        """Fetch cached per-layout tree metadata for a batch shape and device.

        Args:
            batch_size: Batch dimension for the returned tree tensors.
            num_blocks: Number of anchor blocks per sequence.
            device: Device on which tree tensors should live.

        Returns:
            A tree-info object whose broadcasted tensors typically have shapes
            ``relation_map = (batch_size, num_blocks, block_size, block_size)``
            and ``tree_position_ids = (batch_size, num_blocks, block_size)``.
        """
        device_key = (device.type, -1 if device.index is None else device.index)
        cache_key = (batch_size, num_blocks, device_key[0], device_key[1])
        tree_info = self._tree_info_cache.get(cache_key)
        if tree_info is None:
            tree_info = self.tree_processor.build_tree_info(batch_size, num_blocks, device)
            self._tree_info_cache[cache_key] = tree_info
        return tree_info

    def _forward_anchor_chunk(
        self,
        batch: PackedBatch,
        target_ctx_features: torch.Tensor,
        anchor_slice: slice,
        *,
        compute_predictions: bool,
        profile: bool = False,
    ) -> dict[str, Any]:
        """Run the drafter and auxiliary heads on a subset of anchor blocks.

        Args:
            batch: Packed batch whose context tensors have shape
                ``(batch_size, ctx_len)`` and tree tensors have shape
                ``(batch_size, num_anchors, block_size)``.
            target_ctx_features: Tensor of shape ``(batch_size, ctx_len, hidden_size)``
                extracted from the frozen target model.
            anchor_slice: Slice selecting the anchor-block interval processed in
                this call.
            compute_predictions: Whether to decode argmax token predictions for
                acceptance-proxy computation.
            profile: Whether to measure mask, drafter, and CE timings.

        Returns:
            A dictionary containing scalar loss sums, valid-token count, optional
            predictions of shape ``(batch_size, chunk_blocks, block_size)``,
            ``tree_valid_mask`` for the chunk, and optional timing values.
        """
        anchor_positions = batch.anchor_positions[:, anchor_slice]
        anchor_valid_mask = batch.anchor_valid_mask[:, anchor_slice]
        if not anchor_valid_mask.any():
            return {
                "loss_sum": target_ctx_features.new_zeros(()),
                "q_loss_sum": target_ctx_features.new_zeros(()),
                "ar_loss_sum": target_ctx_features.new_zeros(()),
                "valid_count": 0,
                "predictions": None,
                "mask_time": 0.0,
                "drafter_time": 0.0,
                "ce_time": 0.0,
            }

        tree_labels = batch.tree_labels[:, anchor_slice]
        tree_noise_ids = batch.tree_noise_ids[:, anchor_slice]
        tree_position_ids = batch.tree_position_ids[:, anchor_slice]
        tree_cum_probs = batch.tree_cum_probs[:, anchor_slice]
        tree_valid_mask = batch.tree_valid_mask[:, anchor_slice]

        batch_size, num_blocks, block_size = tree_labels.shape
        tree_info = self._get_tree_info(batch_size=batch_size, num_blocks=num_blocks, device=tree_labels.device)
        valid_mask = (
            tree_valid_mask
            & tree_info.non_root_mask.view(1, 1, -1)
            & anchor_valid_mask.unsqueeze(-1)
        )
        flat_valid_mask = valid_mask.reshape(batch_size, num_blocks * block_size)
        if not flat_valid_mask.any():
            return {
                "loss_sum": target_ctx_features.new_zeros(()),
                "q_loss_sum": target_ctx_features.new_zeros(()),
                "ar_loss_sum": target_ctx_features.new_zeros(()),
                "valid_count": 0,
                "predictions": None,
                "tree_valid_mask": tree_valid_mask,
                "mask_time": 0.0,
                "drafter_time": 0.0,
                "ce_time": 0.0,
            }

        noise_embeddings = self.target_embeddings(tree_noise_ids.reshape(batch_size, num_blocks * block_size))
        mask_start = time.perf_counter() if profile else 0.0
        drafter_mask = self._build_drafter_block_mask(
            anchor_positions=anchor_positions,
            document_mask=batch.document_mask,
            context_valid_mask=batch.context_valid_mask,
            tree_valid_mask=tree_valid_mask,
            block_size=block_size,
        )
        # if self.config.verbose:
        #     print("Drafter Mask:", drafter_mask[0,0])
        #     print("Anchor Positions:", anchor_positions[0,0])
        #     print("Tree Position Ids:", tree_position_ids[0,0])
        #     print("Tree Labels:", tree_labels[0,0])


        position_ids_with_seq = torch.cat((
            batch.position_ids,
            tree_position_ids.reshape(batch_size, num_blocks * block_size),
        ), dim=1)
        parent_embeddings = None
        ar_attention_mask = None
        ar_position_ids = None
        if self.has_ar_head:
            parent_token_ids = self._build_parent_token_ids(tree_labels, tree_info)
            parent_embeddings = self.target_embeddings(parent_token_ids.reshape(batch_size, num_blocks * block_size))
            ar_position_ids = tree_position_ids.reshape(batch_size, num_blocks * block_size)
            ar_attention_mask = self._build_ar_block_mask(
                anchor_positions=anchor_positions,
                document_mask=batch.document_mask,
                context_valid_mask=batch.context_valid_mask,
                tree_valid_mask=tree_valid_mask,
                tree_info=tree_info,
                block_size=block_size,
            )
        mask_time = time.perf_counter() - mask_start if profile else 0.0

        drafter_start = time.perf_counter() if profile else 0.0
        draft_hidden_states, _, q_logits, ar_hidden_states = self.drafter_model(
            hidden_states=noise_embeddings,
            position_ids=position_ids_with_seq,
            tree_info=tree_info,
            attention_mask=drafter_mask,
            target_ctx_features=target_ctx_features,
            return_aux=True,
            parent_embeddings=parent_embeddings,
            ar_position_ids=ar_position_ids,
            ar_attention_mask=ar_attention_mask,
        )
        drafter_time = time.perf_counter() - drafter_start if profile else 0.0
        loss_sum, valid_count, predictions, ce_time = self._chunked_loss_and_predictions(
            hidden_states=draft_hidden_states,
            labels=tree_labels.reshape(batch_size, num_blocks * block_size),
            weights=tree_cum_probs.reshape(batch_size, num_blocks * block_size),
            valid_mask=flat_valid_mask,
            prediction_mask=(tree_valid_mask & anchor_valid_mask.unsqueeze(-1)).reshape(batch_size, num_blocks * block_size),
            compute_predictions=compute_predictions,
            profile=profile,
        )
        q_loss_sum = draft_hidden_states.new_zeros(())
        if self.has_q_head:
            if predictions is None:
                raise RuntimeError("Q-head training requires drafter predictions to be computed.")
            q_targets = predictions.eq(
                tree_labels.reshape(batch_size, num_blocks * block_size)
            ).to(torch.float32)
            q_loss = F.binary_cross_entropy_with_logits(
                q_logits.float(),
                q_targets,
                reduction="none",
            )
            q_loss_sum = q_loss[flat_valid_mask].sum().to(draft_hidden_states.dtype)

        ar_loss_sum = draft_hidden_states.new_zeros(())
        if self.has_ar_head:
            ar_loss_sum, _, _, ar_ce_time = self._chunked_loss_and_predictions(
                hidden_states=ar_hidden_states,
                labels=tree_labels.reshape(batch_size, num_blocks * block_size),
                weights=tree_cum_probs.reshape(batch_size, num_blocks * block_size),
                valid_mask=flat_valid_mask,
                prediction_mask=None,
                compute_predictions=False,
                profile=profile,
            )
            ce_time += ar_ce_time
        if predictions is not None:
            predictions = predictions.view(batch_size, num_blocks, block_size)
        return {
            "loss_sum": loss_sum,
            "q_loss_sum": q_loss_sum,
            "ar_loss_sum": ar_loss_sum,
            "valid_count": valid_count,
            "predictions": predictions,
            "tree_valid_mask": tree_valid_mask,
            "mask_time": mask_time,
            "drafter_time": drafter_time,
            "ce_time": ce_time,
        }

    def _build_parent_token_ids(
        self,
        tree_token_ids: torch.Tensor,
        tree_info,
    ) -> torch.Tensor:
        """Build per-node parent-token inputs for the autoregressive head.

        Args:
            tree_token_ids: Tensor of shape ``(batch_size, num_blocks, block_size)``
                containing the token id assigned to each tree node.
            tree_info: Tree metadata whose ``parent_idx`` tensor has shape
                ``(block_size,)``.

        Returns:
            Tensor of shape ``(batch_size, num_blocks, block_size)`` where each
            node stores its parent token id and the root position is unchanged.
        """
        parent_token_ids = tree_token_ids.clone()
        if tree_info.block_size <= 1:
            return parent_token_ids
        parent_idx = tree_info.parent_idx.to(tree_token_ids.device)
        parent_token_ids[..., 1:] = tree_token_ids.index_select(-1, parent_idx[1:])
        return parent_token_ids

    def _chunked_loss_and_predictions(
        self,
        *,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
        valid_mask: torch.Tensor,
        prediction_mask: torch.Tensor | None,
        compute_predictions: bool,
        profile: bool = False,
    ) -> tuple[torch.Tensor, int, torch.Tensor | None, float]:
        """Compute weighted CE loss and optional predictions for a tree chunk.

        Args:
            hidden_states: Tensor of shape ``(batch_size, total_tokens, hidden_size)``.
            labels: Tensor of shape ``(batch_size, total_tokens)`` with target ids.
            weights: Tensor of shape ``(batch_size, total_tokens)`` with per-token
                scalar weights.
            valid_mask: Boolean tensor of shape ``(batch_size, total_tokens)``
                selecting tokens that contribute to loss.
            prediction_mask: Reserved mask argument for prediction filtering. It is
                currently unused by this implementation.
            compute_predictions: Whether to return argmax predictions.
            profile: Whether to measure elapsed CE time.

        Returns:
            A tuple ``(loss_sum, valid_count, predictions, ce_time_sec)`` where
            ``loss_sum`` is a scalar tensor, ``valid_count`` is a Python ``int``,
            ``predictions`` has shape ``(batch_size, total_tokens)`` when
            requested, and ``ce_time_sec`` is a float.
        """
        ce_start = time.perf_counter() if profile else 0.0
        _ = prediction_mask
        flat_valid = valid_mask.reshape(-1)
        if not flat_valid.any():
            return hidden_states.new_zeros(()), 0, None, 0.0

        if self._loss_and_predictions_impl is None:
            raise RuntimeError("Loss/prediction helper must be initialized before use.")

        prediction_chunk_size = self.config.ce_chunk_size or int(hidden_states.shape[0] * hidden_states.shape[1])
        total_loss, valid_count_tensor, prediction_tensor = self._loss_and_predictions_impl(
            hidden_states=hidden_states,
            labels=labels,
            weights=weights,
            valid_mask=valid_mask,
            compute_predictions=compute_predictions,
            prediction_chunk_size=prediction_chunk_size,
        )
        valid_count = int(valid_count_tensor.item())
        predictions = prediction_tensor if compute_predictions else None
        ce_time = time.perf_counter() - ce_start if profile else 0.0
        return total_loss, valid_count, predictions, ce_time

    def _compute_linear_cross_entropy(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Delegate unreduced CE computation to the helper implementation.

        Args:
            hidden_states: Tensor of shape ``(num_tokens, hidden_size)``.
            labels: Tensor of shape ``(num_tokens,)``.

        Returns:
            Tensor of shape ``(num_tokens,)`` containing per-token CE loss values.
        """
        if self._loss_and_predictions_impl is None:
            raise RuntimeError("Loss/prediction helper must be initialized before use.")
        return self._loss_and_predictions_impl._compute_linear_cross_entropy(hidden_states, labels)

    def _get_acceptance_path_tensors(
        self,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build or fetch cached tensors used by the acceptance proxy.

        Args:
            device: Device on which the returned tensors should live.

        Returns:
            A tuple ``(ancestor_mask, node_depths, node_target_indices)`` with
            shapes ``(block_size, block_size)``, ``(block_size,)``, and
            ``(block_size,)`` respectively.
        """
        parent_idx = self.tree_processor.parent_idx
        depth = self.tree_processor.depth
        primary_path_indices = self.tree_processor.primary_path_indices
        parent_key = tuple(int(idx) for idx in parent_idx.tolist())
        depth_key = tuple(int(idx) for idx in depth.tolist())
        primary_key = tuple(int(idx) for idx in primary_path_indices.tolist())
        device_key = (device.type, -1 if device.index is None else device.index)
        cache_key = (parent_key, depth_key, primary_key, device_key[0], device_key[1])
        cached = self._acceptance_path_cache.get(cache_key)
        if cached is not None:
            return cached

        ancestor_mask = self.tree_processor.tree_mask.to(device=device)
        node_depths = depth.to(device=device)
        primary_depths = depth.index_select(0, primary_path_indices)
        node_target_indices = torch.full(
            (int(node_depths.numel()),),
            -1,
            dtype=torch.long,
            device=device,
        )
        for primary_offset, depth_idx in enumerate(primary_depths.tolist()):
            node_target_indices[node_depths == int(depth_idx)] = primary_offset

        cached = (ancestor_mask, node_depths, node_target_indices)
        self._acceptance_path_cache[cache_key] = cached
        return cached

    def _count_valid_targets(
        self,
        batch: PackedBatch,
    ) -> int:
        """Count non-root tree targets that should contribute to loss.

        Args:
            batch: Packed batch with ``tree_valid_mask`` of shape
                ``(batch_size, num_anchors, block_size)`` and
                ``anchor_valid_mask`` of shape ``(batch_size, num_anchors)``.

        Returns:
            Number of valid non-root tree nodes across the packed batch.
        """
        if batch.num_anchors == 0 or batch.tree_valid_mask.numel() == 0:
            return 0
        non_root_mask = self.tree_processor.non_root_mask.to(batch.tree_valid_mask.device).view(1, 1, -1)
        valid_mask = (
            batch.tree_valid_mask
            & non_root_mask
            & batch.anchor_valid_mask.unsqueeze(-1)
        )
        return int(valid_mask.sum().item())

    def _should_log_training_metrics_for_step(self, optimizer_step: int) -> bool:
        """Decide whether full training metrics should be logged at a step.

        Args:
            optimizer_step: Current optimizer step.

        Returns:
            ``True`` when the step hits the regular logging cadence or lies within
            the configured profiling window.
        """
        if self.config.log_every > 0 and optimizer_step % self.config.log_every == 0:
            return True
        return self.config.profile_steps > 0 and optimizer_step <= self.config.profile_steps

    def _prefill_target_context(
        self,
        batch: PackedBatch,
    ) -> torch.Tensor:
        """Run the frozen target model on packed context tokens only.

        Args:
            batch: Packed batch whose context tensors ``input_ids``,
                ``position_ids``, ``document_mask``, and ``context_valid_mask``
                all have shape ``(batch_size, ctx_len)``.

        Returns:
            Context features extracted from the target model with shape
            ``(batch_size, ctx_len, hidden_size)`` in the drafter feature space.
        """
        with torch.no_grad():
            prefill_mask = self._build_prefill_attention_mask(batch.document_mask, batch.context_valid_mask)
            target_out = self.target_model(
                input_ids=batch.input_ids,
                attention_mask=prefill_mask,
                position_ids=batch.position_ids,
                output_hidden_states=True,
                use_cache=False,
            )
            return self.raw_drafter.extract_ctx_features(target_out.hidden_states)

    def _train_batch(
        self,
        batch: PackedBatch,
    ) -> dict[str, Any]:
        """Run one training micro-batch and backpropagate chunked losses.

        Args:
            batch: Packed batch containing context tensors of shape
                ``(batch_size, ctx_len)`` and tree tensors of shape
                ``(batch_size, num_anchors, block_size)``.

        Returns:
            A dictionary with scalar tensors for ``loss``, ``ce_loss``,
            ``q_loss``, and ``ar_loss``; integer counts for valid targets and
            acceptance examples; floating acceptance totals; and optional timing
            breakdowns in seconds.
        """
        batch = batch.to(self.fabric.device)
        if not batch.context_valid_mask.any() or batch.num_anchors == 0:
            zero = torch.zeros((), device=self.fabric.device)
            return {
                "loss": zero,
                "ce_loss": zero,
                "q_loss": zero,
                "ar_loss": zero,
                "valid_count": 0,
                "acceptance_total": 0.0,
                "acceptance_count": 0,
                "prefill_time": 0.0,
                "mask_time": 0.0,
                "drafter_time": 0.0,
                "ce_time": 0.0,
            }

        total_valid_count = self._count_valid_targets(batch)
        if total_valid_count == 0:
            zero = torch.zeros((), device=self.fabric.device)
            return {
                "loss": zero,
                "ce_loss": zero,
                "q_loss": zero,
                "ar_loss": zero,
                "valid_count": 0,
                "acceptance_total": 0.0,
                "acceptance_count": 0,
                "prefill_time": 0.0,
                "mask_time": 0.0,
                "drafter_time": 0.0,
                "ce_time": 0.0,
            }

        profile = self.config.profile_steps > 0 and self.global_step < self.config.profile_steps
        prefill_start = time.perf_counter() if profile else 0.0
        target_ctx_features = self._prefill_target_context(batch)
        prefill_time = time.perf_counter() - prefill_start if profile else 0.0
        total_loss_sum = torch.zeros((), device=self.fabric.device)
        total_q_loss_sum = torch.zeros((), device=self.fabric.device)
        total_ar_loss_sum = torch.zeros((), device=self.fabric.device)
        acceptance_total = 0.0
        acceptance_count = 0
        mask_time = 0.0
        drafter_time = 0.0
        ce_time = 0.0
        anchor_chunk = self.config.anchor_chunk_size or batch.num_anchors

        for start in range(0, batch.num_anchors, anchor_chunk):
            end = min(start + anchor_chunk, batch.num_anchors)
            chunk_result = self._forward_anchor_chunk(
                batch,
                target_ctx_features,
                slice(start, end),
                compute_predictions=True,
                profile=profile,
            )
            if chunk_result["valid_count"] == 0:
                continue
            chunk_loss_sum = chunk_result["loss_sum"]
            chunk_q_loss_sum = chunk_result["q_loss_sum"]
            chunk_ar_loss_sum = chunk_result["ar_loss_sum"]
            mask_time += chunk_result["mask_time"]
            drafter_time += chunk_result["drafter_time"]
            ce_time += chunk_result["ce_time"]
            if chunk_result["predictions"] is not None:
                acceptance_chunk_total, acceptance_chunk_count = self._acceptance_proxy(
                    predictions=chunk_result["predictions"],
                    labels=batch.tree_labels[:, start:end],
                    anchor_valid_mask=batch.anchor_valid_mask[:, start:end],
                    tree_valid_mask=chunk_result["tree_valid_mask"],
                )
                acceptance_total += acceptance_chunk_total
                acceptance_count += acceptance_chunk_count
            scaled_chunk_loss = (
                chunk_loss_sum
                + self.config.q_loss_lambda * chunk_q_loss_sum
                + self.config.ar_loss_lambda * chunk_ar_loss_sum
            ) / total_valid_count / max(self.config.grad_accum_steps, 1)
            self.fabric.backward(scaled_chunk_loss)
            total_loss_sum = total_loss_sum + chunk_loss_sum.detach()
            total_q_loss_sum = total_q_loss_sum + chunk_q_loss_sum.detach()
            total_ar_loss_sum = total_ar_loss_sum + chunk_ar_loss_sum.detach()

        ce_loss = total_loss_sum / total_valid_count
        q_loss = total_q_loss_sum / total_valid_count
        ar_loss = total_ar_loss_sum / total_valid_count
        loss = ce_loss + self.config.q_loss_lambda * q_loss + self.config.ar_loss_lambda * ar_loss
        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "q_loss": q_loss,
            "ar_loss": ar_loss,
            "valid_count": total_valid_count,
            "acceptance_total": acceptance_total,
            "acceptance_count": acceptance_count,
            "prefill_time": prefill_time,
            "mask_time": mask_time,
            "drafter_time": drafter_time,
            "ce_time": ce_time,
        }

    @torch.no_grad()
    def _eval_batch(
        self,
        batch: PackedBatch,
    ) -> dict[str, Any]:
        """Evaluate one packed batch without gradient updates.

        Args:
            batch: Packed batch containing context tensors of shape
                ``(batch_size, ctx_len)`` and tree tensors of shape
                ``(batch_size, num_anchors, block_size)``.

        Returns:
            A dictionary matching :meth:`_train_batch`, except no backward pass is
            performed and profiling times are only populated for drafter-side work.
        """
        batch = batch.to(self.fabric.device)
        if not batch.context_valid_mask.any() or batch.num_anchors == 0:
            zero = torch.zeros((), device=self.fabric.device)
            return {
                "loss": zero,
                "ce_loss": zero,
                "q_loss": zero,
                "ar_loss": zero,
                "valid_count": 0,
                "acceptance_total": 0.0,
                "acceptance_count": 0,
                "prefill_time": 0.0,
                "mask_time": 0.0,
                "drafter_time": 0.0,
                "ce_time": 0.0,
            }

        total_valid_count = self._count_valid_targets(batch)
        if total_valid_count == 0:
            zero = torch.zeros((), device=self.fabric.device)
            return {
                "loss": zero,
                "ce_loss": zero,
                "q_loss": zero,
                "ar_loss": zero,
                "valid_count": 0,
                "acceptance_total": 0.0,
                "acceptance_count": 0,
                "prefill_time": 0.0,
                "mask_time": 0.0,
                "drafter_time": 0.0,
                "ce_time": 0.0,
            }

        target_ctx_features = self._prefill_target_context(batch)
        total_loss_sum = torch.zeros((), device=self.fabric.device)
        total_q_loss_sum = torch.zeros((), device=self.fabric.device)
        total_ar_loss_sum = torch.zeros((), device=self.fabric.device)
        acceptance_total = 0.0
        acceptance_count = 0
        mask_time = 0.0
        drafter_time = 0.0
        ce_time = 0.0
        anchor_chunk = self.config.anchor_chunk_size or batch.num_anchors

        for start in range(0, batch.num_anchors, anchor_chunk):
            end = min(start + anchor_chunk, batch.num_anchors)
            chunk_result = self._forward_anchor_chunk(
                batch,
                target_ctx_features,
                slice(start, end),
                compute_predictions=True,
                profile=False,
            )
            total_loss_sum = total_loss_sum + chunk_result["loss_sum"]
            total_q_loss_sum = total_q_loss_sum + chunk_result["q_loss_sum"]
            total_ar_loss_sum = total_ar_loss_sum + chunk_result["ar_loss_sum"]
            mask_time += chunk_result["mask_time"]
            drafter_time += chunk_result["drafter_time"]
            ce_time += chunk_result["ce_time"]
            if chunk_result["predictions"] is not None:
                acceptance_chunk_total, acceptance_chunk_count = self._acceptance_proxy(
                    predictions=chunk_result["predictions"],
                    labels=batch.tree_labels[:, start:end],
                    anchor_valid_mask=batch.anchor_valid_mask[:, start:end],
                    tree_valid_mask=chunk_result["tree_valid_mask"],
                )
                acceptance_total += acceptance_chunk_total
                acceptance_count += acceptance_chunk_count

        ce_loss = total_loss_sum / total_valid_count
        q_loss = total_q_loss_sum / total_valid_count
        ar_loss = total_ar_loss_sum / total_valid_count
        loss = ce_loss + self.config.q_loss_lambda * q_loss + self.config.ar_loss_lambda * ar_loss
        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "q_loss": q_loss,
            "ar_loss": ar_loss,
            "valid_count": total_valid_count,
            "acceptance_total": acceptance_total,
            "acceptance_count": acceptance_count,
            "prefill_time": 0.0,
            "mask_time": mask_time,
            "drafter_time": drafter_time,
            "ce_time": ce_time,
        }

    def _acceptance_proxy(
        self,
        *,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        anchor_valid_mask: torch.Tensor,
        tree_valid_mask: torch.Tensor,
    ) -> tuple[float, int]:
        """Compute a scalar acceptance proxy from predicted tree tokens.

        Args:
            predictions: Tensor of shape ``(batch_size, num_blocks, block_size)``.
            labels: Tensor of shape ``(batch_size, num_blocks, block_size)`` with
                reference tree token ids.
            anchor_valid_mask: Boolean tensor of shape ``(batch_size, num_blocks)``.
            tree_valid_mask: Boolean tensor of shape
                ``(batch_size, num_blocks, block_size)``.

        Returns:
            A tuple ``(accepted_depth_sum, valid_anchor_count)`` as Python
            ``float`` and ``int`` values.
        """
        if predictions.numel() == 0:
            return 0.0, 0
        primary_indices = self.tree_processor.primary_path_indices.to(predictions.device)
        if primary_indices.numel() <= 1:
            return 0.0, 0
        ancestor_mask, node_depths, node_target_indices = self._get_acceptance_path_tensors(device=predictions.device)
        comparable_mask = node_target_indices >= 0
        if not bool(comparable_mask.any().item()):
            return 0.0, int(anchor_valid_mask.to(torch.bool).sum().item())
        total_tensor, count_tensor = self._acceptance_proxy_impl(
            predictions=predictions,
            labels=labels,
            anchor_valid_mask=anchor_valid_mask,
            tree_valid_mask=tree_valid_mask,
            primary_indices=primary_indices,
            ancestor_mask=ancestor_mask,
            node_depths=node_depths,
            node_target_indices=node_target_indices,
        )
        return float(total_tensor.item()), int(count_tensor.item())

    def fit(self):
        """Train the drafter for the configured number of epochs.

        The method performs gradient accumulation, optimizer stepping, periodic
        validation, checkpointing, and optional speculative decoding evaluation.
        """
        total_optimizer_steps = max(
            1,
            math.ceil(len(self.train_loader) * self.config.num_epochs / max(self.config.grad_accum_steps, 1)),
        )
        optimizer_step = self.global_step
        accumulated_loss = 0.0
        accumulated_q_loss = 0.0
        accumulated_ar_loss = 0.0
        accumulated_acceptance_total = 0.0
        accumulated_acceptance_count = 0
        accumulated_prefill_time = 0.0
        accumulated_mask_time = 0.0
        accumulated_drafter_time = 0.0
        accumulated_ce_time = 0.0
        micro_step = 0

        for epoch_idx in range(self.config.num_epochs):
            self.drafter_model.train()
            for batch in self.train_loader:
                micro_step += 1
                is_final_micro = micro_step % self.config.grad_accum_steps == 0
                sync_context = self.fabric.no_backward_sync(
                    self.drafter_model,
                    enabled=not is_final_micro,
                )
                with sync_context:
                    batch_result = self._train_batch(batch)
                    loss = batch_result["loss"]
                accumulated_loss += float(loss.detach().item())
                accumulated_q_loss += float(batch_result["q_loss"].detach().item())
                accumulated_ar_loss += float(batch_result["ar_loss"].detach().item())
                accumulated_acceptance_total += float(batch_result["acceptance_total"])
                accumulated_acceptance_count += int(batch_result["acceptance_count"])
                accumulated_prefill_time += float(batch_result["prefill_time"])
                accumulated_mask_time += float(batch_result["mask_time"])
                accumulated_drafter_time += float(batch_result["drafter_time"])
                accumulated_ce_time += float(batch_result["ce_time"])

                if not is_final_micro:
                    if self.config.dev_run:
                        break
                    continue

                optimizer_step += 1
                lr = get_lr(
                    optimizer_step,
                    self.config.warmup_steps,
                    total_optimizer_steps,
                    self.config.lr,
                    self.config.min_lr,
                )
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                self.fabric.clip_gradients(self.drafter_model, self.optimizer, max_norm=self.config.grad_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step = optimizer_step

                avg_loss = accumulated_loss / max(self.config.grad_accum_steps, 1)
                avg_q_loss = accumulated_q_loss / max(self.config.grad_accum_steps, 1)
                avg_ar_loss = accumulated_ar_loss / max(self.config.grad_accum_steps, 1)
                avg_acceptance = accumulated_acceptance_total / max(accumulated_acceptance_count, 1)
                accumulated_acceptance_total = 0.0
                accumulated_acceptance_count = 0
                avg_prefill_time = accumulated_prefill_time / max(self.config.grad_accum_steps, 1)
                avg_mask_time = accumulated_mask_time / max(self.config.grad_accum_steps, 1)
                avg_drafter_time = accumulated_drafter_time / max(self.config.grad_accum_steps, 1)
                avg_ce_time = accumulated_ce_time / max(self.config.grad_accum_steps, 1)
                accumulated_prefill_time = 0.0
                accumulated_mask_time = 0.0
                accumulated_drafter_time = 0.0
                accumulated_ce_time = 0.0
                accumulated_loss = 0.0
                accumulated_q_loss = 0.0
                accumulated_ar_loss = 0.0
                if self.fabric.is_global_zero and (
                    self.config.verbose
                    or (self.config.log_every > 0 and self.global_step % self.config.log_every == 0)
                ):
                    print(
                        f"step={self.global_step} loss={avg_loss:.4f} lr={lr:.2e}",
                        flush=True,
                    )
                should_log_step = self._should_log_training_metrics_for_step(self.global_step)
                if should_log_step:
                    self._log(
                        {
                            "train/loss": avg_loss,
                            "train/q_loss": avg_q_loss,
                            "train/ar_loss": avg_ar_loss,
                            'train/acceptance_proxy': avg_acceptance,
                            "train/lr": lr,
                            **(
                                {
                                    "profile/train_prefill_s": avg_prefill_time,
                                    "profile/train_mask_s": avg_mask_time,
                                    "profile/train_drafter_s": avg_drafter_time,
                                    "profile/train_ce_s": avg_ce_time,
                                }
                                if self.config.profile_steps > 0 and self.global_step <= self.config.profile_steps
                                else {}
                            ),
                        },
                        self.global_step,
                    )

                if self.config.eval_every > 0 and self.global_step % self.config.eval_every == 0:
                    metrics = self.validate()
                    if self.fabric.is_global_zero:
                        print(
                            f"[eval step={self.global_step}] "
                            f"loss={metrics['eval/loss']:.4f} "
                            f"acceptance={metrics['eval/mean_acceptance_length']:.3f}",
                            flush=True,
                        )
                    self._log(metrics, self.global_step)

                if self.config.save_every > 0 and self.global_step % self.config.save_every == 0:
                    self.save_checkpoint(step=self.global_step)

                if self.config.dev_run:
                    break
            if self.config.dev_run:
                break

        self.save_checkpoint(tag="final")
        if self.fabric.is_global_zero and self.wandb_run is not None:
            self.wandb_run.finish()

    def save_checkpoint(self, step: int | None = None, tag: str | None = None):
        """Persist the drafter, optimizer, and trainer state to disk.

        Args:
            step: Optional explicit step number used in the checkpoint directory
                name when ``tag`` is not provided.
            tag: Optional directory tag such as ``"final"``.
        """
        if tag is not None:
            checkpoint_dir = self.output_dir / tag
        else:
            checkpoint_dir = self.output_dir / f"checkpoint-{step or self.global_step}"
        if self.fabric.is_global_zero:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "drafter_model": self.drafter_model,
            "optimizer": self.optimizer,
            "global_step": self.global_step,
            "wandb_run_id": self.wandb_run_id,
        }
        self.fabric.save(str(checkpoint_dir / "fabric_ckpt.pt"), state)
        if self.fabric.is_global_zero:
            unwrap_model(self.drafter_model).save_pretrained(str(checkpoint_dir / "hf_draft"))

    def load_checkpoint(self, checkpoint_path):
        """Restore trainer state from a previously saved checkpoint directory.

        Args:
            checkpoint_path: Path to a checkpoint directory containing
                ``fabric_ckpt.pt``.
        """
        checkpoint_dir = Path(checkpoint_path)
        ckpt_file = checkpoint_dir / "fabric_ckpt.pt"
        remainder = self.fabric.load(
            str(ckpt_file),
            {
                "drafter_model": self.drafter_model,
                "optimizer": self.optimizer,
            },
        )
        self.global_step = int(remainder.get("global_step", 0))
        self.wandb_run_id = remainder.get("wandb_run_id")

    @torch.inference_mode()
    def validate(self):
        """Evaluate the drafter on the validation loader and spec-eval suite.

        Returns:
            A flat dictionary of validation metrics including weighted CE losses,
            mean acceptance length, and optional speculative-decoding metrics.
        """
        self.drafter_model.eval()
        total_loss = 0.0
        total_q_loss = 0.0
        total_ar_loss = 0.0
        total_valid = 0
        total_acceptance = 0.0
        total_acceptance_count = 0

        for batch_idx, batch in enumerate(self.eval_loader):
            if batch_idx >= self.config.eval_batches:
                break
            batch_result = self._eval_batch(batch)
            valid_count = int(batch_result["valid_count"])
            total_loss += float(batch_result["loss"].item()) * valid_count
            total_q_loss += float(batch_result["q_loss"].item()) * valid_count
            total_ar_loss += float(batch_result["ar_loss"].item()) * valid_count
            total_valid += valid_count
            total_acceptance += float(batch_result["acceptance_total"])
            total_acceptance_count += int(batch_result["acceptance_count"])
            if self.config.dev_run:
                break

        reduced = self._all_reduce_tensor(
            torch.tensor(
                [
                    total_loss,
                    total_q_loss,
                    total_ar_loss,
                    float(total_valid),
                    total_acceptance,
                    float(total_acceptance_count),
                ],
                dtype=torch.float64,
                device=self.fabric.device,
            ),
            reduce_op="sum",
        )
        total_loss = float(reduced[0].item())
        total_q_loss = float(reduced[1].item())
        total_ar_loss = float(reduced[2].item())
        total_valid = int(round(float(reduced[3].item())))
        total_acceptance = float(reduced[4].item())
        total_acceptance_count = int(round(float(reduced[5].item())))

        metrics: dict[str, Any]
        if total_valid == 0:
            eval_loss = 0.0
            eval_q_loss = 0.0
            eval_ar_loss = 0.0
        else:
            eval_loss = total_loss / total_valid
            eval_q_loss = total_q_loss / total_valid
            eval_ar_loss = total_ar_loss / total_valid
        mean_acceptance = total_acceptance / max(total_acceptance_count, 1)
        metrics = {
            "eval/loss": eval_loss,
            "eval/q_loss": eval_q_loss,
            "eval/ar_loss": eval_ar_loss,
            "eval/mean_acceptance_length": mean_acceptance,
        }
        metrics.update(self._run_speculative_eval_suite())
        self.drafter_model.train()
        return metrics


def build_parser() -> ArgumentParser:
    """Build the command-line parser for trainer and data configuration.

    Returns:
        Configured ``jsonargparse.ArgumentParser`` instance.
    """
    parser = ArgumentParser(description="Train the Tree Flash drafter from Stage 2 HDF5 data.")
    parser.add_class_arguments(TrainerConfig, "trainer")
    parser.add_class_arguments(DataModuleConfig, "data")
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--drafter", type=str, required=True)
    parser.add_argument(
        "--tree_type",
        type=str,
        default="block",
        choices=["block", "branch_off", "prunable"],
    )
    parser.add_argument("--tree_args", type=dict, default=None)
    return parser


def main() -> None:
    """Parse CLI arguments, instantiate the trainer, and start training."""
    parser = build_parser()
    cfg = parser.parse_args()
    trainer_cfg = TrainerConfig(**namespace_to_dict(cfg.trainer))
    data_cfg = DataModuleConfig(**namespace_to_dict(cfg.data))
    drafter = cfg.drafter
    try:
        drafter = json.loads(drafter)
    except json.JSONDecodeError:
        pass
    trainer = Trainer(
        config=trainer_cfg,
        target=cfg.target,
        data=data_cfg,
        drafter=drafter,
        tree_type=cfg.tree_type,
        tree_args=cfg.tree_args,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
