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


def compute_linear_cross_entropy(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    target_lm_head: torch.nn.Module,
) -> torch.Tensor:
    """Compute unreduced token cross-entropy from hidden states and labels.

    Args:
        hidden_states: Hidden activations with shape ``(..., hidden_dim)``.
        labels: Token ids aligned with ``hidden_states.shape[:-1]``.
        target_lm_head: Output projection that maps ``hidden_dim`` features to
            vocabulary logits.

    Returns:
        Tensor of per-token cross-entropy losses with shape ``labels.shape``.
    """
    weight = target_lm_head.weight
    bias = getattr(target_lm_head, "bias", None)
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

    logits = target_lm_head(hidden_states)
    return F.cross_entropy(
        logits.float(),
        labels,
        reduction="none",
    )


class TrainerLossAndPredictions(torch.nn.Module):
    def __init__(self, target_lm_head: torch.nn.Module):
        """Store the LM head used for loss computation and argmax predictions.

        Args:
            target_lm_head: Module mapping hidden states of shape
                ``(..., hidden_dim)`` to logits of shape ``(..., vocab_size)``.
        """
        super().__init__()
        self.target_lm_head = target_lm_head

    def forward(
        self,
        *,
        flat_hidden: torch.Tensor,
        flat_labels: torch.Tensor,
        flat_weights: torch.Tensor,
        flat_valid: torch.Tensor,
        flat_prediction_mask: torch.Tensor,
        compute_predictions: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Compute a weighted loss sum and optional flat predictions.

        Args:
            flat_hidden: Flattened hidden states with shape
                ``(num_tokens, hidden_dim)``.
            flat_labels: Flattened target token ids with shape ``(num_tokens,)``.
            flat_weights: Per-token loss weights with shape ``(num_tokens,)``.
            flat_valid: Boolean mask selecting tokens that contribute to loss,
                shape ``(num_tokens,)``.
            flat_prediction_mask: Boolean mask selecting tokens for argmax
                prediction, shape ``(num_tokens,)``. The full argmax is now
                materialized regardless of this mask and the mask is retained
                only for call-site compatibility.
            compute_predictions: Unused compatibility flag. Full argmax
                predictions are always materialized.

        Returns:
            Tuple ``(total_loss, valid_count, predictions)`` where ``total_loss``
            is a scalar weighted loss sum, ``valid_count`` is a scalar tensor
            containing the number of valid loss positions, and ``predictions`` is
            either ``None`` or a tensor of shape ``(num_tokens,)`` containing
            argmax token ids for every flattened position.
        """
        valid_hidden = flat_hidden[flat_valid]
        valid_labels = flat_labels[flat_valid]
        valid_weights = flat_weights[flat_valid]
        per_token_loss = compute_linear_cross_entropy(valid_hidden, valid_labels, self.target_lm_head)
        total_loss = (per_token_loss * valid_weights).sum()
        valid_count = flat_valid.sum()

        predictions = None
        if compute_predictions:
            prediction_logits = self.target_lm_head(
                flat_hidden.detach().to(self.target_lm_head.weight.dtype)
            )
            predictions = prediction_logits.argmax(dim=-1)

        return total_loss, valid_count, predictions


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


def unwrap_model(module):
    """Remove common Fabric or compile wrappers around a module.

    Args:
        module: Potentially wrapped module instance.

    Returns:
        The innermost exposed module after stripping known wrapper attributes.
    """
    raw = module
    if hasattr(raw, "_forward_module"):
        raw = raw._forward_module
    if hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod
    return raw


def has_pruning_head(module) -> bool:
    """Check whether a drafter module exposes pruning-related heads.

    Args:
        module: Drafter module to inspect.

    Returns:
        ``True`` when the module has either a ``q_head`` or an ``ar_block``.
    """
    return getattr(module, "q_head", None) is not None or getattr(module, "ar_block", None) is not None


def build_prefill_attention_mask(
    document_mask: torch.Tensor,
    valid_mask: torch.Tensor,
):
    """Build a causal flex-attention mask for target-model context prefill.

    Args:
        document_mask: Document ids with shape ``(batch, seq_len)`` used to
            prevent cross-document attention.
        valid_mask: Boolean mask with shape ``(batch, seq_len)`` indicating
            which context positions are real tokens.

    Returns:
        A flex-attention block mask for queries and keys of length ``seq_len``.
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
        """Return whether query ``q_idx`` may attend to key ``kv_idx``.

        Args:
            b: Batch index.
            h: Attention head index or ``None`` for head-independent masks.
            q_idx: Query token index in ``[0, seq_len)``.
            kv_idx: Key/value token index in ``[0, seq_len)``.

        Returns:
            Boolean scalar indicating whether the attention edge is enabled.
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
    """Build the drafter attention mask for tree-token decoding blocks.

    Args:
        anchor_positions: Anchor token indices with shape
            ``(batch, num_blocks)``.
        document_mask: Document ids for context tokens with shape
            ``(batch, ctx_len)``.
        context_valid_mask: Boolean context-token validity mask with shape
            ``(batch, ctx_len)``.
        tree_valid_mask: Boolean tree-token validity mask with shape
            ``(batch, num_blocks, block_size)``.
        block_size: Number of tree nodes per anchor block.

    Returns:
        A flex-attention block mask with query length ``num_blocks * block_size``
        and key length ``ctx_len + num_blocks * block_size``.
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
            h: Attention head index or ``None`` for head-independent masks.
            q_idx: Query tree-token index in ``[0, total_tree_len)``.
            kv_idx: Key/value index in ``[0, ctx_len + total_tree_len)``.

        Returns:
            Boolean scalar indicating whether the attention edge is enabled.
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
    """Build the AR-head attention mask for autoregressive tree scoring.

    Args:
        anchor_positions: Anchor token indices with shape
            ``(batch, num_blocks)``.
        document_mask: Document ids for context tokens with shape
            ``(batch, ctx_len)``.
        context_valid_mask: Boolean context-token validity mask with shape
            ``(batch, ctx_len)``.
        tree_valid_mask: Boolean tree-token validity mask with shape
            ``(batch, num_blocks, block_size)``.
        tree_info: Tree metadata object exposing at least ``tree_mask``.
        block_size: Number of tree nodes per anchor block.

    Returns:
        A flex-attention block mask with query length ``num_blocks * block_size``
        and key length ``ctx_len + num_blocks * block_size``.
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
        """Return whether an AR-tree query may attend to a given key.

        Args:
            b: Batch index.
            h: Attention head index or ``None`` for head-independent masks.
            q_idx: Query tree-token index in ``[0, total_tree_len)``.
            kv_idx: Key/value index in ``[0, ctx_len + total_tree_len)``.

        Returns:
            Boolean scalar indicating whether the attention edge is enabled.
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
    """Compute the scalar learning rate for a warmup-plus-cosine schedule.

    Args:
        step: Current optimizer step.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total optimizer steps in the run.
        lr: Peak learning rate reached after warmup.
        min_lr: Minimum learning rate used at the end of cosine decay.

    Returns:
        Learning rate value for ``step``.
    """
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (lr - min_lr) * cosine


def maybe_cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device=device)


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
        """Initialize models, dataloaders, optimizer, and training state.

        Args:
            config: Trainer hyperparameters and runtime options.
            target: Hugging Face name or path of the frozen target LM.
            data: Dataset and dataloader configuration.
            drafter: Drafter checkpoint path or inline config dictionary.
            tree_type: Tree-construction strategy used by the data pipeline.
            tree_args: Optional tree-processor arguments such as subtree layout
                or pruning settings.

        Returns:
            None. The initializer populates the trainer state in place.
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
        self._acceptance_path_cache: dict[tuple[tuple[int, ...], str, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._build_prefill_attention_mask = build_prefill_attention_mask
        self._build_drafter_block_mask = build_drafter_block_mask
        self._build_ar_block_mask = build_ar_block_mask

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
            self._build_prefill_attention_mask = torch.compile(self._build_prefill_attention_mask, dynamic=True)
            self._build_drafter_block_mask = torch.compile(self._build_drafter_block_mask, dynamic=True)
            self._build_ar_block_mask = torch.compile(self._build_ar_block_mask, dynamic=True)

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
        self._loss_and_predictions = TrainerLossAndPredictions(self.target_lm_head)
        if config.compile and hasattr(torch, "compile"):
            self._loss_and_predictions = torch.compile(self._loss_and_predictions, dynamic=True)

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
        """Instantiate the tree processor used by the dataloaders and trainer.

        Args:
            tree_type: Tree strategy name such as ``"block"`` or
                ``"branch_off"``.
            tree_seq_depth: Number of sequence positions represented per tree.
            tree_args: Additional processor-specific configuration values.

        Returns:
            Tree processor instance matching ``tree_type``.
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
            return PrunableTreeProcessor(
                tree_seq_depth=tree_seq_depth,
                base_tree_type=tree_args.get("base_tree_type", "block"),
                prune_topk=int(tree_args.get("prune_topk", 0)),
                sub_tree_paths=tree_args.get("sub_tree_paths"),
                branching_pattern=tree_args.get("branching_pattern"),
            )
        raise NotImplementedError(
            f"tree_type={tree_type!r} is not implemented. Use 'block', 'branch_off', or 'prunable'."
        )

    def _init_wandb(self) -> None:
        """Start or resume the Weights & Biases run on rank zero.

        Args:
            None.

        Returns:
            None.
        """
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

    def _log(self, metrics: dict[str, float], step: int) -> None:
        """Log scalar metrics to the active Weights & Biases run.

        Args:
            metrics: Mapping from metric names to scalar float values.
            step: Global training step attached to the log event.

        Returns:
            None.
        """
        if self.fabric.is_global_zero and self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)

    def _get_tree_info(
        self,
        *,
        batch_size: int,
        num_blocks: int,
        device: torch.device,
    ):
        """Fetch cached tree metadata for a batch shape and device.

        Args:
            batch_size: Batch dimension used to materialize tree tensors.
            num_blocks: Number of anchor blocks per sample.
            device: Device on which the cached tensors must live.

        Returns:
            Tree-info object produced by ``self.tree_processor.build_tree_info``.
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
        """Run the drafter and losses for a slice of anchor blocks.

        Args:
            batch: Packed batch whose tree tensors have shapes like
                ``tree_labels`` ``(batch, num_anchors, block_size)`` and
                ``anchor_positions`` ``(batch, num_anchors)``.
            target_ctx_features: Target-model context features with shape
                ``(batch, ctx_len, hidden_dim)`` or the drafter-specific context
                layout returned by ``extract_ctx_features``.
            anchor_slice: Slice selecting anchor blocks along the ``num_anchors``
                dimension.
            compute_predictions: Whether to produce full-tree argmax predictions
                for the acceptance proxy.
            profile: Whether to time mask building, drafter forward, and CE.

        Returns:
            Dictionary containing scalar loss sums, the integer valid-token
            count, optional predictions of shape
            ``(batch, sliced_num_blocks, block_size)``, the sliced
            ``tree_valid_mask``, and optional timing metrics in seconds.
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
        if profile:
            maybe_cuda_synchronize(tree_labels.device)
            mask_start = time.perf_counter()
        else:
            mask_start = 0.0
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
        if profile:
            maybe_cuda_synchronize(tree_labels.device)
            mask_time = time.perf_counter() - mask_start
        else:
            mask_time = 0.0

        if profile:
            maybe_cuda_synchronize(tree_labels.device)
            drafter_start = time.perf_counter()
        else:
            drafter_start = 0.0
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
        if profile:
            maybe_cuda_synchronize(tree_labels.device)
            drafter_time = time.perf_counter() - drafter_start
        else:
            drafter_time = 0.0
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
            q_targets = tree_cum_probs.reshape(batch_size, num_blocks * block_size).to(torch.float32)
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
        """Replace each non-root tree token by its parent token id.

        Args:
            tree_token_ids: Token ids with shape ``(batch, num_blocks,
                block_size)``.
            tree_info: Tree metadata exposing ``block_size`` and ``parent_idx``.

        Returns:
            Tensor with the same shape as ``tree_token_ids`` where position ``i``
            stores the token id of its parent node for all non-root nodes.
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
        """Flatten batched token features and compute loss and predictions.

        Args:
            hidden_states: Hidden states with shape ``(batch, total_tokens,
                hidden_dim)``.
            labels: Target token ids with shape ``(batch, total_tokens)``.
            weights: Per-token loss weights with shape ``(batch, total_tokens)``.
            valid_mask: Boolean mask selecting loss positions, shape
                ``(batch, total_tokens)``.
            prediction_mask: Optional boolean mask selecting prediction
                positions, shape ``(batch, total_tokens)``. The full argmax is
                now materialized regardless of this mask and the mask is
                retained only for call-site compatibility.
            compute_predictions: Whether to return full argmax predictions over
                all tree positions.
            profile: Whether to report elapsed CE time in seconds.

        Returns:
            Tuple ``(total_loss, valid_count, predictions, ce_time)`` where
            ``total_loss`` is a scalar weighted loss sum, ``valid_count`` is the
            number of valid tokens, ``predictions`` is either ``None`` or a
            tensor of shape ``(batch, total_tokens)``, and ``ce_time`` is a
            float duration in seconds.
        """
        if profile:
            maybe_cuda_synchronize(hidden_states.device)
            ce_start = time.perf_counter()
        else:
            ce_start = 0.0
        batch_size, total_tokens, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.reshape(batch_size * total_tokens, hidden_size)
        flat_labels = labels.reshape(batch_size * total_tokens)
        flat_weights = weights.reshape(batch_size * total_tokens)
        flat_valid = valid_mask.reshape(batch_size * total_tokens)
        if not flat_valid.any():
            return hidden_states.new_zeros(()), 0, None, 0.0
        total_loss, valid_count_tensor, predictions = self._loss_and_predictions(
            flat_hidden=flat_hidden,
            flat_labels=flat_labels,
            flat_weights=flat_weights,
            flat_valid=flat_valid,
            flat_prediction_mask=flat_valid if prediction_mask is None else prediction_mask.reshape(batch_size * total_tokens),
            compute_predictions=compute_predictions,
        )
        valid_count = int(valid_count_tensor.item())
        if predictions is not None:
            predictions = predictions.view(batch_size, total_tokens)
        if profile:
            maybe_cuda_synchronize(hidden_states.device)
            ce_time = time.perf_counter() - ce_start
        else:
            ce_time = 0.0
        return total_loss, valid_count, predictions, ce_time

    def _compute_linear_cross_entropy(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token LM-head cross-entropy for this trainer.

        Args:
            hidden_states: Hidden activations with shape ``(..., hidden_dim)``.
            labels: Token ids aligned with ``hidden_states.shape[:-1]``.

        Returns:
            Tensor of per-token cross-entropy losses with shape ``labels.shape``.
        """
        return compute_linear_cross_entropy(hidden_states, labels, self.target_lm_head)

    def _get_acceptance_path_tensors(
        self,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Materialize cached root-to-node path tensors for acceptance scoring.

        Args:
            device: Device on which the cached tensors should reside.

        Returns:
            Tuple ``(path_indices, path_mask, path_lengths)`` where
            ``path_indices`` has shape ``(num_nodes, max_path_len)``,
            ``path_mask`` has the same shape and marks valid entries, and
            ``path_lengths`` has shape ``(num_nodes,)``.
        """
        parent_idx = self.tree_processor.parent_idx
        parent_key = tuple(int(idx) for idx in parent_idx.tolist())
        device_key = (device.type, -1 if device.index is None else device.index)
        cache_key = (parent_key, device_key[0], device_key[1])
        cached = self._acceptance_path_cache.get(cache_key)
        if cached is not None:
            return cached

        node_paths: list[list[int]] = []
        max_path_len = 0
        for node_idx in range(int(parent_idx.numel())):
            path: list[int] = []
            cur = node_idx
            while cur > 0:
                path.append(cur)
                cur = int(parent_idx[cur].item())
            path.reverse()
            node_paths.append(path)
            max_path_len = max(max_path_len, len(path))

        path_indices = torch.zeros((len(node_paths), max_path_len), dtype=torch.long, device=device)
        path_mask = torch.zeros((len(node_paths), max_path_len), dtype=torch.bool, device=device)
        path_lengths = torch.zeros((len(node_paths),), dtype=torch.long, device=device)
        for node_idx, path in enumerate(node_paths):
            path_lengths[node_idx] = len(path)
            if not path:
                continue
            path_indices[node_idx, : len(path)] = torch.tensor(path, dtype=torch.long, device=device)
            path_mask[node_idx, : len(path)] = True

        cached = (path_indices, path_mask, path_lengths)
        self._acceptance_path_cache[cache_key] = cached
        return cached

    def _count_valid_targets(
        self,
        batch: PackedBatch,
    ) -> int:
        """Count non-root tree targets that contribute to training loss.

        Args:
            batch: Packed batch with ``tree_valid_mask`` of shape
                ``(batch, num_anchors, block_size)`` and ``anchor_valid_mask`` of
                shape ``(batch, num_anchors)``.

        Returns:
            Integer number of valid non-root target positions in the batch.
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
        """Decide whether training metrics should be logged for a step.

        Args:
            optimizer_step: One-based optimizer step to evaluate.

        Returns:
            ``True`` when standard logging is due or profiling is still active.
        """
        if self.config.log_every > 0 and optimizer_step % self.config.log_every == 0:
            return True
        return self.config.profile_steps > 0 and optimizer_step <= self.config.profile_steps

    def _prefill_target_context(
        self,
        batch: PackedBatch,
    ) -> torch.Tensor:
        """Run the frozen target LM over context tokens and extract features.

        Args:
            batch: Packed batch containing ``input_ids``, ``position_ids``,
                ``document_mask``, and ``context_valid_mask`` with leading shape
                ``(batch, ctx_len)``.

        Returns:
            Context features extracted from the target model. The tensor shape is
            determined by ``self.raw_drafter.extract_ctx_features`` and is
            aligned to the batch context sequence.
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
        *,
        compute_acceptance: bool = True,
    ) -> dict[str, Any]:
        """Run forward, backward, and metric accumulation for one batch.

        Args:
            batch: Packed batch with context tensors of shape ``(batch, ctx_len)``
                and tree tensors of shape ``(batch, num_anchors, block_size)``.
            compute_acceptance: Whether to compute full-tree argmax predictions
                and the acceptance proxy for this batch.

        Returns:
            Dictionary containing scalar loss tensors, valid-target count,
            acceptance aggregates, and optional profiling durations in seconds.
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
        if profile:
            maybe_cuda_synchronize(batch.input_ids.device)
            prefill_start = time.perf_counter()
        else:
            prefill_start = 0.0
        target_ctx_features = self._prefill_target_context(batch)
        if profile:
            maybe_cuda_synchronize(batch.input_ids.device)
            prefill_time = time.perf_counter() - prefill_start
        else:
            prefill_time = 0.0
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
                compute_predictions=compute_acceptance,
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
        """Evaluate one batch without gradient updates.

        Args:
            batch: Packed batch with context tensors of shape ``(batch, ctx_len)``
                and tree tensors of shape ``(batch, num_anchors, block_size)``.

        Returns:
            Dictionary containing scalar loss tensors, valid-target count,
            acceptance aggregates, and timing fields in seconds.
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
        """Approximate acceptance length from predicted tree tokens.

        Args:
            predictions: Predicted token ids with shape ``(batch, num_anchors,
                block_size)``.
            labels: Reference tree token ids with shape ``(batch, num_anchors,
                block_size)``.
            anchor_valid_mask: Boolean anchor validity mask with shape
                ``(batch, num_anchors)``.
            tree_valid_mask: Boolean tree-token validity mask with shape
                ``(batch, num_anchors, block_size)``.

        Returns:
            Tuple ``(total, count)`` where ``total`` is the summed accepted
            prefix length across valid anchors plus one extra sampled token per
            valid anchor, and ``count`` is the number of valid anchors
            contributing to that sum.
        """
        if predictions.numel() == 0:
            return 0.0, 0
        primary_indices = self.tree_processor.primary_path_indices.to(predictions.device)
        valid_mask = anchor_valid_mask.to(torch.bool)
        count = int(valid_mask.sum().item())
        if primary_indices.numel() <= 1:
            return float(count), count
        target_main_path = labels.index_select(-1, primary_indices)[..., 1:]
        path_indices, path_mask, path_lengths = self._get_acceptance_path_tensors(device=predictions.device)
        if path_mask.shape[-1] == 0:
            return float(count), count

        gathered_predictions = predictions.unsqueeze(-2).expand(-1, -1, path_indices.shape[0], -1).gather(
            -1,
            path_indices.view(1, 1, path_indices.shape[0], path_indices.shape[1]).expand(
                predictions.shape[0],
                predictions.shape[1],
                -1,
                -1,
            ),
        )
        compare_depth = min(gathered_predictions.shape[-1], target_main_path.shape[-1])
        gathered_validity = tree_valid_mask.unsqueeze(-2).expand(-1, -1, path_indices.shape[0], -1).gather(
            -1,
            path_indices.view(1, 1, path_indices.shape[0], path_indices.shape[1]).expand(
                tree_valid_mask.shape[0],
                tree_valid_mask.shape[1],
                -1,
                -1,
            ),
        )
        expanded_path_mask = path_mask.view(1, 1, path_indices.shape[0], path_indices.shape[1])
        prefix_matches = (
            gathered_predictions[..., :compare_depth]
            == target_main_path.unsqueeze(-2)[..., :compare_depth]
        ) | ~expanded_path_mask[..., :compare_depth]
        node_matches = prefix_matches.all(dim=-1)
        node_matches = node_matches & (gathered_validity | ~expanded_path_mask).all(dim=-1)
        node_matches = node_matches & (path_lengths.view(1, 1, -1) <= target_main_path.shape[-1])
        node_matches = node_matches & tree_valid_mask[..., : path_indices.shape[0]]

        accepted_depth = (node_matches * path_lengths.view(1, 1, -1)).amax(dim=-1)
        accepted_valid = accepted_depth.masked_select(valid_mask)
        total = float(accepted_valid.sum().item()) + float(count)
        return total, count

    def fit(self):
        """Train the drafter across all epochs, logging, evaluating, and saving.

        Args:
            None.

        Returns:
            None.
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
                upcoming_optimizer_step = optimizer_step + 1
                compute_acceptance = self._should_log_training_metrics_for_step(upcoming_optimizer_step)
                sync_context = self.fabric.no_backward_sync(
                    self.drafter_model,
                    enabled=not is_final_micro,
                )
                with sync_context:
                    batch_result = self._train_batch(batch, compute_acceptance=compute_acceptance)
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
            step: Optional step number used in the checkpoint directory name.
            tag: Optional named directory that overrides the numeric step name.

        Returns:
            None.
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
        """Restore the drafter, optimizer, and trainer state from disk.

        Args:
            checkpoint_path: Directory containing ``fabric_ckpt.pt`` and the HF
                drafter export.

        Returns:
            None. The trainer state is updated in place.
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
        """Evaluate the drafter over the validation dataloader.

        Args:
            None.

        Returns:
            Dictionary of dataset-level scalar metrics including evaluation loss,
            auxiliary losses, and mean acceptance length.
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
            total_loss += float(batch_result["loss"].item()) * max(batch_result["valid_count"], 1)
            total_q_loss += float(batch_result["q_loss"].item()) * max(batch_result["valid_count"], 1)
            total_ar_loss += float(batch_result["ar_loss"].item()) * max(batch_result["valid_count"], 1)
            total_valid += batch_result["valid_count"]
            total_acceptance += batch_result["acceptance_total"]
            total_acceptance_count += batch_result["acceptance_count"]
            if self.config.dev_run:
                break

        self.drafter_model.train()
        if total_valid == 0:
            eval_loss = 0.0
            eval_q_loss = 0.0
            eval_ar_loss = 0.0
        else:
            eval_loss = total_loss / total_valid
            eval_q_loss = total_q_loss / total_valid
            eval_ar_loss = total_ar_loss / total_valid
        mean_acceptance = total_acceptance / max(total_acceptance_count, 1)
        return {
            "eval/loss": eval_loss,
            "eval/q_loss": eval_q_loss,
            "eval/ar_loss": eval_ar_loss,
            "eval/mean_acceptance_length": mean_acceptance,
        }


def build_parser() -> ArgumentParser:
    """Create the CLI parser for trainer and data configuration.

    Args:
        None.

    Returns:
        Argument parser configured with trainer, data, and model arguments.
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
    """Parse CLI arguments, construct the trainer, and start training.

    Args:
        None.

    Returns:
        None.
    """
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
