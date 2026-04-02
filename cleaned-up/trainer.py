from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lightning.fabric import Fabric
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import DataConfig, PackedBatch, build_dataloaders
from model import build_dynamic_tree_info_from_batch, load_drafter_model
from utils import cosine_lr, unwrap_model

try:
    from cut_cross_entropy import linear_cross_entropy as cce_linear_cross_entropy
except ImportError:
    cce_linear_cross_entropy = None


def _safe_div(numerator: float, denominator: int) -> float:
    """Return `numerator / denominator` with a zero fallback."""
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


@dataclass
class TrainerConfig:
    """Minimal trainer configuration for the cleaned-up path."""

    num_epochs: int = 1
    eval_every: int = 1024
    log_every: int = 10
    save_every: int = 1024
    precision: str = "bf16-mixed"
    devices: int = 1
    ddp: bool = False
    lr: float = 6e-4
    warmup_steps: int = 128
    grad_accum_steps: int = 1
    checkpoint_path: str = "cleaned-up-checkpoints"
    compile: bool = False
    grad_clip_norm: float = 1.0
    min_lr: float = 1e-6
    seed: int = 42
    eval_batches: int = 32
    profile_steps: int = 0
    prune_loss_lambda: float = 1.0
    use_chunked_cross_entropy: bool = False
    ce_chunk_size: int | None = None
    wandb_project: str = "tree-flash"
    wandb_run_name: str | None = None
    disable_wandb: bool = False
    resume_from: str | None = None


def build_prefill_attention_mask(
    document_mask: torch.Tensor,
    valid_mask: torch.Tensor,
):
    """Build the packed-context mask for target prefill.

    Input shapes:
    - `document_mask`: `(batch_size, ctx_len)`
    - `valid_mask`: `(batch_size, ctx_len)`
    """
    from torch.nn.attention.flex_attention import create_block_mask

    batch_size, seq_len = document_mask.shape

    def mask_mod(b, h, q_idx, kv_idx):
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
    tree_size: int,
):
    """Build the drafter attention mask over packed context + dynamic trees.

    Shapes:
    - `anchor_positions`: `(batch_size, num_anchors)`
    - `document_mask`: `(batch_size, ctx_len)`
    - `context_valid_mask`: `(batch_size, ctx_len)`
    - `tree_valid_mask`: `(batch_size, num_anchors, tree_size)`
    """
    from torch.nn.attention.flex_attention import create_block_mask

    batch_size, num_anchors = anchor_positions.shape
    flat_anchor_positions = anchor_positions.unsqueeze(-1).expand(-1, -1, tree_size).reshape(batch_size, -1)
    flat_tree_valid = tree_valid_mask.reshape(batch_size, -1)
    ctx_len = document_mask.shape[1]
    total_tree_len = flat_anchor_positions.shape[1]
    ctx_clamp_max = max(ctx_len - 1, 0)
    tree_clamp_max = max(total_tree_len - 1, 0)

    def mask_mod(b, h, q_idx, kv_idx):
        q_valid = flat_tree_valid[b, q_idx]
        invalid_self = (~q_valid) & (kv_idx == (ctx_len + q_idx))

        in_ctx = kv_idx < ctx_len
        ctx_idx = kv_idx.clamp(0, ctx_clamp_max)
        q_anchor = flat_anchor_positions[b, q_idx]
        same_doc = document_mask[b, ctx_idx] == document_mask[b, q_anchor]
        causal_ctx = ctx_idx < q_anchor
        ctx_ok = in_ctx & q_valid & same_doc & causal_ctx & context_valid_mask[b, ctx_idx]

        tree_idx = (kv_idx - ctx_len).clamp(0, tree_clamp_max)
        q_block = q_idx // tree_size
        kv_block = tree_idx // tree_size
        same_tree = q_block == kv_block
        tree_ok = (~in_ctx) & q_valid & same_tree & flat_tree_valid[b, tree_idx]
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


class WeightedCrossEntropy:
    """Compute weighted CE with cut-cross-entropy plus argmax predictions."""

    def __init__(
        self,
        *,
        target_lm_head,
        use_chunked_cross_entropy: bool,
        ce_chunk_size: int | None,
    ) -> None:
        self.target_lm_head = target_lm_head
        self.use_chunked_cross_entropy = use_chunked_cross_entropy
        self.ce_chunk_size = ce_chunk_size

    def _project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.target_lm_head(hidden_states.to(self.target_lm_head.weight.dtype))

    def _compute_cce(self, hidden_states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return unreduced CE values using `cut_cross_entropy`.

        Shapes:
        - `hidden_states`: `(num_tokens, hidden_size)`
        - `labels`: `(num_tokens,)`
        - returns: `(num_tokens,)`
        """
        if cce_linear_cross_entropy is None:
            raise ImportError(
                "cut-cross-entropy is required by cleaned-up/trainer.py. "
                "Install the `cut-cross-entropy` package to run the cleaned-up trainer."
            )
        weight = self.target_lm_head.weight
        bias = getattr(self.target_lm_head, "bias", None)
        hidden_states = hidden_states.to(weight.dtype)
        return cce_linear_cross_entropy(
            hidden_states,
            weight,
            labels,
            bias=bias,
            reduction="none",
            impl="cce" if hidden_states.is_cuda else "torch_compile",
        )

    def _compute_standard_ce(self, hidden_states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return unreduced CE values from the dense LM-head logits path."""
        logits = self._project(hidden_states)
        return F.cross_entropy(logits.float(), labels, reduction="none")

    def __call__(
        self,
        *,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        """Return `(loss_sum, valid_count, predictions)` for one batch.

        Shapes:
        - `hidden_states`: `(batch_size, total_tree_tokens, hidden_size)`
        - `labels`: `(batch_size, total_tree_tokens)`
        - `weights`: `(batch_size, total_tree_tokens)`
        - `valid_mask`: `(batch_size, total_tree_tokens)`
        """
        batch_size, total_tree_tokens, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.reshape(batch_size * total_tree_tokens, hidden_size)
        flat_labels = labels.reshape(-1)
        flat_weights = weights.reshape(-1)
        flat_valid = valid_mask.reshape(-1)
        valid_count = int(flat_valid.sum().item())
        if valid_count == 0:
            return hidden_states.new_zeros(()), 0, torch.zeros_like(labels)

        chunk_size = self.ce_chunk_size or flat_hidden.shape[0]
        predictions = torch.empty((flat_hidden.shape[0],), dtype=torch.long, device=flat_hidden.device)
        for start in range(0, flat_hidden.shape[0], chunk_size):
            end = min(start + chunk_size, flat_hidden.shape[0])
            predictions[start:end] = self._project(flat_hidden[start:end]).argmax(dim=-1)
        predictions = predictions.view(batch_size, total_tree_tokens)

        valid_hidden = flat_hidden[flat_valid]
        valid_labels = flat_labels[flat_valid]
        valid_weights = flat_weights[flat_valid]
        if not self.use_chunked_cross_entropy:
            per_token_loss = self._compute_standard_ce(valid_hidden, valid_labels)
            return (per_token_loss * valid_weights).sum(), valid_count, predictions

        loss_sum = hidden_states.new_zeros(())
        valid_chunk_size = self.ce_chunk_size or valid_hidden.shape[0]
        for start in range(0, valid_hidden.shape[0], valid_chunk_size):
            end = min(start + valid_chunk_size, valid_hidden.shape[0])
            per_token_loss = self._compute_cce(valid_hidden[start:end], valid_labels[start:end])
            loss_sum = loss_sum + (per_token_loss * valid_weights[start:end]).sum()
        return loss_sum, valid_count, predictions


class TreeFlashTrainer:
    """Simplified trainer that only supports dynamic Stage 2 v2 training."""

    def __init__(
        self,
        *,
        config: TrainerConfig,
        target: str,
        drafter: dict[str, Any] | str,
        data: DataConfig,
    ) -> None:
        self.config = config
        self.target_name = target
        self.drafter_source = drafter
        self.data_config = data
        self.global_step = 0
        self.wandb_run = None
        self.wandb_run_id: str | None = None

        strategy = "ddp" if config.devices > 1 or config.ddp else "auto"
        self.fabric = Fabric(
            accelerator="auto",
            devices=config.devices,
            strategy=strategy,
            precision=config.precision,
        )
        self.fabric.launch()
        self.fabric.seed_everything(config.seed)

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

        self.drafter_model = load_drafter_model(drafter, torch_dtype=load_dtype)
        self.raw_drafter = unwrap_model(self.drafter_model)
        self.mask_token_id = self.raw_drafter.mask_token_id
        if self.mask_token_id is None:
            raise ValueError("The drafter config must define `dflash_config.mask_token_id`.")

        if config.compile and hasattr(torch, "compile"):
            self.target_model = torch.compile(self.target_model)
            self.drafter_model = torch.compile(self.drafter_model, dynamic=True)
            self._build_prefill_attention_mask = torch.compile(build_prefill_attention_mask)
            self._build_drafter_block_mask = torch.compile(build_drafter_block_mask)
        else:
            self._build_prefill_attention_mask = build_prefill_attention_mask
            self._build_drafter_block_mask = build_drafter_block_mask

        self.optimizer = torch.optim.AdamW(
            self.drafter_model.parameters(),
            lr=config.lr,
            weight_decay=0.01,
        )

        self.target_model = self.fabric.to_device(self.target_model)
        self.drafter_model, self.optimizer = self.fabric.setup(self.drafter_model, self.optimizer)
        self.raw_drafter = unwrap_model(self.drafter_model)
        self.target_embeddings = self.target_model.get_input_embeddings()
        self.target_lm_head = self.target_model.get_output_embeddings()
        if self.target_lm_head is None:
            raise ValueError("Target model must expose an LM head.")

        self.ce_helper = WeightedCrossEntropy(
            target_lm_head=self.target_lm_head,
            use_chunked_cross_entropy=config.use_chunked_cross_entropy,
            ce_chunk_size=config.ce_chunk_size,
        )
        self.train_loader, self.eval_loader = build_dataloaders(
            config=data,
            mask_token_id=self.mask_token_id,
            pad_token_id=self.pad_token_id,
            num_replicas=max(int(getattr(self.fabric, "world_size", 1)), 1),
            rank=int(getattr(self.fabric, "global_rank", 0)),
        )
        self.train_loader, self.eval_loader = self.fabric.setup_dataloaders(
            self.train_loader,
            self.eval_loader,
            move_to_device=False,
            use_distributed_sampler=False,
        )

        if config.resume_from:
            self.load_checkpoint(config.resume_from)
        if self.fabric.is_global_zero:
            self._init_wandb()

    def _init_wandb(self) -> None:
        if self.config.disable_wandb:
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
                "target": self.target_name,
            },
        )
        self.wandb_run_id = self.wandb_run.id

    def _log(self, metrics: dict[str, Any], step: int) -> None:
        if self.fabric.is_global_zero and self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)

    def _count_valid_targets(self, batch: PackedBatch) -> int:
        non_root_mask = batch.tree_parent_indices >= 0
        valid_mask = batch.tree_valid_mask & non_root_mask & batch.anchor_valid_mask.unsqueeze(-1)
        return int(valid_mask.sum().item())

    def _get_target_decoder_layers(self):
        """Return the frozen target decoder layers when the model layout is known."""
        raw_target = unwrap_model(self.target_model)

        decoder = None
        get_decoder = getattr(raw_target, "get_decoder", None)
        if callable(get_decoder):
            decoder = get_decoder()
        if decoder is None:
            decoder = getattr(raw_target, "model", None)
        if decoder is None:
            base_model = getattr(raw_target, "base_model", None)
            decoder = getattr(base_model, "model", None) if base_model is not None else None

        layers = getattr(decoder, "layers", None)
        return layers

    def _prefill_target_context_selected_layers(
        self,
        batch: PackedBatch,
        prefill_mask,
    ) -> torch.Tensor | None:
        """Capture only the verifier layers requested by the drafter, when possible."""
        target_layer_ids = getattr(self.raw_drafter, "target_layer_ids", None)
        if not target_layer_ids:
            return None

        decoder_layers = self._get_target_decoder_layers()
        if decoder_layers is None:
            return None

        layer_ids = [int(layer_id) for layer_id in target_layer_ids]
        if any(layer_id < 0 or layer_id >= len(decoder_layers) for layer_id in layer_ids):
            return None

        captured_states: dict[int, torch.Tensor] = {}
        hook_handles = []

        def make_hook(layer_idx: int):
            def hook(_module, _inputs, output):
                hidden_states = output[0] if isinstance(output, tuple) else output
                captured_states[layer_idx] = hidden_states

            return hook

        for layer_idx in sorted(set(layer_ids)):
            hook_handles.append(decoder_layers[layer_idx].register_forward_hook(make_hook(layer_idx)))

        raw_target = unwrap_model(self.target_model)
        try:
            raw_target(
                input_ids=batch.input_ids,
                attention_mask=prefill_mask,
                position_ids=batch.position_ids,
                output_hidden_states=False,
                use_cache=False,
            )
        finally:
            for handle in hook_handles:
                handle.remove()

        if any(layer_idx not in captured_states for layer_idx in layer_ids):
            return None
        return torch.cat([captured_states[layer_idx] for layer_idx in layer_ids], dim=-1)

    def _prefill_target_context(self, batch: PackedBatch) -> torch.Tensor:
        """Prefill the frozen target on packed context tokens.

        Output shape:
        - `(batch_size, ctx_len, target_feature_dim)`
        """
        with torch.no_grad():
            prefill_mask = self._build_prefill_attention_mask(batch.document_mask, batch.context_valid_mask)
            selected_ctx_features = self._prefill_target_context_selected_layers(batch, prefill_mask)
            if selected_ctx_features is not None:
                return selected_ctx_features

            target_out = self.target_model(
                input_ids=batch.input_ids,
                attention_mask=prefill_mask,
                position_ids=batch.position_ids,
                output_hidden_states=True,
                use_cache=False,
            )
            return self.raw_drafter.extract_ctx_features(target_out.hidden_states)

    def _build_prune_targets(self, tree_info) -> torch.Tensor:
        """Mark nodes whose full ancestor chain stays on the primary path."""
        off_primary_ancestors = tree_info.tree_mask & (~tree_info.primary_path_mask.unsqueeze(-2))
        return (~off_primary_ancestors.any(dim=-1)).to(torch.float32)

    def _compute_prune_loss_sum(
        self,
        *,
        q_logits: torch.Tensor,
        q_targets: torch.Tensor,
        flat_valid_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute focal BCE for prune-head supervision on valid non-root nodes."""
        q_logits = q_logits.float()
        bce = F.binary_cross_entropy_with_logits(q_logits, q_targets, reduction="none")
        probs = torch.sigmoid(q_logits)
        pt = torch.where(q_targets > 0.5, probs, 1.0 - probs)
        focal_factor = (1.0 - pt).pow(2.0)
        prune_loss = bce * focal_factor
        return prune_loss[flat_valid_mask].sum().to(dtype)

    def _empty_tree_metric_totals(self) -> dict[str, float | int]:
        return {
            "tree_match_count": 0,
            "tree_total_count": 0,
            "main_match_count": 0,
            "main_total_count": 0,
            "acceptance_depth_sum": 0.0,
            "acceptance_anchor_count": 0,
            "sibling_collision_count": 0,
            "sibling_pair_count": 0,
            "useful_anchor_count": 0,
            "usefulness_anchor_count": 0,
        }

    def _compute_tree_metric_totals(
        self,
        *,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
        tree_cum_probs: torch.Tensor,
        tree_valid_mask: torch.Tensor,
        anchor_valid_mask: torch.Tensor,
        tree_info,
    ) -> dict[str, float | int]:
        """Compute tree-structured metric totals from one batch."""
        totals = self._empty_tree_metric_totals()
        totals["tree_match_count"] = int((predictions.eq(labels) & valid_mask).sum().item())
        totals["tree_total_count"] = int(valid_mask.sum().item())
        batch_size, num_anchors, tree_size = predictions.shape
        device = predictions.device
        depth = tree_info.depth.clamp(0, tree_size - 1)
        score_scale = tree_cum_probs.abs().amax().to(torch.float32) + 1.0
        ranking_score = depth.to(torch.float32) * score_scale + tree_cum_probs.to(torch.float32)

        off_main_candidates = valid_mask & (~tree_info.primary_path_mask)
        primary_candidates = tree_valid_mask & tree_info.primary_path_mask

        neg_inf = torch.full_like(ranking_score, float("-inf"))
        off_main_scores = torch.where(off_main_candidates, ranking_score, neg_inf)
        primary_scores = torch.where(primary_candidates, ranking_score, neg_inf)
        has_off_main = off_main_candidates.any(dim=-1)
        selected_off_main = off_main_scores.argmax(dim=-1)
        selected_primary = primary_scores.argmax(dim=-1)
        basis_endpoint = torch.where(has_off_main, selected_off_main, selected_primary)

        basis_endpoint_index = basis_endpoint.unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_anchors, 1, tree_size)
        basis_path_mask = tree_info.tree_mask.gather(2, basis_endpoint_index).squeeze(2) & tree_valid_mask

        depth_one_hot = F.one_hot(depth, num_classes=tree_size).to(torch.bool)
        basis_depth_mask = basis_path_mask.unsqueeze(-1) & depth_one_hot
        basis_depth_exists = basis_depth_mask.any(dim=2)
        basis_target_by_depth = (basis_depth_mask.to(labels.dtype) * labels.unsqueeze(-1)).sum(dim=2)

        node_basis_target = basis_target_by_depth.gather(2, depth)
        node_has_basis_target = basis_depth_exists.gather(2, depth)

        main_comparable_mask = valid_mask & node_has_basis_target
        totals["main_total_count"] = int(main_comparable_mask.sum().item())
        totals["main_match_count"] = int((predictions.eq(node_basis_target) & main_comparable_mask).sum().item())

        node_matches_basis = tree_valid_mask & node_has_basis_target & predictions.eq(node_basis_target)
        ancestor_failure = tree_info.tree_mask & (~node_matches_basis.unsqueeze(-2))
        ancestor_chain_ok = ~ancestor_failure.any(dim=-1)
        accepted_node_mask = valid_mask & ancestor_chain_ok

        best_depth_per_anchor = torch.where(
            accepted_node_mask,
            depth,
            torch.zeros_like(depth),
        ).amax(dim=-1)
        anchor_valid_float = anchor_valid_mask.to(torch.float32)
        totals["acceptance_anchor_count"] = int(anchor_valid_mask.sum().item())
        totals["usefulness_anchor_count"] = int(anchor_valid_mask.sum().item())
        totals["acceptance_depth_sum"] = float(
            (best_depth_per_anchor.to(torch.float32) * anchor_valid_float).sum().item()
        )

        best_depth_mask = accepted_node_mask & depth.eq(best_depth_per_anchor.unsqueeze(-1))
        useful_anchor_mask = (best_depth_mask & (~basis_path_mask)).any(dim=-1) & anchor_valid_mask
        totals["useful_anchor_count"] = int(useful_anchor_mask.sum().item())

        parent_idx = tree_info.parent_idx
        upper_triangle = torch.triu(
            torch.ones((tree_size, tree_size), dtype=torch.bool, device=device),
            diagonal=1,
        ).view(1, 1, tree_size, tree_size)
        sibling_pair_mask = (
            valid_mask.unsqueeze(-1)
            & valid_mask.unsqueeze(-2)
            & parent_idx.unsqueeze(-1).eq(parent_idx.unsqueeze(-2))
            & parent_idx.unsqueeze(-1).ge(0)
            & upper_triangle
        )
        collision_mask = sibling_pair_mask & predictions.unsqueeze(-1).eq(predictions.unsqueeze(-2))
        totals["sibling_pair_count"] = int(sibling_pair_mask.sum().item())
        totals["sibling_collision_count"] = int(collision_mask.sum().item())
        return totals

    def _forward_batch(self, batch: PackedBatch, *, profile: bool) -> dict[str, Any]:
        batch = batch.to(self.fabric.device)
        zero = torch.zeros((), device=self.fabric.device)
        if not batch.context_valid_mask.any() or batch.num_anchors == 0:
            return {
                "loss": zero,
                "ce_loss": zero,
                "prune_loss": zero,
                "valid_count": 0,
                **self._empty_tree_metric_totals(),
                "prefill_time": 0.0,
                "mask_time": 0.0,
                "drafter_time": 0.0,
                "ce_time": 0.0,
            }

        valid_count = self._count_valid_targets(batch)
        if valid_count == 0:
            return {
                "loss": zero,
                "ce_loss": zero,
                "prune_loss": zero,
                "valid_count": 0,
                **self._empty_tree_metric_totals(),
                "prefill_time": 0.0,
                "mask_time": 0.0,
                "drafter_time": 0.0,
                "ce_time": 0.0,
            }

        prefill_start = time.perf_counter() if profile else 0.0
        target_ctx_features = self._prefill_target_context(batch)
        prefill_time = time.perf_counter() - prefill_start if profile else 0.0

        batch_size, num_anchors, tree_size = batch.tree_labels.shape
        tree_info = build_dynamic_tree_info_from_batch(
            tree_parent_indices=batch.tree_parent_indices,
            tree_depths=batch.tree_depths,
            tree_node_ranks=batch.tree_node_ranks,
            tree_position_ids=batch.tree_position_ids,
            tree_valid_mask=batch.tree_valid_mask,
            tree_primary_path_mask=batch.tree_primary_path_mask,
        )
        non_root_mask = tree_info.non_root_mask
        valid_mask = batch.tree_valid_mask & non_root_mask & batch.anchor_valid_mask.unsqueeze(-1)
        flat_valid_mask = valid_mask.reshape(batch_size, num_anchors * tree_size)

        noise_embeddings = self.target_embeddings(batch.tree_noise_ids.reshape(batch_size, num_anchors * tree_size))
        mask_start = time.perf_counter() if profile else 0.0
        drafter_mask = self._build_drafter_block_mask(
            anchor_positions=batch.anchor_positions,
            document_mask=batch.document_mask,
            context_valid_mask=batch.context_valid_mask,
            tree_valid_mask=batch.tree_valid_mask,
            tree_size=tree_size,
        )
        position_ids = torch.cat(
            [batch.position_ids, batch.tree_position_ids.reshape(batch_size, num_anchors * tree_size)],
            dim=1,
        )
        mask_time = time.perf_counter() - mask_start if profile else 0.0

        drafter_start = time.perf_counter() if profile else 0.0
        draft_hidden_states, _, q_logits = self.drafter_model(
            hidden_states=noise_embeddings,
            position_ids=position_ids,
            tree_info=tree_info,
            attention_mask=drafter_mask,
            target_ctx_features=target_ctx_features,
            return_aux=True,
        )
        drafter_time = time.perf_counter() - drafter_start if profile else 0.0

        ce_start = time.perf_counter() if profile else 0.0
        flat_labels = batch.tree_labels.reshape(batch_size, num_anchors * tree_size)
        flat_weights = batch.tree_cum_probs.reshape(batch_size, num_anchors * tree_size)
        loss_sum, valid_count, predictions = self.ce_helper(
            hidden_states=draft_hidden_states,
            labels=flat_labels,
            weights=flat_weights,
            valid_mask=flat_valid_mask,
        )
        q_targets = self._build_prune_targets(tree_info).reshape(batch_size, num_anchors * tree_size)
        prune_loss_sum = self._compute_prune_loss_sum(
            q_logits=q_logits,
            q_targets=q_targets,
            flat_valid_mask=flat_valid_mask,
            dtype=draft_hidden_states.dtype,
        )
        ce_time = time.perf_counter() - ce_start if profile else 0.0

        ce_loss = loss_sum / max(valid_count, 1)
        prune_loss = prune_loss_sum / max(valid_count, 1)
        loss = ce_loss + self.config.prune_loss_lambda * prune_loss
        predictions = predictions.view(batch_size, num_anchors, tree_size)
        metric_totals = self._compute_tree_metric_totals(
            predictions=predictions,
            labels=batch.tree_labels,
            valid_mask=valid_mask,
            tree_cum_probs=batch.tree_cum_probs,
            tree_valid_mask=batch.tree_valid_mask,
            anchor_valid_mask=batch.anchor_valid_mask,
            tree_info=tree_info,
        )
        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "prune_loss": prune_loss,
            "valid_count": valid_count,
            **metric_totals,
            "prefill_time": prefill_time,
            "mask_time": mask_time,
            "drafter_time": drafter_time,
            "ce_time": ce_time,
        }

    @torch.no_grad()
    def _eval_batch(self, batch: PackedBatch) -> dict[str, Any]:
        return self._forward_batch(batch, profile=False)

    def save_checkpoint(self, *, step: int | None = None, tag: str | None = None) -> None:
        checkpoint_dir = self.output_dir / (tag if tag is not None else f"checkpoint-{step or self.global_step}")
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

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint_dir = Path(checkpoint_path)
        remainder = self.fabric.load(
            str(checkpoint_dir / "fabric_ckpt.pt"),
            {"drafter_model": self.drafter_model, "optimizer": self.optimizer},
        )
        self.global_step = int(remainder.get("global_step", 0))
        self.wandb_run_id = remainder.get("wandb_run_id")

    @torch.inference_mode()
    def validate(self) -> dict[str, float]:
        self.drafter_model.eval()
        total_loss = 0.0
        total_ce = 0.0
        total_prune = 0.0
        total_valid = 0
        total_tree_matches = 0
        total_tree_count = 0
        total_main_matches = 0
        total_main_count = 0
        total_acceptance_depth = 0.0
        total_acceptance_anchors = 0
        total_sibling_collisions = 0
        total_sibling_pairs = 0
        total_useful_anchors = 0
        total_usefulness_anchors = 0
        for batch_idx, batch in enumerate(self.eval_loader):
            if batch_idx >= self.config.eval_batches:
                break
            result = self._eval_batch(batch)
            valid_count = int(result["valid_count"])
            total_valid += valid_count
            total_loss += float(result["loss"].item()) * valid_count
            total_ce += float(result["ce_loss"].item()) * valid_count
            total_prune += float(result["prune_loss"].item()) * valid_count
            total_tree_matches += int(result["tree_match_count"])
            total_tree_count += int(result["tree_total_count"])
            total_main_matches += int(result["main_match_count"])
            total_main_count += int(result["main_total_count"])
            total_acceptance_depth += float(result["acceptance_depth_sum"])
            total_acceptance_anchors += int(result["acceptance_anchor_count"])
            total_sibling_collisions += int(result["sibling_collision_count"])
            total_sibling_pairs += int(result["sibling_pair_count"])
            total_useful_anchors += int(result["useful_anchor_count"])
            total_usefulness_anchors += int(result["usefulness_anchor_count"])

        reduced = self.fabric.all_reduce(
            torch.tensor(
                [
                    total_loss,
                    total_ce,
                    total_prune,
                    float(total_valid),
                    float(total_tree_matches),
                    float(total_tree_count),
                    float(total_main_matches),
                    float(total_main_count),
                    total_acceptance_depth,
                    float(total_acceptance_anchors),
                    float(total_sibling_collisions),
                    float(total_sibling_pairs),
                    float(total_useful_anchors),
                    float(total_usefulness_anchors),
                ],
                dtype=torch.float64,
                device=self.fabric.device,
            ),
            reduce_op="sum",
        ) if getattr(self.fabric, "world_size", 1) > 1 else torch.tensor(
            [
                total_loss,
                total_ce,
                total_prune,
                float(total_valid),
                float(total_tree_matches),
                float(total_tree_count),
                float(total_main_matches),
                float(total_main_count),
                total_acceptance_depth,
                float(total_acceptance_anchors),
                float(total_sibling_collisions),
                float(total_sibling_pairs),
                float(total_useful_anchors),
                float(total_usefulness_anchors),
            ],
            dtype=torch.float64,
            device=self.fabric.device,
        )
        total_valid = int(round(float(reduced[3].item())))
        tree_accuracy = _safe_div(float(reduced[4].item()), int(round(float(reduced[5].item()))))
        main_accuracy = _safe_div(float(reduced[6].item()), int(round(float(reduced[7].item()))))
        acceptance_proxy = _safe_div(float(reduced[8].item()), int(round(float(reduced[9].item()))))
        sibling_collision_rate = _safe_div(float(reduced[10].item()), int(round(float(reduced[11].item()))))
        tree_usefulness = _safe_div(float(reduced[12].item()), int(round(float(reduced[13].item()))))
        metrics = {
            "eval/loss": float(reduced[0].item()) / max(total_valid, 1),
            "eval/ce_loss": float(reduced[1].item()) / max(total_valid, 1),
            "eval/prune_loss": float(reduced[2].item()) / max(total_valid, 1),
            "eval/tree_accuracy": tree_accuracy,
            "eval/token_accuracy": tree_accuracy,
            "eval/main_accuracy": main_accuracy,
            "eval/acceptance_proxy": acceptance_proxy,
            "eval/sibling_collision_rate": sibling_collision_rate,
            "eval/tree_usefulness": tree_usefulness,
        }
        self.drafter_model.train()
        return metrics

    def fit(self) -> None:
        total_optimizer_steps = max(
            1,
            math.ceil(len(self.train_loader) * self.config.num_epochs / max(self.config.grad_accum_steps, 1)),
        )
        optimizer_step = self.global_step
        micro_step = 0
        accumulated_loss = 0.0
        accumulated_ce = 0.0
        accumulated_prune = 0.0
        accumulated_tree_matches = 0
        accumulated_tree_count = 0
        accumulated_main_matches = 0
        accumulated_main_count = 0
        accumulated_acceptance_depth = 0.0
        accumulated_acceptance_anchors = 0
        accumulated_sibling_collisions = 0
        accumulated_sibling_pairs = 0
        accumulated_useful_anchors = 0
        accumulated_usefulness_anchors = 0
        accumulated_prefill = 0.0
        accumulated_mask = 0.0
        accumulated_drafter = 0.0
        accumulated_ce_time = 0.0

        for _epoch in range(self.config.num_epochs):
            self.drafter_model.train()
            for batch in self.train_loader:
                micro_step += 1
                is_final_micro = micro_step % self.config.grad_accum_steps == 0
                profile = self.config.profile_steps > 0 and self.global_step < self.config.profile_steps
                sync_context = self.fabric.no_backward_sync(self.drafter_model, enabled=not is_final_micro)
                with sync_context:
                    result = self._forward_batch(batch, profile=profile)
                    scaled_loss = result["loss"] / max(self.config.grad_accum_steps, 1)
                    self.fabric.backward(scaled_loss)

                accumulated_loss += float(result["loss"].detach().item())
                accumulated_ce += float(result["ce_loss"].detach().item())
                accumulated_prune += float(result["prune_loss"].detach().item())
                accumulated_tree_matches += int(result["tree_match_count"])
                accumulated_tree_count += int(result["tree_total_count"])
                accumulated_main_matches += int(result["main_match_count"])
                accumulated_main_count += int(result["main_total_count"])
                accumulated_acceptance_depth += float(result["acceptance_depth_sum"])
                accumulated_acceptance_anchors += int(result["acceptance_anchor_count"])
                accumulated_sibling_collisions += int(result["sibling_collision_count"])
                accumulated_sibling_pairs += int(result["sibling_pair_count"])
                accumulated_useful_anchors += int(result["useful_anchor_count"])
                accumulated_usefulness_anchors += int(result["usefulness_anchor_count"])
                accumulated_prefill += float(result["prefill_time"])
                accumulated_mask += float(result["mask_time"])
                accumulated_drafter += float(result["drafter_time"])
                accumulated_ce_time += float(result["ce_time"])

                if not is_final_micro:
                    continue

                optimizer_step += 1
                lr = cosine_lr(
                    optimizer_step,
                    warmup_steps=self.config.warmup_steps,
                    total_steps=total_optimizer_steps,
                    max_lr=self.config.lr,
                    min_lr=self.config.min_lr,
                )
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                self.fabric.clip_gradients(self.drafter_model, self.optimizer, max_norm=self.config.grad_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step = optimizer_step

                denom = max(self.config.grad_accum_steps, 1)
                tree_accuracy = _safe_div(accumulated_tree_matches, accumulated_tree_count)
                main_accuracy = _safe_div(accumulated_main_matches, accumulated_main_count)
                acceptance_proxy = _safe_div(accumulated_acceptance_depth, accumulated_acceptance_anchors)
                sibling_collision_rate = _safe_div(accumulated_sibling_collisions, accumulated_sibling_pairs)
                tree_usefulness = _safe_div(accumulated_useful_anchors, accumulated_usefulness_anchors)
                train_metrics = {
                    "train/loss": accumulated_loss / denom,
                    "train/ce_loss": accumulated_ce / denom,
                    "train/prune_loss": accumulated_prune / denom,
                    "train/tree_accuracy": tree_accuracy,
                    "train/token_accuracy": tree_accuracy,
                    "train/main_accuracy": main_accuracy,
                    "train/acceptance_proxy": acceptance_proxy,
                    "train/sibling_collision_rate": sibling_collision_rate,
                    "train/tree_usefulness": tree_usefulness,
                    "train/lr": lr,
                }
                if profile:
                    train_metrics.update(
                        {
                            "profile/train_prefill_s": accumulated_prefill / denom,
                            "profile/train_mask_s": accumulated_mask / denom,
                            "profile/train_drafter_s": accumulated_drafter / denom,
                            "profile/train_ce_s": accumulated_ce_time / denom,
                        }
                    )
                if self.fabric.is_global_zero and self.config.log_every > 0 and self.global_step % self.config.log_every == 0:
                    print(
                        f"step={self.global_step} loss={train_metrics['train/loss']:.4f} "
                        f"ce={train_metrics['train/ce_loss']:.4f} prune={train_metrics['train/prune_loss']:.4f} "
                        f"tree_acc={train_metrics['train/tree_accuracy']:.3f} "
                        f"main_acc={train_metrics['train/main_accuracy']:.3f} "
                        f"accept={train_metrics['train/acceptance_proxy']:.3f} "
                        f"sib_coll={train_metrics['train/sibling_collision_rate']:.3f} "
                        f"useful={train_metrics['train/tree_usefulness']:.3f} lr={lr:.2e}",
                        flush=True,
                    )
                if self.config.log_every > 0 and self.global_step % self.config.log_every == 0:
                    self._log(train_metrics, self.global_step)

                accumulated_loss = 0.0
                accumulated_ce = 0.0
                accumulated_prune = 0.0
                accumulated_tree_matches = 0
                accumulated_tree_count = 0
                accumulated_main_matches = 0
                accumulated_main_count = 0
                accumulated_acceptance_depth = 0.0
                accumulated_acceptance_anchors = 0
                accumulated_sibling_collisions = 0
                accumulated_sibling_pairs = 0
                accumulated_useful_anchors = 0
                accumulated_usefulness_anchors = 0
                accumulated_prefill = 0.0
                accumulated_mask = 0.0
                accumulated_drafter = 0.0
                accumulated_ce_time = 0.0

                if self.config.eval_every > 0 and self.global_step % self.config.eval_every == 0:
                    metrics = self.validate()
                    if self.fabric.is_global_zero:
                        print(
                            f"[eval step={self.global_step}] loss={metrics['eval/loss']:.4f} "
                            f"tree_acc={metrics['eval/tree_accuracy']:.3f} "
                            f"main_acc={metrics['eval/main_accuracy']:.3f} "
                            f"accept={metrics['eval/acceptance_proxy']:.3f} "
                            f"sib_coll={metrics['eval/sibling_collision_rate']:.3f} "
                            f"useful={metrics['eval/tree_usefulness']:.3f}",
                            flush=True,
                        )
                    self._log(metrics, self.global_step)

                if self.config.save_every > 0 and self.global_step % self.config.save_every == 0:
                    self.save_checkpoint(step=self.global_step)

        self.save_checkpoint(tag="final")
        if self.fabric.is_global_zero and self.wandb_run is not None:
            self.wandb_run.finish()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the cleaned-up Tree Flash drafter.")
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--drafter", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--eval-path", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default="cleaned-up-checkpoints")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pack-length", type=int, default=3072)
    parser.add_argument("--num-anchors", type=int, default=8)
    parser.add_argument("--training-tree-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--use-chunked-cross-entropy", action="store_true")
    parser.add_argument("--ce-chunk-size", type=int, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="tree-flash")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    drafter: dict[str, Any] | str = args.drafter
    try:
        drafter = json.loads(args.drafter)
    except json.JSONDecodeError:
        drafter = args.drafter

    trainer = TreeFlashTrainer(
        config=TrainerConfig(
            num_epochs=args.num_epochs,
            checkpoint_path=args.checkpoint_path,
            lr=args.lr,
            devices=args.devices,
            precision=args.precision,
            compile=args.compile,
            use_chunked_cross_entropy=args.use_chunked_cross_entropy,
            ce_chunk_size=args.ce_chunk_size,
            disable_wandb=args.disable_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
        ),
        target=args.target,
        drafter=drafter,
        data=DataConfig(
            path=args.data_path,
            eval_path=args.eval_path,
            batch_size=args.batch_size,
            pack_length=args.pack_length,
            num_anchors=args.num_anchors,
            training_tree_size=args.training_tree_size,
        ),
    )
    trainer.fit()


if __name__ == "__main__":
    main()
