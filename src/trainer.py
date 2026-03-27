from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import math

from jsonargparse import ArgumentParser, namespace_to_dict
from lightning.fabric import Fabric
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_pipeline.stage2 import IGNORE_IDX

from .data import DataModuleConfig, PackedBatch, build_dataloaders
from .models import DFlashDraftModel
from .trees import BlockTreeProcessor, BranchOffTreeProcessor


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
    ar_loss_lambda: float = 0.1
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    min_lr: float = 1e-6
    seed: int = 42
    eval_batches: int = 32
    target_attn_implementation: str = "sdpa"
    wandb_project: str = "tree-flash"
    wandb_run_name: str | None = None
    no_wandb: bool = False
    resume_from: str | None = None


def unwrap_model(module):
    raw = module
    if hasattr(raw, "_forward_module"):
        raw = raw._forward_module
    if hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod
    return raw


def build_prefill_attention_mask(
    document_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len = document_mask.shape
    device = document_mask.device
    position_ids = torch.arange(seq_len, device=device)
    q_pos = position_ids.view(1, seq_len, 1)
    k_pos = position_ids.view(1, 1, seq_len)
    same_doc = document_mask.unsqueeze(2) == document_mask.unsqueeze(1)
    causal = k_pos <= q_pos
    key_valid = valid_mask.unsqueeze(1)
    query_valid = valid_mask.unsqueeze(2)
    attend = same_doc & causal & key_valid & query_valid

    eye = torch.eye(seq_len, dtype=torch.bool, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    attend = torch.where(query_valid, attend, eye)

    mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=torch.float32, device=device)
    mask.masked_fill_(~attend.unsqueeze(1), float("-inf"))
    return mask


def build_drafter_block_mask(
    *,
    anchor_positions: torch.Tensor,
    document_mask: torch.Tensor,
    context_valid_mask: torch.Tensor,
    tree_valid_mask: torch.Tensor,
    block_size: int,
):
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


def get_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    lr: float,
    min_lr: float,
) -> float:
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (lr - min_lr) * cosine


class Trainer:
    tree_processor: Any

    def __init__(
        self,
        config: TrainerConfig,
        target: str,
        data: DataModuleConfig,
        drafter: dict[str, Any] | str,
        tree_type: Literal["fixed", "prunable", "block", "every_branch", "loaded"] = "fixed",
        tree_args: dict[str, Any] | None = None,
    ):
        self.config = config
        self.target = target
        self.data_config = data
        self.drafter_source = drafter
        self.tree_type = tree_type
        self.tree_args = tree_args or {}
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

        if tree_type in {"fixed", "block"}:
            self.tree_processor = BlockTreeProcessor(
                tree_seq_depth=data.tree_seq_depth,
                sub_tree_paths=self.tree_args.get("sub_tree_paths"),
            )
        elif tree_type == "branch_off":
            self.tree_processor = BranchOffTreeProcessor(
                tree_seq_depth=data.tree_seq_depth,
                sub_tree_paths=self.tree_args.get("sub_tree_paths"),
                branching_pattern=self.tree_args.get("branching_pattern"),
            )
        else:
            raise NotImplementedError(
                f"tree_type={tree_type!r} is not implemented. Use 'block', 'fixed', or 'branch_off'."
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
            attn_implementation=config.target_attn_implementation,
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

        if config.compile and hasattr(torch, "compile"):
            self.target_model = torch.compile(self.target_model)
            self.drafter_model = torch.compile(self.drafter_model, dynamic=True)

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

    def _init_wandb(self) -> None:
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
        if self.fabric.is_global_zero and self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)

    def _forward_anchor_chunk(
        self,
        batch: PackedBatch,
        target_ctx_features: torch.Tensor,
        anchor_slice: slice,
        *,
        compute_predictions: bool,
    ) -> dict[str, Any]:
        anchor_positions = batch.anchor_positions[:, anchor_slice]
        anchor_valid_mask = batch.anchor_valid_mask[:, anchor_slice]
        if not anchor_valid_mask.any():
            return {
                "loss_sum": target_ctx_features.new_zeros(()),
                "valid_count": 0,
                "predictions": None,
            }

        tree_labels = batch.tree_labels[:, anchor_slice]
        tree_noise_ids = batch.tree_noise_ids[:, anchor_slice]
        tree_position_ids = batch.tree_position_ids[:, anchor_slice]
        tree_cum_probs = batch.tree_cum_probs[:, anchor_slice]
        tree_valid_mask = batch.tree_valid_mask[:, anchor_slice]

        batch_size, num_blocks, block_size = tree_labels.shape
        tree_info = self.tree_processor.build_tree_info(batch_size, num_blocks, tree_labels.device)
        noise_embeddings = self.target_embeddings(tree_noise_ids.reshape(batch_size, num_blocks * block_size))
        drafter_mask = build_drafter_block_mask(
            anchor_positions=anchor_positions,
            document_mask=batch.document_mask,
            context_valid_mask=batch.context_valid_mask,
            tree_valid_mask=tree_valid_mask,
            block_size=block_size,
        )
        draft_hidden_states, _ = self.drafter_model(
            hidden_states=noise_embeddings,
            position_ids=tree_position_ids.reshape(batch_size, num_blocks * block_size),
            tree_info=tree_info,
            attention_mask=drafter_mask,
            target_ctx_features=target_ctx_features,
        )
        valid_mask = (
            tree_valid_mask
            & tree_info.non_root_mask.view(1, 1, -1)
            & anchor_valid_mask.unsqueeze(-1)
        )
        loss_sum, valid_count, predictions = self._chunked_loss_and_predictions(
            hidden_states=draft_hidden_states,
            labels=tree_labels.reshape(batch_size, num_blocks * block_size),
            weights=tree_cum_probs.reshape(batch_size, num_blocks * block_size),
            valid_mask=valid_mask.reshape(batch_size, num_blocks * block_size),
            compute_predictions=compute_predictions,
        )
        if predictions is not None:
            predictions = predictions.view(batch_size, num_blocks, block_size)
        return {
            "loss_sum": loss_sum,
            "valid_count": valid_count,
            "predictions": predictions,
        }

    def _chunked_loss_and_predictions(
        self,
        *,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
        valid_mask: torch.Tensor,
        compute_predictions: bool,
    ) -> tuple[torch.Tensor, int, torch.Tensor | None]:
        batch_size, total_tokens, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.reshape(batch_size * total_tokens, hidden_size)
        flat_labels = labels.reshape(batch_size * total_tokens)
        flat_weights = weights.reshape(batch_size * total_tokens)
        flat_valid = valid_mask.reshape(batch_size * total_tokens)
        if not flat_valid.any():
            return hidden_states.new_zeros(()), 0, None

        chunk_size = self.config.ce_chunk_size or flat_hidden.shape[0]
        total_loss = hidden_states.new_zeros(())
        predictions = (
            torch.full_like(flat_labels, IGNORE_IDX)
            if compute_predictions
            else None
        )
        for start in range(0, flat_hidden.shape[0], chunk_size):
            end = min(start + chunk_size, flat_hidden.shape[0])
            chunk_hidden = flat_hidden[start:end]
            logits = self.target_lm_head(chunk_hidden)
            if compute_predictions and predictions is not None:
                predictions[start:end] = logits.argmax(dim=-1)
            ce = F.cross_entropy(
                logits.float(),
                flat_labels[start:end],
                ignore_index=IGNORE_IDX,
                reduction="none",
            )
            weighted = ce * flat_weights[start:end]
            chunk_valid = flat_valid[start:end]
            if chunk_valid.any():
                total_loss = total_loss + weighted[chunk_valid].sum()
        valid_count = int(flat_valid.sum().item())
        if predictions is not None:
            predictions = predictions.view(batch_size, total_tokens)
        return total_loss, valid_count, predictions

    def _run_batch(
        self,
        batch: PackedBatch,
        *,
        compute_metrics: bool,
    ) -> dict[str, Any]:
        batch = batch.to(self.fabric.device)
        if not batch.context_valid_mask.any() or batch.num_anchors == 0:
            zero = torch.zeros((), device=self.fabric.device)
            return {
                "loss": zero,
                "valid_count": 0,
                "acceptance_total": 0.0,
                "acceptance_count": 0,
            }

        with torch.no_grad():
            prefill_mask = build_prefill_attention_mask(batch.document_mask, batch.context_valid_mask)
            target_out = self.target_model(
                input_ids=batch.input_ids,
                attention_mask=prefill_mask,
                position_ids=batch.position_ids,
                output_hidden_states=True,
                use_cache=False,
            )
            target_ctx_features = self.raw_drafter.extract_ctx_features(target_out.hidden_states)

        total_loss_sum = torch.zeros((), device=self.fabric.device)
        total_valid_count = 0
        acceptance_total = 0.0
        acceptance_count = 0
        anchor_chunk = self.config.anchor_chunk_size or batch.num_anchors

        for start in range(0, batch.num_anchors, anchor_chunk):
            end = min(start + anchor_chunk, batch.num_anchors)
            chunk_result = self._forward_anchor_chunk(
                batch,
                target_ctx_features,
                slice(start, end),
                compute_predictions=compute_metrics,
            )
            total_loss_sum = total_loss_sum + chunk_result["loss_sum"]
            total_valid_count += chunk_result["valid_count"]
            if compute_metrics and chunk_result["predictions"] is not None:
                acceptance_chunk_total, acceptance_chunk_count = self._acceptance_proxy(
                    predictions=chunk_result["predictions"],
                    labels=batch.tree_labels[:, start:end].to(self.fabric.device),
                    anchor_valid_mask=batch.anchor_valid_mask[:, start:end].to(self.fabric.device),
                )
                acceptance_total += acceptance_chunk_total
                acceptance_count += acceptance_chunk_count

        if total_valid_count == 0:
            loss = torch.zeros((), device=self.fabric.device)
        else:
            loss = total_loss_sum / total_valid_count
        return {
            "loss": loss,
            "valid_count": total_valid_count,
            "acceptance_total": acceptance_total,
            "acceptance_count": acceptance_count,
        }

    def _acceptance_proxy(
        self,
        *,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        anchor_valid_mask: torch.Tensor,
    ) -> tuple[float, int]:
        if predictions.numel() == 0:
            return 0.0, 0
        primary_indices = self.tree_processor.primary_path_indices
        if primary_indices.numel() <= 1:
            return 0.0, 0
        future_primary_indices = primary_indices[1:]
        pred_primary = predictions[:, :, future_primary_indices]
        label_primary = labels[:, :, future_primary_indices]

        total = 0.0
        count = 0
        for batch_idx in range(pred_primary.shape[0]):
            for anchor_idx in range(pred_primary.shape[1]):
                if not bool(anchor_valid_mask[batch_idx, anchor_idx].item()):
                    continue
                accepted = 0
                for depth_idx in range(pred_primary.shape[-1]):
                    if pred_primary[batch_idx, anchor_idx, depth_idx] != label_primary[batch_idx, anchor_idx, depth_idx]:
                        break
                    accepted += 1
                total += accepted
                count += 1
        return total, count

    def fit(self):
        total_optimizer_steps = max(
            1,
            math.ceil(len(self.train_loader) * self.config.num_epochs / max(self.config.grad_accum_steps, 1)),
        )
        optimizer_step = self.global_step
        accumulated_loss = 0.0
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
                    batch_result = self._run_batch(batch, compute_metrics=False)
                    loss = batch_result["loss"]
                    scaled_loss = loss / max(self.config.grad_accum_steps, 1)
                    self.fabric.backward(scaled_loss)
                accumulated_loss += float(loss.detach().item())

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
                accumulated_loss = 0.0
                if self.fabric.is_global_zero and (
                    self.config.verbose or self.global_step % self.config.log_every == 0
                ):
                    print(
                        f"step={self.global_step} loss={avg_loss:.4f} lr={lr:.2e}",
                        flush=True,
                    )
                if self.global_step % self.config.log_every == 0:
                    self._log({"train/loss": avg_loss, "train/lr": lr}, self.global_step)

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
        self.drafter_model.eval()
        total_loss = 0.0
        total_valid = 0
        total_acceptance = 0.0
        total_acceptance_count = 0

        for batch_idx, batch in enumerate(self.eval_loader):
            if batch_idx >= self.config.eval_batches:
                break
            batch_result = self._run_batch(batch, compute_metrics=True)
            total_loss += float(batch_result["loss"].item()) * max(batch_result["valid_count"], 1)
            total_valid += batch_result["valid_count"]
            total_acceptance += batch_result["acceptance_total"]
            total_acceptance_count += batch_result["acceptance_count"]
            if self.config.dev_run:
                break

        self.drafter_model.train()
        if total_valid == 0:
            eval_loss = 0.0
        else:
            eval_loss = total_loss / total_valid
        mean_acceptance = total_acceptance / max(total_acceptance_count, 1)
        return {
            "eval/loss": eval_loss,
            "eval/mean_acceptance_length": mean_acceptance,
        }


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train the Tree Flash drafter from Stage 2 HDF5 data.")
    parser.add_class_arguments(TrainerConfig, "trainer")
    parser.add_class_arguments(DataModuleConfig, "data")
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--drafter", type=str, required=True)
    parser.add_argument(
        "--tree_type",
        type=str,
        default="block",
        choices=["fixed", "block", "branch_off", "prunable", "every_branch", "loaded"],
    )
    parser.add_argument("--tree_args", type=dict, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    cfg = parser.parse_args()
    trainer_cfg = TrainerConfig(**namespace_to_dict(cfg.trainer))
    data_cfg = DataModuleConfig(**namespace_to_dict(cfg.data))
    trainer = Trainer(
        config=trainer_cfg,
        target=cfg.target,
        data=data_cfg,
        drafter=cfg.drafter,
        tree_type=cfg.tree_type,
        tree_args=cfg.tree_args,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
