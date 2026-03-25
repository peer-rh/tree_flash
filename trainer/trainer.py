"""
FabricTrainer: Lightning Fabric-based data-parallel trainer for tree-flash.

Responsibilities
----------------
* Load frozen target model and trainable draft + AR head.
* Build DraftWrapper and optionally torch.compile it.
* Run the training loop with gradient accumulation, LR scheduling,
  periodic loss-validation, and periodic spec-decode validation.
* Save checkpoints (draft state dict + optimizer + scheduler) on rank 0.

Device / parallelism strategy
------------------------------
* Target model: placed on device with .to(); NOT passed to fabric.setup(),
  so DDP never touches its parameters.  Weights are frozen before setup.
* DraftWrapper: passed to fabric.setup(model, optimizer) → DDP-wrapped.
  Only draft + ar_head parameters have requires_grad=True; lm_head and
  embed_tokens inside the wrapper are frozen references (no grad, no DDP sync).

torch.compile
-------------
Applied to DraftWrapper.forward() only.  The target model forward is NOT
compiled (it runs in torch.no_grad(); compiling it is optional and separate).
compile_mode is read from TrainConfig.compile_mode.
"""

from __future__ import annotations

import os
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lightning.fabric import Fabric
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.optimization import get_cosine_schedule_with_warmup

from config import TrainConfig
from tree.spec import TreeSpec
from data.dataset import Stage2Dataset
from model.ar_head import ARHead
from model.draft_wrapper import DraftWrapper
from trainer.loss import compute_loss
from trainer.metrics import TreeMetrics, validate_loss, validate_spec
from trainer.bench import run_bench, load_bench_prompts
from dflash.model.utils import extract_context_feature
from model.draft_model import TreeDraftModel


class FabricTrainer:
    """
    Data-parallel trainer using Lightning Fabric.

    Parameters
    ----------
    config : TrainConfig
    """

    def __init__(self, config: TrainConfig) -> None:
        self.config = config

        self.fabric = Fabric(
            accelerator="gpu",
            strategy="ddp",
            devices=config.devices,
            precision=config.precision,
        )
        self.fabric.launch()

        # Populated by setup()
        self.model: DraftWrapper | None = None
        self.target: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler = None
        self.tree_spec: TreeSpec | None = None
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.target_layer_ids: list[int] | None = None
        self.ancestor_matrix: torch.Tensor | None = None
        self.bench_prompts: list | None = None      # populated by setup() if configured

    # ── Setup ────────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Load models, build tree spec, create dataloaders, wrap with Fabric."""
        cfg = self.config
        device = self.fabric.device

        # ── Tree spec ────────────────────────────────────────────────────────
        self.tree_spec = TreeSpec(
            n_subtrees=cfg.n_subtrees,
            sub_tree_paths=cfg.sub_tree_paths,
        )
        tree_size = self.tree_spec.tree_size
        self.fabric.print(
            f"Tree: n_subtrees={cfg.n_subtrees}, sub_tree_paths={cfg.sub_tree_paths}, "
            f"subtree_size={self.tree_spec.subtree_size}, tree_size={tree_size}"
        )

        # ── Target model (frozen) ────────────────────────────────────────────
        # Loaded in bf16; not passed to fabric.setup() — DDP ignores it.
        target = AutoModelForCausalLM.from_pretrained(
            cfg.target_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        target.requires_grad_(False)
        target.eval()
        target = target.to(device)
        self.target = target

        # ── Draft model ──────────────────────────────────────────────────────
        # Either load from a checkpoint or initialise from the target config.
        draft_config = AutoConfig.from_pretrained(cfg.target_model_path)
        if cfg.draft_checkpoint is not None:
            draft = TreeDraftModel.from_pretrained(cfg.draft_checkpoint)
        else:
            draft = TreeDraftModel(draft_config)
            # Re-use target embeddings to initialise (they're shared at inference)
            draft.post_init()

        self.target_layer_ids = draft.target_layer_ids

        # ── AR head ──────────────────────────────────────────────────────────
        ar_head = ARHead(
            config=draft_config,
            num_draft_layers=len(draft.layers),
        )
        # Inject shared lm_head (frozen reference — not a new parameter)
        ar_head.lm_head = target.lm_head

        # ── DraftWrapper ─────────────────────────────────────────────────────
        model = DraftWrapper(
            draft=draft,
            ar_head=ar_head,
            lm_head=target.lm_head,
            embed_tokens=target.model.embed_tokens,
            mask_token_id=draft.mask_token_id,
            adjusted_parent_ids=self.tree_spec.adjusted_parent_ids,
            ctx_len=cfg.ctx_len,
            position_ids_rel=self.tree_spec.position_ids,
        )

        # Freeze lm_head and embed_tokens inside the wrapper — DDP must not
        # try to synchronise their gradients.
        model.lm_head.requires_grad_(False)
        model.embed_tokens.requires_grad_(False)

        # ── torch.compile ────────────────────────────────────────────────────
        if cfg.compile:
            model = torch.compile(model, mode=cfg.compile_mode)

        # ── Optimizer & scheduler ────────────────────────────────────────────
        # Only parameters with requires_grad=True (draft + ar_head)
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.fabric.print(
            f"Trainable parameters: {sum(p.numel() for p in trainable) / 1e6:.1f}M"
        )
        optimizer = torch.optim.AdamW(
            trainable,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            fused=True,   # fused AdamW is faster and compile-friendly
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.warmup_steps,
            num_training_steps=cfg.total_steps,
        )

        # ── Fabric wrapping ──────────────────────────────────────────────────
        self.model, self.optimizer = self.fabric.setup(model, optimizer)
        self.scheduler = scheduler

        # Ancestor matrix on device — passed to compute_loss at each step.
        # Not a model parameter; placed once at setup.
        self.ancestor_matrix = self.tree_spec.ancestor_matrix.to(device)
        # [tree_size, tree_size] bool

        # ── Dataloaders ──────────────────────────────────────────────────────
        train_ds = Stage2Dataset(cfg.data_path, ctx_len=cfg.ctx_len, n_subtrees=cfg.n_subtrees, split="train")
        val_ds   = Stage2Dataset(cfg.data_path, ctx_len=cfg.ctx_len, n_subtrees=cfg.n_subtrees, split="val")

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,   # keep batch size static for torch.compile
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        self.train_loader, self.val_loader = self.fabric.setup_dataloaders(
            train_loader, val_loader
        )

        # ── Benchmark prompts ────────────────────────────────────────────────
        # Loaded once on rank 0; bench is skipped on other ranks.
        if cfg.bench_data_path is not None and self.fabric.is_global_zero:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(cfg.target_model_path)
            self.bench_prompts = load_bench_prompts(
                cfg.bench_data_path,
                tokenizer,
                cfg.bench_n_prompts,
                cfg.ctx_len,
                device,
            )
            self.fabric.print(
                f"Bench: loaded {len(self.bench_prompts)} prompts "
                f"from {cfg.bench_data_path}"
            )

    # ── Training loop ────────────────────────────────────────────────────────

    def fit(self) -> None:
        """Run the full training loop."""
        cfg = self.config
        device = self.fabric.device
        grad_accum = cfg.grad_accum
        output_dir = Path(cfg.output_dir)

        if self.fabric.is_global_zero:
            output_dir.mkdir(parents=True, exist_ok=True)

        step = 0
        running_loss = 0.0
        self.optimizer.zero_grad()

        # Infinite iterator over training data; we control steps explicitly.
        train_iter = iter(self.train_loader)

        while step < cfg.total_steps:
            # ── Fetch batch ──────────────────────────────────────────────────
            try:
                context_ids, tree_tokens, tree_probs = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                context_ids, tree_tokens, tree_probs = next(train_iter)

            # context_ids  : [B, ctx_len]    int64  (on device via Fabric)
            # tree_tokens  : [B, tree_size]  int64  (-1 = IGNORE_IDX)
            # tree_probs   : [B, tree_size]  float32 (individual AR probs)

            # ── Target conditioning (frozen, no grad) ────────────────────────
            with torch.no_grad():
                target_out = self.target(
                    context_ids,
                    output_hidden_states=True,
                    use_cache=False,
                )
                raw_target_hidden = extract_context_feature(
                    target_out.hidden_states, self.target_layer_ids
                )
                # [B, ctx_len, n_feature_layers * H]

            anchor_ids = context_ids[:, -1]  # [B]

            # ── Gradient accumulation ────────────────────────────────────────
            # Suppress DDP all-reduce on accumulation steps; sync on the last.
            is_accumulating = (step % grad_accum) != (grad_accum - 1)

            with self.fabric.no_backward_sync(self.model, enabled=is_accumulating):
                draft_logits, ar_logits = self.model(
                    anchor_ids, raw_target_hidden, tree_tokens
                )
                # draft_logits: [B, tree_size, V]
                # ar_logits:    [B, tree_size, V]

                loss, draft_loss, ar_loss = compute_loss(
                    draft_logits,
                    ar_logits,
                    tree_tokens,
                    tree_probs,
                    cfg.ar_loss_weight,
                    self.ancestor_matrix,
                )

                # Scale loss by accumulation factor so gradients are correct
                self.fabric.backward(loss / grad_accum)

            running_loss += loss.item()

            # ── Optimizer step ───────────────────────────────────────────────
            if not is_accumulating:
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    cfg.max_grad_norm,
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                step += 1

                # ── Logging ──────────────────────────────────────────────────
                if step % cfg.log_every == 0 and self.fabric.is_global_zero:
                    avg_loss = running_loss / cfg.log_every
                    lr = self.scheduler.get_last_lr()[0]
                    self.fabric.print(
                        f"step={step:6d}  loss={avg_loss:.4f}  "
                        f"draft={draft_loss.item():.4f}  "
                        f"ar={ar_loss.item():.4f}  lr={lr:.2e}"
                    )
                    running_loss = 0.0

                # ── Loss validation ───────────────────────────────────────────
                if step % cfg.val_loss_every == 0:
                    self.model.eval()
                    val_metrics = validate_loss(
                        model=self.model,
                        target=self.target,
                        val_loader=self.val_loader,
                        target_layer_ids=self.target_layer_ids,
                        ar_loss_weight=cfg.ar_loss_weight,
                        ancestor_matrix=self.ancestor_matrix,
                        n_steps=cfg.val_steps,
                        device=device,
                    )
                    self.model.train()
                    if self.fabric.is_global_zero:
                        self.fabric.print(
                            f"[val loss]  step={step}  "
                            + "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                        )

                # ── Spec-decode validation ────────────────────────────────────
                if step % cfg.val_spec_every == 0:
                    self.model.eval()
                    spec_metrics = TreeMetrics()
                    validate_spec(
                        model=self.model,
                        target=self.target,
                        val_loader=self.val_loader,
                        target_layer_ids=self.target_layer_ids,
                        tree_spec=self.tree_spec,
                        n_steps=cfg.val_steps,
                        device=device,
                        metrics=spec_metrics,
                    )
                    self.model.train()
                    if self.fabric.is_global_zero:
                        summary = spec_metrics.summarise(
                            self.tree_spec, self.tree_spec.sibling_pairs
                        )
                        scalars = {
                            k: v for k, v in summary.items()
                            if isinstance(v, (int, float))
                        }
                        self.fabric.print(
                            f"[val spec]  step={step}  "
                            + "  ".join(f"{k}={v:.4f}" for k, v in scalars.items())
                        )

                # ── Benchmark ────────────────────────────────────────────────
                if (
                    step % cfg.bench_every == 0
                    and self.bench_prompts
                    and self.fabric.is_global_zero
                ):
                    self.model.eval()
                    bench_metrics = run_bench(
                        model=self.model,
                        target=self.target,
                        prompt_tokens=self.bench_prompts,
                        tree_spec=self.tree_spec,
                        target_layer_ids=self.target_layer_ids,
                        max_new_tokens=cfg.bench_max_new_tokens,
                        n_candidate_tokens=cfg.bench_n_candidate_tokens,
                    )
                    self.model.train()
                    self.fabric.print(
                        f"[bench]  step={step}  "
                        + "  ".join(f"{k}={v:.3f}" for k, v in bench_metrics.items())
                    )

                # ── Checkpoint ───────────────────────────────────────────────
                if step % cfg.save_every == 0:
                    self._save(step, output_dir)

        # Final checkpoint
        self._save(step, output_dir)
        self.fabric.print("Training complete.")

    # ── Checkpoint helpers ───────────────────────────────────────────────────

    def _save(self, step: int, output_dir: Path) -> None:
        """Save draft + ar_head state dict, optimizer state, and scheduler."""
        state = {
            "step": step,
            "model": self.model,          # Fabric handles unwrapping DDP
            "optimizer": self.optimizer,
            "scheduler": self.scheduler.state_dict(),
            "config": self.config,
        }
        save_path = output_dir / f"checkpoint_step{step:07d}.pt"
        self.fabric.save(save_path, state)
        if self.fabric.is_global_zero:
            self.fabric.print(f"Saved checkpoint: {save_path}")

    @classmethod
    def load_checkpoint(
        cls,
        config: TrainConfig,
        checkpoint_path: str,
    ) -> "FabricTrainer":
        """
        Resume training from a saved checkpoint.

        Parameters
        ----------
        config          : TrainConfig (must match the saved config)
        checkpoint_path : path to a .pt file saved by _save()
        """
        trainer = cls(config)
        trainer.setup()
        state = {
            "model": trainer.model,
            "optimizer": trainer.optimizer,
            "scheduler": None,  # loaded below from state dict
            "step": 0,
        }
        remainder = trainer.fabric.load(checkpoint_path, state)
        if "scheduler" in remainder:
            trainer.scheduler.load_state_dict(remainder["scheduler"])
        return trainer
