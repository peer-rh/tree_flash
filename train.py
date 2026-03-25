"""
Entry point for tree-flash training.

Usage
-----
Single GPU (debug):
    python train.py --devices 1 --compile False

Multi-GPU data parallel (e.g. 8 GPUs):
    python train.py --devices 8

Resume from checkpoint:
    python train.py --resume checkpoints/checkpoint_step0050000.pt

All TrainConfig fields can be overridden as CLI arguments.
Array-valued arguments (sub_tree_paths) are passed as space-separated strings:
    python train.py --sub_tree_paths 01 02 14 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import TrainConfig
from trainer.trainer import FabricTrainer


def parse_args() -> tuple[TrainConfig, str | None]:
    """Parse CLI arguments into a TrainConfig and an optional resume path."""
    parser = argparse.ArgumentParser(description="tree-flash trainer")

    # ── Core paths ───────────────────────────────────────────────────────────
    parser.add_argument("--target_model_path", type=str, default=TrainConfig.target_model_path)
    parser.add_argument("--draft_checkpoint",  type=str, default=None)
    parser.add_argument("--data_path",         type=str, default=TrainConfig.data_path)
    parser.add_argument("--output_dir",        type=str, default=TrainConfig.output_dir)
    parser.add_argument("--resume",            type=str, default=None,
                        help="Path to a checkpoint to resume from")

    # ── Data ─────────────────────────────────────────────────────────────────
    parser.add_argument("--ctx_len",    type=int, default=TrainConfig.ctx_len)
    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)

    # ── Tree ─────────────────────────────────────────────────────────────────
    parser.add_argument("--seq_depth",      type=int,   default=TrainConfig.seq_depth)
    parser.add_argument("--sub_tree_paths", type=str,   nargs="*", default=[],
                        help='e.g. --sub_tree_paths 01 02 14 15')

    # ── Training ─────────────────────────────────────────────────────────────
    parser.add_argument("--lr",             type=float, default=TrainConfig.lr)
    parser.add_argument("--weight_decay",   type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--warmup_steps",   type=int,   default=TrainConfig.warmup_steps)
    parser.add_argument("--total_steps",    type=int,   default=TrainConfig.total_steps)
    parser.add_argument("--grad_accum",     type=int,   default=TrainConfig.grad_accum)
    parser.add_argument("--max_grad_norm",  type=float, default=TrainConfig.max_grad_norm)

    # ── Loss ─────────────────────────────────────────────────────────────────
    parser.add_argument("--ar_loss_weight", type=float, default=TrainConfig.ar_loss_weight)

    # ── Validation ───────────────────────────────────────────────────────────
    parser.add_argument("--val_loss_every", type=int, default=TrainConfig.val_loss_every)
    parser.add_argument("--val_spec_every", type=int, default=TrainConfig.val_spec_every)
    parser.add_argument("--val_steps",      type=int, default=TrainConfig.val_steps)
    parser.add_argument("--save_every",     type=int, default=TrainConfig.save_every)

    # ── Hardware ─────────────────────────────────────────────────────────────
    parser.add_argument("--devices",   type=int, default=TrainConfig.devices)
    parser.add_argument("--precision", type=str, default=TrainConfig.precision)
    parser.add_argument("--log_every", type=int, default=TrainConfig.log_every)

    # ── Compile ──────────────────────────────────────────────────────────────
    parser.add_argument("--compile",      type=lambda x: x.lower() != "false",
                        default=TrainConfig.compile,
                        help="Set to False to disable torch.compile")
    parser.add_argument("--compile_mode", type=str, default=TrainConfig.compile_mode)

    args = parser.parse_args()
    resume = args.resume

    config = TrainConfig(
        target_model_path=args.target_model_path,
        draft_checkpoint=args.draft_checkpoint,
        data_path=args.data_path,
        output_dir=args.output_dir,
        ctx_len=args.ctx_len,
        batch_size=args.batch_size,
        seq_depth=args.seq_depth,
        sub_tree_paths=args.sub_tree_paths,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        grad_accum=args.grad_accum,
        max_grad_norm=args.max_grad_norm,
        ar_loss_weight=args.ar_loss_weight,
        val_loss_every=args.val_loss_every,
        val_spec_every=args.val_spec_every,
        val_steps=args.val_steps,
        save_every=args.save_every,
        devices=args.devices,
        precision=args.precision,
        log_every=args.log_every,
        compile=args.compile,
        compile_mode=args.compile_mode,
    )

    return config, resume


def main() -> None:
    config, resume = parse_args()

    if resume is not None:
        print(f"Resuming from {resume}")
        trainer = FabricTrainer.load_checkpoint(config, resume)
    else:
        trainer = FabricTrainer(config)
        trainer.setup()

    trainer.fit()


if __name__ == "__main__":
    main()
