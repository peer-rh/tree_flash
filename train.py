"""
Entry point for tree-flash / DFlash training.

Examples
--------
# Tree-flash (default, 8-GPU):
    python train.py

# DFlash mode (linear chain, no subtrees):
    python train.py --dflash

# Single-GPU debug run:
    python train.py --devices 1 --no-compile

# Custom tree shape:
    python train.py --sub-tree-paths 0-1 0-2 1-3 1-4

# Resume from checkpoint:
    python train.py --resume checkpoints/checkpoint_step0010000.pt

# With periodic generation benchmark:
    python train.py --bench-data-path data/bench_prompts.jsonl
"""

from __future__ import annotations

import argparse

from config import TrainConfig
from trainer.trainer import FabricTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train tree-flash (or DFlash) speculative decoding draft model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    p.add_argument("--target-model",     default=TrainConfig.target_model_path,
                   help="HuggingFace name or local path of the frozen target model.")
    p.add_argument("--data-path",        default=TrainConfig.data_path,
                   help="Path to the stage-2 HDF5 dataset.")
    p.add_argument("--output-dir",       default=TrainConfig.output_dir,
                   help="Directory for checkpoints.")
    p.add_argument("--resume",           default=None,
                   help="Full resume: path to a .pt checkpoint (restores model, "
                        "optimizer, scheduler, and step counter).")
    p.add_argument("--draft-checkpoint", default=None,
                   help="Warm-start draft weights from this checkpoint without "
                        "resuming optimizer/scheduler state.")

    # ── Tree shape ────────────────────────────────────────────────────────────
    p.add_argument("--n-subtrees",       type=int, default=TrainConfig.n_subtrees,
                   help="Length of the primary drafted path (tree depth).")
    p.add_argument("--sub-tree-paths",   nargs="*", default=TrainConfig.sub_tree_paths,
                   metavar="EDGE",
                   help="Subtree shape as parent-child edges, e.g. 0-1 0-2 1-3. "
                        "Ignored when --dflash is set.")
    p.add_argument("--dflash",           action="store_true",
                   help="Train in DFlash mode (linear chain, no subtrees). "
                        "Equivalent to --sub-tree-paths with no arguments.")

    # ── Training hyper-params ─────────────────────────────────────────────────
    p.add_argument("--lr",               type=float, default=TrainConfig.lr)
    p.add_argument("--weight-decay",     type=float, default=TrainConfig.weight_decay)
    p.add_argument("--warmup-steps",     type=int,   default=TrainConfig.warmup_steps)
    p.add_argument("--total-steps",      type=int,   default=TrainConfig.total_steps)
    p.add_argument("--batch-size",       type=int,   default=TrainConfig.batch_size,
                   help="Per-device batch size.")
    p.add_argument("--grad-accum",       type=int,   default=TrainConfig.grad_accum,
                   help="Gradient accumulation steps.")
    p.add_argument("--max-grad-norm",    type=float, default=TrainConfig.max_grad_norm)
    p.add_argument("--ctx-len",          type=int,   default=TrainConfig.ctx_len)
    p.add_argument("--ar-loss-weight",   type=float, default=TrainConfig.ar_loss_weight,
                   help="Weight λ for the AR-head cross-entropy loss term.")

    # ── Validation & saving ───────────────────────────────────────────────────
    p.add_argument("--val-loss-every",   type=int, default=TrainConfig.val_loss_every,
                   help="Steps between loss-only validation passes.")
    p.add_argument("--val-spec-every",   type=int, default=TrainConfig.val_spec_every,
                   help="Steps between full spec-decode validation passes.")
    p.add_argument("--val-steps",        type=int, default=TrainConfig.val_steps,
                   help="Number of batches per validation pass.")
    p.add_argument("--save-every",       type=int, default=TrainConfig.save_every,
                   help="Steps between checkpoint saves.")
    p.add_argument("--log-every",        type=int, default=TrainConfig.log_every)

    # ── Benchmark ─────────────────────────────────────────────────────────────
    p.add_argument("--bench-data-path",  default=None,
                   help="JSONL file with {\"prompt\": \"...\"} lines for the periodic "
                        "generation benchmark. Omit to disable.")
    p.add_argument("--bench-n-prompts",           type=int, default=TrainConfig.bench_n_prompts)
    p.add_argument("--bench-max-new-tokens",      type=int, default=TrainConfig.bench_max_new_tokens)
    p.add_argument("--bench-every",               type=int, default=TrainConfig.bench_every,
                   help="Steps between benchmark runs.")
    p.add_argument("--bench-n-candidate-tokens",  type=int, default=None,
                   help="Pruning budget for bench spec-decode. None = full tree.")

    # ── Hardware ──────────────────────────────────────────────────────────────
    p.add_argument("--devices",    type=int, default=TrainConfig.devices,
                   help="Number of GPUs.")
    p.add_argument("--precision",  default=TrainConfig.precision,
                   choices=["bf16-mixed", "16-mixed", "32-true"])

    # ── torch.compile ─────────────────────────────────────────────────────────
    p.add_argument("--no-compile", action="store_true",
                   help="Disable torch.compile (useful for debugging).")
    p.add_argument("--compile-mode", default=TrainConfig.compile_mode,
                   choices=["default", "reduce-overhead", "max-autotune"])

    return p.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        target_model_path=args.target_model,
        draft_checkpoint=args.draft_checkpoint,
        data_path=args.data_path,
        output_dir=args.output_dir,
        ctx_len=args.ctx_len,
        batch_size=args.batch_size,
        n_subtrees=args.n_subtrees,
        sub_tree_paths=[] if args.dflash else (args.sub_tree_paths or []),
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
        log_every=args.log_every,
        bench_data_path=args.bench_data_path,
        bench_n_prompts=args.bench_n_prompts,
        bench_max_new_tokens=args.bench_max_new_tokens,
        bench_every=args.bench_every,
        bench_n_candidate_tokens=args.bench_n_candidate_tokens,
        devices=args.devices,
        precision=args.precision,
        compile=not args.no_compile,
        compile_mode=args.compile_mode,
    )


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    if args.resume is not None:
        trainer = FabricTrainer.load_checkpoint(cfg, args.resume)
    else:
        trainer = FabricTrainer(cfg)
        trainer.setup()

    trainer.fit()


if __name__ == "__main__":
    main()
