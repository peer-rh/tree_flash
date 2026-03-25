from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    """
    Flat configuration for tree-flash training.

    All shape-determining constants (ctx_len, tree_size) must be consistent
    with the stage-2 dataset and the TreeSpec derived from seq_depth /
    sub_tree_paths.
    """

    # ── Paths ────────────────────────────────────────────────────────────────
    target_model_path: str = "z-lab/dflash-qwen3-8b"
    draft_checkpoint: Optional[str] = None          # None → init from target
    data_path: str = "data/stage2.h5"              # HDF5 stage-2 dataset
    output_dir: str = "checkpoints"

    # ── Data ─────────────────────────────────────────────────────────────────
    ctx_len: int = 512          # fixed context length (padded in stage-2)
    batch_size: int = 4         # per-device batch size

    # ── Tree ─────────────────────────────────────────────────────────────────
    n_subtrees: int = 8          # primary path length used during training
    # Default subtree: root has children 1,2,3; node 1→(4,5); node 2→(6,7)
    # subtree_size=7, tree_size = n_subtrees + n_subtrees*7 = 8*8 = 64
    sub_tree_paths: list[str] = field(
        default_factory=lambda: ["0-1", "0-2", "0-3", "1-4", "1-5", "2-6", "2-7"]
    )

    # ── Training ─────────────────────────────────────────────────────────────
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    total_steps: int = 50_000
    grad_accum: int = 4         # gradient accumulation steps
    max_grad_norm: float = 1.0

    # ── Loss ─────────────────────────────────────────────────────────────────
    ar_loss_weight: float = 0.1     # λ: weight for AR-head loss term

    # ── Validation ───────────────────────────────────────────────────────────
    val_loss_every: int = 250       # steps between loss-only val passes
    val_spec_every: int = 1_000     # steps between full spec-decode val passes
    val_steps: int = 64             # number of batches per val pass
    save_every: int = 5_000

    # ── Benchmark (generation quality) ───────────────────────────────────────
    # bench_data_path: JSONL with {"prompt": "..."} lines (pre-formatted).
    # Set to None to disable.  Only rank-0 runs the benchmark.
    bench_data_path: Optional[str] = None
    bench_n_prompts: int = 20           # prompts loaded at setup time
    bench_max_new_tokens: int = 128
    bench_every: int = 2_000            # steps between benchmark passes
    bench_n_candidate_tokens: Optional[int] = None  # None = no pruning

    # ── Fabric / hardware ────────────────────────────────────────────────────
    devices: int = 8
    precision: str = "bf16-mixed"
    log_every: int = 10

    # ── torch.compile ────────────────────────────────────────────────────────
    compile: bool = True
    compile_mode: str = "default"   # "default" | "reduce-overhead" | "max-autotune"
