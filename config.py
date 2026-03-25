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
    seq_depth: int = 8
    # e.g. ["01","02","03","14","15","26","27"] for a depth-2 sub_tree
    sub_tree_paths: list[str] = field(default_factory=list)

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

    # ── Fabric / hardware ────────────────────────────────────────────────────
    devices: int = 8
    precision: str = "bf16-mixed"
    log_every: int = 10

    # ── torch.compile ────────────────────────────────────────────────────────
    compile: bool = True
    compile_mode: str = "default"   # "default" | "reduce-overhead" | "max-autotune"
