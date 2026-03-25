"""
Benchmark harness for tree-flash / DFlash training.

Measures actual multi-step speculative decoding quality on a fixed set of
held-out prompts.  Designed to run periodically during training (rank 0 only)
to track acceptance length as a function of training progress.

Metrics reported
----------------
bench_acceptance_length  : mean tokens accepted per decode step (incl. bonus)
                           target = n_subtrees + 1 (perfect acceptance)
                           baseline = 1.0 (target-only greedy)
bench_steps              : total decode steps across all prompts (sanity check)

DFlash mode
-----------
With sub_tree_paths=[] the tree degenerates to a linear chain; tree acceptance
reduces to standard linear speculative decoding.  The same function works for
both modes.

Usage (standalone)
------------------
    python -m trainer.bench \\
        --model  Qwen/Qwen3-8B \\
        --checkpoint checkpoints/checkpoint_step0010000.pt \\
        --prompts data/bench_prompts.jsonl \\
        --n-prompts 20 --max-new-tokens 128
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer

from tree.spec import TreeSpec
from trainer.spec_decode import tree_spec_decode


# ── Core benchmark function ───────────────────────────────────────────────────

@torch.no_grad()
def run_bench(
    model: nn.Module,
    target: nn.Module,
    prompt_tokens: list[Tensor],        # list of [1, S] int64, already on device
    tree_spec: TreeSpec,
    target_layer_ids: list[int],
    max_new_tokens: int = 128,
    n_candidate_tokens: int | None = None,
    temperature: float = 0.0,
) -> dict[str, float]:
    """
    Run multi-step tree speculative decoding on a fixed prompt set.

    Parameters
    ----------
    model               : DraftWrapper in eval mode
    target              : frozen target LM
    prompt_tokens       : pre-tokenized prompts, each [1, S] on device
    tree_spec           : TreeSpec describing the tree shape
    target_layer_ids    : layer indices for extract_context_feature
    max_new_tokens      : max tokens generated per prompt
    n_candidate_tokens  : pruning budget (None = full tree)
    temperature         : 0.0 for greedy acceptance

    Returns
    -------
    dict with keys:
        bench_acceptance_length  float  — mean accepted tokens per decode step
        bench_steps              int    — total steps across all prompts
    """
    was_training = model.training
    model.eval()

    total_acceptance = 0.0
    total_steps = 0

    for ctx in prompt_tokens:
        _, stats = tree_spec_decode(
            context_ids=ctx,
            model=model,
            target=target,
            tree_spec=tree_spec,
            target_layer_ids=target_layer_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            n_candidate_tokens=n_candidate_tokens,
        )
        for s in stats:
            total_acceptance += s["acceptance_length"]
        total_steps += len(stats)

    if was_training:
        model.train()

    denom = max(total_steps, 1)
    return {
        "bench_acceptance_length": total_acceptance / denom,
        "bench_steps": total_steps,
    }


# ── Prompt loading ────────────────────────────────────────────────────────────

def load_bench_prompts(
    path: str,
    tokenizer,
    n_prompts: int,
    ctx_len: int,
    device: torch.device,
) -> list[Tensor]:
    """
    Load up to ``n_prompts`` prompts from a JSONL file and tokenize them.

    Each line must be a JSON object with a ``"prompt"`` key containing a
    pre-formatted prompt string (same format as stage-1 output).

    Sequences longer than ctx_len are left-truncated.

    Returns
    -------
    list of [1, S] int64 tensors on ``device``, S <= ctx_len
    """
    prompts: list[Tensor] = []
    with open(path) as f:
        for line in f:
            if len(prompts) >= n_prompts:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("prompt", "")
            if not text:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            # Left-truncate to ctx_len
            if len(ids) > ctx_len:
                ids = ids[-ctx_len:]
            t = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            prompts.append(t)
    return prompts


# ── Standalone CLI ────────────────────────────────────────────────────────────

def _main() -> None:
    import argparse
    from transformers import AutoModelForCausalLM, AutoConfig
    from config import TrainConfig
    from model.draft_model import TreeDraftModel
    from model.ar_head import ARHead
    from model.draft_wrapper import DraftWrapper
    from trainer.trainer import FabricTrainer

    parser = argparse.ArgumentParser(
        description="Standalone benchmark: measure acceptance length from a checkpoint."
    )
    parser.add_argument("--model",       required=True, help="Target model HF name or path")
    parser.add_argument("--checkpoint",  required=True, help="Path to .pt checkpoint")
    parser.add_argument("--prompts",     required=True, help="JSONL bench prompts file")
    parser.add_argument("--n-prompts",   type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--n-subtrees",  type=int, default=8)
    parser.add_argument("--sub-tree-paths", nargs="*",
                        default=["0-1","0-2","0-3","1-4","1-5","2-6","2-7"])
    parser.add_argument("--n-candidate-tokens", type=int, default=None,
                        help="Pruning budget (None = full tree)")
    parser.add_argument("--ctx-len",     type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = TrainConfig(
        target_model_path=args.model,
        n_subtrees=args.n_subtrees,
        sub_tree_paths=args.sub_tree_paths or [],
        ctx_len=args.ctx_len,
        draft_checkpoint=args.checkpoint,
        devices=1,
    )
    trainer = FabricTrainer(cfg)
    trainer.setup()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_tokens = load_bench_prompts(
        args.prompts, tokenizer, args.n_prompts, args.ctx_len, device
    )
    print(f"Loaded {len(prompt_tokens)} prompts")

    metrics = run_bench(
        model=trainer.model,
        target=trainer.target,
        prompt_tokens=prompt_tokens,
        tree_spec=trainer.tree_spec,
        target_layer_ids=trainer.target_layer_ids,
        max_new_tokens=args.max_new_tokens,
        n_candidate_tokens=args.n_candidate_tokens,
    )
    print(
        f"bench_acceptance_length={metrics['bench_acceptance_length']:.3f}  "
        f"bench_steps={metrics['bench_steps']}"
    )


if __name__ == "__main__":
    _main()
