from __future__ import annotations

import argparse
import ast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import build_inference_tree, load_drafter_model
from src.spec_decode import speculative_generate_from_ids


@torch.inference_mode()
def generate(
    *,
    target: str,
    drafter: str,
    prompt: str,
    tree_seq_depth: int,
    branching_pattern: list[list[int]],
    candidate_tree_size: int,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
):
    """Run the cleaned-up inference path.

    This module intentionally exposes exactly one inference configuration:

    - draft with one fixed branch-off tree
    - prune with the q-head to `candidate_tree_size`
    - verify with the target model
    """
    tokenizer = AutoTokenizer.from_pretrained(target)
    target_model = AutoModelForCausalLM.from_pretrained(
        target,
        torch_dtype=torch.bfloat16,
        attn_implementation="flex_attention",
    )
    drafter_model = load_drafter_model(drafter, torch_dtype=torch.bfloat16)
    target_model.eval()
    drafter_model.eval()
    tree = build_inference_tree(
        tree_seq_depth=tree_seq_depth,
        branching_pattern=branching_pattern,
        candidate_tree_size=candidate_tree_size,
    )
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    return speculative_generate_from_ids(
        target_model=target_model,
        drafter_model=drafter_model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        tree_processor=tree,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run cleaned-up Tree Flash inference.")
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--drafter", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--tree-seq-depth", type=int, default=4)
    parser.add_argument("--branching-pattern", type=str, default="[[0,1,2],[0,1],[0]]")
    parser.add_argument("--candidate-tree-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    branching_pattern = ast.literal_eval(args.branching_pattern)
    result = generate(
        target=args.target,
        drafter=args.drafter,
        prompt=args.prompt,
        tree_seq_depth=args.tree_seq_depth,
        branching_pattern=branching_pattern,
        candidate_tree_size=args.candidate_tree_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(result.text)


if __name__ == "__main__":
    main()
