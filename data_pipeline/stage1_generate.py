"""
Stage 1: Synthetic dataset generation.

Uses vLLM to generate completions from the target model, storing per-token
logprobs for anchor selection in Stage 2, and top-b token IDs/probs at each
position for level-0 tree construction in Stage 2 (avoiding a second forward
pass through the model).

Output format (one JSON-lines file per dataset split):
{
    "prompt_token_ids": List[int],
    "completion_token_ids": List[int],
    "chosen_logprobs": List[float],   # log p(chosen_token[t]) at each position
    "top_token_ids": List[List[int]], # top-b token IDs at each position
    "top_logprobs": List[List[float]] # log probs for the top-b tokens
}
"""

import argparse
import json
import math
import os
from pathlib import Path

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from sys import path as sys_path
sys_path.insert(0, str(Path(__file__).parent.parent / "dflash"))
from model.utils import load_and_process_dataset


def build_prompts(dataset, tokenizer, enable_thinking: bool = False):
    """Convert dataset turns into tokenized chat prompts."""
    prompts = []
    for instance in dataset:
        messages = []
        for turn in instance["turns"]:
            messages.append({"role": "user", "content": turn})
            # For multi-turn we only generate the first turn here;
            # extend to multi-turn by iterating if needed.
            break
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prompts.append(text)
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Target model path or HF name, e.g. Qwen/Qwen3-8B")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name: gsm8k, math500, humaneval, alpaca, ...")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write output JSONL files")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-b", type=int, default=4,
                        help="Number of top tokens to store per position (used for "
                             "level-0 tree construction in Stage 2)")
    parser.add_argument("--tp-size", type=int, default=1,
                        help="Tensor parallel size for vLLM")
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = Path(args.output_dir) / f"{args.dataset}.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_and_process_dataset(args.dataset)
    if args.max_samples and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    prompts = build_prompts(dataset, tokenizer, args.enable_thinking)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tp_size,
        # Enable prefix caching so Stage 2 tree generation can reuse KV cache
        # when processing the same sequences again.
        enable_prefix_caching=True,
    )

    # logprobs=top_b: vLLM returns the top-b log probs at each token position.
    # The chosen token is always included in the top-b (it has the highest prob
    # under greedy decoding at temperature=0).
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        logprobs=args.top_b,
        # prompt_logprobs not needed — we only care about completion positions.
    )

    outputs = llm.generate(prompts, sampling_params)

    written = 0
    with open(output_path, "w") as f:
        for output, instance in zip(outputs, dataset):
            completion = output.outputs[0]

            if completion.logprobs is None:
                continue

            prompt_token_ids = list(output.prompt_token_ids)
            completion_token_ids = list(completion.token_ids)

            # Per-position top-b token IDs and log probs.
            # completion.logprobs is a list (length = num_generated_tokens) of
            # Dict[int, Logprob] sorted by logprob descending.
            top_token_ids = []
            top_logprobs = []
            chosen_logprobs = []

            for pos_logprobs in completion.logprobs:
                # vLLM Logprob objects have .logprob (float) and .rank (int).
                sorted_items = sorted(
                    pos_logprobs.items(),
                    key=lambda kv: kv[1].logprob,
                    reverse=True,
                )
                ids = [tok_id for tok_id, _ in sorted_items]
                lps = [lp.logprob for _, lp in sorted_items]
                top_token_ids.append(ids)
                top_logprobs.append(lps)
                # Chosen token is always the one with the highest rank under
                # greedy. Under temperature=0 this is ids[0].
                chosen_logprobs.append(lps[0])

            record = {
                "prompt_token_ids": prompt_token_ids,
                "completion_token_ids": completion_token_ids,
                "chosen_logprobs": chosen_logprobs,
                "top_token_ids": top_token_ids,
                "top_logprobs": top_logprobs,
            }
            f.write(json.dumps(record) + "\n")
            written += 1

    print(f"Wrote {written} records to {output_path}")


if __name__ == "__main__":
    main()
