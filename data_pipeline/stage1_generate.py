"""
Stage 1: Synthetic dataset generation.

Uses vLLM to generate completions from the target model for the DFlash
training mixture (Nemotron V2 + CodeAlpaca), producing JSONL files consumed
by Stage 2.

Output format (one JSONL file per shard):
{
    "prompt":   str,   # chat-template-formatted prompt fed to vLLM
    "response": str    # decoded model completion
}

Stage 2 (stage2_trees.py) tokenizes prompt and response independently; the
prompt text already contains all special tokens from the chat template so
it is encoded with add_special_tokens=False.
"""

import argparse
import json
import os
import random
from pathlib import Path

from datasets import load_dataset, concatenate_datasets
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dflash_prompts(
    tokenizer,
    num_samples: int = 800_000,
    max_seq_len: int = 3072,
) -> list[str]:
    """
    Load the DFlash training mixture: Nemotron V2 + CodeAlpaca.

    Returns a list of chat-template-formatted prompt strings, shuffled and
    filtered to at most num_samples entries whose token length is ≤ 66% of
    max_seq_len (leaving headroom for the completion).
    """
    print("Loading Nemotron and CodeAlpaca datasets...")
    prompts: list[list[dict]] = []

    # CodeAlpaca (~20 k samples)
    try:
        code_ds = load_dataset("HuggingFaceH4/CodeAlpaca_20k", split="train")
        for row in code_ds:
            content = row.get("prompt", "")
            prompts.append([{"role": "user", "content": content}])
    except Exception as e:
        print(f"Warning: could not load CodeAlpaca: {e}")

    # Nemotron V2 (streamed to avoid downloading the full ~6 M dataset)
    try:
        nemo_ds = concatenate_datasets(
            load_dataset(
                "nvidia/Nemotron-Post-Training-Dataset-v2",
                split=["chat", "math", "code", "stem"],
            )
        )
        nemo_ds = nemo_ds.shuffle(seed=42)
        for row in nemo_ds:
            if "messages" not in row or not row["messages"]:
                continue
            message = next(
                (m for m in row["messages"] if m["role"] == "user"), None
            )
            if message is None:
                continue
            if len(message["content"]) >= max_seq_len * 4:
                # character-level pre-filter (rough, before tokenisation)
                continue
            prompts.append([{"role": "user", "content": message["content"]}])
            if len(prompts) >= num_samples * 2:
                break
    except Exception as e:
        print(f"Warning: could not load Nemotron: {e}")

    random.seed(42)
    random.shuffle(prompts)

    print("Applying chat template...")
    formatted: list[str] = [
        tokenizer.apply_chat_template(
            p, tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]

    # Filter by token length, keeping prompts that leave room for a completion
    token_lengths = [
        len(ids)
        for ids in tokenizer(formatted, add_special_tokens=False)["input_ids"]
    ]
    max_prompt_tokens = int(max_seq_len * 0.66)
    filtered = [
        fmt
        for fmt, tlen in zip(formatted, token_lengths)
        if tlen <= max_prompt_tokens
    ][:num_samples]

    print(f"Prompts after filtering: {len(filtered)} / {len(formatted)}")
    return filtered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1: generate prompt/response pairs with vLLM."
    )
    parser.add_argument(
        "--model", required=True,
        help="Target model HF name or local path, e.g. Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write output JSONL shard(s)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=800_000,
        help="Total prompt/response pairs to generate (default: 800 000)",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=3072,
        help="Max full-sequence token length; prompts > 66%% of this are dropped "
             "(default: 3072)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=2048,
        help="Maximum completion tokens per sample (default: 2048)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
    )
    parser.add_argument(
        "--tp-size", type=int, default=1,
        help="Tensor parallel size for vLLM (default: 1)",
    )
    parser.add_argument(
        "--enable-thinking", action="store_true",
        help="Pass enable_thinking=True to apply_chat_template (Qwen3 thinking mode)",
    )
    parser.add_argument(
        "--shard-size", type=int, default=100_000,
        help="Write a new JSONL shard every N records (default: 100 000)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    prompts = load_dflash_prompts(
        tokenizer,
        num_samples=args.num_samples,
        max_seq_len=args.max_seq_len,
    )

    if args.enable_thinking:
        # Re-apply template with enable_thinking flag
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for p in prompts
        ]

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tp_size,
        enable_prefix_caching=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )

    print(f"Generating completions for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)

    shard_idx = 0
    written_total = 0
    written_shard = 0
    fout = open(Path(args.output_dir) / f"shard_{shard_idx:04d}.jsonl", "w")

    for prompt_text, output in zip(prompts, outputs):
        completion = output.outputs[0]
        response_text = completion.text

        record = {"prompt": prompt_text, "response": response_text}
        fout.write(json.dumps(record) + "\n")
        written_total += 1
        written_shard += 1

        if written_shard >= args.shard_size:
            fout.close()
            print(f"Shard {shard_idx}: {written_shard} records")
            shard_idx += 1
            written_shard = 0
            fout = open(Path(args.output_dir) / f"shard_{shard_idx:04d}.jsonl", "w")

    fout.close()
    print(f"Done. Total records: {written_total}  Shards: {shard_idx + 1}")


if __name__ == "__main__":
    main()
