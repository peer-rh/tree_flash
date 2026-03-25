# tree_flash

Research project extending DFlash with tree-structured speculative decoding.

## Project Goal

Improve DFlash by replacing its linear block-draft acceptance with tree-structured verification, increasing expected acceptance length without changing draft cost.

## Background

**DFlash** (Chen, Liang, Liu 2026 — arXiv:2602.06036): A speculative decoding method that drafts an entire block of tokens in a single parallel forward pass using a block diffusion model. The draft model denoises a masked block conditioned on KV-injected hidden states from the target model. Achieves ~6× speedup on Qwen3-8B, ~2.5× over EAGLE-3.

Source code cloned to `dflash/` from https://github.com/z-lab/dflash.

## DFlash Architecture Summary

- Draft model: 5 layers, block_size=16, shares embeddings + LM head with target
- Feature extraction: 5 uniformly-sampled target hidden states → concat → linear project
- KV injection: each draft attention layer's K/V = cat(target_hidden, current_hidden)
- Non-causal (bidirectional within block)
- Training: ~800K samples (Nemotron v2 + CodeAlpaca), exponentially decaying cross-entropy

## Current DFlash Acceptance (linear)

```
for each step:
  block = [last_token, MASK, MASK, ..., MASK]  # block_size tokens
  draft_tokens = draft_model(block, target_hidden)  # one parallel pass
  target_logits = target_model(block_with_draft_fills)
  accept greedily: take tokens while draft[i] == argmax(target[i])
  advance by acceptance_length + 1
```

## Tree Extension Idea

Replace linear acceptance with tree-structured verification:
- Draft a primary path of sequential tokens from the current anchor
- Attach an alternative continuation subtree at each node on that primary path
- Represent each attached subtree as an arbitrary rooted tree over relative paths
- Verify multiple paths in one target pass with tree attention mask
- Accept the longest valid path (more accepted tokens per verification step)

## Tree v1

`Tree v1` is parameterized by `seq_depth` and `sub_tree`.

- `seq_depth` is the length of the primary drafted path starting from the current anchor.
- `sub_tree` is an arbitrary rooted tree shape attached at each node of that primary path.
- A convenient representation of `sub_tree` is a set of relative paths. For example, `01, 02, 03, 14, 15, 26, 27` defines a depth-2 subtree where node `0` has children `1, 2, 3`, node `1` has children `4, 5`, and node `2` has children `6, 7`.
- The full verification tree is the primary path plus one attached copy of `sub_tree` at every position along that path.

## Key Files

- `dflash/model/dflash.py` — draft model architecture + `spec_generate`
- `dflash/model/utils.py` — feature extraction, sampling, dataset loaders
- `dflash/benchmark.py` — transformers-backend benchmarking
- `dflash/benchmark_sglang.py` — SGLang benchmarking

## Benchmarks Used by DFlash

Math: GSM8K, MATH-500, AIME24, AIME25
Code: HumanEval, MBPP, LiveCodeBench, SWE-Bench
Chat: MT-Bench, Alpaca

## Models

Qwen3-8B / Qwen3-4B / LLaMA-3.1-8B / Qwen3-Coder / gpt-oss variants
HuggingFace collection: https://huggingface.co/collections/z-lab/dflash

## Compute

To be filled in.
