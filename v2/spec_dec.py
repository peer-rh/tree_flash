"""
v2/spec_dec.py — Tree speculative decoding with efficient KV cache management.

Usage:
    python v2/spec_dec.py \
        --target-model Qwen/Qwen3-8B \
        --draft-model checkpoints/run1/final/hf_draft \
        --tree-pos-emb-path checkpoints/run1/final/tree_pos_emb.pt \
        --tree-seq-depth 4 \
        --prompt "What is the capital of France?" \
        --max-new-tokens 256 \
        --temperature 0.0 \
        --dtype bfloat16
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "dflash"))
sys.path.insert(0, str(_ROOT / "v2"))

from model.dflash import DFlashDraftModel  # noqa: E402
from model.utils import extract_context_feature, load_and_process_dataset, sample  # noqa: E402
from stage2 import SubTreeInfo, DEFAULT_SUB_TREE_PATHS  # noqa: E402
from stage3 import TreePositionEmbedding  # noqa: E402


# ---------------------------------------------------------------------------
# Precomputed tree constants
# ---------------------------------------------------------------------------

def build_tree_constants(
    st_info: SubTreeInfo,
    tree_seq_depth: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute flat-index helpers for the full tree.

    Returns:
        pos_offsets   [tree_size]  — d + depth_of[v] for each flat index
        vertex_ids    [tree_size]  — subtree vertex v for each flat index
        parent_flat   [tree_size]  — parent flat index, -1 for root
        anc_mask      [tree_size, tree_size] bool — anc_mask[k, q] = k is ancestor-or-self of q
    """
    st_size = st_info.size
    tree_size = tree_seq_depth * st_size

    pos_offsets = torch.zeros(tree_size, dtype=torch.long, device=device)
    vertex_ids = torch.zeros(tree_size, dtype=torch.long, device=device)
    parent_flat = torch.full((tree_size,), -1, dtype=torch.long, device=device)

    for d in range(tree_seq_depth):
        for v in range(st_size):
            flat = d * st_size + v
            pos_offsets[flat] = d + st_info.depth_of[v]
            vertex_ids[flat] = v
            if d == 0 and v == 0:
                parent_flat[flat] = -1
            elif v == 0:
                parent_flat[flat] = (d - 1) * st_size  # vertex 0 at previous depth
            else:
                parent_flat[flat] = d * st_size + st_info.parent_map[v]

    # Build ancestor mask: anc_mask[k, q] = True if k is ancestor-or-self of q
    anc_mask = torch.zeros(tree_size, tree_size, dtype=torch.bool, device=device)
    parent_flat_list = parent_flat.tolist()
    for q_flat in range(tree_size):
        cur = q_flat
        while cur >= 0:
            anc_mask[cur, q_flat] = True
            cur = parent_flat_list[cur]

    return pos_offsets, vertex_ids, parent_flat, anc_mask


# ---------------------------------------------------------------------------
# Draft phase
# ---------------------------------------------------------------------------

def draft_tree(
    draft_model: DFlashDraftModel,
    target_model: nn.Module,
    tree_pos_emb: TreePositionEmbedding | None,
    target_hidden_new: torch.Tensor,  # [1, new_ctx_len, multi_H]
    anchor_token_id: int,
    seq_start: int,                   # absolute position of tree root (= current anchor)
    past_kv_draft: DynamicCache,
    st_info: SubTreeInfo,
    tree_seq_depth: int,
    mask_token_id: int,
    temperature: float,
    pos_offsets: torch.Tensor,        # [tree_size]
    vertex_ids_flat: torch.Tensor,    # [tree_size]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single draft forward pass over the full tree.

    Returns:
        tree_token_ids  [tree_size]     — drafted tokens (root overwritten to anchor)
        tree_logits     [tree_size, V]  — raw logits
    """
    device = target_hidden_new.device
    st_size = st_info.size
    tree_size = tree_seq_depth * st_size

    # Noise ids: root = known anchor, all others = mask
    noise_ids = torch.full((1, tree_size), mask_token_id, dtype=torch.long, device=device)
    noise_ids[0, 0] = anchor_token_id

    # Noise embeddings + optional initial position bias
    noise_embeds = target_model.model.embed_tokens(noise_ids)  # [1, tree_size, H]
    if tree_pos_emb is not None:
        initial_bias = tree_pos_emb.get_initial_bias(vertex_ids_flat.unsqueeze(0))
        if initial_bias is not None:
            noise_embeds = noise_embeds + initial_bias

    # Position ids: [draft_cache_len .. seq_start-1] for new context, then tree positions
    draft_cache_len = past_kv_draft.get_seq_length()
    if draft_cache_len < seq_start:
        ctx_pos_ids = torch.arange(
            draft_cache_len, seq_start, dtype=torch.long, device=device
        ).unsqueeze(0)  # [1, new_ctx_len]
    else:
        ctx_pos_ids = torch.zeros((1, 0), dtype=torch.long, device=device)

    tree_pos_ids = (seq_start + pos_offsets).unsqueeze(0)  # [1, tree_size]
    full_pos_ids = torch.cat([ctx_pos_ids, tree_pos_ids], dim=1)  # [1, new_ctx_len + tree_size]

    # DFlashDraftModel.forward: hidden_states starts as noise_embedding,
    # target_hidden is only used for K/V injection. Output shape = [1, tree_size, H].
    draft_hidden = draft_model(
        target_hidden=target_hidden_new,
        noise_embedding=noise_embeds,
        position_ids=full_pos_ids,
        past_key_values=past_kv_draft,
        use_cache=True,
        is_causal=False,
    )  # [1, tree_size, H]

    # Crop draft KV cache: discard tree block entries, keep only context
    past_kv_draft.crop(seq_start)

    logits = target_model.lm_head(draft_hidden)  # [1, tree_size, V]
    tree_logits = logits[0]                       # [tree_size, V]

    tree_token_ids = sample(logits, temperature)[0]  # [tree_size]
    tree_token_ids[0] = anchor_token_id              # root is always the known anchor

    return tree_token_ids, tree_logits


# ---------------------------------------------------------------------------
# Verification phase
# ---------------------------------------------------------------------------

def build_verification_mask(
    anc_mask: torch.Tensor,  # [tree_size, tree_size] bool: anc_mask[k, q] = k is anc-or-self of q
    prefix_len: int,
    tree_size: int,
    device: torch.device,
) -> torch.Tensor:
    """4D additive attention mask [1, 1, tree_size, prefix_len + tree_size].

    Each tree query q attends to:
    - Full cached prefix (all prefix_len entries).
    - Tree entries k where k is an ancestor-or-self of q.
    """
    # Left (prefix): 0.0 → all attendable
    left = torch.zeros(1, 1, tree_size, prefix_len, dtype=torch.float32, device=device)
    # Right (tree): anc_mask[k, q] = k is ancestor-or-self of q
    # For a [q, k] additive mask: 0 if q can attend to k, -inf otherwise.
    # q can attend to k iff anc_mask[k, q] is True → we need mask[q, k] = not anc_mask[k, q]
    can_attend = anc_mask.T  # [tree_size(q), tree_size(k)]: True if q can attend to k
    right = torch.where(can_attend, torch.zeros(1, device=device), torch.full((1,), float("-inf"), device=device))
    right = right.unsqueeze(0).unsqueeze(0)             # [1, 1, tree_size, tree_size]

    return torch.cat([left, right], dim=-1)             # [1, 1, tree_size, prefix_len + tree_size]


def verify_tree(
    target_model: nn.Module,
    tree_token_ids: torch.Tensor,    # [tree_size]
    tree_pos_ids: torch.Tensor,      # [tree_size]
    past_kv_target: DynamicCache,
    anc_mask: torch.Tensor,          # [tree_size, tree_size]
    temperature: float,
) -> tuple[torch.Tensor, tuple]:
    """Single target forward over all tree nodes with ancestor attention.

    Returns:
        target_samples  [tree_size]  — target's prediction at each tree position
        hidden_states   tuple        — all layer hidden states (for feature extraction)
    """
    device = tree_token_ids.device
    tree_size = tree_token_ids.shape[0]
    prefix_len = past_kv_target.get_seq_length()

    mask_4d = build_verification_mask(anc_mask, prefix_len, tree_size, device)
    cache_position = torch.arange(prefix_len, prefix_len + tree_size, device=device)

    out = target_model(
        input_ids=tree_token_ids.unsqueeze(0),
        position_ids=tree_pos_ids.unsqueeze(0),
        attention_mask=mask_4d,
        past_key_values=past_kv_target,
        use_cache=True,
        cache_position=cache_position,
        output_hidden_states=True,
    )

    target_samples = sample(out.logits, temperature)[0]  # [tree_size]
    return target_samples, out.hidden_states


# ---------------------------------------------------------------------------
# Acceptance phase
# ---------------------------------------------------------------------------

def accept_tree(
    tree_token_ids: torch.Tensor,  # [tree_size]
    target_samples: torch.Tensor,  # [tree_size] — target's prediction given each tree node
    parent_flat: torch.Tensor,     # [tree_size], -1 for root
    tree_seq_depth: int,
    st_size: int,
) -> tuple[list[int], int]:
    """Greedy DFS: accept a node if its token matches target's prediction at its parent.

    The accepted path forms a chain root → … → deepest accepted node.
    All nodes on the path are at consecutive absolute positions (start, start+1, …).

    Returns:
        accepted_flat_indices  — flat indices in accepted order (includes root)
        bonus_token_id         — target's prediction at the last accepted node
    """
    tree_size = tree_seq_depth * st_size
    tok_list = tree_token_ids.tolist()
    tgt_list = target_samples.tolist()
    parent_list = parent_flat.tolist()

    # Build children map
    children_of: dict[int, list[int]] = {}
    for flat in range(tree_size):
        p = int(parent_list[flat])
        if p >= 0:
            children_of.setdefault(p, []).append(flat)

    accepted: list[int] = [0]
    current = 0

    while True:
        kids = children_of.get(current, [])
        if not kids:
            break
        target_pred = int(tgt_list[current])
        next_node = None
        for kid in kids:
            if int(tok_list[kid]) == target_pred:
                next_node = kid
                break
        if next_node is None:
            break
        accepted.append(next_node)
        current = next_node

    bonus_token = int(tgt_list[current])
    return accepted, bonus_token


# ---------------------------------------------------------------------------
# KV cache cleanup
# ---------------------------------------------------------------------------

def cleanup_cache(
    past_kv: DynamicCache,
    prefix_len: int,
    accepted_flat_indices: list[int],
) -> None:
    """Gather prefix + accepted path K/V; discard rejected tree entries in-place."""
    keep = torch.tensor(accepted_flat_indices, dtype=torch.long)
    for layer_idx in range(len(past_kv.key_cache)):
        k = past_kv.key_cache[layer_idx]    # [B, H, prefix_len + tree_size, D]
        v = past_kv.value_cache[layer_idx]
        prefix_k, tree_k = k[:, :, :prefix_len, :], k[:, :, prefix_len:, :]
        prefix_v, tree_v = v[:, :, :prefix_len, :], v[:, :, prefix_len:, :]
        sel = keep.to(k.device)
        past_kv.key_cache[layer_idx] = torch.cat([prefix_k, tree_k[:, :, sel, :]], dim=2)
        past_kv.value_cache[layer_idx] = torch.cat([prefix_v, tree_v[:, :, sel, :]], dim=2)


# ---------------------------------------------------------------------------
# Main decode loop
# ---------------------------------------------------------------------------

@torch.inference_mode()
def tree_spec_generate(
    target_model: nn.Module,
    draft_model: DFlashDraftModel,
    tree_pos_emb: TreePositionEmbedding | None,
    input_ids: torch.Tensor,   # [1, prompt_len]
    max_new_tokens: int,
    st_info: SubTreeInfo,
    tree_seq_depth: int,
    temperature: float = 0.0,
    stop_token_ids: list[int] | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """Tree speculative decoding with KV cache reuse.

    Returns:
        output_ids         [1, total_len]
        acceptance_lengths list of accepted-path lengths per draft step
    """
    device = input_ids.device
    prompt_len = input_ids.shape[1]
    max_length = prompt_len + max_new_tokens

    st_size = st_info.size
    tree_size = tree_seq_depth * st_size
    mask_token_id = draft_model.mask_token_id

    pos_offsets, vertex_ids_flat, parent_flat, anc_mask = build_tree_constants(
        st_info, tree_seq_depth, device
    )

    # Output buffer (large enough to never overflow)
    output_ids = torch.full(
        (1, max_length + tree_size), mask_token_id, dtype=torch.long, device=device
    )
    output_ids[0, :prompt_len] = input_ids[0]

    past_kv_target = DynamicCache()
    past_kv_draft = DynamicCache()

    # --- Prefill ---
    out = target_model(
        input_ids,
        position_ids=torch.arange(prompt_len, device=device).unsqueeze(0),
        past_key_values=past_kv_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    output_ids[0, prompt_len] = sample(out.logits[:, -1:, :], temperature)[0, 0]
    # features for all prompt positions → used in first draft call
    target_hidden_new = extract_context_feature(out.hidden_states, draft_model.target_layer_ids)

    acceptance_lengths: list[int] = []
    start = prompt_len  # current anchor position

    while start < max_length:
        anchor_token_id = int(output_ids[0, start].item())

        # --- Draft: one parallel forward over the full tree ---
        tree_token_ids, _ = draft_tree(
            draft_model=draft_model,
            target_model=target_model,
            tree_pos_emb=tree_pos_emb,
            target_hidden_new=target_hidden_new,
            anchor_token_id=anchor_token_id,
            seq_start=start,
            past_kv_draft=past_kv_draft,
            st_info=st_info,
            tree_seq_depth=tree_seq_depth,
            mask_token_id=mask_token_id,
            temperature=temperature,
            pos_offsets=pos_offsets,
            vertex_ids_flat=vertex_ids_flat,
        )

        # --- Verify: one target forward with ancestor attention ---
        tree_pos_ids = start + pos_offsets  # [tree_size], absolute positions
        prefix_len_before = past_kv_target.get_seq_length()  # = start
        target_samples, hidden_states = verify_tree(
            target_model=target_model,
            tree_token_ids=tree_token_ids,
            tree_pos_ids=tree_pos_ids,
            past_kv_target=past_kv_target,
            anc_mask=anc_mask,
            temperature=temperature,
        )

        # --- Accept: greedy DFS ---
        accepted_indices, bonus_token = accept_tree(
            tree_token_ids=tree_token_ids,
            target_samples=target_samples,
            parent_flat=parent_flat,
            tree_seq_depth=tree_seq_depth,
            st_size=st_size,
        )
        accept_len = len(accepted_indices)  # includes root

        # --- Write tokens ---
        # accepted_indices[0] = root (anchor already at output_ids[0, start])
        # accepted_indices[i] for i>0: token at position start + i
        for i, flat_idx in enumerate(accepted_indices[1:], 1):
            pos = start + i
            if pos < max_length:
                output_ids[0, pos] = tree_token_ids[flat_idx]
        # bonus token becomes the new anchor
        bonus_pos = start + accept_len
        if bonus_pos < max_length:
            output_ids[0, bonus_pos] = bonus_token

        # --- KV cache cleanup: keep prefix + accepted path ---
        cleanup_cache(past_kv_target, prefix_len_before, accepted_indices)
        # past_kv_target now has prefix_len_before + accept_len = start + accept_len entries

        # --- Update target_hidden_new for next draft step ---
        # Features for the accepted nodes (positions start..start+accept_len-1)
        target_features = extract_context_feature(hidden_states, draft_model.target_layer_ids)
        feat_indices = torch.tensor(accepted_indices, dtype=torch.long, device=device)
        target_hidden_new = target_features[:, feat_indices, :]  # [1, accept_len, multi_H]

        start += accept_len  # new anchor = bonus token position
        acceptance_lengths.append(accept_len)

        # --- Stop check ---
        if stop_token_ids is not None:
            generated = output_ids[0, prompt_len:start]
            stop_t = torch.tensor(stop_token_ids, device=device)
            if torch.isin(generated, stop_t).any():
                break

    # Trim to actual generated length
    end = min(start, max_length)
    out_ids = output_ids[0, :end]

    # Truncate at first stop token
    if stop_token_ids is not None:
        stop_t = torch.tensor(stop_token_ids, device=device)
        stop_mask = torch.isin(out_ids[prompt_len:], stop_t)
        if stop_mask.any():
            first = stop_mask.nonzero(as_tuple=True)[0][0].item()
            out_ids = out_ids[:prompt_len + int(first) + 1]

    return out_ids.unsqueeze(0), acceptance_lengths


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def load_models(args) -> tuple[nn.Module, DFlashDraftModel, TreePositionEmbedding | None]:
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading target model: {args.target_model}")
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=torch_dtype, attn_implementation="sdpa"
    ).to(device).eval()

    print(f"Loading draft model: {args.draft_model}")
    draft = DFlashDraftModel.from_pretrained(args.draft_model, torch_dtype=torch_dtype).to(device).eval()

    tree_pos_emb = None
    if args.tree_pos_emb_path is not None:
        print(f"Loading TreePositionEmbedding: {args.tree_pos_emb_path}")
        tree_pos_emb = torch.load(args.tree_pos_emb_path, map_location=device, weights_only=False)
        tree_pos_emb.eval()

    return target, draft, tree_pos_emb


def run_benchmark(args, target, draft, tree_pos_emb, tokenizer, st_info):
    dataset = load_and_process_dataset(args.benchmark)
    prompts = [row["turns"][0] for row in dataset]
    if args.benchmark_n is not None:
        prompts = prompts[:args.benchmark_n]

    device = next(target.parameters()).device
    total_tokens = 0
    all_accept_lens: list[int] = []
    t0 = time.perf_counter()

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

        output_ids, accept_lens = tree_spec_generate(
            target_model=target,
            draft_model=draft,
            tree_pos_emb=tree_pos_emb,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            st_info=st_info,
            tree_seq_depth=args.tree_seq_depth,
            temperature=args.temperature,
            stop_token_ids=[tokenizer.eos_token_id],
        )
        n_new = output_ids.shape[1] - input_ids.shape[1]
        total_tokens += n_new
        all_accept_lens.extend(accept_lens)

    elapsed = time.perf_counter() - t0
    mean_accept = sum(all_accept_lens) / max(len(all_accept_lens), 1)
    print(f"\n=== Benchmark: {args.benchmark} | {len(prompts)} prompts ===")
    print(f"  Mean acceptance length : {mean_accept:.2f}")
    print(f"  Total new tokens       : {total_tokens}")
    print(f"  Elapsed                : {elapsed:.1f}s")
    print(f"  Tokens/s               : {total_tokens / max(elapsed, 1e-6):.1f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tree speculative decoding inference.")
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--tree-pos-emb-path", default=None,
                        help="Path to saved TreePositionEmbedding .pt file")
    parser.add_argument("--sub-tree-paths", nargs="+", default=DEFAULT_SUB_TREE_PATHS)
    parser.add_argument("--tree-seq-depth", type=int, default=4)
    parser.add_argument("--prompt", default=None, help="Single prompt string")
    parser.add_argument("--prompt-file", default=None, help="File with one prompt per line")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--benchmark", default=None,
                        choices=["gsm8k", "math500", "aime24", "aime25",
                                 "humaneval", "mbpp", "lbpp", "swe-bench",
                                 "livecodebench", "alpaca", "mt-bench"],
                        help="Run on a standard benchmark dataset")
    parser.add_argument("--benchmark-n", type=int, default=None,
                        help="Limit number of benchmark prompts")
    args = parser.parse_args()

    target, draft, tree_pos_emb = load_models(args)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    st_info = SubTreeInfo(args.sub_tree_paths)
    device = next(target.parameters()).device

    if args.benchmark is not None:
        run_benchmark(args, target, draft, tree_pos_emb, tokenizer, st_info)
        return

    # Collect prompts
    prompts: list[str] = []
    if args.prompt is not None:
        prompts = [args.prompt]
    elif args.prompt_file is not None:
        with open(args.prompt_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        parser.error("Provide --prompt, --prompt-file, or --benchmark")

    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

        t0 = time.perf_counter()
        output_ids, accept_lens = tree_spec_generate(
            target_model=target,
            draft_model=draft,
            tree_pos_emb=tree_pos_emb,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            st_info=st_info,
            tree_seq_depth=args.tree_seq_depth,
            temperature=args.temperature,
            stop_token_ids=[tokenizer.eos_token_id],
        )
        elapsed = time.perf_counter() - t0

        n_new = output_ids.shape[1] - input_ids.shape[1]
        mean_accept = sum(accept_lens) / max(len(accept_lens), 1)
        response = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)

        print(f"\n--- Prompt {i} ---")
        print(prompt)
        print(f"\n--- Response ---")
        print(response)
        print(f"\n[{n_new} tokens | {elapsed:.2f}s | {n_new/max(elapsed,1e-6):.1f} tok/s"
              f" | mean accept len {mean_accept:.2f}]")


if __name__ == "__main__":
    main()
