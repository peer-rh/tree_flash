from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache

from .models import DFlashDraftModel
from .trainer import has_pruning_head, unwrap_model
from .trees import (
    BlockTreeProcessor,
    BranchOffTreeProcessor,
    PrunableTreeProcessor,
    TreeInfo,
    subset_tree_info,
)


@dataclass
class SpecDecodeResult:
    token_ids: torch.Tensor
    text: str
    acceptance_lengths: list[int]
    drafted_tokens: int
    committed_tokens: int


def sample_from_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def gather_token_probability(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature > 0:
        logits = logits / temperature
    probs = torch.softmax(logits.float(), dim=-1)
    return probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1).to(torch.float32)


def build_tree_processor(
    *,
    tree_type: str,
    tree_seq_depth: int,
    sub_tree_paths: Sequence[str] | None = None,
    tree_args: dict[str, Any] | None = None,
):
    tree_args = tree_args or {}
    if tree_type in {"fixed", "block"}:
        return BlockTreeProcessor(
            tree_seq_depth=tree_seq_depth,
            sub_tree_paths=sub_tree_paths or tree_args.get("sub_tree_paths"),
        )
    if tree_type == "branch_off":
        return BranchOffTreeProcessor(
            tree_seq_depth=tree_seq_depth,
            sub_tree_paths=sub_tree_paths or tree_args.get("sub_tree_paths"),
            branching_pattern=tree_args.get("branching_pattern"),
        )
    if tree_type == "prunable":
        return PrunableTreeProcessor(
            tree_seq_depth=tree_seq_depth,
            base_tree_type=tree_args.get("base_tree_type", "block"),
            prune_topk=int(tree_args.get("prune_topk", 0)),
            sub_tree_paths=sub_tree_paths or tree_args.get("sub_tree_paths"),
            branching_pattern=tree_args.get("branching_pattern"),
        )
    raise NotImplementedError(
        f"tree_type={tree_type!r} is not implemented for speculative decoding."
    )

def build_verifier_score_mod(
    *,
    tree_info: TreeInfo,
    prefix_len: int,
):
    tree_mask = tree_info.tree_mask
    block_size = tree_info.block_size

    def score_mod(score, B, H, Q, KV):
        is_prefix = KV < prefix_len
        tree_kv = KV - prefix_len
        valid_tree = (tree_kv >= 0) & (tree_kv < block_size)
        q_idx = Q.clamp(0, block_size - 1)
        kv_idx = tree_kv.clamp(0, block_size - 1)
        legal_tree = valid_tree & tree_mask[q_idx, kv_idx]
        return torch.where(is_prefix | legal_tree, score, torch.full_like(score, float("-inf")))

    return score_mod


def trim_dynamic_cache(
    past_key_values: Cache,
    *,
    prefix_len: int,
    keep_tree_indices: list[int],
) -> Cache:
    keep = torch.tensor(keep_tree_indices, dtype=torch.long)
    for layer in past_key_values.layers:
        if not getattr(layer, "is_initialized", False) or layer.keys is None or layer.values is None:
            continue
        device = layer.keys.device
        select = keep.to(device)
        prefix_keys = layer.keys[:, :, :prefix_len, :]
        prefix_values = layer.values[:, :, :prefix_len, :]
        tree_keys = layer.keys[:, :, prefix_len:, :]
        tree_values = layer.values[:, :, prefix_len:, :]
        layer.keys = torch.cat([prefix_keys, tree_keys.index_select(2, select)], dim=2)
        layer.values = torch.cat([prefix_values, tree_values.index_select(2, select)], dim=2)
        if hasattr(layer, "cumulative_length"):
            layer.cumulative_length = int(layer.keys.shape[-2])
    return past_key_values


def choose_deepest_valid_node(
    accepted_mask: torch.Tensor,
    tree_info: TreeInfo,
) -> int:
    valid_nodes = []
    for node_idx in range(tree_info.block_size):
        path_mask = tree_info.tree_mask[node_idx]
        if bool(accepted_mask[path_mask].all().item()):
            valid_nodes.append(node_idx)
    if not valid_nodes:
        return 0
    return max(valid_nodes, key=lambda idx: (int(tree_info.depth[idx].item()), -idx))


def gather_path_indices(
    node_idx: int,
    tree_info: TreeInfo,
) -> list[int]:
    path = []
    cur = node_idx
    while cur >= 0:
        path.append(cur)
        cur = int(tree_info.parent_idx[cur].item())
    return list(reversed(path))


def select_pruned_keep_indices(
    *,
    tree_info: TreeInfo,
    q_scores: torch.Tensor,
    prune_topk: int,
) -> list[int]:
    if prune_topk < 0:
        raise ValueError(f"prune_topk must be >= 0, got {prune_topk}.")
    if prune_topk == 0 or tree_info.block_size <= 1:
        return [0]

    num_candidates = tree_info.block_size - 1
    topk = min(prune_topk, num_candidates)
    ranked = torch.topk(q_scores[1:], k=topk).indices + 1
    kept = {0}
    for node_idx in ranked.tolist():
        kept.update(gather_path_indices(int(node_idx), tree_info))
    return sorted(kept)


def prune_drafted_tree(
    *,
    tree_token_ids: torch.Tensor,
    draft_logits: torch.Tensor,
    draft_token_probs: torch.Tensor,
    tree_info: TreeInfo,
    q_scores: torch.Tensor,
    prune_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, TreeInfo]:
    keep_indices = select_pruned_keep_indices(
        tree_info=tree_info,
        q_scores=q_scores,
        prune_topk=prune_topk,
    )
    if len(keep_indices) == tree_info.block_size:
        return tree_token_ids, draft_logits, draft_token_probs, tree_info

    keep = torch.tensor(keep_indices, dtype=torch.long, device=tree_token_ids.device)
    pruned_tree_info = subset_tree_info(tree_info, keep)
    return (
        tree_token_ids.index_select(0, keep),
        draft_logits.index_select(0, keep),
        draft_token_probs.index_select(0, keep),
        pruned_tree_info,
    )


def build_tree_parent_token_ids(
    tree_token_ids: torch.Tensor,
    tree_info: TreeInfo,
) -> torch.Tensor:
    parent_token_ids = tree_token_ids.clone()
    if tree_info.block_size <= 1:
        return parent_token_ids
    parent_idx = tree_info.parent_idx.to(tree_token_ids.device)
    parent_token_ids[1:] = tree_token_ids.index_select(0, parent_idx[1:])
    return parent_token_ids


def build_pruning_scores(
    *,
    raw_drafter,
    backbone_hidden: torch.Tensor,
    tree_token_ids: torch.Tensor,
    tree_info: TreeInfo,
    target_embeddings,
    target_lm_head,
    temperature: float,
) -> torch.Tensor | None:
    if getattr(raw_drafter, "ar_fusion", None) is not None:
        parent_token_ids = build_tree_parent_token_ids(tree_token_ids, tree_info)
        parent_embeddings = target_embeddings(parent_token_ids.unsqueeze(0))
        ar_hidden_states = raw_drafter.build_ar_hidden_states(backbone_hidden, parent_embeddings)
        ar_logits = target_lm_head(ar_hidden_states)[0]
        ar_scores = gather_token_probability(ar_logits, tree_token_ids, temperature)
        ar_scores[0] = 1.0
        return ar_scores
    if getattr(raw_drafter, "q_head", None) is not None:
        return torch.sigmoid(raw_drafter.q_head(backbone_hidden)[0, :, 0]).to(torch.float32)
    return None


def draft_tree(
    *,
    drafter_model,
    raw_drafter,
    target_embeddings,
    target_lm_head,
    tree_processor,
    target_ctx_features,
    drafter_cache: Cache,
    current_root_token: int,
    root_position: int,
    temperature: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, TreeInfo, torch.Tensor | None]:
    tree_info = tree_processor.build_tree_info(batch_size=1, num_blocks=1, device=device)
    tree_len = tree_info.block_size
    pre_len = drafter_cache.get_seq_length()
    noise_ids = torch.full((1, tree_len), raw_drafter.mask_token_id, dtype=torch.long, device=device)
    noise_ids[0, 0] = current_root_token
    noise_embeddings = target_embeddings(noise_ids)
    position_ids = (tree_info.depth.view(1, -1) + root_position).to(device)
    draft_hidden_states, backbone_hidden = drafter_model(
        hidden_states=noise_embeddings,
        position_ids=position_ids,
        tree_info=tree_info,
        target_ctx_features=target_ctx_features,
        past_key_values=drafter_cache,
        use_cache=True,
    )
    drafter_cache.crop(
        pre_len + target_ctx_features.shape[1],
    )
    draft_logits = target_lm_head(draft_hidden_states)[0]
    tree_token_ids = sample_from_logits(draft_logits, temperature)
    tree_token_ids[0] = current_root_token
    draft_token_probs = gather_token_probability(draft_logits, tree_token_ids, temperature)
    draft_token_probs[0] = 1.0
    pruning_scores = build_pruning_scores(
        raw_drafter=raw_drafter,
        backbone_hidden=backbone_hidden,
        tree_token_ids=tree_token_ids,
        tree_info=tree_info,
        target_embeddings=target_embeddings,
        target_lm_head=target_lm_head,
        temperature=temperature,
    )
    return tree_token_ids, draft_logits, draft_token_probs, tree_info, pruning_scores


def verify_tree(
    *,
    target_model,
    raw_drafter,
    tree_token_ids: torch.Tensor,
    tree_info: TreeInfo,
    root_position: int,
    target_cache: Cache,
    temperature: float,
) -> tuple[Cache, torch.Tensor, torch.Tensor]:
    prefix_len = target_cache.get_seq_length()
    score_mod = build_verifier_score_mod(
        tree_info=tree_info,
        prefix_len=prefix_len,
    )
    device = tree_token_ids.device
    position_ids = (tree_info.depth.view(1, -1) + root_position).to(device)
    cache_position = torch.arange(prefix_len, prefix_len + tree_info.block_size, device=device)
    outputs = target_model(
        input_ids=tree_token_ids.unsqueeze(0),
        position_ids=position_ids,
        attention_mask=None,
        past_key_values=target_cache,
        use_cache=True,
        cache_position=cache_position,
        output_hidden_states=True,
        score_mod=score_mod,
    )
    tree_ctx_features = raw_drafter.extract_ctx_features(outputs.hidden_states)
    return outputs.past_key_values, outputs.logits[0], tree_ctx_features


def build_acceptance_mask(
    *,
    draft_logits: torch.Tensor,
    draft_token_probs: torch.Tensor,
    verifier_logits: torch.Tensor,
    tree_token_ids: torch.Tensor,
    tree_info: TreeInfo,
    temperature: float,
) -> torch.Tensor:
    accepted = torch.zeros((tree_info.block_size,), dtype=torch.bool, device=tree_token_ids.device)
    accepted[0] = True
    if temperature <= 0:
        for node_idx in range(1, tree_info.block_size):
            parent_idx = int(tree_info.parent_idx[node_idx].item())
            target_token = verifier_logits[parent_idx].argmax(dim=-1)
            accepted[node_idx] = tree_token_ids[node_idx] == target_token
        return accepted

    for node_idx in range(1, tree_info.block_size):
        parent_idx = int(tree_info.parent_idx[node_idx].item())
        target_prob = gather_token_probability(
            verifier_logits[parent_idx].unsqueeze(0),
            tree_token_ids[node_idx].view(1),
            temperature,
        )[0]
        draft_prob = draft_token_probs[node_idx].clamp_min(1e-8)
        accept_prob = torch.clamp(target_prob / draft_prob, max=1.0)
        accepted[node_idx] = torch.rand((), device=tree_token_ids.device) <= accept_prob
    return accepted


@torch.inference_mode()
def speculative_generate(
    *,
    target_model,
    drafter_model,
    tokenizer,
    prompt: str,
    tree_processor,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> SpecDecodeResult:
    device = next(target_model.parameters()).device
    raw_drafter = unwrap_model(drafter_model)
    if isinstance(tree_processor, PrunableTreeProcessor) and not has_pruning_head(raw_drafter):
        raise ValueError(
            "tree_type='prunable' requires a drafter checkpoint/config with use_q_head=True or use_ar_head=True."
        )
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    target_cache = DynamicCache()
    prefill_outputs = target_model(
        input_ids=prompt_ids,
        output_hidden_states=True,
        use_cache=True,
        past_key_values=target_cache,
    )
    target_cache = prefill_outputs.past_key_values
    target_ctx_features = raw_drafter.extract_ctx_features(prefill_outputs.hidden_states)
    drafter_cache = DynamicCache()
    current_root_token = int(sample_from_logits(prefill_outputs.logits[:, -1, :], temperature)[0].item())
    output_ids = torch.cat(
        [prompt_ids[0], torch.tensor([current_root_token], dtype=torch.long, device=device)],
        dim=0,
    )
    input_len = output_ids.shape[0]
    acceptance_lengths: list[int] = []
    drafted_tokens = 0
    committed_tokens = 1
    eos_token_id = tokenizer.eos_token_id

    target_embeddings = target_model.get_input_embeddings()
    target_lm_head = target_model.get_output_embeddings()
    if target_lm_head is None:
        raise ValueError("Target model must expose an LM head for speculative decoding.")

    while committed_tokens < max_new_tokens:
        if eos_token_id is not None and current_root_token == eos_token_id:
            break
            
        prefix_len = target_cache.get_seq_length()

        root_position = output_ids.shape[0] - 1
        tree_token_ids, draft_logits, draft_token_probs, tree_info, pruning_scores = draft_tree(
            drafter_model=drafter_model,
            raw_drafter=raw_drafter,
            target_embeddings=target_embeddings,
            target_lm_head=target_lm_head,
            tree_processor=tree_processor,
            drafter_cache=drafter_cache,
            current_root_token=current_root_token,
            root_position=root_position,
            temperature=temperature,
            target_ctx_features=target_ctx_features,
            device=device,
        )
        drafted_tokens += tree_info.block_size - 1
        if isinstance(tree_processor, PrunableTreeProcessor):
            if pruning_scores is None:
                raise ValueError("Prunable speculative decoding requires q_head or AR-head scores.")
            tree_token_ids, draft_logits, draft_token_probs, tree_info = prune_drafted_tree(
                tree_token_ids=tree_token_ids,
                draft_logits=draft_logits,
                draft_token_probs=draft_token_probs,
                tree_info=tree_info,
                q_scores=pruning_scores,
                prune_topk=tree_processor.prune_topk,
            )

        updated_cache, verifier_logits, tree_ctx_features = verify_tree(
            target_model=target_model,
            raw_drafter=raw_drafter,
            tree_token_ids=tree_token_ids,
            tree_info=tree_info,
            root_position=root_position,
            target_cache=target_cache,
            temperature=temperature,
        )
        accepted_mask = build_acceptance_mask(
            draft_logits=draft_logits,
            draft_token_probs=draft_token_probs,
            verifier_logits=verifier_logits,
            tree_token_ids=tree_token_ids,
            tree_info=tree_info,
            temperature=temperature,
        )
        deepest_idx = choose_deepest_valid_node(accepted_mask, tree_info)
        accepted_path = gather_path_indices(deepest_idx, tree_info)
        acceptance_lengths.append(len(accepted_path) - 1)

        bonus_token = int(sample_from_logits(verifier_logits[deepest_idx].unsqueeze(0), temperature)[0].item())
        committed_path_tokens = tree_token_ids[accepted_path[1:]]
        step_tokens = torch.cat(
            [
                committed_path_tokens,
                torch.tensor([bonus_token], dtype=torch.long, device=device),
            ],
            dim=0,
        )
        remaining = max_new_tokens - committed_tokens
        if step_tokens.numel() > remaining:
            step_tokens = step_tokens[:remaining]
        output_ids = torch.cat([output_ids, step_tokens], dim=0)
        committed_tokens += int(step_tokens.numel())

        target_ctx_features = tree_ctx_features
        
        target_cache = trim_dynamic_cache(
            updated_cache,
            prefix_len=prefix_len,
            keep_tree_indices=accepted_path,
        )
        current_root_token = bonus_token
        if eos_token_id is not None and (output_ids[input_len:] == eos_token_id).any():
            break

    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return SpecDecodeResult(
        token_ids=output_ids.detach().cpu(),
        text=text,
        acceptance_lengths=acceptance_lengths,
        drafted_tokens=drafted_tokens,
        committed_tokens=committed_tokens,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tree speculative decoding.")
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--tree-type",
        default="block",
        choices=["fixed", "block", "branch_off", "prunable"],
    )
    parser.add_argument("--tree-seq-depth", type=int, default=4)
    parser.add_argument("--sub-tree-paths", nargs="*", default=None)
    parser.add_argument("--tree-args", type=str, default="{}")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    tree_args = json.loads(args.tree_args)
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=dtype_map[args.dtype],
        attn_implementation="flex_attention",
    ).to(device)
    target_model.eval()

    drafter_model = DFlashDraftModel.from_pretrained(
        args.draft_model,
        torch_dtype=dtype_map[args.dtype],
    ).to(device)
    drafter_model.eval()

    tree_processor = build_tree_processor(
        tree_type=args.tree_type,
        tree_seq_depth=args.tree_seq_depth,
        sub_tree_paths=args.sub_tree_paths,
        tree_args=tree_args,
    )
    result = speculative_generate(
        target_model=target_model,
        drafter_model=drafter_model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        tree_processor=tree_processor,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(result.text, flush=True)


if __name__ == "__main__":
    main()
