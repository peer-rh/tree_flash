from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from v2.stage2 import (
    IGNORE_IDX,
    DEFAULT_SUB_TREE_PATHS,
    SubTreeInfo,
    build_arg_parser,
    build_batch,
    build_step_attention_mask,
    generate_trees,
    iter_position_chunks,
    summarize_lm_head_chunk,
)


@dataclass
class DummyBaseOutput:
    last_hidden_state: torch.Tensor
    past_key_values: object | None = None


@dataclass
class DummyCausalLMOutput:
    logits: torch.Tensor
    past_key_values: object | None = None


class DummyBaseModel(nn.Module):
    def __init__(self, owner: "DummyCausalLM") -> None:
        super().__init__()
        self.owner = owner

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> DummyBaseOutput:
        hidden_states = self.owner.hidden_from_inputs(input_ids, position_ids=position_ids)
        cache = None
        if use_cache:
            prev = 0 if past_key_values is None else int(past_key_values)
            cache = prev + input_ids.shape[1]
        return DummyBaseOutput(last_hidden_state=hidden_states, past_key_values=cache)


class DummyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 16, max_positions: int = 512) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.config = SimpleNamespace(vocab_size=vocab_size, _attn_implementation="sdpa")
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(max_positions, hidden_size)
        self.mix = nn.Linear(hidden_size, hidden_size, bias=False)
        self.base_model = DummyBaseModel(self)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        with torch.no_grad():
            self.token_embed.weight.copy_(
                torch.arange(vocab_size * hidden_size, dtype=torch.float32).view(vocab_size, hidden_size) / 100.0
            )
            self.pos_embed.weight.copy_(
                torch.arange(max_positions * hidden_size, dtype=torch.float32).view(max_positions, hidden_size) / 250.0
            )
            self.mix.weight.copy_(torch.eye(hidden_size, dtype=torch.float32) * 0.5)
            self.lm_head.weight.copy_(
                torch.arange(vocab_size * hidden_size, dtype=torch.float32).view(vocab_size, hidden_size) / 150.0
            )

    def hidden_from_inputs(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        hidden_states = self.token_embed(input_ids) + self.pos_embed(position_ids)
        return torch.tanh(self.mix(hidden_states))

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> DummyCausalLMOutput:
        base_out = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        return DummyCausalLMOutput(
            logits=self.lm_head(base_out.last_hidden_state),
            past_key_values=base_out.past_key_values,
        )


@torch.inference_mode()
def generate_trees_dense_reference(batch, model, n_subtrees: int, st_info: SubTreeInfo):
    input_ids = batch["input_ids"]
    is_response = batch["is_response"]
    document_mask = batch["document_mask"]
    bsz, seq_len = input_ids.shape
    device = input_ids.device
    valid_tokens = document_mask >= 0

    subtree_ids = torch.full((bsz, seq_len, st_info.size), IGNORE_IDX, dtype=torch.long, device=device)
    subtree_ar_probs = torch.zeros((bsz, seq_len, st_info.size), dtype=torch.float32, device=device)
    subtree_ids[:, :, 0] = torch.where(valid_tokens, input_ids, torch.full_like(input_ids, IGNORE_IDX))

    out = model(
        input_ids=input_ids,
        attention_mask=valid_tokens.long(),
        use_cache=True,
    )
    logits = out.logits
    kv_cache = out.past_key_values
    vocab_size = logits.shape[-1]
    log_denom = torch.logsumexp(logits, dim=-1)
    use_flex = getattr(getattr(model, "config", None), "_attn_implementation", None) == "flex_attention"

    token_probs = torch.zeros((bsz, seq_len), dtype=torch.float32, device=device)
    if seq_len > 0:
        token_probs[:, 0] = 1.0
    if seq_len > 1:
        next_token_logits = logits[:, :-1].gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        token_probs[:, 1:] = (next_token_logits - log_denom[:, :-1]).exp().to(torch.float32)
    subtree_ar_probs[:, :, 0] = torch.where(valid_tokens, token_probs, torch.zeros_like(token_probs))

    valid_response = is_response & valid_tokens
    response_counts = valid_response.sum(dim=1)
    if int(response_counts.min().item()) > 0:
        score = torch.where(
            valid_response,
            1.0 - token_probs,
            torch.full_like(token_probs, float("-inf")),
        )
        n_select = min(n_subtrees, int(response_counts.min().item()))
        subtree_anchors = score.topk(n_select, dim=1).indices.sort(dim=1).values
    else:
        subtree_anchors = torch.empty((bsz, 0), dtype=torch.long, device=device)

    depth1_slots = st_info.nodes_at_depth.get(1, [])
    if depth1_slots:
        request_k = min(len(depth1_slots) + 1, vocab_size)
        cand_vals, cand_ids = logits.topk(request_k, dim=-1)
        has_next = torch.zeros((bsz, seq_len), dtype=torch.bool, device=device)
        excluded_token = torch.full((bsz, seq_len), IGNORE_IDX, dtype=torch.long, device=device)
        if seq_len > 1:
            has_next[:, :-1] = valid_tokens[:, 1:]
            excluded_token[:, :-1] = torch.where(
                valid_tokens[:, 1:],
                input_ids[:, 1:],
                torch.full_like(input_ids[:, 1:], IGNORE_IDX),
            )

        for batch_idx in range(bsz):
            for pos in range(seq_len):
                if not valid_tokens[batch_idx, pos]:
                    continue

                filled = 0
                banned = int(excluded_token[batch_idx, pos].item()) if has_next[batch_idx, pos] else None
                for cand_idx in range(request_k):
                    tok = int(cand_ids[batch_idx, pos, cand_idx].item())
                    if banned is not None and tok == banned:
                        continue

                    slot = depth1_slots[filled]
                    subtree_ids[batch_idx, pos, slot] = tok
                    subtree_ar_probs[batch_idx, pos, slot] = float(
                        torch.exp(cand_vals[batch_idx, pos, cand_idx] - log_denom[batch_idx, pos]).item()
                    )
                    filled += 1
                    if filled >= len(depth1_slots):
                        break

    ancestor_map = st_info.ancestor_map.to(device)
    current_depth = 1
    current_vertices_list = st_info.non_leaf_at_depth.get(current_depth, [])
    if subtree_anchors.shape[1] > 0 and current_vertices_list:
        current_vertices = torch.tensor(current_vertices_list, dtype=torch.long, device=device)
        anchor_idx = subtree_anchors.unsqueeze(-1).expand(bsz, subtree_anchors.shape[1], len(current_vertices_list))
        vertex_idx = current_vertices.view(1, 1, -1).expand(bsz, subtree_anchors.shape[1], -1)
        batch_idx = torch.arange(bsz, device=device).view(bsz, 1, 1).expand_as(anchor_idx)
        next_input_ids = subtree_ids[batch_idx, anchor_idx, vertex_idx].reshape(bsz, -1)
        root_positions = anchor_idx.reshape(bsz, -1)
        vertex_ids = current_vertices.repeat(subtree_anchors.shape[1])
        position_ids = root_positions + current_depth
        cached_root_positions = None
        cached_vertex_ids = None

        while next_input_ids.shape[1] > 0:
            current_vertex_ids = vertex_ids.unsqueeze(0).expand(bsz, -1)
            if cached_root_positions is None:
                all_tree_root_positions = root_positions
                all_tree_vertex_ids = current_vertex_ids
            else:
                all_tree_root_positions = torch.cat([cached_root_positions, root_positions], dim=1)
                all_tree_vertex_ids = torch.cat([cached_vertex_ids, current_vertex_ids], dim=1)
            attention_mask = build_step_attention_mask(
                root_positions=root_positions,
                query_vertex_ids=vertex_ids,
                tree_root_positions=all_tree_root_positions,
                tree_vertex_ids=all_tree_vertex_ids,
                document_mask=document_mask,
                valid_tokens=valid_tokens,
                ancestor_map=ancestor_map,
                ctx_len=seq_len,
                use_flex=use_flex,
            )

            out = model(
                next_input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=kv_cache,
                use_cache=True,
                kernel_options={
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "BLOCK_M1": 32,
                    "BLOCK_N1": 64,
                    "BLOCK_M2": 64,
                    "BLOCK_N2": 32,
                },
            )
            new_logits = out.logits
            kv_cache = out.past_key_values
            cached_root_positions = all_tree_root_positions
            cached_vertex_ids = all_tree_vertex_ids

            next_tokens = []
            next_roots = []
            next_vertices = []
            next_non_leaf = set(st_info.non_leaf_at_depth.get(current_depth + 1, []))
            batch_rows = torch.arange(bsz, device=device)

            for col, parent_vertex in enumerate(vertex_ids.tolist()):
                children = st_info.children_map.get(parent_vertex, [])
                if not children:
                    continue

                child_k = len(children)
                _, child_ids = new_logits[:, col, :].topk(child_k, dim=-1)
                child_probs = F.softmax(new_logits[:, col, :].float(), dim=-1).gather(-1, child_ids)

                for child_rank, child_vertex in enumerate(children):
                    anchor_pos = root_positions[:, col]
                    subtree_ids[batch_rows, anchor_pos, child_vertex] = child_ids[:, child_rank]
                    subtree_ar_probs[batch_rows, anchor_pos, child_vertex] = child_probs[:, child_rank].to(
                        torch.float32
                    )
                    if child_vertex in next_non_leaf:
                        next_tokens.append(child_ids[:, child_rank])
                        next_roots.append(anchor_pos)
                        next_vertices.append(child_vertex)

            if not next_tokens:
                break

            current_depth += 1
            next_input_ids = torch.stack(next_tokens, dim=1)
            root_positions = torch.stack(next_roots, dim=1)
            vertex_ids = torch.tensor(next_vertices, dtype=torch.long, device=device)
            position_ids = root_positions + current_depth

    result = dict(batch)
    result["subtree_ids"] = subtree_ids.masked_fill(~valid_tokens.unsqueeze(-1), IGNORE_IDX)
    result["subtree_ar_probs"] = subtree_ar_probs * valid_tokens.unsqueeze(-1)
    return result


@pytest.mark.parametrize("chunk_size", [1, 2, 5])
def test_summarize_lm_head_chunk_matches_full_projection(chunk_size: int) -> None:
    torch.manual_seed(0)
    hidden_states = torch.randn(2, 5, 7)
    lm_head = nn.Linear(7, 11, bias=False)
    gather_ids = torch.tensor([[1, 3, 5, 7, 9], [0, 2, 4, 6, 8]], dtype=torch.long)
    topk = 4

    full_logits = lm_head(hidden_states)
    full_log_denom = torch.logsumexp(full_logits, dim=-1)
    full_gathered = full_logits.gather(-1, gather_ids.unsqueeze(-1)).squeeze(-1)
    full_top_vals, full_top_ids = full_logits.topk(topk, dim=-1)
    full_top_probs = F.softmax(full_logits.float(), dim=-1).gather(-1, full_top_ids)

    log_denom_chunks = []
    gathered_chunks = []
    top_vals_chunks = []
    top_ids_chunks = []
    top_probs_chunks = []

    for pos_chunk in iter_position_chunks(hidden_states.shape[1], chunk_size):
        outputs = summarize_lm_head_chunk(
            hidden_states[:, pos_chunk, :],
            lm_head,
            gather_token_ids=gather_ids[:, pos_chunk],
            topk=topk,
            compute_top_probs=True,
        )
        log_denom_chunks.append(outputs.log_denom)
        gathered_chunks.append(outputs.gathered_logits)
        top_vals_chunks.append(outputs.top_vals)
        top_ids_chunks.append(outputs.top_ids)
        top_probs_chunks.append(outputs.top_probs)

    torch.testing.assert_close(torch.cat(log_denom_chunks, dim=1), full_log_denom)
    torch.testing.assert_close(torch.cat(gathered_chunks, dim=1), full_gathered)
    torch.testing.assert_close(torch.cat(top_vals_chunks, dim=1), full_top_vals)
    torch.testing.assert_close(torch.cat(top_ids_chunks, dim=1), full_top_ids)
    torch.testing.assert_close(torch.cat(top_probs_chunks, dim=1), full_top_probs)


@pytest.mark.parametrize("chunk_size", [1, 2, 128])
def test_generate_trees_chunked_matches_dense_reference(chunk_size: int) -> None:
    model = DummyCausalLM(vocab_size=29, hidden_size=12)
    st_info = SubTreeInfo(DEFAULT_SUB_TREE_PATHS)
    batch = build_batch(
        [
            ([1, 2], [3, 4, 5, 6, 7]),
            ([2, 3, 4], [5, 6, 7, 8, 9]),
        ],
        pad_token_id=0,
        device=torch.device("cpu"),
    )

    expected = generate_trees_dense_reference(dict(batch), model, n_subtrees=3, st_info=st_info)
    actual = generate_trees(dict(batch), model, n_subtrees=3, st_info=st_info, logit_chunk_size=chunk_size)

    torch.testing.assert_close(actual["subtree_ids"], expected["subtree_ids"])
    torch.testing.assert_close(actual["subtree_ar_probs"], expected["subtree_ar_probs"], atol=1e-6, rtol=1e-6)


def test_build_arg_parser_accepts_logit_chunk_size() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--model",
            "dummy-model",
            "--data-dir",
            "data/stage1",
            "--output",
            "data/stage2.h5",
            "--logit-chunk-size",
            "64",
        ]
    )
    assert args.logit_chunk_size == 64
