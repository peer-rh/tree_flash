from __future__ import annotations
import math

import argparse
import json
import random
from collections import defaultdict
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

IGNORE_IDX = -1
DEFAULT_SUB_TREE_PATHS = ["0-1", "0-2", "0-3", "1-4", "1-5", "2-6", "2-7"]


@dataclass
class SubTreeInfo:
    edge_list: InitVar[Sequence[str | tuple[int, int]]]

    paths: list[tuple[int, int]] = field(init=False)
    size: int = field(init=False)
    ancestor_map: torch.Tensor = field(init=False)  # [size, size], ancestor-or-self
    parent_map: dict[int, int] = field(init=False)
    children_map: dict[int, list[int]] = field(init=False)
    depth_of: list[int] = field(init=False)
    nodes_at_depth: dict[int, list[int]] = field(init=False)
    non_leaf_at_depth: dict[int, list[int]] = field(init=False)
    max_depth: int = field(init=False)

    def __post_init__(self, edge_list: Sequence[str | tuple[int, int]]) -> None:
        parsed_paths: list[tuple[int, int]] = []
        parent_map: dict[int, int] = {}
        all_nodes: set[int] = {0}

        for edge in edge_list:
            if isinstance(edge, str):
                parts = edge.split("-")
                if len(parts) != 2:
                    raise ValueError(f"Subtree edge must be 'X-Y', got {edge!r}")
                parent, child = int(parts[0]), int(parts[1])
            else:
                parent, child = int(edge[0]), int(edge[1])

            if child == 0:
                raise ValueError("Subtree root must remain node 0")
            if child in parent_map:
                raise ValueError(f"Node {child} has multiple parents")

            parsed_paths.append((parent, child))
            parent_map[child] = parent
            all_nodes.add(parent)
            all_nodes.add(child)

        sorted_nodes = sorted(all_nodes)
        expected_nodes = list(range(len(sorted_nodes)))
        if sorted_nodes != expected_nodes:
            raise ValueError(
                f"Subtree node ids must be contiguous from 0, got {sorted_nodes}"
            )

        self.paths = parsed_paths
        self.parent_map = parent_map
        self.size = len(sorted_nodes)

        children_map: dict[int, list[int]] = defaultdict(list)
        for parent, child in parsed_paths:
            children_map[parent].append(child)
        self.children_map = {node: sorted(children_map.get(node, [])) for node in range(self.size)}

        depth_of = [0] * self.size
        for node in range(1, self.size):
            cur = node
            depth = 0
            seen: set[int] = set()
            while cur != 0:
                if cur in seen:
                    raise ValueError("Cycle detected in subtree edge list")
                seen.add(cur)
                if cur not in parent_map:
                    raise ValueError(f"Node {cur} is disconnected from root 0")
                cur = parent_map[cur]
                depth += 1
            depth_of[node] = depth
        self.depth_of = depth_of
        self.max_depth = max(depth_of, default=0)

        nodes_at_depth: dict[int, list[int]] = defaultdict(list)
        non_leaf_at_depth: dict[int, list[int]] = defaultdict(list)
        for node, depth in enumerate(depth_of):
            nodes_at_depth[depth].append(node)
            if self.children_map.get(node):
                non_leaf_at_depth[depth].append(node)
        self.nodes_at_depth = {depth: sorted(nodes) for depth, nodes in nodes_at_depth.items()}
        self.non_leaf_at_depth = {
            depth: sorted(nodes) for depth, nodes in non_leaf_at_depth.items()
        }

        anc = torch.zeros(self.size, self.size, dtype=torch.bool)
        for q in range(self.size):
            cur = q
            while True:
                anc[cur, q] = True
                if cur == 0:
                    break
                cur = parent_map[cur]
        self.ancestor_map = anc



def build_step_attention_mask(
    *,
    root_positions: torch.Tensor,           # [B, Q]
    query_vertex_ids: torch.Tensor,        # [Q]
    tree_root_positions: torch.Tensor,     # [B, K_tree]
    tree_vertex_ids: torch.Tensor,         # [B, K_tree]
    document_mask: torch.Tensor,           # [B, S]
    valid_tokens: torch.Tensor,            # [B, S]
    ancestor_map: torch.Tensor,            # [st_size, st_size]
    ctx_len: int,
    use_flex: bool,
):
    device = root_positions.device
    B, q_count = root_positions.shape
    k_count = tree_root_positions.shape[1]

    if use_flex:
        try:
            from torch.nn.attention.flex_attention import create_block_mask
        except ImportError as exc:
            raise RuntimeError(
                "Flex attention is not available in this PyTorch build. "
                "Use a PyTorch version with torch.nn.attention.flex_attention "
                "or rerun with --attn-implementation sdpa."
            ) from exc
        

        # ancestor_map = ancestor_map.T.clone()
        ancestor_map = torch.eye(ancestor_map.shape[0], dtype=torch.bool, device=device) 
        print(query_vertex_ids, tree_vertex_ids)
        def mask_mod(b, h, q_idx, kv_idx):
            q_root = root_positions[b, q_idx]

            in_ctx = kv_idx < ctx_len
            ctx_idx = kv_idx.clamp(0, max(ctx_len - 1, 0))
            same_doc = document_mask[b, ctx_idx] == document_mask[b, q_root]
            causal_ctx = ctx_idx <= q_root
            ctx_mask = in_ctx & same_doc & causal_ctx & valid_tokens[b, ctx_idx]

            tree_idx = (kv_idx - ctx_len).clamp(0, max(k_count - 1, 0))
            same_tree = tree_root_positions[b, tree_idx] == q_root
            key_vertex = tree_vertex_ids[b, tree_idx]
            query_vertex = query_vertex_ids[q_idx]
            tree_mask = (~in_ctx) & same_tree & ancestor_map[query_vertex, key_vertex]
            # return ctx_mask | tree_mask
            return ctx_mask | tree_mask

        return create_block_mask(
            mask_mod,
            B=B,
            H=None,
            Q_LEN=q_count,
            KV_LEN=ctx_len + k_count,
            device=device,
            BLOCK_SIZE=128
        )

    ctx_pos = torch.arange(ctx_len, device=device).view(1, 1, ctx_len)
    root_docs = document_mask.gather(1, root_positions)
    ctx_attend = (
        (document_mask.unsqueeze(1) == root_docs.unsqueeze(-1))
        & (ctx_pos <= root_positions.unsqueeze(-1))
        & valid_tokens.unsqueeze(1)
    )

    key_vertices = tree_vertex_ids.unsqueeze(1).expand(B, q_count, k_count)
    query_vertices = query_vertex_ids.view(1, q_count, 1).expand(B, q_count, k_count)
    same_tree = tree_root_positions.unsqueeze(1) == root_positions.unsqueeze(-1)
    tree_attend = same_tree & ancestor_map[key_vertices, query_vertices]

    attend = torch.cat([ctx_attend, tree_attend], dim=-1)
    mask_4d = torch.zeros(
        (B, 1, q_count, ctx_len + k_count),
        dtype=torch.float32,
        device=device,
    )
    mask_4d.masked_fill_(~attend.unsqueeze(1), float("-inf"))
    return mask_4d


@torch.inference_mode()
def generate_trees(batch, model, n_subtrees: int, st_info: SubTreeInfo):
    input_ids = batch["input_ids"]  # [B, S]
    is_response = batch["is_response"]  # [B, S]
    document_mask = batch["document_mask"]  # [B, S], -1 for padding
    B, S = input_ids.shape
    print(input_ids.shape)
    device = input_ids.device
    valid_tokens = document_mask >= 0

    subtree_ids = torch.full(
        (B, S, st_info.size),
        fill_value=IGNORE_IDX,
        dtype=torch.long,
        device=device,
    )
    subtree_ar_probs = torch.zeros((B, S, st_info.size), dtype=torch.float32, device=device)
    subtree_ids[:, :, 0] = torch.where(valid_tokens, input_ids, torch.full_like(input_ids, IGNORE_IDX))

    out = model(
        input_ids=input_ids,
        attention_mask=valid_tokens.long(),
        use_cache=True,
        output_hidden_states=False,
    )
    logits = out.logits  # [B, S, V]
    kv_cache = out.past_key_values
    vocab_size = logits.shape[-1]
    log_denom = torch.logsumexp(logits, dim=-1)  # [B, S]
    use_flex = getattr(getattr(model, "config", None), "_attn_implementation", None) == "flex_attention"

    token_probs = torch.zeros((B, S), dtype=torch.float32, device=device)
    if S > 0:
        token_probs[:, 0] = 1.0
    if S > 1:
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
        subtree_anchors = torch.empty((B, 0), dtype=torch.long, device=device)

    print(subtree_ar_probs[0, subtree_anchors[0], 0])
    depth1_slots = st_info.nodes_at_depth.get(1, [])
    if depth1_slots:
        request_k = min(len(depth1_slots) + 1, vocab_size)
        cand_vals, cand_ids = logits.topk(request_k, dim=-1)
        has_next = torch.zeros((B, S), dtype=torch.bool, device=device)
        excluded_token = torch.full((B, S), IGNORE_IDX, dtype=torch.long, device=device)
        if S > 1:
            has_next[:, :-1] = valid_tokens[:, 1:]
            excluded_token[:, :-1] = torch.where(
                valid_tokens[:, 1:],
                input_ids[:, 1:],
                torch.full_like(input_ids[:, 1:], IGNORE_IDX),
            )
        
        for b in range(B):
            for pos in range(S):
                if not valid_tokens[b, pos]:
                    continue

                filled = 0
                banned = int(excluded_token[b, pos].item()) if has_next[b, pos] else None
                for cand_idx in range(request_k):
                    tok = int(cand_ids[b, pos, cand_idx].item())
                    if banned is not None and tok == banned:
                        continue

                    slot = depth1_slots[filled]
                    subtree_ids[b, pos, slot] = tok
                    subtree_ar_probs[b, pos, slot] = float(
                        torch.exp(cand_vals[b, pos, cand_idx] - log_denom[b, pos]).item()
                    )
                    filled += 1
                    if filled >= len(depth1_slots):
                        break

    ancestor_map = st_info.ancestor_map.to(device)
    current_depth = 1
    current_vertices_list = st_info.non_leaf_at_depth.get(current_depth, [])
    if subtree_anchors.shape[1] > 0 and current_vertices_list:
        current_vertices = torch.tensor(current_vertices_list, dtype=torch.long, device=device)
        anchor_idx = subtree_anchors.unsqueeze(-1).expand(B, subtree_anchors.shape[1], len(current_vertices_list))
        vertex_idx = current_vertices.view(1, 1, -1).expand(B, subtree_anchors.shape[1], -1)
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand_as(anchor_idx)
        next_input_ids = subtree_ids[batch_idx, anchor_idx, vertex_idx].reshape(B, -1)
        root_positions = anchor_idx.reshape(B, -1)
        vertex_ids = current_vertices.repeat(subtree_anchors.shape[1])
        position_ids = root_positions + current_depth
        cached_root_positions: torch.Tensor | None = None
        cached_vertex_ids: torch.Tensor | None = None

        while next_input_ids.shape[1] > 0:
            current_vertex_ids = vertex_ids.unsqueeze(0).expand(B, -1)
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
                ctx_len=S,
                use_flex=use_flex,
            )
            print(attention_mask)
            print(position_ids.shape, next_input_ids.shape, kv_cache.get_seq_length())

            out = model(
                next_input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=kv_cache,
                use_cache=True,
            )
            new_logits = out.logits
            kv_cache = out.past_key_values
            cached_root_positions = all_tree_root_positions
            cached_vertex_ids = all_tree_vertex_ids

            next_tokens: list[torch.Tensor] = []
            next_roots: list[torch.Tensor] = []
            next_vertices: list[int] = []
            next_non_leaf = set(st_info.non_leaf_at_depth.get(current_depth + 1, []))
            batch_rows = torch.arange(B, device=device)

            for col, parent_vertex in enumerate(vertex_ids.tolist()):
                children = st_info.children_map.get(parent_vertex, [])
                if not children:
                    continue

                child_k = len(children)
                child_vals, child_ids = new_logits[:, col, :].topk(child_k, dim=-1)
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

    subtree_ids = subtree_ids.masked_fill(~valid_tokens.unsqueeze(-1), IGNORE_IDX)
    subtree_ar_probs = subtree_ar_probs * valid_tokens.unsqueeze(-1)
    batch["subtree_ids"] = subtree_ids
    batch["subtree_ar_probs"] = subtree_ar_probs
    return batch


def iter_jsonl_dir(data_dir: Path, max_len: int | None) -> Iterator[dict]:
    files = sorted(data_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in {data_dir}")
    lines = []
    for path in files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data =  json.loads(line)
                    if max_len is not None and (len(data.get("prompt", "")) + len(data.get("response", ""))) > max_len:
                        continue
                    lines.append(data)
    sorted_lines = sorted(lines, key=lambda x: - len(x.get("prompt", "")) - len(x.get("response", "")))
    return iter(sorted_lines)


def build_batch(
    examples: Sequence[tuple[list[int], list[int]]],
    pad_token_id: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    max_len = max(len(prompt_ids) + len(response_ids) for prompt_ids, response_ids in examples)
    max_len = math.ceil(max_len / 128) * 128
    B = len(examples)

    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    is_response = torch.zeros((B, max_len), dtype=torch.bool)
    document_mask = torch.full((B, max_len), -1, dtype=torch.long)

    for row, (prompt_ids, response_ids) in enumerate(examples):
        full_ids = prompt_ids + response_ids
        seq_len = len(full_ids)
        prompt_len = len(prompt_ids)
        input_ids[row, :seq_len] = torch.tensor(full_ids, dtype=torch.long)
        is_response[row, prompt_len:seq_len] = True
        document_mask[row, :seq_len] = 0

    return {
        "input_ids": input_ids.to(device),
        "is_response": is_response.to(device),
        "document_mask": document_mask.to(device),
    }


def flush_hdf5(
    hf: h5py.File,
    prompt_buf: list[np.ndarray],
    response_buf: list[np.ndarray],
    tree_buf: list[np.ndarray],
    prob_buf: list[np.ndarray],
    n_seqs_written: int,
    n_rows_written: int,
) -> tuple[int, int]:
    if not prompt_buf:
        return n_seqs_written, n_rows_written

    ds_prompt = hf["prompt_ids"]
    ds_response = hf["response_ids"]
    ds_trees = hf["sub_trees"]
    ds_probs = hf["sub_trees_ar_probs"]
    ds_offsets = hf["sequence_offsets"]

    n_new_seqs = len(prompt_buf)
    new_seq_total = n_seqs_written + n_new_seqs
    ds_prompt.resize(new_seq_total, axis=0)
    ds_response.resize(new_seq_total, axis=0)
    for idx, arr in enumerate(prompt_buf):
        ds_prompt[n_seqs_written + idx] = arr
    for idx, arr in enumerate(response_buf):
        ds_response[n_seqs_written + idx] = arr

    combined_trees = np.concatenate(tree_buf, axis=0)
    combined_probs = np.concatenate(prob_buf, axis=0)
    new_rows = combined_trees.shape[0]
    ds_trees.resize(n_rows_written + new_rows, axis=0)
    ds_probs.resize(n_rows_written + new_rows, axis=0)
    ds_trees[n_rows_written : n_rows_written + new_rows] = combined_trees
    ds_probs[n_rows_written : n_rows_written + new_rows] = combined_probs

    offsets = [n_rows_written]
    for arr in tree_buf:
        offsets.append(offsets[-1] + arr.shape[0])
    ds_offsets.resize(new_seq_total + 1, axis=0)
    ds_offsets[n_seqs_written + 1 : new_seq_total + 1] = np.array(offsets[1:], dtype=np.int64)

    prompt_buf.clear()
    response_buf.clear()
    tree_buf.clear()
    prob_buf.clear()
    return new_seq_total, n_rows_written + new_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 v2: batched subtree generation with KV caching.")
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument("--data-dir", required=True, help="Directory of stage-1 JSONL shards")
    parser.add_argument("--output", required=True, help="Output HDF5 path")
    parser.add_argument(
        "--sub-tree-paths",
        nargs="+",
        default=DEFAULT_SUB_TREE_PATHS,
        help='Subtree edges as "X-Y" strings',
    )
    parser.add_argument("--n-subtrees", type=int, default=64, help="Anchor positions per sequence")
    parser.add_argument("--batch-size", type=int, default=1, help="Sequences per batch")
    parser.add_argument("--max-sequences", type=int, default=None, help="Optional cap on sequences")
    parser.add_argument(
        "--attn-implementation",
        choices=["flex_attention", "sdpa"],
        default="flex_attention",
        help="Attention backend for the HF model",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model load dtype",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="Optional max prompt+response length; longer ones are skipped",
    )
    args = parser.parse_args()

    if args.batch_size > 1:
        print(
            "Note: batch_size > 1 uses padded sequence batches; anchor count is clipped "
            "to the minimum valid response-token count in the batch.",
            flush=True,
        )

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
    )
    if tokenizer.pad_token_id is not None and getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "resize_token_embeddings") and len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()

    st_info = SubTreeInfo(args.sub_tree_paths)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vlen_int64 = h5py.vlen_dtype(np.int64)
    prompt_buf: list[np.ndarray] = []
    response_buf: list[np.ndarray] = []
    tree_buf: list[np.ndarray] = []
    prob_buf: list[np.ndarray] = []
    pending: list[tuple[list[int], list[int]]] = []
    FLUSH_EVERY = 128

    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("prompt_ids", shape=(0,), maxshape=(None,), dtype=vlen_int64)
        hf.create_dataset("response_ids", shape=(0,), maxshape=(None,), dtype=vlen_int64)
        hf.create_dataset(
            "sub_trees",
            shape=(0, st_info.size),
            maxshape=(None, st_info.size),
            dtype="int64",
            chunks=(512, st_info.size),
            compression="lzf",
        )
        hf.create_dataset(
            "sub_trees_ar_probs",
            shape=(0, st_info.size),
            maxshape=(None, st_info.size),
            dtype="float32",
            chunks=(512, st_info.size),
            compression="lzf",
        )
        hf.create_dataset("sequence_offsets", shape=(1,), maxshape=(None,), dtype="int64")
        hf["sequence_offsets"][0] = 0
        hf.attrs["sub_tree_paths"] = np.array(args.sub_tree_paths, dtype=h5py.string_dtype())

        n_seqs_written = 0
        n_rows_written = 0

        for record in iter_jsonl_dir(Path(args.data_dir), args.max_len):
            if args.max_sequences is not None and n_seqs_written + len(pending) >= args.max_sequences:
                break

            prompt_ids = tokenizer.encode(record["prompt"], add_special_tokens=False)
            response_ids = tokenizer.encode(record["response"], add_special_tokens=False)
            if not response_ids:
                continue

            pending.append((prompt_ids, response_ids))
            if len(pending) < args.batch_size:
                continue

            batch = build_batch(pending, tokenizer.pad_token_id, device)
            batch = generate_trees(batch, model, args.n_subtrees, st_info)

            for row, (prompt_ids_row, response_ids_row) in enumerate(pending):
                prompt_len = len(prompt_ids_row)
                response_len = len(response_ids_row)
                seq_slice = slice(prompt_len, prompt_len + response_len)
                subtrees = batch["subtree_ids"][row, seq_slice].detach().cpu().numpy().astype(np.int64, copy=False)
                subtree_probs = (
                    batch["subtree_ar_probs"][row, seq_slice].detach().cpu().numpy().astype(np.float32, copy=False)
                )

                prompt_buf.append(np.asarray(prompt_ids_row, dtype=np.int64))
                response_buf.append(np.asarray(response_ids_row, dtype=np.int64))
                tree_buf.append(subtrees)
                prob_buf.append(subtree_probs)

            pending.clear()

            if len(prompt_buf) >= FLUSH_EVERY:
                n_seqs_written, n_rows_written = flush_hdf5(
                    hf,
                    prompt_buf,
                    response_buf,
                    tree_buf,
                    prob_buf,
                    n_seqs_written,
                    n_rows_written,
                )
                print(f"Sequences: {n_seqs_written}  Response rows: {n_rows_written}", flush=True)

        if pending:
            batch = build_batch(pending, tokenizer.pad_token_id, device)
            batch = generate_trees(batch, model, args.n_subtrees, st_info)

            for row, (prompt_ids_row, response_ids_row) in enumerate(pending):
                prompt_len = len(prompt_ids_row)
                response_len = len(response_ids_row)
                seq_slice = slice(prompt_len, prompt_len + response_len)
                subtrees = batch["subtree_ids"][row, seq_slice].detach().cpu().numpy().astype(np.int64, copy=False)
                subtree_probs = (
                    batch["subtree_ar_probs"][row, seq_slice].detach().cpu().numpy().astype(np.float32, copy=False)
                )

                prompt_buf.append(np.asarray(prompt_ids_row, dtype=np.int64))
                response_buf.append(np.asarray(response_ids_row, dtype=np.int64))
                tree_buf.append(subtrees)
                prob_buf.append(subtree_probs)

        n_seqs_written, n_rows_written = flush_hdf5(
            hf,
            prompt_buf,
            response_buf,
            tree_buf,
            prob_buf,
            n_seqs_written,
            n_rows_written,
        )

    print(f"Done. Sequences: {n_seqs_written}  Response rows: {n_rows_written}")


if __name__ == "__main__":
    torch.compiler.reset()
    main()
