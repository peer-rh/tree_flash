"""Stage 3: Multi-anchor tree trainer with flex attention.

Supports: Lightning Fabric (DDP), torch.compile, bf16-mixed, checkpoint resume, Wandb.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lightning.fabric import Fabric
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Path setup so we can import from sibling packages
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "dflash"))
sys.path.insert(0, str(_ROOT / "v2"))

from model.dflash import DFlashDraftModel  # noqa: E402
from model.utils import extract_context_feature  # noqa: E402
from stage2 import SubTreeInfo, IGNORE_IDX, DEFAULT_SUB_TREE_PATHS  # noqa: E402

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TreeFlashHDF5Dataset(Dataset):
    def __init__(self, hdf5_path: str):
        self.path = hdf5_path
        with h5py.File(hdf5_path, "r") as hf:
            self.offsets = hf["sequence_offsets"][:]  # [N+1]
        self.n = len(self.offsets) - 1

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        with h5py.File(self.path, "r") as hf:
            prompt_ids = torch.tensor(hf["prompt_ids"][idx], dtype=torch.long)
            response_ids = torch.tensor(hf["response_ids"][idx], dtype=torch.long)
            start = int(self.offsets[idx])
            end = int(self.offsets[idx + 1])
            sub_trees = torch.tensor(hf["sub_trees"][start:end], dtype=torch.long)
            sub_trees_ar_probs = torch.tensor(hf["sub_trees_ar_probs"][start:end], dtype=torch.float32)
        return {
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "sub_trees": sub_trees,                    # [S_R, st_size]
            "sub_trees_ar_probs": sub_trees_ar_probs,  # [S_R, st_size]
        }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class TreeFlashCollator:
    def __init__(
        self,
        max_ctx_len: int,
        num_anchors: int,
        tree_seq_depth: int,
        st_info: SubTreeInfo,
        mask_token_id: int,
        pad_token_id: int,
    ):
        self.max_ctx_len = max_ctx_len
        self.num_anchors = num_anchors
        self.tree_seq_depth = tree_seq_depth
        self.st_info = st_info
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.block_per_anchor = tree_seq_depth * st_info.size

        # Precompute cumprob path for each subtree node:
        # path from node to root (list of node ids, from node upward)
        self._node_to_root_path: list[list[int]] = []
        for v in range(st_info.size):
            path = []
            cur = v
            while cur != 0:
                path.append(cur)
                cur = st_info.parent_map[cur]
            path.append(0)
            self._node_to_root_path.append(list(reversed(path)))  # root-first

    def _compute_cumprobs(
        self,
        ar_probs: torch.Tensor,  # [tree_seq_depth, st_size]
    ) -> torch.Tensor:
        """Returns cumulative probability weights for all tree_seq_depth*st_size nodes."""
        tsd = self.tree_seq_depth
        st_size = self.st_info.size
        weights = torch.zeros(tsd * st_size, dtype=torch.float32)

        # primary-path cumprob at depth d = product of root node probs 0..d
        primary_cumprob = 1.0
        for d in range(tsd):
            primary_cumprob = primary_cumprob * float(ar_probs[d, 0].item())
            # within subtree at depth d, walk each node's path from root
            for v in range(st_size):
                idx = d * st_size + v
                if v == 0:
                    weights[idx] = primary_cumprob
                else:
                    # multiply primary_cumprob by intra-subtree path probs
                    # path from root (node 0) to v, excluding root itself
                    path = self._node_to_root_path[v]  # [0, ..., v]
                    cp = primary_cumprob
                    for node in path[1:]:  # skip root
                        cp = cp * float(ar_probs[d, node].item())
                    weights[idx] = cp
        return weights

    def __call__(self, samples: list[dict]) -> dict:
        max_ctx = self.max_ctx_len
        pad = self.pad_token_id
        tsd = self.tree_seq_depth
        st_size = self.st_info.size
        block_per_anchor = self.block_per_anchor

        batch_ctx_ids = []
        batch_ctx_doc_mask = []
        batch_ctx_valid = []
        batch_block_token_ids = []
        batch_block_noise_ids = []
        batch_block_anchor_pos = []
        batch_block_vertex_ids = []
        batch_block_position_ids = []
        batch_block_cumprob_weights = []
        batch_block_valid = []

        for sample in samples:
            prompt_ids = sample["prompt_ids"]    # [P]
            response_ids = sample["response_ids"]  # [R]
            sub_trees = sample["sub_trees"]        # [R, st_size]
            sub_trees_ar_probs = sample["sub_trees_ar_probs"]  # [R, st_size]

            full_ids = torch.cat([prompt_ids, response_ids])  # [P+R]
            seq_len = len(full_ids)
            prompt_len = len(prompt_ids)
            response_len = len(response_ids)

            # Context buffer
            ctx_len = min(seq_len, max_ctx)
            ctx_ids = torch.full((max_ctx,), pad, dtype=torch.long)
            ctx_ids[:ctx_len] = full_ids[:ctx_len]

            ctx_doc = torch.full((max_ctx,), -1, dtype=torch.long)
            ctx_doc[:ctx_len] = 0

            ctx_valid = torch.zeros(max_ctx, dtype=torch.bool)
            ctx_valid[:ctx_len] = True

            # ---- sample anchors ----
            valid_anchors = []
            for a_local in range(response_len - tsd + 1):
                a_ctx = prompt_len + a_local
                if a_ctx >= max_ctx:
                    break
                if int(sub_trees[a_local, 0].item()) == IGNORE_IDX:
                    continue
                valid_anchors.append((a_local, a_ctx))

            if valid_anchors:
                n_select = min(self.num_anchors, len(valid_anchors))
                chosen = random.sample(valid_anchors, n_select)
                chosen.sort(key=lambda x: x[0])
            else:
                chosen = []

            # ---- build block tensors ----
            n_anchors = len(chosen)
            total_block = n_anchors * block_per_anchor

            block_token_ids = torch.full((total_block,), IGNORE_IDX, dtype=torch.long)
            block_noise_ids = torch.full((total_block,), self.mask_token_id, dtype=torch.long)
            block_anchor_pos = torch.zeros(total_block, dtype=torch.long)
            block_vertex_ids = torch.zeros(total_block, dtype=torch.long)
            block_position_ids = torch.zeros(total_block, dtype=torch.long)
            block_cumprob_w = torch.zeros(total_block, dtype=torch.float32)
            block_valid_mask = torch.zeros(total_block, dtype=torch.bool)

            for anc_i, (a_local, a_ctx) in enumerate(chosen):
                base = anc_i * block_per_anchor

                tree_slice = sub_trees[a_local : a_local + tsd]           # [tsd, st_size]
                probs_slice = sub_trees_ar_probs[a_local : a_local + tsd]  # [tsd, st_size]

                cumprobs = self._compute_cumprobs(probs_slice)  # [tsd * st_size]

                for d in range(tsd):
                    for v in range(st_size):
                        flat = base + d * st_size + v
                        tok = int(tree_slice[d, v].item())
                        block_token_ids[flat] = tok
                        if v == 0:
                            block_noise_ids[flat] = tok if tok != IGNORE_IDX else self.mask_token_id
                        else:
                            block_noise_ids[flat] = self.mask_token_id
                        block_anchor_pos[flat] = a_ctx
                        block_vertex_ids[flat] = v
                        block_position_ids[flat] = a_ctx + d + self.st_info.depth_of[v]
                        block_cumprob_w[flat] = float(cumprobs[d * st_size + v].item())
                        block_valid_mask[flat] = (tok != IGNORE_IDX)

            batch_ctx_ids.append(ctx_ids)
            batch_ctx_doc_mask.append(ctx_doc)
            batch_ctx_valid.append(ctx_valid)
            batch_block_token_ids.append(block_token_ids)
            batch_block_noise_ids.append(block_noise_ids)
            batch_block_anchor_pos.append(block_anchor_pos)
            batch_block_vertex_ids.append(block_vertex_ids)
            batch_block_position_ids.append(block_position_ids)
            batch_block_cumprob_weights.append(block_cumprob_w)
            batch_block_valid.append(block_valid_mask)

        # ---- pad block dimension across batch ----
        max_block = max(t.shape[0] for t in batch_block_token_ids) if batch_block_token_ids else 0

        def pad_block(tensors: list[torch.Tensor], fill) -> torch.Tensor:
            out = []
            for t in tensors:
                pad_len = max_block - t.shape[0]
                if pad_len > 0:
                    t = torch.cat([t, torch.full((pad_len,), fill, dtype=t.dtype)])
                out.append(t)
            return torch.stack(out)

        return {
            "ctx_input_ids": torch.stack(batch_ctx_ids),                          # [B, max_ctx]
            "ctx_document_mask": torch.stack(batch_ctx_doc_mask),                 # [B, max_ctx]
            "ctx_valid": torch.stack(batch_ctx_valid),                            # [B, max_ctx]
            "block_token_ids": pad_block(batch_block_token_ids, IGNORE_IDX),      # [B, max_block]
            "block_noise_ids": pad_block(batch_block_noise_ids, self.mask_token_id),
            "block_anchor_positions": pad_block(batch_block_anchor_pos, 0),
            "block_vertex_ids": pad_block(batch_block_vertex_ids, 0),
            "block_position_ids": pad_block(batch_block_position_ids, 0),
            "block_cumprob_weights": pad_block(batch_block_cumprob_weights, 0.0),
            "block_valid": pad_block(batch_block_valid, False),
        }


# ---------------------------------------------------------------------------
# Tree Position Embeddings
# ---------------------------------------------------------------------------

REL_SELF = 0
REL_PARENT = 1
REL_CHILD = 2
REL_SIBLING = 3
REL_ANCESTOR = 4
REL_DESCENDANT = 5
REL_OTHER = 6
NUM_RELATIONS = 7


def build_relation_matrix(st_info: SubTreeInfo) -> torch.Tensor:
    """Returns [st_size, st_size] int tensor of relation type from q to k."""
    n = st_info.size
    rel = torch.full((n, n), REL_OTHER, dtype=torch.long)
    for q in range(n):
        for k in range(n):
            if q == k:
                rel[q, k] = REL_SELF
            elif st_info.parent_map.get(q) == k:
                rel[q, k] = REL_PARENT
            elif st_info.parent_map.get(k) == q:
                rel[q, k] = REL_CHILD
            elif (
                q != 0
                and k != 0
                and st_info.parent_map.get(q) == st_info.parent_map.get(k)
            ):
                rel[q, k] = REL_SIBLING
            elif bool(st_info.ancestor_map[k, q].item()):  # k is ancestor of q
                rel[q, k] = REL_ANCESTOR
            elif bool(st_info.ancestor_map[q, k].item()):  # q is ancestor of k
                rel[q, k] = REL_DESCENDANT
    return rel


class TreePositionEmbedding(nn.Module):
    def __init__(
        self,
        st_size: int,
        hidden_size: int,
        num_heads: int,
        st_info: SubTreeInfo,
        use_initial: bool = True,
        use_relational: bool = True,
    ):
        super().__init__()
        self.use_initial = use_initial
        self.use_relational = use_relational
        self.num_heads = num_heads
        self.st_size = st_size

        if use_initial:
            self.vertex_embedding = nn.Embedding(st_size, hidden_size)
            nn.init.zeros_(self.vertex_embedding.weight)

        if use_relational:
            self.relation_bias = nn.Embedding(NUM_RELATIONS, num_heads)
            nn.init.zeros_(self.relation_bias.weight)
            rel_matrix = build_relation_matrix(st_info)
            self.register_buffer("relation_matrix", rel_matrix)  # [st_size, st_size]

    def get_initial_bias(self, vertex_ids: torch.Tensor) -> torch.Tensor:
        """vertex_ids: [B, block_len] -> [B, block_len, hidden_size]"""
        if not self.use_initial:
            return None
        return self.vertex_embedding(vertex_ids % self.st_size)

    def get_score_mod(self, block_vertex_ids_captured: torch.Tensor, ctx_len: int, block_len: int):
        """Returns a score_mod closure for flex_attention."""
        if not self.use_relational:
            return None

        rel_matrix = self.relation_matrix  # [st_size, st_size]
        relation_bias = self.relation_bias  # nn.Embedding

        def score_mod(score, b, h, q_idx, kv_idx):
            in_ctx = kv_idx < ctx_len
            blk_idx = (kv_idx - ctx_len).clamp(min=0, max=block_len - 1)
            q_v = block_vertex_ids_captured[b, q_idx] % self.st_size
            k_v = block_vertex_ids_captured[b, blk_idx] % self.st_size
            rel_type = rel_matrix[q_v, k_v]
            bias = relation_bias(rel_type)[h]
            return score + torch.where(in_ctx, torch.zeros_like(bias), bias)

        return score_mod


# ---------------------------------------------------------------------------
# Flex attention helpers
# ---------------------------------------------------------------------------

def build_flex_mask(
    ctx_input_ids_shape,
    block_anchor_positions: torch.Tensor,  # [B, block_len]
    ctx_document_mask: torch.Tensor,       # [B, ctx_len]
    ctx_valid: torch.Tensor,               # [B, ctx_len]
    block_valid: torch.Tensor,             # [B, block_len]
    ctx_len: int,
    block_len: int,
    device: torch.device,
):
    from torch.nn.attention.flex_attention import create_block_mask

    B = block_anchor_positions.shape[0]

    _anc = block_anchor_positions
    _doc = ctx_document_mask
    _ctx_valid = ctx_valid
    _blk_valid = block_valid

    def mask_mod(b, h, q_idx, kv_idx):
        in_ctx = kv_idx < ctx_len

        # Context side
        ctx_pos = kv_idx.clamp(max=ctx_len - 1)
        q_anchor = _anc[b, q_idx]
        same_doc = _doc[b, ctx_pos] == _doc[b, q_anchor]
        causal = ctx_pos < q_anchor
        ctx_ok = in_ctx & same_doc & causal & _ctx_valid[b, ctx_pos]

        # Block side: full bidirectional within same anchor
        blk_idx = (kv_idx - ctx_len).clamp(min=0, max=block_len - 1)
        same_anchor = _anc[b, blk_idx] == q_anchor
        tree_ok = (~in_ctx) & same_anchor & _blk_valid[b, blk_idx]

        return ctx_ok | tree_ok

    block_mask = create_block_mask(
        mask_mod,
        B=B,
        H=None,
        Q_LEN=block_len,
        KV_LEN=ctx_len + block_len,
        device=device,
        BLOCK_SIZE=128,
    )
    return block_mask


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def forward_and_loss(
    batch: dict,
    target_model,
    draft_model: DFlashDraftModel,
    tree_pos_emb: TreePositionEmbedding,
    ctx_len: int,
    use_flex: bool,
) -> torch.Tensor:
    # Fabric handles device placement via setup_dataloaders, so tensors are already on device
    ctx_input_ids = batch["ctx_input_ids"]           # [B, ctx_len]
    ctx_valid = batch["ctx_valid"]                   # [B, ctx_len]
    ctx_doc_mask = batch["ctx_document_mask"]        # [B, ctx_len]
    block_noise_ids = batch["block_noise_ids"]       # [B, block_len]
    block_token_ids = batch["block_token_ids"]       # [B, block_len]
    block_anchor_pos = batch["block_anchor_positions"]
    block_vertex_ids = batch["block_vertex_ids"]
    block_position_ids = batch["block_position_ids"]
    block_cumprob_w = batch["block_cumprob_weights"]
    block_valid = batch["block_valid"]

    B, block_len = block_noise_ids.shape
    device = block_noise_ids.device

    if block_len == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # A: Target forward (frozen)
    with torch.no_grad():
        target_out = target_model(
            input_ids=ctx_input_ids,
            attention_mask=ctx_valid.long(),
            output_hidden_states=True,
        )
        # [B, ctx_len, n_layers * hidden_size]
        target_hidden = extract_context_feature(
            target_out.hidden_states, draft_model.target_layer_ids
        )

    # B: Noise embeddings + initial tree position bias
    noise_embeds = target_model.model.embed_tokens(block_noise_ids)  # [B, block_len, H]
    if tree_pos_emb.use_initial:
        noise_embeds = noise_embeds + tree_pos_emb.get_initial_bias(block_vertex_ids)

    # C: Attention mask
    if use_flex:
        attn_mask = build_flex_mask(
            ctx_input_ids.shape,
            block_anchor_pos,
            ctx_doc_mask,
            ctx_valid,
            block_valid,
            ctx_len,
            block_len,
            device,
        )
        score_mod_fn = tree_pos_emb.get_score_mod(block_vertex_ids, ctx_len, block_len)
    else:
        attn_mask = _build_dense_mask(
            block_anchor_pos, ctx_doc_mask, ctx_valid, block_valid,
            ctx_len, block_len, device
        )
        score_mod_fn = None

    # D: Draft forward
    draft_hidden = draft_model(
        position_ids=block_position_ids,
        attention_mask=attn_mask,
        noise_embedding=noise_embeds,
        target_hidden=target_hidden,
    )  # [B, block_len, H]

    vocab_size = target_model.config.vocab_size
    logits = target_model.lm_head(draft_hidden)  # [B, block_len, V]

    # E: CumProds-weighted cross-entropy
    flat_logits = logits.reshape(-1, vocab_size).float()
    flat_labels = block_token_ids.reshape(-1)
    flat_weights = block_cumprob_w.reshape(-1)
    flat_valid = block_valid.reshape(-1)

    loss_flat = F.cross_entropy(flat_logits, flat_labels, ignore_index=IGNORE_IDX, reduction="none")
    weighted = loss_flat * flat_weights
    valid_losses = weighted[flat_valid & (flat_labels != IGNORE_IDX)]
    if valid_losses.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return valid_losses.mean()


def _build_dense_mask(
    block_anchor_pos, ctx_doc_mask, ctx_valid, block_valid,
    ctx_len, block_len, device
) -> torch.Tensor:
    """Dense 4D additive mask [B, 1, block_len, ctx_len+block_len] for debugging."""
    B = block_anchor_pos.shape[0]
    full_len = ctx_len + block_len

    ctx_pos = torch.arange(ctx_len, device=device).view(1, 1, ctx_len)
    q_anchor = block_anchor_pos.unsqueeze(-1)  # [B, block_len, 1]

    # context side
    same_doc = (
        ctx_doc_mask.unsqueeze(1) == ctx_doc_mask.gather(1, block_anchor_pos).unsqueeze(-1)
    )  # [B, block_len, ctx_len]
    causal = ctx_pos < q_anchor  # [B, block_len, ctx_len]
    ctx_ok = same_doc & causal & ctx_valid.unsqueeze(1)  # [B, block_len, ctx_len]

    # block side
    anc_q = block_anchor_pos.unsqueeze(-1)   # [B, block_len, 1]
    anc_k = block_anchor_pos.unsqueeze(1)    # [B, 1, block_len]
    tree_ok = (anc_q == anc_k) & block_valid.unsqueeze(1)  # [B, block_len, block_len]

    attend = torch.cat([ctx_ok, tree_ok], dim=-1)  # [B, block_len, full_len]
    mask4d = torch.zeros(B, 1, block_len, full_len, device=device)
    mask4d.masked_fill_(~attend.unsqueeze(1), float("-inf"))
    return mask4d


# ---------------------------------------------------------------------------
# Eval metrics
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate(
    eval_loader: DataLoader,
    target_model,
    draft_model: DFlashDraftModel,
    tree_pos_emb: TreePositionEmbedding,
    st_info: SubTreeInfo,
    ctx_len: int,
    use_flex: bool,
    max_batches: int = 50,
) -> dict:
    draft_model.eval()
    tree_pos_emb.eval()

    total_loss = 0.0
    n_loss = 0

    st_size = st_info.size
    eq_counts = torch.zeros(st_size, st_size, dtype=torch.float32)
    eq_total = torch.zeros(st_size, st_size, dtype=torch.float32)
    sibling_eq_sum = 0.0
    sibling_eq_count = 0

    accept_lens = []
    node_hist = torch.zeros(st_size, dtype=torch.long)
    leftmost_count = 0
    total_trees = 0

    sibling_pairs = []
    for v in range(st_size):
        for u in range(v + 1, st_size):
            if (
                v != 0 and u != 0
                and st_info.parent_map.get(v) == st_info.parent_map.get(u)
            ):
                sibling_pairs.append((v, u))

    leftmost_nodes = set()
    cur = 0
    while True:
        leftmost_nodes.add(cur)
        children = st_info.children_map.get(cur, [])
        if not children:
            break
        cur = children[0]

    for batch_idx, batch in enumerate(eval_loader):
        if batch_idx >= max_batches:
            break

        ctx_input_ids = batch["ctx_input_ids"]
        ctx_valid = batch["ctx_valid"]
        ctx_doc_mask = batch["ctx_document_mask"]
        block_noise_ids = batch["block_noise_ids"]
        block_token_ids = batch["block_token_ids"]
        block_anchor_pos = batch["block_anchor_positions"]
        block_vertex_ids = batch["block_vertex_ids"]
        block_position_ids = batch["block_position_ids"]
        block_cumprob_w = batch["block_cumprob_weights"]
        block_valid = batch["block_valid"]

        B, block_len = block_noise_ids.shape
        device = block_noise_ids.device
        if block_len == 0:
            continue

        # Target forward
        target_out = target_model(
            input_ids=ctx_input_ids,
            attention_mask=ctx_valid.long(),
            output_hidden_states=True,
        )
        target_hidden = extract_context_feature(target_out.hidden_states, draft_model.target_layer_ids)

        # Noise embeds
        noise_embeds = target_model.model.embed_tokens(block_noise_ids)
        if tree_pos_emb.use_initial:
            noise_embeds = noise_embeds + tree_pos_emb.get_initial_bias(block_vertex_ids)

        if use_flex:
            attn_mask = build_flex_mask(
                ctx_input_ids.shape, block_anchor_pos, ctx_doc_mask,
                ctx_valid, block_valid, ctx_len, block_len, device,
            )
        else:
            attn_mask = _build_dense_mask(
                block_anchor_pos, ctx_doc_mask, ctx_valid, block_valid,
                ctx_len, block_len, device,
            )

        draft_hidden = draft_model(
            position_ids=block_position_ids,
            attention_mask=attn_mask,
            noise_embedding=noise_embeds,
            target_hidden=target_hidden,
        )
        logits = target_model.lm_head(draft_hidden)  # [B, block_len, V]
        vocab_size = target_model.config.vocab_size

        # Loss
        flat_logits = logits.reshape(-1, vocab_size).float()
        flat_labels = block_token_ids.reshape(-1)
        flat_weights = block_cumprob_w.reshape(-1)
        flat_valid = block_valid.reshape(-1)
        loss_flat = F.cross_entropy(flat_logits, flat_labels, ignore_index=IGNORE_IDX, reduction="none")
        weighted = loss_flat * flat_weights
        valid_mask = flat_valid & (flat_labels != IGNORE_IDX)
        if valid_mask.any():
            total_loss += weighted[valid_mask].mean().item()
            n_loss += 1

        # Per-anchor metrics
        predicted = logits.argmax(dim=-1)  # [B, block_len]

        # Equality heatmap
        for b in range(B):
            unique_anchors = block_anchor_pos[b].unique()
            for anc in unique_anchors:
                anc_mask = block_anchor_pos[b] == anc
                anc_pred = predicted[b][anc_mask]
                anc_vids = block_vertex_ids[b][anc_mask]
                anc_valid = block_valid[b][anc_mask]

                n_anc = anc_pred.shape[0]
                for i in range(n_anc):
                    if not anc_valid[i]:
                        continue
                    vi = int(anc_vids[i].item()) % st_size
                    for j in range(n_anc):
                        if not anc_valid[j]:
                            continue
                        vj = int(anc_vids[j].item()) % st_size
                        eq_total[vi, vj] += 1
                        if anc_pred[i] == anc_pred[j]:
                            eq_counts[vi, vj] += 1

        # Sibling equality rate
        for b in range(B):
            unique_anchors = block_anchor_pos[b].unique()
            for anc in unique_anchors:
                anc_mask = block_anchor_pos[b] == anc
                anc_pred = predicted[b][anc_mask]
                anc_valid_m = block_valid[b][anc_mask]

                n_copies = anc_pred.shape[0] // st_size
                for copy_i in range(n_copies):
                    off = copy_i * st_size
                    for sv, su in sibling_pairs:
                        if off + sv >= len(anc_pred) or off + su >= len(anc_pred):
                            continue
                        if anc_valid_m[off + sv] and anc_valid_m[off + su]:
                            sibling_eq_sum += float(anc_pred[off + sv] == anc_pred[off + su])
                            sibling_eq_count += 1

        # Acceptance length simulation
        for b in range(B):
            unique_anchors = block_anchor_pos[b].unique()
            for anc in unique_anchors:
                anc_mask_bool = block_anchor_pos[b] == anc
                anc_pred = predicted[b][anc_mask_bool]
                anc_labels = block_token_ids[b][anc_mask_bool]
                anc_valid_m = block_valid[b][anc_mask_bool]

                n_copies = anc_pred.shape[0] // st_size
                last_accepted_vertex = 0
                found_accept = False

                for copy_i in range(n_copies):
                    off = copy_i * st_size
                    root_idx = off  # vertex 0
                    if root_idx >= len(anc_pred):
                        break
                    if not anc_valid_m[root_idx]:
                        break
                    if anc_pred[root_idx] == anc_labels[root_idx]:
                        accept_lens.append(copy_i + 1)
                        last_accepted_vertex = 0
                        found_accept = True
                    else:
                        break

                if not found_accept:
                    accept_lens.append(0)

                node_hist[last_accepted_vertex] += 1
                if last_accepted_vertex in leftmost_nodes:
                    leftmost_count += 1
                total_trees += 1

    metrics = {
        "eval_loss": total_loss / max(n_loss, 1),
        "mean_acceptance_length": float(np.mean(accept_lens)) if accept_lens else 0.0,
        "sibling_equality_rate": sibling_eq_sum / max(sibling_eq_count, 1),
        "leftmost_path_pct": leftmost_count / max(total_trees, 1),
        "node_acceptance_histogram": node_hist.tolist(),
    }

    heatmap = torch.where(eq_total > 0, eq_counts / eq_total, torch.zeros_like(eq_counts))
    metrics["equality_heatmap"] = heatmap.tolist()

    sib_mask = torch.zeros(st_size, st_size, dtype=torch.bool)
    for v, u in sibling_pairs:
        sib_mask[v, u] = True
        sib_mask[u, v] = True
    sibling_heatmap = torch.where(sib_mask & (eq_total > 0), eq_counts / eq_total, torch.zeros_like(eq_counts))
    metrics["sibling_equality_heatmap"] = sibling_heatmap.tolist()

    return metrics


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float, min_lr: float = 1e-6) -> float:
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (lr - min_lr) * cosine


# ---------------------------------------------------------------------------
# Wandb helpers
# ---------------------------------------------------------------------------

_wandb = None  # lazy import


def _init_wandb(args, wandb_run_id: str | None):
    global _wandb
    if args.no_wandb:
        return None
    import wandb
    _wandb = wandb
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
        id=wandb_run_id or wandb.util.generate_id(),
        resume="allow",
    )
    return run.id


def _log_wandb(metrics: dict, step: int):
    if _wandb is not None:
        _wandb.log(metrics, step=step)


def _finish_wandb():
    if _wandb is not None:
        _wandb.finish()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 3: tree flash trainer")
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--data", required=True, help="Stage 2 HDF5 path")
    parser.add_argument("--eval-data", default=None, help="Optional separate eval HDF5")
    parser.add_argument(
        "--sub-tree-paths", nargs="+", default=DEFAULT_SUB_TREE_PATHS,
        help='Subtree edges as "X-Y" strings',
    )
    parser.add_argument("--num-anchors", type=int, default=8)
    parser.add_argument("--tree-seq-depth", type=int, default=4)
    parser.add_argument("--max-ctx-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--tree-pos-emb",
        choices=["initial", "relational", "both", "none"],
        default="both",
    )
    parser.add_argument(
        "--attn-impl",
        choices=["flex", "dense"],
        default="flex",
    )
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # Fabric / DDP
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs")
    parser.add_argument(
        "--precision",
        choices=["bf16-mixed", "16-mixed", "32-true"],
        default="bf16-mixed",
    )

    # torch.compile
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile on models")

    # Checkpoint resume
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint dir to resume from")

    # Wandb
    parser.add_argument("--wandb-project", type=str, default="tree-flash")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")

    args = parser.parse_args()

    # ---- Fabric setup ----
    strategy = "ddp" if args.devices > 1 else "auto"
    fabric = Fabric(
        accelerator="auto",
        devices=args.devices,
        strategy=strategy,
        precision=args.precision,
    )
    fabric.launch()
    fabric.seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    if fabric.is_global_zero:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Derive dtype for model weight loading ----
    dtype_map = {"bf16-mixed": torch.bfloat16, "16-mixed": torch.float16, "32-true": torch.float32}
    load_dtype = dtype_map[args.precision]

    # ---- Load target model (frozen) ----
    if fabric.is_global_zero:
        print("Loading target model...", flush=True)
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=load_dtype,
        output_hidden_states=False,
    )
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # ---- Load draft model (trainable) ----
    if fabric.is_global_zero:
        print("Loading draft model...", flush=True)
    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_model, torch_dtype=load_dtype,
    )
    draft_model.train()

    mask_token_id = draft_model.mask_token_id
    if mask_token_id is None:
        raise ValueError("Draft model config must have mask_token_id in dflash_config")

    # ---- torch.compile (before Fabric setup) ----
    if args.compile:
        if fabric.is_global_zero:
            print("Compiling models with torch.compile...", flush=True)
        target_model = torch.compile(target_model)
        draft_model = torch.compile(draft_model, dynamic=True)

    # ---- SubTree info ----
    st_info = SubTreeInfo(args.sub_tree_paths)
    st_size = st_info.size

    # ---- Tree position embeddings ----
    use_initial = args.tree_pos_emb in ("initial", "both")
    use_relational = args.tree_pos_emb in ("relational", "both")
    num_heads = target_model.config.num_attention_heads

    # TODO: Shouldn't this be layer wise
    tree_pos_emb = TreePositionEmbedding(
        st_size=st_size,
        hidden_size=target_model.config.hidden_size,
        num_heads=num_heads,
        st_info=st_info,
        use_initial=use_initial,
        use_relational=use_relational,
    )

    # ---- Optimizer ----
    trainable_params = list(draft_model.parameters()) + list(tree_pos_emb.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # ---- Fabric setup for models/optimizer ----
    # target_model is frozen, just move to device
    target_model = fabric.to_device(target_model)
    # draft_model + optimizer are wrapped together for DDP + precision
    draft_model, optimizer = fabric.setup(draft_model, optimizer)
    # tree_pos_emb has params in the optimizer, wrap for DDP
    tree_pos_emb = fabric.setup_module(tree_pos_emb)

    # ---- Dataset + DataLoader ----
    dataset = TreeFlashHDF5Dataset(args.data)
    collator = TreeFlashCollator(
        max_ctx_len=args.max_ctx_len,
        num_anchors=args.num_anchors,
        tree_seq_depth=args.tree_seq_depth,
        st_info=st_info,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
    )

    train_size = int(0.95 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset_default = torch.utils.data.random_split(
        dataset, [train_size, eval_size], generator=torch.Generator().manual_seed(args.seed)
    )

    if args.eval_data:
        eval_dataset = TreeFlashHDF5Dataset(args.eval_data)
    else:
        eval_dataset = eval_dataset_default

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    # Fabric wraps dataloaders for distributed sampling + device placement
    train_loader = fabric.setup_dataloaders(train_loader)
    eval_loader = fabric.setup_dataloaders(eval_loader)

    use_flex = args.attn_impl == "flex"
    ctx_len = args.max_ctx_len

    # ---- Resume from checkpoint ----
    start_step = 0
    wandb_run_id = None

    if args.resume:
        ckpt_path = Path(args.resume) / "fabric_ckpt.pt"
        if ckpt_path.exists():
            if fabric.is_global_zero:
                print(f"Resuming from {ckpt_path}", flush=True)
            remainder = fabric.load(str(ckpt_path), {
                "draft_model": draft_model,
                "tree_pos_emb": tree_pos_emb,
                "optimizer": optimizer,
            })
            start_step = remainder.get("step", 0)
            wandb_run_id = remainder.get("wandb_run_id", None)
        else:
            if fabric.is_global_zero:
                print(f"Warning: checkpoint not found at {ckpt_path}, training from scratch", flush=True)

    # ---- Wandb init (rank 0 only) ----
    if fabric.is_global_zero:
        wandb_run_id = _init_wandb(args, wandb_run_id)

    # ---- Training loop ----
    step = start_step
    accum_loss = 0.0
    optimizer.zero_grad()

    if fabric.is_global_zero:
        print(
            f"Training for {args.max_steps} steps (starting at {start_step}), "
            f"batch_size={args.batch_size}, accum={args.grad_accum_steps}, "
            f"devices={args.devices}, precision={args.precision}",
            flush=True,
        )

    while step < args.max_steps:
        for batch in train_loader:
            if step >= args.max_steps:
                break

            # Update LR
            cur_lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            draft_model.train()
            tree_pos_emb.train()

            # Gradient accumulation: disable sync for non-final micro-steps
            is_accumulating = (step + 1) % args.grad_accum_steps != 0
            with fabric.no_backward_sync(draft_model, enabled=is_accumulating):
                loss = forward_and_loss(
                    batch, target_model, draft_model, tree_pos_emb,
                    ctx_len, use_flex,
                )
                scaled_loss = loss / args.grad_accum_steps
                fabric.backward(scaled_loss)

            accum_loss += loss.item()

            if not is_accumulating:
                fabric.clip_gradients(draft_model, optimizer, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                avg_loss = accum_loss / args.grad_accum_steps
                if fabric.is_global_zero:
                    print(f"step={step+1}  loss={avg_loss:.4f}  lr={cur_lr:.2e}", flush=True)
                    # TODO: Set log interval
                    _log_wandb({
                        "train/loss": avg_loss,
                        "train/lr": cur_lr,
                    }, step=step + 1)
                accum_loss = 0.0

            # Eval
            if (step + 1) % args.eval_every == 0:
                fabric.barrier()
                draft_model.eval()
                tree_pos_emb.eval()
                with torch.no_grad():
                    metrics = evaluate(
                        eval_loader, target_model, draft_model, tree_pos_emb,
                        st_info, ctx_len, use_flex, args.eval_batches,
                    )
                if fabric.is_global_zero:
                    print(
                        f"[eval step={step+1}] "
                        f"loss={metrics['eval_loss']:.4f}  "
                        f"accept_len={metrics['mean_acceptance_length']:.3f}  "
                        f"sibling_eq={metrics['sibling_equality_rate']:.3f}  "
                        f"leftmost_pct={metrics['leftmost_path_pct']:.3f}",
                        flush=True,
                    )
                    _log_wandb({
                        "eval/loss": metrics["eval_loss"],
                        "eval/acceptance_length": metrics["mean_acceptance_length"],
                        "eval/sibling_equality_rate": metrics["sibling_equality_rate"],
                        "eval/leftmost_path_pct": metrics["leftmost_path_pct"],
                    }, step=step + 1)

                    # Save heatmap
                    heatmap_path = output_dir / f"heatmap_step{step+1}.json"
                    with open(heatmap_path, "w") as f:
                        json.dump({
                            "step": step + 1,
                            "equality_heatmap": metrics["equality_heatmap"],
                            "sibling_equality_heatmap": metrics["sibling_equality_heatmap"],
                            "node_acceptance_histogram": metrics["node_acceptance_histogram"],
                        }, f)
                fabric.barrier()

            # Save checkpoint
            if (step + 1) % args.save_every == 0:
                ckpt_dir = output_dir / f"checkpoint-{step+1}"
                if fabric.is_global_zero:
                    ckpt_dir.mkdir(parents=True, exist_ok=True)

                # Fabric checkpoint (for resume)
                state = {
                    "draft_model": draft_model,
                    "tree_pos_emb": tree_pos_emb,
                    "optimizer": optimizer,
                    "step": step + 1,
                    "wandb_run_id": wandb_run_id,
                }
                fabric.save(str(ckpt_dir / "fabric_ckpt.pt"), state)

                # HF-format save for inference (rank 0 only)
                if fabric.is_global_zero:
                    raw_draft = draft_model._forward_module if hasattr(draft_model, "_forward_module") else draft_model
                    if hasattr(raw_draft, "_orig_mod"):
                        raw_draft = raw_draft._orig_mod  # unwrap torch.compile
                    raw_draft.save_pretrained(str(ckpt_dir / "hf_draft"))
                    torch.save(
                        tree_pos_emb.state_dict()
                        if not hasattr(tree_pos_emb, "_forward_module")
                        else tree_pos_emb._forward_module.state_dict(),
                        str(ckpt_dir / "tree_pos_emb.pt"),
                    )
                    print(f"Saved checkpoint to {ckpt_dir}", flush=True)

                fabric.barrier()

            step += 1

    # Final save
    final_dir = output_dir / "final"
    if fabric.is_global_zero:
        final_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "draft_model": draft_model,
        "tree_pos_emb": tree_pos_emb,
        "optimizer": optimizer,
        "step": step,
        "wandb_run_id": wandb_run_id,
    }
    fabric.save(str(final_dir / "fabric_ckpt.pt"), state)

    if fabric.is_global_zero:
        raw_draft = draft_model._forward_module if hasattr(draft_model, "_forward_module") else draft_model
        if hasattr(raw_draft, "_orig_mod"):
            raw_draft = raw_draft._orig_mod
        raw_draft.save_pretrained(str(final_dir / "hf_draft"))
        torch.save(
            tree_pos_emb.state_dict()
            if not hasattr(tree_pos_emb, "_forward_module")
            else tree_pos_emb._forward_module.state_dict(),
            str(final_dir / "tree_pos_emb.pt"),
        )
        print(f"Training complete. Final model at {final_dir}", flush=True)

    if fabric.is_global_zero:
        _finish_wandb()


if __name__ == "__main__":
    main()
