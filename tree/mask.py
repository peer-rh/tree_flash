"""
Attention mask construction for tree-structured verification.

During training the verifier (target model) is NOT called — CumProd weights
are pre-computed in stage-2.  This module is used only during validation and
inference to implement tree-structured token verification.

Two masks are provided:
  build_tree_attn_mask  — dense additive mask for HuggingFace SDPA models
  build_flex_block_mask — sparse BlockMask for flex_attention (faster at scale)
"""

from __future__ import annotations
import torch
from torch import Tensor


def build_tree_attn_mask(
    ancestor_matrix: Tensor,
    ctx_len: int,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """
    Build a dense additive attention mask for tree-structured verification.
    Suitable for passing as ``attention_mask`` to a HuggingFace model.

    Layout: query positions are the tree nodes [0..tree_size-1];
            key   positions are [context (0..ctx_len-1), tree (ctx_len..ctx_len+tree_size-1)].

    Mask convention (additive):
        0.0       → position is attended to
        -inf      → position is masked out

    Parameters
    ----------
    ancestor_matrix : [tree_size, tree_size] bool
        ancestor_matrix[k, q] = True iff tree node k is an ancestor-or-self of q.
        Produced by ``TreeSpec.ancestor_matrix``.
    ctx_len         : number of context tokens (always attended to)
    batch_size      : B — mask is expanded over the batch dimension
    device          : target device

    Returns
    -------
    mask : [B, 1, tree_size, ctx_len + tree_size]
        Ready for HuggingFace SDPA with 4-D attention mask support.
    """
    tree_size = ancestor_matrix.shape[0]

    # Context columns: all queries can attend → fill with 0 (attend)
    ctx_part = torch.zeros(tree_size, ctx_len, dtype=torch.float32, device=device)
    # [tree_size, ctx_len]

    # Tree columns: query q attends to key k iff k is ancestor-or-self of q
    # ancestor_matrix[k, q] → transpose to [q, k]
    tree_part = ancestor_matrix.T.to(device=device, dtype=torch.float32)
    # [tree_size, tree_size], True = attend
    # Convert to additive: True → 0.0, False → -inf
    tree_part = torch.where(tree_part, torch.zeros_like(tree_part), torch.full_like(tree_part, float("-inf")))
    # [tree_size, tree_size]

    # Concatenate along key dimension
    mask = torch.cat([ctx_part, tree_part], dim=1)  # [tree_size, ctx_len + tree_size]

    # Expand to [B, 1, tree_size, ctx_len + tree_size]
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)


def build_flex_block_mask(
    ancestor_matrix: Tensor,
    ctx_len: int,
    total_len: int,
    device: torch.device,
):
    """
    Build a ``BlockMask`` for ``torch.nn.attention.flex_attention``.

    Use this for the verification forward pass when you have direct control
    over the attention computation (e.g. a custom model or a patched layer).
    For standard HuggingFace models use ``build_tree_attn_mask`` instead.

    Parameters
    ----------
    ancestor_matrix : [tree_size, tree_size] bool
        ancestor_matrix[k, q] = True iff tree node k is an ancestor-or-self of q.
    ctx_len         : number of context token positions
    total_len       : ctx_len + tree_size (Q_LEN = KV_LEN for this mask)
    device          : target device

    Returns
    -------
    BlockMask — pass to flex_attention as ``block_mask=...``
    """
    from torch.nn.attention.flex_attention import create_block_mask

    tree_size = ancestor_matrix.shape[0]
    # Move to device for indexing inside mask_fn
    anc = ancestor_matrix.to(device)  # [tree_size, tree_size]

    def mask_fn(b: Tensor, h: Tensor, q_idx: Tensor, k_idx: Tensor) -> Tensor:
        """
        Returns True if query q_idx should attend to key k_idx.

        q_idx, k_idx are scalar int tensors provided by flex_attention internals.
        Context keys (k_idx < ctx_len) are always attended to.
        Tree keys are attended to iff the key node is an ancestor-or-self of the query node.
        """
        in_ctx = k_idx < ctx_len
        # Clamp to valid tree range (safe even when k_idx is a context index, because
        # the result is overridden by `in_ctx` via the bitwise-OR below)
        k_tree = (k_idx - ctx_len).clamp(0, tree_size - 1)
        q_tree = (q_idx - ctx_len).clamp(0, tree_size - 1)
        return in_ctx | anc[k_tree, q_tree]

    return create_block_mask(
        mask_fn,
        B=None,          # same mask for all batch elements
        H=None,          # same mask for all heads
        Q_LEN=total_len,
        KV_LEN=total_len,
        device=device,
        _compile=True,
    )
