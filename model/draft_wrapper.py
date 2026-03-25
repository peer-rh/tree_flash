"""
Thin wrapper combining DFlashDraftModel + ARHead for joint training.

Key insight from reading dflash/model/dflash.py:
    DFlashDraftModel.forward() returns backbone_hs (the normed final hidden
    states) as a plain Tensor — NOT a CausalLMOutputWithPast.
    The lm_head is called EXTERNALLY: target.lm_head(backbone_hs).
    The embedding table lives on the target model: target.model.embed_tokens.

This wrapper therefore needs references to:
    draft        — DFlashDraftModel (trainable)
    ar_head      — ARHead (trainable)
    lm_head      — shared Linear [H → V] from target (frozen weights)
    embed_tokens — shared Embedding [V → H] from target (frozen weights)

All of these are stored as plain attributes (not nn.Parameters for the frozen
ones) so that DDP only syncs gradients for draft + ar_head.

torch.compile compatibility
---------------------------
* All tensor shapes are static given fixed ctx_len, tree_size, batch_size.
* No Python-level control flow on tensor values.
* adjusted_parent_ids is a registered buffer (static index tensor).
* noise_ids is constructed inside forward via torch.full with a compile-time
  constant (mask_token_id); torch.compile handles this via specialisation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .ar_head import ARHead


class DraftWrapper(nn.Module):
    """
    Joint forward pass: draft diffusion model + AR pruning head.

    Parameters
    ----------
    draft                : DFlashDraftModel — the 5-layer bidirectional drafter
    ar_head              : ARHead — tree-topology-aware pruning head
    lm_head              : nn.Linear [H → V] — shared with target, kept frozen
    embed_tokens         : nn.Embedding [V → H] — shared with target, kept frozen
    mask_token_id        : int — token ID for masked (noisy) positions
    adjusted_parent_ids  : [tree_size] long — indexes into [anchor, tree_token_0, …]
                           root → 0 (anchor), others → parent_ids[i] + 1
    ctx_len              : int — fixed context length (used to build position_ids)
    position_ids_rel     : [tree_size] long — relative (depth-based) position ids
                           from TreeSpec; absolute = ctx_len + position_ids_rel
    """

    def __init__(
        self,
        draft: nn.Module,
        ar_head: ARHead,
        lm_head: nn.Linear,
        embed_tokens: nn.Embedding,
        mask_token_id: int,
        adjusted_parent_ids: Tensor,   # [tree_size]
        ctx_len: int,
        position_ids_rel: Tensor,      # [tree_size]
    ) -> None:
        super().__init__()

        # Trainable submodules — DDP will sync their gradients
        self.draft = draft
        self.ar_head = ar_head

        # Frozen shared modules — stored as plain attributes so DDP ignores them.
        # We call .requires_grad_(False) on them at setup time (see trainer).
        self.lm_head = lm_head
        self.embed_tokens = embed_tokens

        self.mask_token_id = mask_token_id

        # Static index tensors: registered as buffers so they move with .to(device)
        # and are visible to torch.compile without being treated as Parameters.
        self.register_buffer("adjusted_parent_ids", adjusted_parent_ids)
        # [tree_size]

        # Absolute position IDs: [1, tree_size], broadcast over batch
        abs_pos = (ctx_len + position_ids_rel).unsqueeze(0)  # [1, tree_size]
        self.register_buffer("position_ids", abs_pos)
        # [1, tree_size]

    def forward(
        self,
        anchor_ids: Tensor,          # [B]          — last context token id per sample
        raw_target_hidden: Tensor,   # [B, ctx_len, n_feature_layers * H]
        tree_tokens: Tensor,         # [B, tree_size] — ground-truth tokens (teacher forcing)
    ) -> tuple[Tensor, Tensor]:
        """
        One training forward pass.

        Noise schedule: all tree positions are fully masked (mask_token_id).
        The draft model denoises them conditioned on raw_target_hidden.

        Parameters
        ----------
        anchor_ids        : [B]
            Token id of the last accepted context token (used as the "parent"
            of the tree root in the AR head).
        raw_target_hidden : [B, ctx_len, n_feature_layers * H]
            Concatenated hidden states from target_layer_ids target layers.
            Passed directly to DFlashDraftModel.forward() which projects them
            internally (via self.fc + self.hidden_norm).
        tree_tokens       : [B, tree_size]
            Ground-truth token ids at each tree node.  Used only for building
            parent token embeddings (teacher forcing for AR head).

        Returns
        -------
        draft_logits : [B, tree_size, V]
        ar_logits    : [B, tree_size, V]
        """
        B, tree_size = tree_tokens.shape

        # ── Noise embedding ──────────────────────────────────────────────────
        # All tree positions masked; shape [B, tree_size]
        noise_ids = torch.full(
            (B, tree_size),
            self.mask_token_id,
            dtype=torch.long,
            device=tree_tokens.device,
        )
        noise_embedding = self.embed_tokens(noise_ids)
        # [B, tree_size, H]

        # ── Draft model: bidirectional over ALL tree nodes ───────────────────
        # position_ids is [1, tree_size]; expand to [B, tree_size]
        backbone_hs = self.draft(
            position_ids=self.position_ids.expand(B, -1),
            noise_embedding=noise_embedding,
            target_hidden=raw_target_hidden,   # projected inside DFlashDraftModel
            use_cache=False,
        )
        # backbone_hs: [B, tree_size, H]

        # ── Draft logits via shared lm_head ──────────────────────────────────
        draft_logits = self.lm_head(backbone_hs)
        # [B, tree_size, V]

        # ── AR head: tree-topology-aware ─────────────────────────────────────
        # Build extended token sequence: [anchor, tree_token_0, …, tree_token_{T-1}]
        # Shape: [B, 1 + tree_size]
        extended_ids = torch.cat([anchor_ids.unsqueeze(1), tree_tokens], dim=1)

        # For each node i: index = adjusted_parent_ids[i]
        #   root (i=0)  → index 0  → anchor token embedding
        #   others      → index parent_ids[i] + 1 → parent tree_token embedding
        # adjusted_parent_ids: [tree_size]; indexing gives [B, tree_size]
        parent_embeds = self.embed_tokens(extended_ids[:, self.adjusted_parent_ids])
        # [B, tree_size, H]

        ar_logits = self.ar_head(backbone_hs, parent_embeds)
        # [B, tree_size, V]

        return draft_logits, ar_logits

    # ── Inference helpers ────────────────────────────────────────────────────

    def infer_forward(
        self,
        anchor_ids: Tensor,            # [B] — last accepted context token id
        raw_target_hidden: Tensor,     # [B, ctx_len, n_feature_layers * H]
        draft_past_kv=None,            # DynamicCache | None
        position_ids: Tensor | None = None,  # [B, n_ctx + tree_size] — override for multi-step
    ) -> tuple[Tensor, Tensor]:
        """
        Inference-time draft pass.  No teacher forcing; AR head not called.

        Use this inside validate_spec and the spec-decode loop.  The training
        ``forward()`` requires ground-truth ``tree_tokens`` for parent embeds
        (teacher forcing); here we skip the AR head entirely.

        Parameters
        ----------
        anchor_ids        : [B]
        raw_target_hidden : [B, ctx_len, n_feature_layers * H]
        draft_past_kv     : optional DynamicCache for multi-step decode;
                            pass None on the first step (empty cache)

        Returns
        -------
        draft_logits : [B, tree_size, V]
        backbone_hs  : [B, tree_size, H]  — returned for optional AR-head
                       q-value computation during pruning (Exp 2+)
        """
        B = anchor_ids.shape[0]
        tree_size = self.position_ids.shape[1]
        device = anchor_ids.device

        noise_ids = torch.full(
            (B, tree_size), self.mask_token_id, dtype=torch.long, device=device
        )
        noise_embedding = self.embed_tokens(noise_ids)  # [B, tree_size, H]

        pos = self.position_ids.expand(B, -1) if position_ids is None else position_ids
        # pos: [B, tree_size] for single-step, [B, n_ctx+tree_size] for multi-step decode
        backbone_hs = self.draft(
            position_ids=pos,
            noise_embedding=noise_embedding,
            target_hidden=raw_target_hidden,
            past_key_values=draft_past_kv,
            use_cache=(draft_past_kv is not None),
        )  # [B, tree_size, H]

        draft_logits = self.lm_head(backbone_hs)  # [B, tree_size, V]
        return draft_logits, backbone_hs

    def ar_qvalues(
        self,
        backbone_hs: Tensor,    # [B, tree_size, H]
        anchor_ids: Tensor,     # [B] — true anchor (context last token)
        draft_tokens: Tensor,   # [B, tree_size] — sampled draft tokens
    ) -> Tensor:                # [B, tree_size, V]
        """
        AR head forward at inference time using sampled draft tokens as parents.

        Called after ``infer_forward`` + sampling when pruning is active (Exp 2+).
        Not needed for Exp 1 (no pruning).

        Parameters
        ----------
        backbone_hs  : [B, tree_size, H] from infer_forward()
        anchor_ids   : [B] — used as the parent embed of the root node
        draft_tokens : [B, tree_size] — draft model's sampled tokens
        """
        # Build [anchor, tree_token_0, …, tree_token_{T-1}]: [B, 1 + tree_size]
        extended_ids = torch.cat([anchor_ids.unsqueeze(1), draft_tokens], dim=1)
        parent_embeds = self.embed_tokens(extended_ids[:, self.adjusted_parent_ids])
        # [B, tree_size, H]
        return self.ar_head(backbone_hs, parent_embeds)  # [B, tree_size, V]
