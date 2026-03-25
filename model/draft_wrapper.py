"""
Thin wrapper combining TreeDraftModel + ARHead for joint training.

Key insight from reading dflash/model/dflash.py:
    DFlashDraftModel.forward() returns backbone_hs (the normed final hidden
    states) as a plain Tensor — NOT a CausalLMOutputWithPast.
    The lm_head is called EXTERNALLY: target.lm_head(backbone_hs).
    The embedding table lives on the target model: target.model.embed_tokens.

Shared backbone design:
    layers[0..N-2]  →  shared_hs
    shared_hs  →  layers[N-1]  →  norm  →  lm_head  →  draft_logits
    shared_hs + parent_proj(parent_embeds)  →  ar_layer  →  norm  →  lm_head  →  ar_logits

This wrapper therefore needs references to:
    draft        — TreeDraftModel (trainable)
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
    draft                : TreeDraftModel — the bidirectional drafter
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

        # Scratch space for the most-recent infer_forward context; used by ar_qvalues.
        # Not a buffer — cleared each step, not meant for checkpointing.
        self._infer_ctx: tuple[Tensor, Tensor, tuple[Tensor, Tensor]] | None = None

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
        raw_target_hidden : [B, ctx_len, n_feature_layers * H]
        tree_tokens       : [B, tree_size] — ground-truth tokens; used to build
                            parent token embeddings (teacher forcing for AR head)

        Returns
        -------
        draft_logits : [B, tree_size, V]
        ar_logits    : [B, tree_size, V]
        """
        B, tree_size = tree_tokens.shape

        # ── Noise embedding ──────────────────────────────────────────────────
        noise_ids = torch.full(
            (B, tree_size),
            self.mask_token_id,
            dtype=torch.long,
            device=tree_tokens.device,
        )
        noise_embedding = self.embed_tokens(noise_ids)  # [B, tree_size, H]

        # ── Shared backbone: layers[0..N-2] ──────────────────────────────────
        pos = self.position_ids.expand(B, -1)  # [B, tree_size]
        shared_hs, target_hidden_proj, pos_emb = self.draft.shared_forward(
            position_ids=pos,
            noise_embedding=noise_embedding,
            target_hidden=raw_target_hidden,
            use_cache=False,
        )
        # shared_hs:          [B, tree_size, H]
        # target_hidden_proj: [B, ctx_len, H]
        # pos_emb:            (cos, sin)

        # ── Diffusion head: layers[N-1] + norm + lm_head ─────────────────────
        backbone_hs = self.draft.diffusion_head(
            shared_hs=shared_hs,
            target_hidden_proj=target_hidden_proj,
            position_embeddings=pos_emb,
            use_cache=False,
        )
        draft_logits = self.lm_head(backbone_hs)  # [B, tree_size, V]

        # ── AR head: tree-topology-aware ─────────────────────────────────────
        # Build extended token sequence: [anchor, tree_token_0, …, tree_token_{T-1}]
        extended_ids = torch.cat([anchor_ids.unsqueeze(1), tree_tokens], dim=1)
        # For each node i: adjusted_parent_ids[i] indexes into extended_ids
        #   root (i=0)  → 0  → anchor embedding
        #   others      → parent_ids[i] + 1  → parent tree_token embedding
        parent_embeds = self.embed_tokens(extended_ids[:, self.adjusted_parent_ids])
        # [B, tree_size, H]

        ar_logits = self.ar_head(
            shared_hs=shared_hs,
            parent_embeds=parent_embeds,
            target_hidden_proj=target_hidden_proj,
            position_embeddings=pos_emb,
        )  # [B, tree_size, V]

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

        After this call, (shared_hs, target_hidden_proj, pos_emb) are cached in
        self._infer_ctx so that ar_qvalues() can be called without re-running the
        shared backbone (Exp 2+ pruning).

        Returns
        -------
        draft_logits : [B, tree_size, V]
        backbone_hs  : [B, tree_size, H]  normed diffusion-head output
        """
        B = anchor_ids.shape[0]
        tree_size = self.position_ids.shape[1]
        device = anchor_ids.device

        noise_ids = torch.full(
            (B, tree_size), self.mask_token_id, dtype=torch.long, device=device
        )
        noise_embedding = self.embed_tokens(noise_ids)  # [B, tree_size, H]

        pos = self.position_ids.expand(B, -1) if position_ids is None else position_ids

        shared_hs, target_hidden_proj, pos_emb = self.draft.shared_forward(
            position_ids=pos,
            noise_embedding=noise_embedding,
            target_hidden=raw_target_hidden,
            past_key_values=draft_past_kv,
            use_cache=(draft_past_kv is not None),
        )

        backbone_hs = self.draft.diffusion_head(
            shared_hs=shared_hs,
            target_hidden_proj=target_hidden_proj,
            position_embeddings=pos_emb,
            past_key_values=draft_past_kv,
            use_cache=(draft_past_kv is not None),
        )

        # Cache context for optional ar_qvalues() call
        self._infer_ctx = (shared_hs, target_hidden_proj, pos_emb)

        draft_logits = self.lm_head(backbone_hs)  # [B, tree_size, V]
        return draft_logits, backbone_hs

    def ar_qvalues(
        self,
        anchor_ids: Tensor,     # [B] — true anchor (context last token)
        draft_tokens: Tensor,   # [B, tree_size] — sampled draft tokens
    ) -> Tensor:                # [B, tree_size, V]
        """
        AR head forward at inference time using sampled draft tokens as parents.

        Must be called after infer_forward() in the same step (uses cached context).
        Not needed for Exp 1 (no pruning).

        Parameters
        ----------
        anchor_ids   : [B]
        draft_tokens : [B, tree_size] — draft model's sampled tokens
        """
        assert self._infer_ctx is not None, "Call infer_forward() before ar_qvalues()"
        shared_hs, target_hidden_proj, pos_emb = self._infer_ctx

        extended_ids = torch.cat([anchor_ids.unsqueeze(1), draft_tokens], dim=1)
        parent_embeds = self.embed_tokens(extended_ids[:, self.adjusted_parent_ids])
        # [B, tree_size, H]

        return self.ar_head(
            shared_hs=shared_hs,
            parent_embeds=parent_embeds,
            target_hidden_proj=target_hidden_proj,
            position_embeddings=pos_emb,
        )  # [B, tree_size, V]
