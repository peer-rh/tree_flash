"""
Validation metrics for tree-flash.

Two kinds of validation
-----------------------
1. validate_loss   — fast; computes CumProd-weighted loss on held-out stage-2
                     batches.  Run every val_loss_every steps.

2. validate_spec   — full tree speculative-decoding pass on validation context
                     batches; records acceptance statistics.
                     Run every val_spec_every steps.

Spec-decode metrics (from experiments.md)
------------------------------------------
* average_acceptance_length        — mean number of accepted tokens per step
* equality_heatmap                 — [tree_size, tree_size] float32
                                     (i,j) = P(draft_token[i] == draft_token[j])
                                     averaged over all steps (unmasked)
* sibling_equality_heatmap         — same matrix, non-sibling entries zeroed out
* sibling_equality_rate            — scalar mean over all sibling pairs
* final_node_histogram             — [tree_size] int64 counts
* pct_final_on_leftmost            — float ∈ [0,1]

KV-cache strategy
-----------------
Verifier (target model):
    The context is prefilled once with use_cache=True, yielding a DynamicCache.
    The tree-token verification pass reuses this cache:
        target(input_ids=draft_tokens [B,T],
               past_key_values=target_kv,
               attention_mask=[B,1,T, ctx_len+T],   ← tree mask covers both halves
               use_cache=False)
    Q length = T; KV length = ctx_len (cached) + T (new) = ctx_len + T.
    This avoids recomputing context K,V on every validation step.

Drafter (DFlashDraftModel):
    The draft model conditions on target_hidden (not on past token K,V) for the
    context.  On the first decode step the draft KV cache is empty; it grows on
    subsequent steps as accepted tokens accumulate.  DraftWrapper.infer_forward()
    accepts an optional draft_past_kv argument for multi-step extension (pass
    None for single-step validation as done here).

GPU-optimised acceptance
------------------------
tree_accept() replaces the sequential Python loop with a single matmul:

    rejected_count[b, q] = Σ_k  (~accepted[b, k])  ×  ancestor_matrix[k, q]

path_accepted[b, q] = (rejected_count[b, q] == 0)
One [B, T] × [T, T] FP32 matmul; no Python-level iteration over tree nodes.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import torch
from torch import Tensor

from tree.spec import TreeSpec
from tree.mask import build_tree_attn_mask
from dflash.model.utils import extract_context_feature


@dataclass
class TreeMetrics:
    """Accumulated validation statistics, updated in-place by validate_spec."""

    # Running sums / counts for averaging
    total_acceptance: float = 0.0
    total_steps: int = 0

    # [tree_size, tree_size] float32 — accumulated equality counts
    equality_sum: Tensor | None = None
    equality_count: int = 0

    # [tree_size] int64 — histogram of final accepted node indices
    final_node_hist: Tensor | None = None

    # Scalar counts for leftmost-path metric
    final_on_leftmost: int = 0
    final_total: int = 0

    def reset(self) -> None:
        self.total_acceptance = 0.0
        self.total_steps = 0
        self.equality_sum = None
        self.equality_count = 0
        self.final_node_hist = None
        self.final_on_leftmost = 0
        self.final_total = 0

    def summarise(self, tree_spec: TreeSpec, sibling_pairs: Tensor) -> dict:
        """
        Return a dict of scalar / tensor metrics ready for logging.

        Parameters
        ----------
        tree_spec     : TreeSpec — provides leftmost_path() and tree_size
        sibling_pairs : [n_pairs, 2] long — from tree_spec.sibling_pairs
        """
        T = tree_spec.tree_size
        out: dict = {}

        # Average acceptance length
        if self.total_steps > 0:
            out["avg_acceptance_length"] = self.total_acceptance / self.total_steps
        else:
            out["avg_acceptance_length"] = 0.0

        # Equality heatmaps
        if self.equality_sum is not None and self.equality_count > 0:
            heatmap = self.equality_sum / self.equality_count  # [T, T] float
            out["equality_heatmap"] = heatmap  # unmasked

            # Sibling-masked: keep only sibling-pair entries
            sib_heatmap = torch.zeros_like(heatmap)
            if sibling_pairs.shape[0] > 0:
                i_idx = sibling_pairs[:, 0]  # [n_pairs]
                j_idx = sibling_pairs[:, 1]  # [n_pairs]
                sib_heatmap[i_idx, j_idx] = heatmap[i_idx, j_idx]
                sib_heatmap[j_idx, i_idx] = heatmap[j_idx, i_idx]
            out["sibling_equality_heatmap"] = sib_heatmap

            if sibling_pairs.shape[0] > 0:
                sib_vals = heatmap[sibling_pairs[:, 0], sibling_pairs[:, 1]]
                out["sibling_equality_rate"] = sib_vals.mean().item()
            else:
                out["sibling_equality_rate"] = 0.0
        else:
            out["equality_heatmap"] = torch.zeros(T, T)
            out["sibling_equality_heatmap"] = torch.zeros(T, T)
            out["sibling_equality_rate"] = 0.0

        # Final node histogram
        if self.final_node_hist is not None:
            out["final_node_histogram"] = self.final_node_hist
        else:
            out["final_node_histogram"] = torch.zeros(T, dtype=torch.long)

        # Leftmost path %
        if self.final_total > 0:
            out["pct_final_on_leftmost"] = self.final_on_leftmost / self.final_total
        else:
            out["pct_final_on_leftmost"] = 0.0

        return out


# ── Loss validation ──────────────────────────────────────────────────────────

@torch.no_grad()
def validate_loss(
    model: torch.nn.Module,
    target: torch.nn.Module,
    val_loader,
    target_layer_ids: list[int],
    ar_loss_weight: float,
    n_steps: int,
    device: torch.device,
) -> dict[str, float]:
    """
    Compute CumProd-weighted loss on held-out stage-2 data.

    Parameters
    ----------
    model           : DraftWrapper (already fabric.setup'd)
    target          : frozen target model
    val_loader      : DataLoader yielding (context_ids, tree_tokens, cumprod_weights)
    target_layer_ids: list of target layer indices for feature extraction
    ar_loss_weight  : λ for AR head loss term
    n_steps         : number of batches to evaluate
    device          : device for tensors

    Returns
    -------
    {"val_loss": float, "val_draft_loss": float, "val_ar_loss": float}
    """
    from trainer.loss import compute_loss

    total, draft_total, ar_total = 0.0, 0.0, 0.0
    count = 0

    for step, (context_ids, tree_tokens, cumprod_weights) in enumerate(val_loader):
        if step >= n_steps:
            break

        context_ids = context_ids.to(device)       # [B, ctx_len]
        tree_tokens = tree_tokens.to(device)        # [B, tree_size]
        cumprod_weights = cumprod_weights.to(device) # [B, tree_size]

        # Target model: get conditioning hidden states
        target_out = target(
            context_ids,
            output_hidden_states=True,
            use_cache=False,
        )
        raw_target_hidden = extract_context_feature(
            target_out.hidden_states, target_layer_ids
        )  # [B, ctx_len, n_feature_layers * H]

        anchor_ids = context_ids[:, -1]  # [B] — last context token

        draft_logits, ar_logits = model(anchor_ids, raw_target_hidden, tree_tokens)
        # draft_logits: [B, tree_size, V]
        # ar_logits:    [B, tree_size, V]

        loss, d_loss, a_loss = compute_loss(
            draft_logits, ar_logits, tree_tokens, cumprod_weights, ar_loss_weight
        )

        total += loss.item()
        draft_total += d_loss.item()
        ar_total += a_loss.item()
        count += 1

    denom = max(count, 1)
    return {
        "val_loss": total / denom,
        "val_draft_loss": draft_total / denom,
        "val_ar_loss": ar_total / denom,
    }


# ── Vectorised acceptance ─────────────────────────────────────────────────────

def tree_accept(
    accepted: Tensor,           # [B, tree_size] bool
    ancestor_matrix: Tensor,    # [tree_size, tree_size] bool — same device as accepted
) -> Tensor:                    # [B, tree_size] bool
    """
    GPU-vectorised tree path acceptance.

    For each node q: path_accepted[b, q] = True iff every node on the path
    from the root to q (inclusive) was individually accepted.

    The sequential BFS loop:
        for i in range(T):
            path_accepted[:, i] = accepted[:, i] & path_accepted[:, parent[i]]

    is equivalent to checking whether the count of *rejected* ancestors is zero:
        rejected_count[b, q] = Σ_k  (~accepted[b, k]) * ancestor_matrix[k, q]
        path_accepted[b, q]  = (rejected_count[b, q] == 0)

    This is one [B, T] × [T, T] float32 matmul — fully on GPU, no Python loop.

    Parameters
    ----------
    accepted        : [B, tree_size] bool
        accepted[b, i] = (draft_token[b, i] == target_token[b, i])
    ancestor_matrix : [tree_size, tree_size] bool (device-resident)
        ancestor_matrix[k, q] = True iff node k is an ancestor-or-self of node q.

    Returns
    -------
    path_accepted : [B, tree_size] bool
    """
    # (~accepted): [B, T] float  ×  ancestor_matrix: [T, T] float  →  [B, T] float
    rejected_count = (~accepted).to(torch.float32) @ ancestor_matrix.to(torch.float32)
    # rejected_count[b, q] = number of rejected nodes on the path root → q
    return rejected_count < 0.5   # True iff no ancestor was rejected


# ── Spec-decode validation ────────────────────────────────────────────────────

@torch.no_grad()
def validate_spec(
    model: torch.nn.Module,
    target: torch.nn.Module,
    val_loader,
    target_layer_ids: list[int],
    tree_spec: TreeSpec,
    n_steps: int,
    device: torch.device,
    metrics: TreeMetrics,
) -> None:
    """
    Run one tree-spec-decode step per validation batch and update ``metrics``.

    This is a fast single-step validator: one draft + verify cycle per context,
    suitable for tracking training progress.  For full multi-step generation
    use ``trainer.spec_decode.tree_spec_decode``.

    Per batch
    ---------
    1. Prefill target with KV cache (context once, reused for verification).
    2. Draft: one bidirectional pass over all tree nodes (DraftWrapper.infer_forward).
    3. Verify: target forward using prefilled KV + tree attention mask.
       Q = T tree tokens; KV = ctx_len cached + T new → mask [B,1,T,ctx_len+T].
    4. Acceptance (correct tree semantics):
       verify_logits[b, i] predicts the token AFTER node i (= child position).
       To accept node i we check the PARENT's logit:
           root  → prefill logits at last context position
           node i → verify_logits[b, parent_ids[i]]
       adjusted_parent_ids indexes both into one concatenated tensor.
       path_accepted via tree_accept() — one matmul, no Python loop.
    5. Accumulate metrics entirely on GPU; only .cpu() for final scalars.

    Parameters
    ----------
    model           : DraftWrapper (already fabric.setup'd, eval mode)
    target          : frozen target model
    val_loader      : yields (context_ids, tree_tokens, cumprod_weights);
                      cumprod_weights are ignored here
    target_layer_ids: layer indices for extract_context_feature
    tree_spec       : TreeSpec for this experiment
    n_steps         : number of batches to process
    device          : GPU device
    metrics         : TreeMetrics — updated IN PLACE
    """
    T = tree_spec.tree_size

    # Move static tree tensors to device once, outside the loop
    anc_mat  = tree_spec.ancestor_matrix.to(device)        # [T, T] bool
    depths   = tree_spec.depths.to(device)                  # [T] long
    adj_par  = tree_spec.adjusted_parent_ids.to(device)     # [T] long
    leftmost = tree_spec.ancestor_matrix.new_zeros(T, dtype=torch.bool)
    for idx in tree_spec.leftmost_path():
        leftmost[idx] = True
    # leftmost: [T] bool — True for nodes on the primary (left-most) path

    # Depth-based position ids for tree tokens (relative; add ctx_len at call site)
    tree_depth_pos = tree_spec.position_ids.to(device)  # [T]

    # Pre-build the tree attn mask template; ctx_len is fixed across batches
    ctx_len: int | None = None
    attn_mask_cache: Tensor | None = None

    for step, (context_ids, _tree_tokens, _cumprod) in enumerate(val_loader):
        if step >= n_steps:
            break

        context_ids = context_ids.to(device)   # [B, ctx_len]
        B, CL = context_ids.shape

        # Build attention mask once (same ctx_len throughout)
        if ctx_len is None or CL != ctx_len:
            ctx_len = CL
            # [B, 1, T, ctx_len + T] — additive float mask (0 / -inf)
            attn_mask_cache = build_tree_attn_mask(anc_mat, ctx_len, B, device)

        # ── 1. Prefill target: cache context K,V ─────────────────────────────
        target_out = target(
            context_ids,
            output_hidden_states=True,
            use_cache=True,          # ← cache context K,V; reused for verification
        )
        target_kv = target_out.past_key_values
        # target_kv holds K,V for all ctx_len context tokens across all layers

        # anchor_logits: target's prediction for the first tree node (root)
        anchor_logits = target_out.logits[:, -1:, :]
        # [B, 1, V] — logits at last context position

        raw_target_hidden = extract_context_feature(
            target_out.hidden_states, target_layer_ids
        )  # [B, ctx_len, n_feature_layers * H]

        anchor_ids = context_ids[:, -1]          # [B]

        # ── 2. Draft: bidirectional over all T tree nodes ─────────────────────
        # infer_forward() skips teacher forcing and the AR head;
        # draft KV cache is None for single-step validation (first decode step).
        draft_logits, _backbone_hs = model.infer_forward(
            anchor_ids=anchor_ids,
            raw_target_hidden=raw_target_hidden,
            draft_past_kv=None,
        )
        # draft_logits: [B, T, V]
        draft_tokens = draft_logits.argmax(dim=-1)   # [B, T]

        # ── 3. Verify: reuse prefilled context KV, apply tree mask ────────────
        # Position IDs for tree tokens: absolute depth-based positions
        tree_pos = (ctx_len + tree_depth_pos).unsqueeze(0).expand(B, -1)
        # [B, T]

        # Pass only the T new tree tokens; the context K,V come from target_kv.
        # attention_mask [B,1,T,ctx_len+T]: covers both the cached context keys
        # (all attend) and the tree keys (ancestor-or-self only).
        verify_out = target(
            input_ids=draft_tokens,       # [B, T]
            position_ids=tree_pos,        # [B, T]
            past_key_values=target_kv,    # DynamicCache with ctx_len entries
            attention_mask=attn_mask_cache,
            use_cache=False,
        )
        verify_logits = verify_out.logits   # [B, T, V]

        # ── 4. Vectorised acceptance ──────────────────────────────────────────
        # verify_logits[b, i] predicts the token AFTER draft_tokens[b, i].
        # To accept node i we use the PARENT's logit:
        #   root  → anchor_logits[b, 0]  (target prediction for first new token)
        #   node i → verify_logits[b, parent_ids[i]]
        # adjusted_parent_ids maps each node to index in [anchor_logit, verify_logits].
        extended_logits = torch.cat([anchor_logits, verify_logits], dim=1)
        # [B, 1+T, V]
        parent_preds = extended_logits[:, adj_par, :].argmax(dim=-1)
        # [B, T] — target's greedy prediction at each node's parent position
        accepted     = (draft_tokens == parent_preds)        # [B, T] bool
        path_accepted = tree_accept(accepted, anc_mat)       # [B, T] bool

        # Final accepted node: deepest node whose full path is accepted.
        # depth_score[b, i] = depths[i] if path_accepted[b,i] else -1
        depth_score  = torch.where(
            path_accepted,
            depths.unsqueeze(0).expand(B, -1).float(),   # [B, T]
            depths.new_full((1, T), -1).float(),
        )  # [B, T]
        final_nodes = depth_score.argmax(dim=1)            # [B]

        # Acceptance length = depth of final node + 1 (nodes on accepted path)
        acceptance_lengths = depths[final_nodes] + 1       # [B]

        # ── 5. Accumulate metrics (GPU → CPU only for scalars) ────────────────
        metrics.total_acceptance += acceptance_lengths.sum().item()
        metrics.total_steps      += B

        # Equality matrix: [B, T, T] → mean over B → [T, T], then to CPU
        # (draft_tokens[:, i] == draft_tokens[:, j]) for all node pairs (i, j)
        eq_mean = (
            (draft_tokens.unsqueeze(2) == draft_tokens.unsqueeze(1))  # [B, T, T]
            .float()
            .mean(dim=0)   # [T, T]
            .cpu()
        )
        if metrics.equality_sum is None:
            metrics.equality_sum = eq_mean
        else:
            metrics.equality_sum = metrics.equality_sum + eq_mean
        metrics.equality_count += 1

        # Final node histogram: bincount over the batch (no Python loop)
        if metrics.final_node_hist is None:
            metrics.final_node_hist = torch.zeros(T, dtype=torch.long)
        metrics.final_node_hist += torch.bincount(
            final_nodes.cpu(), minlength=T
        )  # [T]

        # Leftmost path %: count how many final nodes fall on the primary path
        # leftmost: [T] bool (device); final_nodes: [B] (device)
        on_left = leftmost[final_nodes]        # [B] bool
        metrics.final_on_leftmost += on_left.sum().item()
        metrics.final_total       += B
