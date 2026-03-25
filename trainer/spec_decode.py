"""
Full multi-step tree speculative decoding.

Overview
--------
tree_spec_decode() runs the complete generate loop:
    prefill → [draft → verify → accept → trim cache → bonus] × steps → output

KV-cache strategy
-----------------
Target model — SelectiveCache
    Prefill once: context K,V cached normally.
    Verification: use_cache=True so ALL T tree tokens' K,V are captured.
    After acceptance: call cache.keep_positions([ctx | accepted_path]) to
    discard the non-accepted tree tokens in-place — one index-select per
    layer, no extra forward pass.
    Bonus token: run target on the 1 bonus token to extend the cache by 1
    and get next-step anchor logits.  This is the ONLY extra target call
    per step — saving acceptance_len target forward passes vs. the naive
    approach of re-running all accepted tokens.

Draft model
    No inter-step KV cache.  Context conditioning flows through the growing
    raw_target_hidden, which is extended each step using:
      • verify_out hidden states at accepted path indices (tree-attention,
        matches DFlash's practice of reusing verification hidden states)
      • bonus_out hidden states (causal, 1 token)

Position IDs (rotary correctness)
    Draft: full range [0 … n_ctx + T - 1] is passed as position_ids so the
    rotary embedding covers both the context (target_hidden) K-vectors and
    the tree (noise) K-vectors — same pattern as DFlash's spec_generate.
        ctx positions  : 0 … n_ctx-1
        tree positions : n_ctx + depth[i]  for node i
    Target verification: tree token positions only (context is in cache).

Acceptance logic
----------------
verify_logits[b, i] = target's predicted distribution for the token AFTER
draft_tokens[b, i], i.e. the distribution over node i's *child*, not node i.
To accept node i we use the PARENT's predicted distribution:
    root (i=0)  → anchor_logits (prefill output at last context position,
                  updated each step via bonus_out)
    node i > 0  → verify_logits[b, parent_ids[i]]
adjusted_parent_ids from TreeSpec indexes both into one concatenated tensor
[anchor_logits, verify_logits] with a single gather.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import DynamicCache

from tree.spec import TreeSpec
from tree.mask import build_tree_attn_mask
from dflash.model.utils import extract_context_feature
from trainer.metrics import tree_accept


# ── Selective KV cache ────────────────────────────────────────────────────────

class SelectiveCache(DynamicCache):
    """
    DynamicCache that can discard arbitrary cached positions in-place.

    After tree verification (use_cache=True), the cache contains K,V for ALL
    T tree tokens.  keep_positions() retains only the indices we want (context
    tokens + accepted path tokens) and drops the rest — no extra forward pass.

    The result is a valid DynamicCache whose get_seq_length() reflects the
    trimmed length, ready for subsequent use_cache=True calls.
    """

    def keep_positions(self, positions: Tensor) -> None:
        """
        Retain only the specified sequence positions, discarding the rest.

        Parameters
        ----------
        positions : [k] long (device-resident)
            Indices into the full cached sequence [0, seq_len) to keep,
            in the desired order (should be ascending for causal attention).

        Effect
        ------
        Each layer's key_cache and value_cache is replaced by the selected
        rows along the sequence dimension.  Shapes go from
            [B, n_heads, seq_len, head_dim]  →  [B, n_heads, k, head_dim]
        """
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx]   = self.key_cache[layer_idx][:, :, positions, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, positions, :]


# ── Sampling helper ───────────────────────────────────────────────────────────

def _sample(logits: Tensor, temperature: float) -> Tensor:
    """
    Sample a token from logits.

    Parameters
    ----------
    logits      : [..., V]
    temperature : 0.0 → greedy argmax; >0 → softmax sampling

    Returns
    -------
    token ids with the same leading shape as logits[..., 0]
    """
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    probs = F.softmax(logits / temperature, dim=-1)
    flat  = probs.reshape(-1, probs.shape[-1])
    return torch.multinomial(flat, num_samples=1).squeeze(-1).reshape(probs.shape[:-1])


# ── Main decode loop ──────────────────────────────────────────────────────────

@torch.no_grad()
def tree_spec_decode(
    context_ids: Tensor,                  # [1, n_ctx]
    model: "DraftWrapper",                # noqa: F821
    target: "torch.nn.Module",            # noqa: F821
    tree_spec: TreeSpec,
    target_layer_ids: list[int],
    max_new_tokens: int,
    temperature: float = 0.0,
    stop_token_ids: list[int] | None = None,
) -> tuple[Tensor, list[dict]]:
    """
    Generate up to ``max_new_tokens`` new tokens using tree speculative decoding.

    Only batch size 1 is supported.  Variable per-sample acceptance lengths
    make batched multi-step KV management extremely complex; B=1 is the
    standard setting for latency benchmarking.

    Parameters
    ----------
    context_ids     : [1, n_ctx] — tokenised prompt (on the correct device)
    model           : DraftWrapper — trained draft model (eval mode)
    target          : frozen target LM (HuggingFace, Qwen3 / LLaMA style)
    tree_spec       : TreeSpec for the current experiment tree shape
    target_layer_ids: indices of target hidden layers used for draft conditioning
    max_new_tokens  : maximum tokens to generate
    temperature     : 0.0 → greedy; >0 → softmax sampling with rejection
    stop_token_ids  : optional EOS / stop token ids; terminates early

    Returns
    -------
    output_ids : [1, n_ctx + n_generated] — full sequence including prompt
    step_stats : list of per-step dicts with keys
                    "acceptance_length"  int  — accepted tokens this step (incl. bonus)
                    "final_node"         int  — tree node index of deepest accepted node
                    "draft_tokens"       [T]  — draft at this step (CPU tensor)
    """
    assert context_ids.shape[0] == 1, "tree_spec_decode supports batch size 1 only"
    device = context_ids.device
    T = tree_spec.tree_size

    # ── Precompute static tree tensors (device-resident) ──────────────────────
    anc_mat = tree_spec.ancestor_matrix.to(device)       # [T, T] bool
    depths  = tree_spec.depths.to(device)                 # [T] long
    pos_rel = tree_spec.position_ids.to(device)           # [T] long  (= depths)
    adj_par = tree_spec.adjusted_parent_ids.to(device)    # [T] long
    stop_ids = (
        torch.tensor(stop_token_ids, dtype=torch.long, device=device)
        if stop_token_ids else None
    )

    # paths_to[q] = node indices from root to q in order (for accepted token extraction)
    paths_to: list[list[int]] = [
        list(reversed(tree_spec.path_to_root(q))) for q in range(T)
    ]

    # ── Initial prefill ────────────────────────────────────────────────────────
    # Use SelectiveCache from the start so subsequent keep_positions() calls
    # work uniformly (prefill populates it exactly like DynamicCache).
    target_kv = SelectiveCache()
    prefill_out = target(
        context_ids,                          # [1, n_ctx]
        past_key_values=target_kv,
        use_cache=True,
        output_hidden_states=True,
    )
    # target_kv now holds context K,V for n_ctx positions

    raw_target_hidden = extract_context_feature(
        prefill_out.hidden_states, target_layer_ids
    )  # [1, n_ctx, n_feat_layers * H]

    # anchor_logits: target's greedy prediction for the first tree root token
    anchor_logits = prefill_out.logits[:, -1, :]   # [1, V]

    # Sample and cache the first new token, then use it as the initial anchor.
    # Running target on it (use_cache=True) correctly appends its K,V and gives
    # us the anchor_logits for the first tree's root acceptance check.
    first_token = _sample(anchor_logits, temperature)    # [1]
    output_ids  = torch.cat([context_ids, first_token.unsqueeze(0)], dim=1)
    # [1, n_ctx+1]; grows each step

    first_out = target(
        first_token.unsqueeze(0),                   # [1, 1]
        past_key_values=target_kv,
        use_cache=True,
        output_hidden_states=True,
    )
    anchor_logits     = first_out.logits[:, -1, :]  # [1, V] — prediction for root
    raw_target_hidden = torch.cat(
        [raw_target_hidden,
         extract_context_feature(first_out.hidden_states, target_layer_ids)],
        dim=1,
    )  # [1, n_ctx+1, n_feat*H]
    anchor_id   = first_token    # [1]
    n_generated = 1              # the first token already generated
    step_stats: list[dict] = []

    # ── Decode loop ────────────────────────────────────────────────────────────
    while n_generated < max_new_tokens:
        n_ctx = raw_target_hidden.shape[1]   # grows each step

        # ── Draft pass ────────────────────────────────────────────────────────
        # Position IDs must span ALL n_ctx context positions + T tree positions
        # so rotary embeddings cover both k_ctx and k_noise in DFlash attention.
        #   ctx positions  : 0 … n_ctx-1
        #   tree positions : n_ctx + depth[i]  for node i
        ctx_pos  = torch.arange(n_ctx, device=device).unsqueeze(0)  # [1, n_ctx]
        tree_pos = (n_ctx + pos_rel).unsqueeze(0)                    # [1, T]
        full_pos = torch.cat([ctx_pos, tree_pos], dim=1)             # [1, n_ctx+T]

        draft_logits, _backbone_hs = model.infer_forward(
            anchor_ids=anchor_id,
            raw_target_hidden=raw_target_hidden,
            draft_past_kv=None,
            position_ids=full_pos,       # full range for correct rotary PE
        )
        # draft_logits: [1, T, V]

        draft_tokens = _sample(draft_logits, temperature)  # [1, T]

        # ── Verification pass ─────────────────────────────────────────────────
        # use_cache=True: all T tree tokens' K,V are appended to target_kv.
        # We'll discard the non-accepted ones after acceptance (keep_positions).
        attn_mask = build_tree_attn_mask(anc_mat, n_ctx, B=1, device=device)
        # [1, 1, T, n_ctx+T]: context keys fully attended, tree keys ancestor-masked

        verify_out = target(
            input_ids=draft_tokens,              # [1, T]
            position_ids=tree_pos,               # [1, T]
            past_key_values=target_kv,           # SelectiveCache, n_ctx entries
            attention_mask=attn_mask,
            use_cache=True,                      # ← capture all T tree K,V
            output_hidden_states=True,           # ← for raw_target_hidden update
        )
        verify_logits = verify_out.logits        # [1, T, V]
        # target_kv now has n_ctx + T entries

        # ── Acceptance ────────────────────────────────────────────────────────
        # verify_logits[0, i] predicts what comes after draft_tokens[0, i].
        # To accept node i we use the PARENT's predicted distribution:
        #   root  → anchor_logits  (updated each step via bonus_out)
        #   node i → verify_logits[0, parent_ids[i]]
        # adjusted_parent_ids maps [root→0, other→parent+1] into
        # [anchor_logits, verify_logits[0,0], …, verify_logits[0,T-1]].
        extended_logits = torch.cat(
            [anchor_logits.unsqueeze(1), verify_logits], dim=1
        )  # [1, 1+T, V]
        parent_logits = extended_logits[:, adj_par, :]   # [1, T, V]

        if temperature == 0.0:
            target_pred = parent_logits.argmax(dim=-1)           # [1, T]
            accepted    = (draft_tokens == target_pred)          # [1, T] bool
        else:
            # Rejection sampling: accept node i w.p. min(1, p_target / p_draft)
            draft_probs  = F.softmax(draft_logits  / temperature, dim=-1)  # [1, T, V]
            target_probs = F.softmax(parent_logits / temperature, dim=-1)  # [1, T, V]
            draft_p  = draft_probs.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)
            target_p = target_probs.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)
            accept_prob = (target_p / draft_p.clamp(min=1e-9)).clamp(max=1.0)
            accepted    = torch.rand_like(accept_prob) < accept_prob        # [1, T] bool

        # path_accepted[b, q] = True iff every ancestor of q (incl. q) was accepted
        path_accepted = tree_accept(accepted, anc_mat)   # [1, T] bool

        # Final accepted node: deepest with a fully-accepted path
        depth_score = torch.where(
            path_accepted,
            depths.float().unsqueeze(0),              # [1, T]
            depths.new_full((1, T), -1).float(),
        )  # [1, T]
        final_node_idx = int(depth_score.argmax(dim=1).item())

        # ── Trim cache: discard non-accepted tree tokens ──────────────────────
        # path_nodes: indices into the T-length tree-token input of the verify pass
        # (= the accepted node indices, in order from root to final_node)
        path_nodes = paths_to[final_node_idx]              # list[int], len = depth+1

        # Build keep_positions: context (0..n_ctx-1) + accepted path (n_ctx + node_idx)
        # The accepted path nodes were appended to the cache at positions
        # n_ctx + path_nodes[0], n_ctx + path_nodes[1], ... (in tree-input order).
        ctx_keep  = torch.arange(n_ctx, device=device)
        tree_keep = torch.tensor([n_ctx + i for i in path_nodes], device=device)
        keep_pos  = torch.cat([ctx_keep, tree_keep])      # [n_ctx + depth+1]
        target_kv.keep_positions(keep_pos)
        # cache is now n_ctx + len(path_nodes) = n_ctx + depth+1 entries

        # ── Bonus token ───────────────────────────────────────────────────────
        # Sample from verify_logits at the final accepted node (target's prediction
        # for what comes after the deepest accepted token).
        if temperature == 0.0:
            bonus_token = verify_logits[0, final_node_idx, :].argmax().view(1)
        else:
            bonus_probs = F.softmax(
                verify_logits[0, final_node_idx, :] / temperature, dim=-1
            )
            if path_accepted[0, final_node_idx]:
                # All path tokens accepted: sample directly from target
                adjusted_probs = bonus_probs
            else:
                # Last token rejected: sample from (p_target - p_draft)+
                fin_draft_probs = F.softmax(
                    draft_logits[0, final_node_idx, :] / temperature, dim=-1
                )
                adjusted_probs = (bonus_probs - fin_draft_probs).clamp(min=0.0)
                adjusted_probs /= adjusted_probs.sum().clamp(min=1e-9)
            bonus_token = torch.multinomial(adjusted_probs, 1)    # [1]

        # Run target on the bonus token (1 token) to:
        #   1. Append its K,V to the cache for the next step's verification.
        #   2. Get anchor_logits for the next tree's root acceptance check.
        #   3. Get its hidden states for draft conditioning.
        bonus_pos = torch.tensor([[n_ctx + len(path_nodes)]], device=device)
        # = [[n_ctx + depth_of_final + 1]]  (position right after accepted path)

        bonus_out = target(
            bonus_token.unsqueeze(0),                   # [1, 1]
            position_ids=bonus_pos,                      # [1, 1]
            past_key_values=target_kv,
            use_cache=True,
            output_hidden_states=True,
        )
        anchor_logits = bonus_out.logits[:, -1, :]       # [1, V] — next root prediction

        # Update raw_target_hidden:
        #   accepted path hidden states from verify_out (tree-attention, consistent
        #   with DFlash's practice of reusing verification outputs directly)
        #   + bonus token hidden states from bonus_out (causal)
        path_t = torch.tensor(path_nodes, device=device)
        accepted_hs = extract_context_feature(
            verify_out.hidden_states, target_layer_ids
        )[:, path_t, :]                                   # [1, depth+1, n_feat*H]
        bonus_hs = extract_context_feature(
            bonus_out.hidden_states, target_layer_ids
        )                                                  # [1, 1, n_feat*H]
        raw_target_hidden = torch.cat(
            [raw_target_hidden, accepted_hs, bonus_hs], dim=1
        )  # [1, n_ctx + depth+2, n_feat*H]

        # ── Update output and anchor ──────────────────────────────────────────
        accepted_token_ids = draft_tokens[0, path_t]      # [depth+1]
        new_tokens = torch.cat(
            [accepted_token_ids, bonus_token], dim=0
        ).unsqueeze(0)                                     # [1, depth+2]
        output_ids  = torch.cat([output_ids, new_tokens], dim=1)
        anchor_id   = bonus_token                          # [1]
        acceptance_length = len(path_nodes) + 1           # accepted + bonus

        n_generated += acceptance_length

        step_stats.append({
            "acceptance_length": acceptance_length,
            "final_node":        final_node_idx,
            "draft_tokens":      draft_tokens[0].cpu(),
        })

        # ── Early stopping ────────────────────────────────────────────────────
        if stop_ids is not None:
            if torch.isin(output_ids[0, context_ids.shape[1]:], stop_ids).any():
                break

    output_ids = output_ids[:, : context_ids.shape[1] + max_new_tokens]
    return output_ids, step_stats
