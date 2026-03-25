"""
Full multi-step tree speculative decoding.

Overview
--------
tree_spec_decode() runs the complete generate loop:
    prefill → [draft → (prune) → verify → accept → trim cache → bonus] × steps → output

Pruning
-------
Optional: set n_candidate_tokens < tree_size to enable AR-head pruning before
verification.  After drafting all T nodes, the AR head scores each node and
the n_candidate_tokens nodes with the highest path-acceptance probability are
selected for verification.  The rest are discarded.

Pruning criterion
-----------------
For each node q, its path-acceptance score is the product of AR probabilities
along the root-to-q path:

    path_log_p[q] = Σ_{k: ancestor_matrix[k,q]} log p_ar(draft_token[k])

Since path_log_p[child] ≤ path_log_p[parent], taking the top-k by score is
always a topologically valid subtree (every selected node's parent is also
selected).  No tree-validity post-processing is needed.

KV-cache strategy
-----------------
Target model — SelectiveCache
    Prefill once: context K,V cached normally.
    Verification: use_cache=True so all k (pruned) tree tokens' K,V are captured.
    After acceptance: call cache.keep_positions([ctx | accepted_path]) to
    discard the non-accepted tree tokens in-place — one index-select per
    layer, no extra forward pass.
    Bonus token: run target on the 1 bonus token to extend the cache by 1
    and get next-step anchor logits.

Draft model
    No inter-step KV cache.  Context conditioning flows through the growing
    raw_target_hidden, extended each step using:
      • verify_out hidden states at accepted path indices
      • bonus_out hidden states (causal, 1 token)

Position IDs (rotary correctness)
    Draft: full range [0 … n_ctx + T - 1] is passed as position_ids so the
    rotary embedding covers both the context (target_hidden) K-vectors and
    the tree (noise) K-vectors.
        ctx positions  : 0 … n_ctx-1
        tree positions : n_ctx + depth[i]  for node i
    Target verification: pruned tree token positions only (context is in cache).

Acceptance logic
----------------
verify_logits[b, i] = target's predicted distribution for the token AFTER
draft_tokens[b, i], i.e. the distribution over node i's *child*, not node i.
To accept node i we use the PARENT's predicted distribution:
    root (i=0)  → anchor_logits (prefill output at last context position,
                  updated each step via bonus_out)
    node i > 0  → verify_logits[b, parent_ids[i]]
adjusted_parent_ids (remapped to the pruned subgraph) indexes both into one
concatenated tensor [anchor_logits, verify_logits] with a single gather.
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
    k verified tree tokens.  keep_positions() retains only the indices we want
    (context tokens + accepted path tokens) and drops the rest — no extra
    forward pass.
    """

    def keep_positions(self, positions: Tensor) -> None:
        """
        Retain only the specified sequence positions, discarding the rest.

        Parameters
        ----------
        positions : [k] long (device-resident)
            Indices into the full cached sequence [0, seq_len) to keep,
            in ascending order.
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


# ── Pruning ───────────────────────────────────────────────────────────────────

def _prune_tree(
    ar_logits: Tensor,        # [1, T, V]
    draft_tokens: Tensor,     # [1, T]
    ancestor_matrix: Tensor,  # [T, T] bool
    n_candidates: int,
) -> Tensor:
    """
    Select the ``n_candidates`` nodes that maximise expected acceptance length.

    Score for node q:
        path_log_p[q] = Σ_{k: ancestor_matrix[k,q]=True} log p_ar(draft_token[k])

    Because path_log_p[child] ≤ path_log_p[parent] (multiplying by a prob ≤ 1),
    top-k by path score always forms a topologically valid subtree — no
    tree-validity check is needed.

    Returns
    -------
    pruned_ids : [k] long, sorted ascending (preserves cache-position order)
    """
    T = ar_logits.shape[1]
    n_candidates = min(n_candidates, T)

    log_p = F.log_softmax(ar_logits[0], dim=-1)                              # [T, V]
    node_log_p = log_p.gather(1, draft_tokens[0].unsqueeze(1)).squeeze(1)    # [T]

    # Sum log probs over ancestor path → log path-acceptance probability per node
    path_log_p = node_log_p @ ancestor_matrix.to(dtype=torch.float32)        # [T]

    top_k = path_log_p.topk(n_candidates).indices   # [k]
    return top_k.sort().values                        # [k] ascending


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
    n_candidate_tokens: int | None = None,
) -> tuple[Tensor, list[dict]]:
    """
    Generate up to ``max_new_tokens`` new tokens using tree speculative decoding.

    Only batch size 1 is supported.

    Parameters
    ----------
    context_ids         : [1, n_ctx] — tokenised prompt (on the correct device)
    model               : DraftWrapper — trained draft model (eval mode)
    target              : frozen target LM (HuggingFace, Qwen3 / LLaMA style)
    tree_spec           : TreeSpec for the current experiment tree shape
    target_layer_ids    : indices of target hidden layers used for draft conditioning
    max_new_tokens      : maximum tokens to generate
    temperature         : 0.0 → greedy; >0 → softmax sampling with rejection
    stop_token_ids      : optional EOS / stop token ids; terminates early
    n_candidate_tokens  : if set and < tree_size, prune the draft tree to this
                          many nodes using the AR head before verification.
                          None (default) → verify the full tree.

    Returns
    -------
    output_ids : [1, n_ctx + n_generated] — full sequence including prompt
    step_stats : list of per-step dicts with keys
                    "acceptance_length"  int  — accepted tokens this step (incl. bonus)
                    "final_node"         int  — original tree node index of deepest accepted node
                    "draft_tokens"       [T]  — full draft at this step (CPU tensor)
                    "n_verified"         int  — number of nodes sent to target
    """
    assert context_ids.shape[0] == 1, "tree_spec_decode supports batch size 1 only"
    device = context_ids.device
    T = tree_spec.tree_size
    do_prune = n_candidate_tokens is not None and n_candidate_tokens < T

    # ── Precompute static tree tensors (device-resident) ──────────────────────
    anc_mat     = tree_spec.ancestor_matrix.to(device)     # [T, T] bool
    depths      = tree_spec.depths.to(device)               # [T] long
    pos_rel     = tree_spec.position_ids.to(device)         # [T] long
    adj_par     = tree_spec.adjusted_parent_ids.to(device)  # [T] long
    parent_ids_ = tree_spec.parent_ids.to(device)           # [T] long  (-1 for root)
    stop_ids = (
        torch.tensor(stop_token_ids, dtype=torch.long, device=device)
        if stop_token_ids else None
    )

    # paths_to[q] = node indices from root to q in order (original indices)
    paths_to: list[list[int]] = [
        list(reversed(tree_spec.path_to_root(q))) for q in range(T)
    ]

    # ── Initial prefill ────────────────────────────────────────────────────────
    target_kv = SelectiveCache()
    prefill_out = target(
        context_ids,
        past_key_values=target_kv,
        use_cache=True,
        output_hidden_states=True,
    )

    raw_target_hidden = extract_context_feature(
        prefill_out.hidden_states, target_layer_ids
    )  # [1, n_ctx, n_feat_layers * H]

    anchor_logits = prefill_out.logits[:, -1, :]   # [1, V]

    first_token = _sample(anchor_logits, temperature)   # [1]
    output_ids  = torch.cat([context_ids, first_token.unsqueeze(0)], dim=1)

    first_out = target(
        first_token.unsqueeze(0),
        past_key_values=target_kv,
        use_cache=True,
        output_hidden_states=True,
    )
    anchor_logits     = first_out.logits[:, -1, :]
    raw_target_hidden = torch.cat(
        [raw_target_hidden,
         extract_context_feature(first_out.hidden_states, target_layer_ids)],
        dim=1,
    )
    anchor_id   = first_token
    n_generated = 1
    step_stats: list[dict] = []

    # ── Decode loop ────────────────────────────────────────────────────────────
    while n_generated < max_new_tokens:
        n_ctx = raw_target_hidden.shape[1]

        # ── Draft pass ────────────────────────────────────────────────────────
        ctx_pos  = torch.arange(n_ctx, device=device).unsqueeze(0)   # [1, n_ctx]
        tree_pos = (n_ctx + pos_rel).unsqueeze(0)                     # [1, T]
        full_pos = torch.cat([ctx_pos, tree_pos], dim=1)              # [1, n_ctx+T]

        draft_logits, _backbone_hs = model.infer_forward(
            anchor_ids=anchor_id,
            raw_target_hidden=raw_target_hidden,
            draft_past_kv=None,
            position_ids=full_pos,
        )
        # draft_logits: [1, T, V]
        draft_tokens = _sample(draft_logits, temperature)   # [1, T]

        # ── Pruning (optional) ────────────────────────────────────────────────
        if do_prune:
            ar_logits  = model.ar_qvalues(anchor_id, draft_tokens)   # [1, T, V]
            pruned_ids = _prune_tree(ar_logits, draft_tokens, anc_mat, n_candidate_tokens)
            # pruned_ids: [k] sorted ascending
        else:
            pruned_ids = torch.arange(T, device=device)              # [T]

        k = pruned_ids.shape[0]

        # ── Build pruned-subgraph quantities ──────────────────────────────────
        # Ancestor sub-matrix for the k verified nodes
        pruned_anc = anc_mat[pruned_ids][:, pruned_ids]              # [k, k] bool

        # Depths and position ids for the pruned nodes
        pruned_depths = depths[pruned_ids]                            # [k]
        pruned_pos_ids = (n_ctx + pruned_depths).unsqueeze(0)         # [1, k]

        # Mapping: original node index → position in pruned array (for cache trim)
        pos_map = torch.full((T,), -1, dtype=torch.long, device=device)
        pos_map[pruned_ids] = torch.arange(k, device=device)

        # Adjusted parent ids remapped to the pruned subgraph
        # For pruned node at position j (original = pruned_ids[j]):
        #   root (parent_ids_ == -1)  → 0      (anchor slot in extended_logits)
        #   otherwise                 → pos_map[parent] + 1
        par_orig = parent_ids_[pruned_ids]                            # [k]
        adj_par_k = torch.where(
            par_orig < 0,
            torch.zeros(k, dtype=torch.long, device=device),
            pos_map[par_orig.clamp(min=0)] + 1,
        )  # [k]

        # ── Verification pass ─────────────────────────────────────────────────
        attn_mask = build_tree_attn_mask(pruned_anc, n_ctx, B=1, device=device)
        # [1, 1, k, n_ctx+k]

        verify_out = target(
            input_ids=draft_tokens[:, pruned_ids],   # [1, k]
            position_ids=pruned_pos_ids,              # [1, k]
            past_key_values=target_kv,
            attention_mask=attn_mask,
            use_cache=True,
            output_hidden_states=True,
        )
        verify_logits = verify_out.logits            # [1, k, V]
        # target_kv now has n_ctx + k entries;
        # position j in the new k entries = pruned node pruned_ids[j]

        # ── Acceptance ────────────────────────────────────────────────────────
        extended_logits = torch.cat(
            [anchor_logits.unsqueeze(1), verify_logits], dim=1
        )  # [1, 1+k, V]
        parent_logits = extended_logits[:, adj_par_k, :]   # [1, k, V]

        if temperature == 0.0:
            target_pred = parent_logits.argmax(dim=-1)                     # [1, k]
            accepted    = (draft_tokens[:, pruned_ids] == target_pred)     # [1, k] bool
        else:
            draft_probs_k  = F.softmax(draft_logits[:, pruned_ids] / temperature, dim=-1)
            target_probs_k = F.softmax(parent_logits / temperature, dim=-1)
            tokens_k = draft_tokens[:, pruned_ids]                         # [1, k]
            draft_p  = draft_probs_k.gather(-1, tokens_k.unsqueeze(-1)).squeeze(-1)
            target_p = target_probs_k.gather(-1, tokens_k.unsqueeze(-1)).squeeze(-1)
            accept_prob = (target_p / draft_p.clamp(min=1e-9)).clamp(max=1.0)
            accepted    = torch.rand_like(accept_prob) < accept_prob       # [1, k] bool

        # path_accepted[b, j] = True iff every ancestor of pruned node j was accepted
        path_accepted = tree_accept(accepted, pruned_anc)   # [1, k] bool

        # Final accepted node: deepest pruned node with a fully-accepted path
        depth_score = torch.where(
            path_accepted,
            pruned_depths.float().unsqueeze(0),
            pruned_depths.new_full((1, k), -1).float(),
        )  # [1, k]
        final_node_pruned_pos = int(depth_score.argmax(dim=1).item())
        final_node_orig       = int(pruned_ids[final_node_pruned_pos].item())

        # ── Trim cache: discard non-accepted tree tokens ──────────────────────
        # path_nodes_orig: original node indices from root to final accepted node
        path_nodes_orig = paths_to[final_node_orig]   # list[int]

        ctx_keep  = torch.arange(n_ctx, device=device)
        # Cache position of pruned_ids[j] = n_ctx + j  (sequential in verify call)
        tree_keep = torch.tensor(
            [n_ctx + int(pos_map[p].item()) for p in path_nodes_orig],
            device=device,
        )
        target_kv.keep_positions(torch.cat([ctx_keep, tree_keep]))

        # ── Bonus token ───────────────────────────────────────────────────────
        if temperature == 0.0:
            bonus_token = verify_logits[0, final_node_pruned_pos, :].argmax().view(1)
        else:
            bonus_probs = F.softmax(
                verify_logits[0, final_node_pruned_pos, :] / temperature, dim=-1
            )
            if path_accepted[0, final_node_pruned_pos]:
                adjusted_probs = bonus_probs
            else:
                fin_draft_probs = F.softmax(
                    draft_logits[0, final_node_orig, :] / temperature, dim=-1
                )
                adjusted_probs = (bonus_probs - fin_draft_probs).clamp(min=0.0)
                adjusted_probs /= adjusted_probs.sum().clamp(min=1e-9)
            bonus_token = torch.multinomial(adjusted_probs, 1)   # [1]

        bonus_pos = torch.tensor([[n_ctx + len(path_nodes_orig)]], device=device)
        bonus_out = target(
            bonus_token.unsqueeze(0),
            position_ids=bonus_pos,
            past_key_values=target_kv,
            use_cache=True,
            output_hidden_states=True,
        )
        anchor_logits = bonus_out.logits[:, -1, :]

        # ── Update raw_target_hidden ──────────────────────────────────────────
        # Positions of accepted path nodes in the verify_out hidden states
        path_pruned_pos = torch.tensor(
            [int(pos_map[p].item()) for p in path_nodes_orig], device=device
        )
        accepted_hs = extract_context_feature(
            verify_out.hidden_states, target_layer_ids
        )[:, path_pruned_pos, :]                            # [1, depth+1, n_feat*H]
        bonus_hs = extract_context_feature(
            bonus_out.hidden_states, target_layer_ids
        )                                                    # [1, 1, n_feat*H]
        raw_target_hidden = torch.cat(
            [raw_target_hidden, accepted_hs, bonus_hs], dim=1
        )

        # ── Update output and anchor ──────────────────────────────────────────
        path_t = torch.tensor(path_nodes_orig, device=device)
        accepted_token_ids = draft_tokens[0, path_t]       # [depth+1]
        new_tokens  = torch.cat([accepted_token_ids, bonus_token], dim=0).unsqueeze(0)
        output_ids  = torch.cat([output_ids, new_tokens], dim=1)
        anchor_id   = bonus_token
        acceptance_length = len(path_nodes_orig) + 1

        n_generated += acceptance_length

        step_stats.append({
            "acceptance_length": acceptance_length,
            "final_node":        final_node_orig,
            "draft_tokens":      draft_tokens[0].cpu(),
            "n_verified":        k,
        })

        # ── Early stopping ────────────────────────────────────────────────────
        if stop_ids is not None:
            if torch.isin(output_ids[0, context_ids.shape[1]:], stop_ids).any():
                break

    output_ids = output_ids[:, : context_ids.shape[1] + max_new_tokens]
    return output_ids, step_stats
