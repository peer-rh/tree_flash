# Tree-Flash: Method

Tree-Flash extends DFlash speculative decoding by replacing its linear block draft with a tree-structured draft. Instead of verifying one candidate sequence per step, the target model verifies an entire tree of candidates in a single forward pass, accepting the longest valid path and increasing expected tokens generated per step.

---

## 1. Background: DFlash

DFlash (Chen, Liang, Liu 2026) is a speculative decoding method that drafts an entire block of `block_size` tokens in one parallel forward pass using a bidirectional (non-causal) diffusion model.

**Key properties:**
- The draft model denoises a fully-masked block conditioned on hidden states extracted from the target model.
- KV injection: each draft attention layer's keys and values are the concatenation of projected target hidden states and the current draft hidden states, giving the draft model direct access to the target's representations without causal constraints.
- At decode time the accepted tokens advance the sequence and the draft model runs again from the new position.
- DFlash achieves ~6× wall-clock speedup on Qwen3-8B and ~2.5× over EAGLE-3.

**Limitation addressed here:** DFlash accepts tokens linearly — one path through the block. If the draft diverges from the target at position *k*, all tokens at positions *k+1, …* are discarded. Tree drafting recovers some of this waste by exploring multiple continuations simultaneously.

---

## 2. Draft Model Architecture

The draft model is `TreeDraftModel` (extending `DFlashDraftModel`), a 5-layer transformer with bidirectional (non-causal) self-attention over the draft tokens. The layers are split into a **shared backbone** (layers 0..N-2) and two independent **heads** (diffusion head and AR head) that both operate on the shared backbone output.

### 2.1 Feature extraction

At each forward pass the target model runs on the context and returns hidden states from all its layers. A fixed subset of `n_feature_layers` target layers (uniformly spaced) are concatenated along the feature dimension and projected down to the draft model's hidden size:

```
raw_target_hidden : [B, ctx_len, n_feature_layers × H_target]
    → fc (Linear, no bias)
    → hidden_norm (RMSNorm)
    → target_hidden_proj : [B, ctx_len, H]
```

### 2.2 KV injection

Each draft attention layer concatenates the projected target hidden states with the current draft hidden states to form its keys and values:

```
k = cat([k_proj(target_hidden_proj),  k_proj(noise_embedding)],  dim=seq)
v = cat([v_proj(target_hidden_proj),  v_proj(noise_embedding)],  dim=seq)
q = q_proj(noise_embedding)
```

The query attends over the full `ctx_len + tree_size` sequence, giving every draft position access to the complete context. Rotary position embeddings are applied to both the context and tree token positions using a unified position ID vector that spans the full range.

### 2.3 Bidirectional attention over the tree

Unlike standard causal LMs, the draft model applies **no causal mask** — all tree nodes attend to all other tree nodes and to all context tokens. This means the draft model has a global view of every tree node it is denoising simultaneously, which is appropriate because all positions start masked and are denoised in one pass.

### 2.4 Shared backbone + split heads

`TreeDraftModel.forward()` is split into two steps:

```
layers[0..N-2]  →  shared_hs          [B, tree_size, H]  (shared backbone output)

shared_hs  →  layers[N-1, diffusion]  →  norm  →  lm_head  →  draft_logits
shared_hs  →  ar_layer  (in ARHead)   →  norm  →  lm_head  →  ar_logits
```

Exposed methods on `TreeDraftModel`:
- `shared_forward(position_ids, noise_embedding, target_hidden)` → `(shared_hs, target_hidden_proj, position_embeddings)`
- `diffusion_head(shared_hs, target_hidden_proj, position_embeddings)` → `backbone_hs` (normed, ready for lm_head)

Sharing all-but-last layers between diffusion and AR paths maximises parameter efficiency while allowing each head to specialise its final transformer block.

### 2.5 Output

Draft token logits are obtained by applying the **shared** target LM head to the diffusion head output:

```
backbone_hs  = diffusion_head(shared_hs, ...)       # [B, tree_size, H]
draft_logits = lm_head(backbone_hs)                 # [B, tree_size, V]
```

Sharing the LM head keeps the draft and target models in the same vocabulary space without adding parameters.

---

## 3. AR Pruning Head

The AR head (`ARHead`) is the **only tree-topology-aware component** in the draft model. It takes the **shared backbone output** and injects the **embedding of each node's parent token** before running a dedicated final transformer block, making its predictions conditional on the particular path taken to reach each node.

```
ar_input = shared_hs + parent_proj(parent_embeds)   [B, tree_size, H]
ar_hs    = ar_layer(ar_input, target_hidden_proj, position_embeddings)
ar_logits = lm_head(norm(ar_hs))                    [B, tree_size, V]
```

Components:
- `parent_proj`: `Linear(H → H, bias=False)` — projects parent token embedding into the hidden space
- `ar_layer`: `Qwen3DFlashDecoderLayer` with `layer_idx = num_draft_layers` — a full transformer block with KV injection, distinct from the diffusion head's final layer
- `norm`: `RMSNorm(H)` — independent from the diffusion head's norm
- `lm_head`: shared with the target model (same as draft's lm_head)

The AR head uses the same shared `lm_head` as the draft model. During training it is supervised with the same CumProd-weighted cross-entropy as the draft, scaled by `ar_loss_weight` (λ = 0.1).

During inference the AR head provides **q-values** for pruning: given the sampled draft tokens as parent embeddings (not teacher-forced), it estimates how likely each subtree path is to be accepted, enabling the model to discard low-probability branches before sending the tree to the target for verification (Experiment 2+).

---

## 4. Tree Structure

### 4.1 Tree v1 parameterisation

The verification tree is defined by two parameters:

- **`n_subtrees`** (training-time): the number of primary-path nodes used in each training item and during inference. The primary path is assembled directly from the response at dataset-load time — it is not stored in stage 2.
- **`sub_tree_paths`** (stage-2 and training): a list of `"X-Y"` edge strings defining an arbitrary rooted subtree, where `X` is the parent node index and `Y` is the child node index (0 = attachment root). These are the **diverging alternatives** generated in stage 2.

The full tree is constructed by:
1. Building a primary path of `n_subtrees` nodes: `0 → 1 → 2 → … → n_subtrees-1` — taken from `response[t : t + n_subtrees]`.
2. Attaching one copy of the subtree at **every** primary-path node. Subtree node 0 is the primary-path node itself (the attachment point); subtree nodes 1, 2, … become its additional children, with tokens diverging from the response.

This gives:
```
tree_size = n_subtrees + n_subtrees × subtree_size
```
where `subtree_size` is the number of non-root subtree nodes.

**Default tree:** `n_subtrees=8`, `sub_tree_paths=["0-1","0-2","0-3","1-4","1-5","2-6","2-7"]`

The subtree has root (attachment point) with 3 children (1, 2, 3); node 1 has children (4, 5); node 2 has children (6, 7); node 3 is a leaf. `subtree_size=7`, `tree_size = 8 + 8×7 = 64`.

```
anchor
  └─ primary node 0  [response[t]]
       ├─ primary node 1  [response[t+1]]
       │    ├─ primary node 2  …  (primary path continues for n_subtrees nodes)
       │    ├─ alt A  (subtree depth-1, diverges from response[t+2])
       │    │    ├─ alt A1  (subtree depth-2)
       │    │    └─ alt A2
       │    └─ alt B
       │         ├─ alt B1
       │         └─ alt B2
       │    └─ alt C  (leaf)
       ├─ alt A  (subtree copy at primary node 0, diverges from response[t+1])
       │    ├─ alt A1
       │    └─ alt A2
       └─ alt B  …
```

Each primary path node has an identical subtree copy attached (same structure, independently filled tokens). The depth-1 nodes of each subtree always diverge from the next response token; deeper nodes are sampled freely.

### 4.2 Derived quantities (TreeSpec)

`TreeSpec.__post_init__` pre-computes all derived quantities needed throughout training and inference:

| Tensor | Shape | Description |
|---|---|---|
| `parent_ids` | `[T]` | Parent node index; −1 for root |
| `depths` | `[T]` | Depth of each node (root = 0) |
| `ancestor_matrix` | `[T, T]` bool | `[k, q]` = True iff k is ancestor-or-self of q |
| `adjusted_parent_ids` | `[T]` | Index into `[anchor, tree_token_0, …]`; root → 0 (anchor slot) |
| `position_ids` | `[T]` | Depth-based relative position IDs; add `ctx_len` for absolute positions |
| `sibling_pairs` | `[n_pairs, 2]` | All (i, j) pairs sharing the same parent, i < j |

### 4.3 Attention mask for verification

During the tree verification pass, each tree token is allowed to attend to:
- **All context tokens** (which are held in the target's KV cache).
- **Its own ancestors and itself** in the tree (not siblings or descendants).

This is exactly the `ancestor_matrix` transposed. Two mask formats are supported:
- `build_tree_attn_mask`: dense `[B, 1, T, ctx_len + T]` additive float mask (0 / −∞) for HuggingFace SDPA.
- `build_flex_block_mask`: sparse `BlockMask` for `torch.nn.attention.flex_attention` (preferred at scale).

---

## 5. Data Pipeline

### Stage 1 — Response generation (`stage1_generate.py`)

Stage 1 generates the raw training corpus using the target model via vLLM.

**Input:** Nemotron V2 (chat, math, code, STEM splits) + CodeAlpaca 20k, up to 800 k prompts after filtering. Prompts are formatted with the target model's chat template and filtered to ≤ 66% of `max_seq_len` tokens to leave headroom for the generated response.

**Process:** vLLM runs the target model (temperature = 1.0) to generate completions.

**Output:** sharded JSONL files, each line:
```json
{"prompt": "<formatted prompt string>", "response": "<generated completion>"}
```

### Stage 2 — Continuation tree generation (`stage2_trees.py`)

Stage 2 runs the target model on each (prompt, response) pair to generate the ground-truth continuation trees used for training.

**Input:** directory of Stage 1 JSONL shards. Both `prompt` and `response` are encoded with `add_special_tokens=False` (the chat template already inserts all special tokens).

**Output:** A single HDF5 file with the following datasets:

| Dataset | Shape | Dtype | Description |
|---|---|---|---|
| `prompt_ids` | `[N]` vlen | int64 | Tokenised prompt per sequence |
| `response_ids` | `[N]` vlen | int64 | Tokenised response per sequence |
| `response_probs` | `[T]` flat | float32 | `p(response[t] \| context[:t])` for every response position; all sequences concatenated |
| `continuation_trees` | `[T, subtree_size]` | int64 | Subtree tokens at each position; −1 = IGNORE_IDX for non-selected positions |
| `continuation_trees_probs` | `[T, subtree_size]` | float32 | Individual AR probability of each subtree token; 0.0 = IGNORE_IDX |
| `selected_positions` | `[N]` vlen | int64 | Response-relative indices of selected subtree positions per sequence |
| `sequence_offsets` | `[N+1]` | int64 | Row-pointer index: sequence n occupies `[offsets[n]:offsets[n+1]]` |

`T = Σ S_R_n` (sum of all response lengths). Row `offsets[n] + t` corresponds to response position `t` of sequence `n`.

**What stage 2 stores at position t:**

Stage 2 stores only the **diverging alternatives** — not the primary path, which is just `response[t : t + n_subtrees]` and is assembled at training time directly from `response_ids`.

The subtree at position t:
- **Depth-1 nodes**: the top-k most-likely tokens from `logits_after(response[t])`, **excluding** `response[t+1]` (which is already on the primary path). Extracted from the base forward pass.
- **Depth-2 nodes**: for each depth-1 parent with token X, the context `[prompt + response[0..t]] + [X]` is run through the target model; the final-position logits provide the depth-2 children. No exclusion needed at depth ≥ 2.

**Position selection:**

One forward pass over `prompt + response` gives `response_probs[t]` for all positions. The *uncertainty score* is:

```
score(t) = 1 − response_probs[t]
```

Positions where the response token was hard to predict are most valuable — a subtree there covers more probability mass. The top `num_trees_per_seq` positions (by descending score) are selected; the rest have IGNORE_IDX / 0.0 in `continuation_trees` / `continuation_trees_probs`. `response_probs` is always populated for all positions.

Valid positions for selection: `0 ≤ t < S_R` (any response position). The constraint `t + n_subtrees ≤ S_R` is enforced at training time, not at stage-2 time, because `n_subtrees` is a training-time parameter independent of stage 2.

**Probabilities:**

`continuation_trees_probs[t][node]` stores the individual AR probability of each subtree token:
```
p_target(token | context + ancestor path to node)
```
These are **individual** per-node probabilities, not cumulative products. The training loss computes cumprods on the fly. `response_probs` similarly stores individual (not cumulative) per-position probs for the primary path nodes.

---

## 6. Training

### 6.1 Loss function

Training minimises a **CumProd-weighted cross-entropy** over all tree nodes jointly for both the draft model and the AR head:

```
L = L_draft + λ · L_ar
```

CumProd weights are computed on-the-fly from the individual probabilities stored in the dataset:

```
log_cumprod[b, q] = Σ_{k: ancestor_matrix[k,q]=True}  log(tree_probs[b, k])
cumprod_weight[b, q] = exp(log_cumprod[b, q])
```

This is one `[B, T] × [T, T]` matrix multiply using the `ancestor_matrix` from `TreeSpec`.

```
         Σ_{b,i}  w[b,i] · CE(draft_logits[b,i], tree_tokens[b,i])
L_draft = ─────────────────────────────────────────────────────────
                         Σ_{b,i}  w[b,i]

         Σ_{b,i}  w[b,i] · CE(ar_logits[b,i], tree_tokens[b,i])
L_ar    = ─────────────────────────────────────────────────────────
                         Σ_{b,i}  w[b,i]
```

Positions with `tree_tokens == −1` (IGNORE_IDX) are excluded via `ignore_index=−1` in `F.cross_entropy` and naturally receive zero weight (their `tree_probs = 0.0 → cumprod = 0.0`).

**Motivation:** Nodes on high-probability paths — those that will actually be reached and verified at test time — receive proportionally higher loss weight, concentrating capacity on the paths that matter.

### 6.2 Training forward pass

The `Stage2Dataset` explodes the per-sequence HDF5 data into individual `(context_ids, tree_tokens, tree_probs)` training examples. For each selected `(sequence n, anchor t)` pair where `t + n_subtrees ≤ S_R`:

```
context_ids = (prompt_ids[n] + response_ids[n][:t])[-ctx_len:]      [ctx_len]

# Full tree assembled on the fly:
base = offsets[n]
tree_tokens[:n_subtrees]                  = response_ids[n][t : t+n_subtrees]    (primary path)
tree_tokens[n_subtrees + i*ss : ... ]     = continuation_trees[base+t+i]         (subtree at node i)

tree_probs[:n_subtrees]                   = response_probs[base+t : base+t+n_subtrees]
tree_probs[n_subtrees + i*ss : ... ]      = continuation_trees_probs[base+t+i]
```

where `ss = subtree_size`. Subtree slots are IGNORE_IDX / 0.0 for primary-path nodes `t+i` that were not selected in stage 2; these receive near-zero cumprod weight and are effectively ignored by the loss.

Each training step:

1. **Target model** (frozen, `torch.no_grad()`): forward pass on `context_ids` with `output_hidden_states=True`. Extract `raw_target_hidden [B, ctx_len, n_feat × H]` by concatenating the `n_feature_layers` uniformly-sampled hidden states.

2. **DraftWrapper.forward** (trainable):
   - Embed `noise_ids` (all mask tokens) → `noise_embedding [B, tree_size, H]`.
   - **Shared backbone**: `shared_hs, target_hidden_proj, pos_emb = draft.shared_forward(...)`.
   - **Diffusion head**: `backbone_hs = draft.diffusion_head(shared_hs, ...)`. Draft logits: `lm_head(backbone_hs)`.
   - **AR head**: construct parent embeddings by embedding `[anchor_id, tree_tokens_0, …]` indexed with `adjusted_parent_ids`. AR logits: `ar_head(shared_hs, parent_embeds, target_hidden_proj, pos_emb)`.

3. **Loss**: `compute_loss(draft_logits, ar_logits, tree_tokens, tree_probs, λ, ancestor_matrix)`.

Teacher forcing is used for the AR head at train time — parent embeddings come from the ground-truth `tree_tokens`, not from the draft's own predictions. At inference the AR head uses sampled draft tokens as parent embeddings.

### 6.3 Infrastructure

**Lightning Fabric** manages data-parallel training across 8 GPUs (DDP strategy, bf16-mixed precision):

- The **target model** is placed on the device with `.to()` and is never passed to `fabric.setup()`. DDP does not touch its parameters — it has no gradients.
- The **DraftWrapper** (draft + AR head) is wrapped with `fabric.setup(model, optimizer)`. Only `draft` and `ar_head` parameters have `requires_grad=True`; `lm_head` and `embed_tokens` are frozen references inside the wrapper so DDP skips their gradient synchronisation.
- The **`ancestor_matrix`** `[tree_size, tree_size]` is placed on device once at setup time; passed to `compute_loss` each step.
- **`torch.compile`** is applied to `DraftWrapper.forward()`, which has static shapes (fixed `batch_size`, `ctx_len`, `tree_size`) enabling full graph compilation.

**Optimiser:** AdamW (fused), cosine LR schedule with warmup.

**Gradient accumulation:** configurable via `grad_accum`; `fabric.no_backward_sync` suppresses DDP all-reduce on accumulation micro-steps.

### 6.4 Validation

Two validation modes run periodically during training:

**`validate_loss`** (every `val_loss_every` steps): runs the same training forward pass on held-out data; reports `val_loss`, `val_draft_loss`, `val_ar_loss`. Fast.

**`validate_spec`** (every `val_spec_every` steps): measures actual speculative decoding quality on held-out contexts. For each batch:
1. Prefill target with `use_cache=True` → `target_kv`.
2. Draft pass (`infer_forward`, no teacher forcing, no AR head).
3. Verification pass (`use_cache=False`, tree attention mask).
4. Acceptance and metric accumulation (see Section 7.2 for acceptance details).

Metrics reported: average acceptance length, equality heatmap, sibling equality rate, final-node histogram, % of steps where the accepted path lies on the primary (left-most) path.

---

## 7. Speculative Decoding

### 7.1 Overview

At inference, tree-flash runs in a loop:

```
prefill → repeat {
    draft tree  →  verify tree  →  accept path  →  extend output
}
```

The key advantage over sequential speculative decoding is that a single target forward pass verifies the entire tree (all `tree_size` nodes), and the longest valid path from root to any accepted node is taken, including any node in any branch — not just the primary path.

### 7.2 Acceptance logic

Acceptance in a tree differs from the linear case because the target model's output at position `i` predicts the **token after** node `i` — that is, the distribution for node `i`'s child, not for node `i` itself. To decide whether to accept node `i`, we use the **parent's** output:

- **Root** (node 0): accepted if `draft_tokens[0] == target_logits[last_context_position].argmax()`.
- **Node `i` (i > 0)**: accepted if `draft_tokens[i] == verify_logits[parent_ids[i]].argmax()`.

`adjusted_parent_ids` encodes these indices into a single concatenated tensor `[anchor_logits, verify_logits_0, …, verify_logits_{T-1}]`, so acceptance reduces to one gather and one argmax:

```python
extended_logits = cat([anchor_logits.unsqueeze(1), verify_logits], dim=1)  # [B, 1+T, V]
parent_preds    = extended_logits[:, adjusted_parent_ids, :].argmax(dim=-1) # [B, T]
accepted        = (draft_tokens == parent_preds)                            # [B, T]
```

For temperature > 0, standard rejection sampling is used:
```
accept node i with probability min(1, p_target(draft_tokens[i]) / p_draft(draft_tokens[i]))
```

**Path acceptance** (vectorised): a node's full path from the root must be accepted. This uses a single matrix multiply instead of a sequential BFS loop:

```python
rejected_count = (~accepted).float() @ ancestor_matrix.float()   # [B, T] × [T, T] → [B, T]
path_accepted  = rejected_count < 0.5                            # [B, T]
```

`rejected_count[b, q]` counts rejected ancestors of node `q` in sample `b`; a node is path-accepted iff this count is zero.

**Final node selection**: the deepest path-accepted node across all branches.

**Bonus token**: sampled from `verify_logits[final_node]` — the target's prediction for the position immediately after the accepted path.

### 7.3 KV cache management (SelectiveCache)

Efficient multi-step decoding reuses target model KV cache across steps. The key challenge with trees is that verifying all `T` nodes with `use_cache=True` pollutes the cache with non-accepted tokens. We solve this with `SelectiveCache`:

1. **Prefill** target on the context → `target_kv` (context K,V cached, `ctx_len` entries).
2. **Draft** pass: draft model runs over all `T` tree positions with growing `raw_target_hidden`.
3. **Verify** with `use_cache=True`: all `T` tree tokens' K,V are appended to `target_kv`.
4. **Trim** `target_kv.keep_positions([0…ctx_len−1] + [ctx_len + i for i in accepted_path])`: one index-select per layer, discarding non-accepted K,V in place. The accepted path nodes were encoded with depth-based position IDs (`ctx_len + depth[i]`), which are sequential along any root-to-node path, so the trimmed cache remains causally consistent.
5. **Bonus token** (`1` token): run target causally to append its K,V and obtain the next step's `anchor_logits`.
6. **Update `raw_target_hidden`**: accepted path hidden states are extracted from `verify_out.hidden_states` at path node indices; bonus hidden states come from the bonus forward pass. Both are concatenated to `raw_target_hidden` for the next draft step.

Net cost per step: 1 full tree verification pass + 1 single-token forward pass.

### 7.4 Draft position IDs

The draft model's rotary embeddings must cover both the context K-vectors and the tree K-vectors. At step `s` with accumulated context length `n_ctx`, the full position ID vector is:

```
[0, 1, …, n_ctx−1,   n_ctx + depth[0], n_ctx + depth[1], …, n_ctx + depth[T−1]]
 └─────────────────────┘   └──────────────────────────────────────────────────────┘
    context positions                    tree token positions
```

This ensures all K,V vectors in the draft attention get correct rotary encoding, matching the pattern used in DFlash's own `spec_generate`.

---

## 8. Experiment Plan

See [experiments.md](experiments.md) for the full ablation plan. In brief:

| Exp | Question | Variable |
|---|---|---|
| 1 | Does tree drafting improve acceptance length over linear? | Tree v1, no pruning |
| 2 | Which pruning method is best? | Q_Logit vs AR_Head head |
| 3 | How to maximise sibling diversity? | Aux KL loss vs deduplicated sampling |

**Metrics:** average acceptance length, sibling equality rate, equality heatmap, final-node histogram, % of accepted paths on the primary path.

---

## 9. File Map

```
config.py                       Training hyperparameters (TrainConfig)
train.py                        CLI entry point

tree/
  spec.py                       TreeSpec: tree structure + all derived tensors
  mask.py                       Tree attention masks (dense + Flex Attention)

model/
  draft_model.py                TreeDraftModel: shared backbone + split heads
  ar_head.py                    AR pruning head (transformer block + shared lm_head)
  draft_wrapper.py              DraftWrapper: joint forward for training + infer helpers

trainer/
  loss.py                       CumProd-weighted CE (cumprods computed on-the-fly)
  metrics.py                    TreeMetrics, validate_loss, validate_spec, tree_accept
  spec_decode.py                Full multi-step tree speculative decoding (SelectiveCache)
  trainer.py                    FabricTrainer: DDP training loop

data/
  dataset.py                    Stage2Dataset: HDF5-backed dataset, assembles full tree (primary path + subtrees) at load time

data_pipeline/
  stage1_generate.py            vLLM response generation from Nemotron + CodeAlpaca
  stage2_trees.py               Per-position subtree generation (diverging alternatives + response_probs)

dflash/                         DFlash source (read-only reference)
  model/dflash.py               DFlashDraftModel architecture
  model/utils.py                extract_context_feature, sampling utilities
```
