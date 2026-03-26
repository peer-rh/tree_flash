# v2 — Tree-Flash: Tree Speculative Decoding

## TODO

- [ ] **Run stage 1** to generate prompt/response pairs (Nemotron-v2 + CodeAlpaca)
- [ ] **Run stage 2** to generate sub-tree data from stage 1 output
- [ ] **Run stage 3** to train draft model + TreePositionEmbedding
- [ ] **Experiment 1** (Basic PoC): verify tree drafting improves acceptance length over linear DFlash
- [ ] **Experiment 2** (Tree Position Embeddings): Initial Bias vs Relational Bias vs Both
- [ ] **Experiment 3** (Pruning): Q_Logit vs AR_Head heads
- [ ] **Experiment 4** (Tree Shape): Tree v1 vs mass-front-loaded tree
- [ ] **Experiment 5** (Sibling Deduplication): Aux Loss vs Deduplicated Sampling
- [ ] **Correctness test** for `spec_dec.py`: greedy output at temperature=0 must match target-only generation exactly
- [ ] **Speed benchmark**: tree spec_dec vs linear DFlash `spec_generate` vs target-only on GSM8K
- [x] Wire per-layer `score_mod` into DFlashDraftModel so each layer uses its own relational bias — done via `TreeDFlashDraftModel` subclass in `v2/model.py`
- [ ] Evaluate whether CumProds decay smoothing is needed (see `experiments.md` note on DFlash comparison)
- [ ] Add `--prompt-file` batch mode to `spec_dec.py` benchmark output (tokens/sec per prompt)
- [x] Stage 2: multi-GPU data-parallel generation via `torchrun` with rank-local temp HDF5 shards merged into one final output

---

## Overview

This directory implements a three-stage pipeline to train and run **tree speculative decoding** on top of DFlash. The idea: instead of verifying a single linear block of tokens per step (as in DFlash), we draft a tree of candidate continuations and verify the entire tree in one target forward pass, then accept the longest valid path. This increases the expected number of accepted tokens per verification step without increasing draft cost.

The DFlash base (`dflash/`) is kept unchanged. All new code lives in `v2/`.

---

## Stage 1 — Sequence Generation (`dflash/` standard pipeline)

Stage 1 is handled by the existing DFlash stage-1 tooling. It produces a directory of JSONL shards, each record containing:

```json
{"prompt": "<chat-template prompt>", "response": "<model continuation>"}
```

- Source datasets: Nemotron-v2 and CodeAlpaca
- Generation: temperature T=1, standard autoregressive sampling
- The concatenation `prompt + response` must be a valid tokenized sequence for the LLM (chat template applied)

---

## Stage 2 — Sub-Tree Generation (`v2/stage2.py`)

For each prompt+response pair from stage 1, stage 2 runs the target model and builds a **sub-tree** at each response token position. The sub-tree encodes the most likely alternative continuations starting from that position.

### Sub-tree structure

A sub-tree is parameterised by a set of edges `X-Y` (default: `["0-1","0-2","0-3","1-4","1-5","2-6","2-7"]`), defining a rooted tree of 8 nodes:

```
        0
      / | \
     1  2  3
    /\ /\
   4 5 6 7
```

Node 0 is the **root**, set to the actual token at that response position (`x_t`). Depth-1 nodes (1, 2, 3) are the top-k next-token alternatives (excluding the original `x_{t+1}`). Depth-2 nodes (4, 5, 6, 7) are the top-k continuations of each depth-1 node, generated with a second target forward pass using ancestor-based attention.

### Key data structures

**`SubTreeInfo`** — parsed sub-tree topology:
- `parent_map: dict[int,int]` — vertex → parent vertex
- `children_map: dict[int,list[int]]` — vertex → sorted children
- `ancestor_map: Tensor[st_size, st_size]` bool — `ancestor_map[k, q]` = k is ancestor-or-self of q
- `depth_of: list[int]` — depth of each vertex in the sub-tree
- `nodes_at_depth`, `non_leaf_at_depth` — vertex sets grouped by depth

**`build_step_attention_mask`** — builds the 4D verification mask used during tree expansion: each query vertex attends to its ancestors within the same sub-tree copy, and causally to the prefix context.

### Output HDF5 schema

```
prompt_ids          vlen int64   [N]            per-sequence
response_ids        vlen int64   [N]            per-sequence
sub_trees           int64        [R_total, st_size]  per response token
sub_trees_ar_probs  float32      [R_total, st_size]  AR probs for each node
sequence_offsets    int64        [N+1]          row ranges into sub_trees
```

`sub_trees_ar_probs[t, v]` is the autoregressive probability of node `v`'s token given its parent's context — used as CumProds weights during training.

When launched with `torchrun`, Stage 2 runs one worker per GPU. Each rank writes a temporary HDF5 shard under `<output>.parts/`, and rank 0 merges those shards back into one final HDF5 with the same schema shown above. The merged file remains directly compatible with Stage 3 and `viz_tree.py`.

### Anchor selection

At each sequence, the `n_subtrees` response positions with the **lowest** AR probability for the next token are selected as anchors — these are the positions where the model is least certain, so alternative branches are most valuable.

### Running stage 2

```bash
python v2/stage2.py \
  --model Qwen/Qwen3-8B \
  --data-dir data/stage1/ \
  --output data/stage2_v2.h5 \
  --n-subtrees 64 \
  --batch-size 1 \
  --attn-implementation sdpa
```

Multi-GPU data parallel:
```bash
torchrun --standalone --nproc_per_node=4 v2/stage2.py \
  --model Qwen/Qwen3-8B \
  --data-dir data/stage1/ \
  --output data/stage2_v2.h5 \
  --n-subtrees 64 \
  --batch-size 1 \
  --attn-implementation sdpa
```

Visualise a sub-tree:
```bash
python v2/viz_tree.py data/stage2_v2.h5 --model Qwen/Qwen3-8B --seq 3 --pos 12
```

---

## Stage 3 — Training (`v2/stage3.py`)

Trains the DFlash draft model (`DFlashDraftModel`) plus a new `TreePositionEmbedding` module using the stage-2 HDF5 data. Supports Lightning Fabric (DDP), torch.compile, bf16-mixed precision, checkpoint resume, and Wandb logging.

### What is trained

1. **`DFlashDraftModel`** — 5-layer Qwen3-based model (already pre-trained by DFlash). Fine-tuned here with the tree-structured loss. Shares embedding table and LM head with the frozen target model. KV injection: each draft attention layer concatenates `k_proj(target_hidden)` and `k_proj(noise_embedding)` as keys, similarly for values.

2. **`TreePositionEmbedding`** — new module, two optional components:
   - **Initial Bias** (`use_initial=True`): a learned `nn.Embedding(st_size, hidden_size)` indexed by vertex id. Added to the noise embedding before the draft forward so siblings get distinct input representations.
   - **Relational Attention Bias** (`use_relational=True`): a learned `nn.Embedding(NUM_RELATIONS, num_draft_layers * num_heads)` (7 relation types × draft layers × heads). Each draft layer gets its own slice of `num_heads` bias scalars per relation type. Applied as a per-head score bias in flex attention via `score_mod`. `get_score_mod(layer_idx)` returns a closure for a specific layer; `get_score_mods()` returns the full list. `TreeDFlashDraftModel` (in `v2/model.py`) routes each layer's closure via the `score_mods` parameter.

### Training data format

`TreeFlashCollator` processes raw HDF5 sequences into packed training batches. For each sequence:
1. Randomly sample `num_anchors` valid anchor positions from the response (a valid anchor has sub-tree data and fits within `max_ctx_len`).
2. For each anchor, flatten the sub-tree across `tree_seq_depth` primary-path positions into a block of `tree_seq_depth × st_size` tokens.
3. Build:
   - `block_noise_ids`: vertex 0 (primary path token) set to the actual token; all other vertices set to `mask_token_id`
   - `block_token_ids`: actual tokens for all vertices (target labels)
   - `block_position_ids`: `anchor_ctx_pos + d + depth_of[v]` for each flat index `(d, v)` — siblings share the same absolute position
   - `block_cumprob_weights`: cumulative product of AR probs along the path from root to each node (used for loss weighting)

### Attention masks during training

**Context side** (causal): query token at position `anchor_ctx_pos` attends to all prefix tokens `< anchor_ctx_pos` in the same document.

**Block side** (bidirectional): all tokens within the same anchor's tree block attend to each other fully — equivalent to block-diagonal bidirectional attention per anchor. Implemented via flex attention `mask_mod`.

The relational attention bias is applied via flex attention's `score_mod`, passed as `score_mod=` kwarg to the draft forward.

### Loss

**CumProds-weighted cross-entropy** (same strategy as DFlash, extended to trees):

```
loss = mean over valid nodes of [cumprob(node) × CE(logits[node], label[node])]
```

`cumprob(node)` = product of AR probabilities along the path from root to node. This gives high weight to high-probability branches and low weight to unlikely branches, matching the distribution of nodes that will actually be visited at inference time.

Cross-entropy is computed in chunks of 64 tokens to avoid materializing the full `[B, block_len, V]` logits tensor. The `lm_head` is applied per-chunk, and loss is accumulated across chunks. This reduces peak GPU memory by ~200-300MB for large vocabularies (e.g. Qwen3's 152K vocab).

### Training loop

- **LR schedule**: linear warmup then cosine decay
- **Gradient accumulation**: `--grad-accum-steps` micro-steps before an optimizer step; both `draft_model` and `tree_pos_emb` are wrapped in `fabric.no_backward_sync` so DDP all-reduces happen only on the final micro-step — `grad_accum_steps - 1` redundant all-reduces per optimizer step are suppressed for both modules
- **Gradient clipping**: max norm 1.0, applied to `draft_model` and `tree_pos_emb` independently before each optimizer step
- **Wandb log interval**: `--log-every N` (default 10) controls how often train metrics are sent to Wandb; the per-step console print is unaffected
- **Eval** every `--eval-every` steps: computes loss + 5 metrics (see below)
- **Checkpointing** every `--save-every` steps: saves `fabric_ckpt.pt` (Fabric state dict for resume) and `hf_draft/` + `tree_pos_emb.pt` (HF format for inference)

### Eval metrics

| Metric | Description |
|---|---|
| `eval_loss` | CumProds-weighted CE on eval set |
| `mean_acceptance_length` | Simulated acceptance depth (how many consecutive primary-path nodes the draft gets right) |
| `sibling_equality_rate` | Fraction of sibling pairs that predict the same token — lower is better (more diverse) |
| `leftmost_path_pct` | Fraction of trees where the final accepted node is on the leftmost path |
| `equality_heatmap` | `[st_size, st_size]` matrix: `%` same predicted token for each vertex pair |
| `sibling_equality_heatmap` | Same but masked to sibling pairs only |

### Running stage 3

```bash
python v2/stage3.py \
  --target-model Qwen/Qwen3-8B \
  --draft-model z-lab/dflash-qwen3-8b \
  --data data/stage2_v2.h5 \
  --output-dir checkpoints/run1 \
  --tree-pos-emb both \
  --tree-seq-depth 4 \
  --num-anchors 8 \
  --batch-size 2 \
  --grad-accum-steps 8 \
  --lr 1e-4 \
  --max-steps 10000 \
  --devices 4 \
  --precision bf16-mixed \
  --compile \
  --wandb-project tree-flash \
  --wandb-run-name run1 \
  --log-every 10
```

Resume:
```bash
python v2/stage3.py ... --resume checkpoints/run1/checkpoint-5000
```

---

## Inference — Tree Speculative Decoding (`v2/spec_dec.py`)

Implements efficient tree speculative decoding at inference time. Each decode step:

1. **Draft** — one parallel forward pass of the draft model over the full tree (`tree_seq_depth × st_size` tokens). Uses bidirectional attention (same as training).
2. **Verify** — one target model forward over all tree nodes with an ancestor-based attention mask. Each node attends to its ancestors only (not siblings), mirroring how the tree would be built autoregressively.
3. **Accept** — greedy DFS walk: a node is accepted if its drafted token matches the target's prediction at its parent. The accepted path always spans consecutive absolute sequence positions.
4. **Cleanup** — KV cache surgery: gather only the prefix + accepted path entries from the target KV cache, discarding all rejected branches.
5. **Advance** — the bonus token (target's prediction at the last accepted node) becomes the new anchor.

### KV cache management

**Target cache**: after verification the cache holds `prefix_len + tree_size` entries. `cleanup_cache()` uses `torch.index_select` to gather only the `accept_len` accepted tree entries, reducing to `prefix_len + accept_len` entries.

**Draft cache** (context reuse): the draft model reuses previously computed K/V for the context via `DynamicCache`. Each step passes only the newly accepted tokens' features as `target_hidden_new`, appending them to the draft cache. After the draft forward, the tree block K/V is cropped from the draft cache (`DynamicCache.crop(seq_start)`), keeping only the accumulated context.

### Position IDs

Tree tokens at flat index `flat = d * st_size + v` get absolute position `seq_start + d + depth_of[v]`. Siblings (same `d`, different `v` with same depth) share the same position ID, giving them identical rotary embeddings — they are alternatives at the same sequence position.

### Precomputed constants (`build_tree_constants`)

- `pos_offsets[flat]` = `d + depth_of[v]` — used to compute absolute position IDs
- `vertex_ids[flat]` = `v` — used for tree position embedding lookup
- `parent_flat[flat]` — parent's flat index (−1 for root)
- `anc_mask[k, q]` — `[tree_size, tree_size]` bool: k is ancestor-or-self of q — used to build the verification attention mask

### Running inference

```bash
python v2/spec_dec.py \
  --target-model Qwen/Qwen3-8B \
  --draft-model checkpoints/run1/final/hf_draft \
  --tree-pos-emb-path checkpoints/run1/final/tree_pos_emb.pt \
  --tree-seq-depth 4 \
  --prompt "Solve: x^2 - 5x + 6 = 0" \
  --max-new-tokens 512 \
  --temperature 0.0

# Benchmark on GSM8K
python v2/spec_dec.py \
  --target-model Qwen/Qwen3-8B \
  --draft-model checkpoints/run1/final/hf_draft \
  --tree-seq-depth 4 \
  --benchmark gsm8k \
  --benchmark-n 100 \
  --max-new-tokens 512
```

---

## File Index

| File | Purpose |
|---|---|
| `v2/model.py` | `TreeDFlashDraftModel` subclass: routes per-layer `score_mod` closures from `TreePositionEmbedding` to individual draft layers |
| `v2/stage2.py` | Sub-tree generation: runs target model on stage-1 data, expands tree of alternatives at low-confidence positions, writes HDF5 |
| `v2/stage3.py` | Training: chunked CumProds-weighted CE loss, flex attention, Lightning Fabric DDP, Wandb, checkpoint resume |
| `v2/spec_dec.py` | Inference: tree speculative decoding with KV cache reuse and cleanup |
| `v2/viz_tree.py` | CLI visualiser: renders a single sub-tree from a stage-2 HDF5 file |
| `v2/summary.md` | This file |
| `method_v2.md` | High-level method description (stages 1–3) |
| `experiments.md` | Experiment plan and design decision catalogue |

---

## Design Decisions Summary

### Tree v1 (current)

Primary path of `seq_depth` positions; attach the same `sub_tree` shape at every node along that path. All attached copies share the same shape, so they can be expanded in a single parallel draft pass.

### Loss scaling: CumProds

Each node's loss weight = product of AR probabilities of all ancestors. Root has weight 1. This directly extends DFlash's exponential decay to trees: likely branches get higher loss weight, unlikely branches get lower weight, and computation can be lazily skipped for near-zero-weight branches.

### Tree position embeddings

Two components jointly address the "sibling disambiguation" problem — without positional information, sibling nodes would have identical inputs and would tend to produce the same token:
- **Initial Bias**: a per-vertex learned embedding added to the input noise token embedding
- **Relational Attention Bias**: per-head attention score bias based on the structural relationship between query and key vertices (parent/child/sibling/ancestor/etc.)

### Verification attention mask

At inference, each tree node attends only to its ancestors (not siblings). This matches the correct autoregressive semantics: node (d, v) should be conditioned on the path from the root to its parent, not on siblings that represent alternative paths.

At training, the draft model uses fully bidirectional attention within each anchor's tree block. This is intentional: the draft model sees all siblings simultaneously and can coordinate them to be diverse.
