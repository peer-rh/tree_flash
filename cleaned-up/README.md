# Tree Flash Cleaned-Up

This directory is a small, readable rewrite of the current project that keeps
only the method described in [writeup.md](../writeup.md).

The goal is overview first:

- one offline tree format
- one training path
- one inference path
- one optional performance toggle for chunked cross entropy
- no anchor chunking
- no backward-compatibility code

The code in this directory is meant to be run from the repository root, for
example:

```bash
python cleaned-up/trainer.py --target Qwen/Qwen3-4B --drafter path/to/drafter
python cleaned-up/infer.py --target Qwen/Qwen3-4B --drafter path/to/drafter --prompt "Hello"
```

## Files

- `data.py`
  - Stage 1 JSONL helpers
  - Stage 2 v2 HDF5 helpers
  - sequence-tree dataset reader
  - packed batch construction
  - dynamic training-subtree sampling
- `model.py`
  - rank-aware relation ids
  - dynamic `TreeInfo` builder for padded trees
  - narrow wrapper around the existing drafter model
  - fixed inference-tree builder
- `trainer.py`
  - simplified Fabric trainer
  - target prefill
  - drafter forward
  - weighted CE
  - prune-head loss
  - W&B logging
- `infer.py`
  - one speculative decoding entrypoint
  - fixed tree + q-head pruning + target verification
- `spec_decode.py`
  - local cleaned-up speculative decoding core
  - target-cache + drafter-cache handling
  - q-head-only pruning and tree verification
- `utils.py`
  - tiny shared helpers (`unwrap_model`, LR schedule, sampling)

## Data Flow

### 1. Stage 1

Stage 1 stores prompt/response pairs as JSONL:

```json
{"prompt": "...", "response": "..."}
```

`data.py` includes small helpers to read and write this format.

### 2. Stage 2

The only offline tree format kept here is the Stage 2 v2 sequence-tree format.
Each stored sequence contains:

- `main_path_ids`: token ids for the prompt + response main path
- `response_start_position`: first token index that belongs to the response
- `anchors`: one tree per main-path anchor position

Each stored node contains:

- `token_id`
- `parent_index`
- `depth`
- `local_prob`
- `path_prob`
- `rank`
- `main_path_position`
- `is_main_path`

`data.py` wraps the existing Stage 2 v2 HDF5 writer helpers instead of
re-implementing the low-level file layout.

### 3. Training

The training loader reads one Stage 2 v2 file and builds a `PackedBatch`.

Important shapes:

- `input_ids`: `(batch_size, pack_length)`
- `position_ids`: `(batch_size, pack_length)`
- `document_mask`: `(batch_size, pack_length)`
- `context_valid_mask`: `(batch_size, pack_length)`
- `anchor_positions`: `(batch_size, num_anchors)`
- `tree_labels`: `(batch_size, num_anchors, tree_size)`
- `tree_noise_ids`: `(batch_size, num_anchors, tree_size)`
- `tree_position_ids`: `(batch_size, num_anchors, tree_size)`
- `tree_cum_probs`: `(batch_size, num_anchors, tree_size)`
- `tree_valid_mask`: `(batch_size, num_anchors, tree_size)`
- `tree_parent_indices`: `(batch_size, num_anchors, tree_size)`
- `tree_depths`: `(batch_size, num_anchors, tree_size)`
- `tree_node_ranks`: `(batch_size, num_anchors, tree_size)`
- `tree_primary_path_mask`: `(batch_size, num_anchors, tree_size)`

The trainer does exactly four hot-path steps:

1. prefill the frozen target model on the packed context
2. build dynamic tree relations for the sampled training subtrees
3. run the drafter once over all anchors in the batch
4. compute weighted CE + prune-head BCE

There is no anchor chunking in this rewrite.

The only optional loss toggle is:

- `use_chunked_cross_entropy=False`: compute CE in one pass
- `use_chunked_cross_entropy=True`: compute CE in chunks of `ce_chunk_size`

### 4. Inference

Inference keeps one path:

1. build one fixed draft tree
2. draft all tree nodes in parallel
3. prune to `candidate_tree_size` using the q-head
4. verify with the target model
5. commit the deepest accepted path

`infer.py` intentionally reuses the existing low-level speculative-decoding
helpers from the local `spec_decode.py` module so the cleaned-up code stays
self-contained and easy to scan.

This cleaned-up speculative decoder is intentionally narrower than `src/`:

- it keeps target-model and drafter caching
- it uses q-head pruning only
- it does not carry AR-head branches or legacy inference support

## Compile Notes

This rewrite is intentionally compile-friendly:

- the training hot path is structurally fixed
- feature branches from the original trainer are removed
- data and tree semantics are explicit in the batch tensors
- all major functions document expected shapes

The q-head is required here. AR-head support is intentionally excluded.
