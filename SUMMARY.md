# Tree Flash - Codebase Summary

Tree Flash trains a drafter model to propose token trees for speculative decoding. A frozen target LLM verifies candidates via rejection sampling, accepting the longest valid path to accelerate inference.

## Directory Structure

```
tree_flash/
├── src/                              # Core training & inference
│   ├── data.py                       # HDF5 data loading, packing, collation
│   ├── trainer.py                    # Lightning training loop
│   ├── spec_decode.py                # Speculative decoding inference
│   ├── dflash_eval.py                # DFlash benchmark evaluation
│   ├── eagle3_eval.py                # EAGLE-3 evaluation compatibility
│   ├── models/
│   │   └── dflash.py                 # DFlash drafter model (Qwen3-based)
│   └── trees/
│       ├── blocked.py                # BlockTreeProcessor - linear chains
│       ├── branch_off.py             # BranchOffTreeProcessor - branching trees
│       ├── prunable.py               # PrunableTreeProcessor - trees with pruning
│       ├── var_tree.py               # VarTreeProcessor - dynamic variable trees
│       └── relation_ids.py           # Tree relation encoding (parent, child, sibling, etc.)
├── data_pipeline/                    # Data generation
│   ├── stage1.py                     # Generate responses with target model
│   ├── stage2.py                     # Generate alternative subtrees
│   └── stage2_v2.py                  # Rich tree generation (v2)
├── tests/                            # Test suite
│   ├── test_data_and_trees.py
│   ├── test_dflash_and_trainer.py
│   ├── test_stage2_v2.py
│   └── test_visualize_stage2_v2_tree.py
├── experiments/                      # Experiment configs and runs
├── pyproject.toml                    # Dependencies and metadata
├── method.md                         # Detailed method documentation
├── writeup.md                        # Project writeup
└── TODO.md                           # Architecture and design notes
```

## Architecture

### Training Pipeline

```
Frozen Target Model  →  Stage 2 HDF5 Data (sequence trees)
                              ↓
                     DataLoader (PackedBatch)
                              ↓
                        Trainer Loop:
                     1. Extract target context features
                     2. Build tree labels from frozen target
                     3. Build attention masks (document & tree-aware)
                     4. Drafter forward pass
                     5. Compute losses (CE + optional q-loss + AR-loss)
                     6. Backprop through drafter only
```

### Inference Pipeline

```
Prompt → Target prefill → Loop:
  1. Drafter generates tree (all nodes in parallel)
  2. Optional: prune with q-head scores
  3. Target verifies candidates (tree-causal attention)
  4. Rejection sampling: accept where p_target ≥ p_draft
  5. Advance to deepest accepted node
  → Output sequence
```

## Key Components

### Data (`src/data.py`)
- **PackedBatch**: Training batch with packed sequences, document masks, anchor positions, tree labels, and path probabilities.
- **Stage2Dataset / Stage2V2Dataset**: HDF5-backed datasets.
- **PackedBatchCollator**: Packs multiple documents into fixed-length sequences with document isolation.

### Tree Processors (`src/trees/`)
All processors produce a `TreeInfo` dataclass containing: tree attention mask, parent indices, depths, relation map, position IDs, and primary path info.

| Processor | Description |
|-----------|-------------|
| `BlockTreeProcessor` | Simple linear chains (baseline) |
| `BranchOffTreeProcessor` | Multi-branch trees via branching patterns |
| `PrunableTreeProcessor` | Wraps any processor with top-k pruning |
| `VarTreeProcessor` | Dynamic trees from Stage 2 v2 data |

### Drafter Model (`src/models/dflash.py`)
Qwen3-based transformer with:
- **Context projection**: Compresses target hidden states into drafter space
- **Tree-aware attention**: Supports relation-based attention biasing
- **Optional heads**: q-head (confidence scoring for pruning), AR-head (autoregressive refinement)

### Training (`src/trainer.py`)
Lightning Fabric-based trainer. Losses:
1. **Cross-entropy**: Drafter predictions vs. teacher labels, weighted by path probability
2. **Q-loss** (optional): Binary CE on confidence scores (1 if argmax matches teacher)
3. **AR-loss** (optional): CE from autoregressive refinement layer

### Speculative Decoding (`src/spec_decode.py`)
- `speculative_generate()`: Main inference loop with tree drafting, verification, and acceptance sampling
- `prune_drafted_tree()`: Prune tree using q-head or AR-head scores
- `verify_tree()`: Target model verification with tree-causal masking

### Relation Encoding (`src/trees/relation_ids.py`)
Vocabulary of ~55 relation types (self, parent-k, child-k, sibling, ancestor, descendant, other) used as attention biases.

## Data Pipeline

1. **Stage 1** (`data_pipeline/stage1.py`): Generate responses to prompts using target model → HDF5
2. **Stage 2** (`data_pipeline/stage2.py`): Build alternative continuation subtrees at response positions → fixed-shape HDF5
3. **Stage 2 v2** (`data_pipeline/stage2_v2.py`): Richer tree generation with flex-attention, variable-shape trees → HDF5

## Dependencies

- `torch>=2.9.1`, `transformers>=4.55.0`, `lightning>=2.4.0`
- `h5py`, `datasets`, `cut-cross-entropy`, `jsonargparse`, `wandb`

## Entry Points

- **Training**: `src/trainer.py` (jsonargparse CLI)
- **Evaluation**: `src/dflash_eval.py`, `src/eagle3_eval.py`
- **Data generation**: `data_pipeline/stage1.py`, `data_pipeline/stage2.py`, `data_pipeline/stage2_v2.py`
