# Experiments Plan
## Datasets
- **Math**: GSM8K, MATH-500, AIME24, AIME25
- **Code**: HumanEval, MBPP, LiveCodeBench, SWE-Bench
- **Chat**: MT-Bench, Alpaca

## Metrics
- Average acceptance length
- Unmasked equality heat map over all node pairs (% same token sampled)
- Sibling-masked equality heat map
- Scalar sibling equality rate
- Histogram of which node is the final accepted node
- % of cases where final accepted node lies on the left-most path

## Ablation Setting
- Target: Qwen 3 4B
- 100K Samples
- 512 Anchor Positions
- Tree Depth of 8
- Sequence Length of 3072
- 6 Epochs
- Batch Size 72
- LR: 6e-4 with 128 warmup steps with Cosine decay
- AdamW

## Experiments

### Experiment 1: Basic Proof Of Concept
**Goal:** Tree drafting improves acceptance length over sequence drafting.

| Component | Choice |
|---|---|
| Tree | Tree v1 |
| Pruning | None |
| SD Optimisation | None |
| Loss Scaling | CumProds |
| Sibling Deduplication | None |
| Tree Position Embeddings | Both |

### Experiment 2: Tree Position Embedding
**Goal:** Are both needed or is one enough

| Component | Choice |
|---|---|
| Tree | Tree v1 |
| Pruning | None |
| SD Optimisation | None |
| Loss Scaling | CumProds |
| Sibling Deduplication | None |
| Tree Position Embeddings | Initial Bias vs Relational Bias |

### Experiment 3: Pruning
**Goal:** Find the best pruning method.
 - Also try to find the best tree size for verification
| Component | Choice |
|---|---|
| Tree | Tree v1 |
| Pruning | Q_Logit vs AR_Head |
| SD Optimisation | Pruning_Probs (+ quick check with Uniform) |
| Loss Scaling | CumProds |
| Sibling Deduplication | None |
| Tree Position Embeddings | -- |

### Experiment 4: Tree Shape
**Goal:** Find a more suitable tree shape

| Component | Choice |
|---|---|
| Tree | Tree v1 vs a tree which moves mass to earlier tokens |
| Pruning | -- |
| SD Optimisation | Pruning_Probs |
| Loss Scaling | CumProds |
| Sibling Deduplication | None |
| Tree Position Embeddings | -- |

### Experiment 5: Sibling Deduplication
**Goal:** Find the best method for maximising diversity across siblings.

| Component | Choice |
|---|---|
| Tree | Tree v1 |
| Pruning | -- |
| SD Optimisation | Pruning_Probs |
| Loss Scaling | CumProds |
| Sibling Deduplication | Aux Loss (+ run baseline checkpoint with Deduplicated Sampling) |
| Tree Position Embeddings | -- |

---

## Design Decisions

### Tree v1
Parameterised by `seq_depth` and `sub_tree`.
- Draft a primary path of length `seq_depth` from the current anchor (the left-most path of the full tree).
- Attach the same `sub_tree` at every node along that primary path.
- `sub_tree` is a rooted tree represented as relative paths, e.g. `01, 02, 03, 14, 15, 26, 27` — node 0 has children 1,2,3; node 1 has children 4,5; node 2 has children 6,7.
- Efficient: all attached subtrees share the same shape and are expanded in one parallel pass.

### Pruning
- **None** — pass the full tree to the verifier.
- **Q_Logit** — `linear(drafter_hs) → [0,1]` estimates per-token correctness probability. Keep the subtree maximising expected accepted depth; prune to `n_candidate_tokens`.
- **AR_Head** — `ar_layer(backbone_hs, input_embds_of_parent)` predicts the upcoming distribution as a q-value. The drafter has a backbone → diffusion head → LM head pipeline; AR_Head reuses backbone hidden states plus the sampled token embeddings. Can also be used to expand tree width.
    - backbone_hs is the second to last layer from the diffusion module 

### Speculative Decoding Optimisation
Improves the probability distribution used in rejection sampling (inference-time, drafter-side).
- **None** — treat every token with p=1.
- **Uniform** — split drafter probability equally across siblings (e.g. two siblings each get p=0.5).
- **Pruning_Probs** — use the pruning head's probability estimates directly.

### Loss Scaling
> TODO: We might have to introduce ablation which measures this against the decay used in DFlash
> Alternatively introduce smoothing parameter?
- **CumProds** — root node has p=1; each node's loss is scaled by the cumulative product of verifier probabilities along its path from root. Directly extends DFlash's loss scaling to the tree setting and enables lazy evaluation of low-probability branches.

### Sibling Deduplication
Prevents siblings from drafting the same token, wasting tree capacity.
- **None**
- **Deduplicated Sampling** — sample siblings without replacement to guarantee no repeated tokens.
- **Aux Loss** — KL divergence penalty to push sibling logit distributions apart.

### Tree Position Embeddings
- **Initial Bias** - A embedding from vertex_idx which gets added to the noise_embds
- **Relational Attention Bias** - A head wise attention bias based on the relationship (parent, child-1, sibling, ancestor, ...)
- **Both** - Use Both

### Loss Masking
> TODO: Must we ablate this? (Currently Ignore)
- **Ignore** - We ignore non-generated tokens (low weighting)
- **Pad Token** - We align to predict mask tokens



