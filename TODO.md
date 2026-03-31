# Pruning

# Tree Embedding
- We want a fuller tree generation and also the ability to not always generate the full tree. 
- We want to somehow encode the tree structure in the transformer. Each token should know it's position and be able to use the information in the attention mechanism
- Therefore we some sort of tree embeddings which are able to encode all kind of tree positions (i.e ancestors path)
- Since trees can have up a depth of 16 and up to 4 paths at each node we might have > 4^15 many possible path combinations, therefore we need some generalized method
## Existing Methods

### 1. Tree-shaped Causal Masks (SpecInfer, Medusa, Eagle)
- Used by most speculative decoding systems: SpecInfer [1], Medusa [2], Eagle [3], Sequoia [4]
- No special tree positional encoding — reuse standard sequential position IDs
- Each token gets the position index it would have if only its root-to-leaf path existed (i.e. all tokens at depth d get position `anchor + d`)
- Tree structure is encoded purely through the attention mask: each token attends only to its ancestors
- **Limitation**: Tokens on different branches at the same depth share position IDs and are only distinguished by the mask, so the model has no explicit representation of branch identity

### 2. Factored / Compositional Tree Position Encoding (Shiv & Quirk, 2019)
- "Novel Positional Encodings to Enable Tree-Based Transformers" [5], NeurIPS 2019
- Represent a node's tree position as the path from root: `(c_1, c_2, ..., c_d)` where `c_i ∈ {0,...,k-1}` is the child index at depth `i`
- Position encoding is the sum of per-level embeddings: `PE(path) = Σ_i E_i(c_i)`
- Each `E_i` is a separate embedding table (or sinusoidal function) for depth level `i`
- Parameters scale as `O(D × k × d_model)` instead of `O(k^D × d_model)` — fully general for arbitrary tree positions
- Closely related to what we already do with `tree_pos_embd`, but our current version assigns a single learned embedding per tree-slot rather than factoring by (depth, child_index)

### 3. Relative Tree Position Bias
- Analogous to T5-style relative position bias [6] or ALiBi [7], but using tree-structural relationships instead of sequential distance
- Encode pairwise relationships between nodes as attention bias terms. Possible relations:
  - **Tree distance**: `d(i,j) = depth(i) + depth(j) - 2 * depth(LCA(i,j))`
  - **Categorical relations**: self, parent, child, sibling, ancestor, descendant, unrelated
  - **Shared prefix length**: length of common ancestor path
- Added as bias to attention logits: `attn(i,j) = q_i · k_j / √d + b(rel(i,j))`
- This is what we already implement with `use_additive_tree_pos_bias` and the `relation_map` in `TreeInfo`

### 4. Multi-dimensional / Tree-RoPE
- Extend RoPE [8] to tree structures by treating each depth level as a separate rotary dimension
- Standard RoPE encodes scalar position `m` as rotation: `R(m, θ)`
- Tree-RoPE encodes path `(c_1, ..., c_d)` by composing rotations across disjoint subspaces of the head dimension: `R_tree = ⊗_i R(c_i, θ_i)`
- Each depth level `i` gets its own frequency band `θ_i` and rotates a dedicated slice of the embedding
- Preserves RoPE's key property: the dot product `R_tree(path_a)^T R_tree(path_b)` depends only on the relative path difference, so relative tree position is naturally captured
- Requires partitioning head dimensions across depth levels — with depth 16 and typical head_dim=128, that's 8 dims per level, which may be tight
- Related: multi-dimensional RoPE is used for 2D/3D positions in vision and video transformers (same principle, different structure)

### 5. Graph Transformer Positional Encodings (applied to trees)
- Trees are a special case of graphs, so graph PE methods apply:
  - **Laplacian Eigenvector PE** [9]: Use eigenvectors of the graph Laplacian as node features. For trees, these capture hierarchical structure. Computed once, used as input features.
  - **Random Walk Structural Encoding (RWSE)** [10]: Encode each node by the diagonal of random walk matrices `[RW, RW^2, ..., RW^k]`. Captures local neighborhood structure.
- These are general-purpose but expensive to compute and don't exploit the specific regularity of k-ary trees
- More relevant for arbitrary graph structures than for fixed-shape speculation trees

### 6. Recursive / Hierarchical Position Encoding
- Define position encoding recursively from parent: `e(node) = f(e(parent), c_i)`
- `f` can be a linear transform, rotation, or small MLP
- Natural for trees and handles arbitrary depth
- Downside: sequential dependency along root-to-node path prevents parallelism during encoding
- Related to how recursive neural networks process tree structures

### 7. Path Hashing / Fixed Encoding Schemes
- Encode the root-to-node path `(c_1, ..., c_d)` as a base-k number and use standard sinusoidal encoding on it
- Or use learned hash functions that map paths to fixed-size vectors while preserving locality (siblings → similar encodings)
- Simple but doesn't naturally preserve tree structure in the embedding space (e.g., depth-1 siblings may have very different encodings)

### Summary / Relevance
- Methods 1 (mask-only) and 2 (factored PE) are the most common in practice
- Method 3 (relative bias) is what we already have; the question is whether the relation categories are rich enough
- Method 4 (Tree-RoPE) is the most promising unexplored direction for us, since we already use RoPE for sequence positions — extending it to tree structure would unify both

### References
- [1] Miao et al., "SpecInfer: Accelerating Generative LLM Serving with Tree-based Speculative Inference and Verification", ASPLOS 2024. https://arxiv.org/abs/2305.09781
- [2] Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads", ICML 2024. https://arxiv.org/abs/2401.10774
- [3] Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty", ICML 2024. https://arxiv.org/abs/2401.15077
- [4] Chen et al., "Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding", NeurIPS 2024. https://arxiv.org/abs/2402.12374
- [5] Shiv & Quirk, "Novel Positional Encodings to Enable Tree-Based Transformers", NeurIPS 2019. https://proceedings.neurips.cc/paper/2019/hash/6e0917469214d8fbd8c517dcdc6b8dcf-Abstract.html
- [6] Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", JMLR 2020. https://arxiv.org/abs/1910.10683
- [7] Press et al., "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation", ICLR 2022. https://arxiv.org/abs/2108.12409
- [8] Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", Neurocomputing 2024. https://arxiv.org/abs/2104.09864
- [9] Dwivedi & Bresson, "A Generalization of Transformer Networks to Graphs", DLG-AAAI 2021. https://arxiv.org/abs/2012.09699
- [10] Dwivedi et al., "Graph Neural Networks with Learnable Structural and Positional Representations", ICLR 2022. https://arxiv.org/abs/2110.07875

## Some things to think about
- A smart way to construct attention head biases. Either learnable or fixed
- Can we perhaps use rotations, similar to rope extended to tree structure



# Data Generation 
- Essentially the idea is that we 

- We generate the most continuations at the most valuable locations. 
    - We pick alpha * S many positions where alternative to the actual token is very promising
    - We then generate the continuation trees there again at the "most" valuable positions
- We then for training get the tree at this anchor with some size 

# Method v2
## Stage 2
- We generate alternative continuation trees with the verifier
- We pick all $x_t$, where $p(x_t+1 | x_1:t) <= alpha$ and set these as our tree anchors
    - i.e the tokens where a alternative continuation tree make the most sense
    - We limit this to $k$ many (else pick the k lowest)
    - Note that this should only be done to the response tokens
- From these tokens we sample alternative token continuation trees which are rooted in the anchor
    - Essentially we want to have it so that all trees are likely to be generated by the LLM 
    - We make sure that the tokens right after are not $x_t+1$
- The way these trees are shaped is based on the likelyhood of each token
    - We set a `num_attend_tokens_per_anchor` which controls how many tokens we have in the tree which are non-leaf nodes
    - We always attend to the token which hasn't been attended to yet and is most likely to be reached by a random walk given the model's probability
        - I.e if a token is at depth 2 of the tree and it has $p = 0.9$ and it's parent has $p=0.5$ then it has a chance of being $0.45$ of being part of the random walk
        - In other words the cumprod of it's ancestors where the anchor/root has $p=1$
- For all nodes (including the response main path) we also make sure that all childrens are stored so that $>= alpha$ of the continuation space is saved
- This produces a set of trees plus the original input ids
    - Note that this one large "sequence tree", where the input_ids will be referred to as "main path"
- TODO: How can we efficiently save this dataset
- We store for each token the following
    - The actual token
    - The probability the model has assigned to this token
    - The ranking it has gotten by the model (i.e top-1, top-2, ...)

## Training
- We sample random anchor positions in from the main path response tokens
- We then pick a sub tree from the sequence tree rooted at the anchor position of size `training_tree_size` which is chosen proportional to the probability the model has assigned to the tokens
- We use the top_k info from this tree and the shape to construct the tree positional embedding
- Do training as before 