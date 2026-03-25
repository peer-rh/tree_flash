# Stage 1 - Sequence generation
- Inputs:
    - model: string
    - num_samples: int
    - temperature $T$: float
    - max_seq_len: int
    - max_new_tokens: int
- We use Nemotron-v2 and CodeAlpaca datasets to gather `num_samples` prompts
- We then generate continuations for each of the prompts using $T=1$ and save them
- Outputs jsonl files with `prompt`, `response`
    - Note that these must include chat template so `cat(prompt, response)` must be a fully valid sequence for the LLM

# Stage 2 - Sub Tree generation
- Inputs:
    - model: string
    - subtree: list[tuple[int, int]]
        - Example [(0,1), (0,2), (1, 3), (1, 4)] means subtree_size=5, 
        - 0 is always root node
    - n_subtrees: int
- For each prompt+response pair from stage 1 we do the following:
    - We run the LLM on prompt+response and gather positions top n_subtree positions $t$ in response where $p[x_t+1]$ is the lowest
    - For each of these positions tree we generate the alternative continuation subtree, this means we set node $0$ to $x_t$ and then set the children of node $0$ to the most likely next tokens exluding $x_t+1$ we then call the model to generate the rest of the tree
- Note that for sub_trees every root node should be set to the corresponding node
    - Also the AR prob should be set according to the sequence
- Outputs:
    - prompts: N x [S_P]
    - responses: N x [S_R]
    - sub_trees: N x [S_R, sub_tree_size]
    - sub_trees_ar_probs: N x [S_R, sub_tree_size]

# Stage 3 - Training
- Inputs:
    - Target Model
    - Drafter Model (To be trained)
    - num_anchors: int
    - tree_seq_depth: int
1. Randomly sample valid anchors 
2. Gather the continuation trees with each anchor being assign sub_trees[anchor:anchor+tree_seq_depth]. This forms a valid tree
    - The tree has size sub_tree_size * num_anchors
3. Compute the cumulative probability of each token, where we start out with the root node of the full tree (x_anchor) with $p=1$
4. Run the verifier to get the target_ctx_features
5. Run the drafter so that it attends causal to target_ctx_featues and fuly inside each tree (bidirectional attention)
    - More specifically attend to target_ctx_features if $t < anchor$, since $x_anchor$ is part of the block
    - The input noise_embds are the input embedding of the tree root being set to $x_anchor$ and all other tokens are the mask token id
    - We also have to add a learned tree_position_encoding so that sibling don't generate the same tokens
6. Compute the loss and backward