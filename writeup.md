# Tree Flash
## Data
- Data is generated in 2 stages
- Stage 1 generates answers to a set of prompts using the Target Model
    - Here we can use vLLM to get good throughput
    - We store them as 'prompt' and 'response'
- Stage 2 then generates alternative continuation trees
    - We select all response positions $x_t$, where $p_T(x_{t+1} | x_{1:t}) <= \alpha$. This ensures the alternative continuations trees capture the model's behaviour (In practice we limit the total number of trees)
        - We still make it so that every token has children so that $alpha$ prob. space is covered. Only some might not get chosen as branch-off-positions if we have to many of these positions
    - For all 'branch-off positions' $x_t$ we do the following:
        - Let $T = \{x_t\}$
        - Add all tokes $x^{(i)}_{t+1}$ to $T$, where $(i)$ is the rank in the prob. ordering, s.t $p(x_{t+1}) sum_i p(x^{(i)}_{t+1}) \geq alpha$ and $x^{(i)}_{t+1} \neq x_{t+1}$
        - Repeat for `n_attend_tokens` steps
            - Select the token in $T$ which has not been processed yet and has the highest 'path-probability' (i.e the AR probs multiplied starting at $x_t$)
            - Add this token to the input sequence and run the model on it to get prob. for children
            - Add all tokes $x^{(i)}_{t+i}$ to $T$, where $(i)$ is the rank in the prob. ordering, s.t $p(x_{t+i}) sum_i p(x^{(i)}_{t+i}) \geq alpha$ 
        - This then gives us a subtree rooted at $x_t$
    - The subtrees + the main sequence form one large 'sequence tree'
    - Implementation Details
        - We can use Flex-Attention and a sparse block mask to only have to do n_attend_tokens + 1 many model calls
    - We store the following information for each node
        - The rank
        - The probability that this token would be sampled given it's ancestors
        - The actual token
        - It's ancestor path and whether it lies on the main path
## Training
- Given the full 'sequence tree' from stage 2 we do the following
    - We sample anchor positions $x_t$ from the main sequence and so that it is part of the response
    - From $x_t$ we 'grow' a subtree of the sequence tree where tokens are sampled proportional to their path-prob to the $x_t$. This subtree then always has size `training_tree_size`
- We then do the following:
    1. Run the target model on the main path to get the target context features
    2. We run the drafter model with a sparse attention mask with all sampled subtrees in sequence so that:
        - It can attend to all target context features which are strictly before it's root node
        - Fully bidirectional inside of the tree
        - No attention to other trees
        - The input is a the input encoding of form `target.embd_tokens([x_t, MASK_ID, MASK_ID, ..., MASK_ID])` (i.e. only root is known)
        - Additionally the model has a additive head wise attention bias which receives as input a 'relation_map' between all tree nodes. The possible relations are
            - `ancestor` & `descendant`
            - `child-{k}` & `parent-k`, where the $k$ refers to the rank of the child
            - `sibling-{i}-{j}` where $i,j$ are the respective ranks
            - `unrelated`
        - We also have the prune-head which is a linear layer which predicts the probability that this token is correct
    3. We do cross entropy from this tree to the actual tree labels 
        - We weight this loss based on the path probability of each token, i.e the path probabilities multiplied from $x_t$ to this token
        - The prune head gets trained with BCE on whether it and it's ancestors are equal to the main path starting from $x_t$ (so kind-off an acceptance proxy)
        - We use focal loss so that deep tokens also get a decent representation
    

## Inference
- We draft a tree of size `tree-inference-size` and shape/relation-map `inference-tree` using the drafter
- We prune this tree using the prune head predictions to a size of `candidate-tree-size`
- We then verify the tree with the target model and keep the longest accepted sequence