"""
- We select all response positions $x_t$, where $p_T(x_{t+1} | x_{1:t}) <= \alpha$. This ensures the alternative continuations trees capture the model's behaviour (In practice we limit the total number of trees)
        - We still make it so that every token has children so that $alpha$ prob. space is covered. Only some might not get chosen as branch-off-positions if we have to many of these positions
    - For all 'branch-off positions' $x_t$ we do the following:
        - Let $T = {x_t}$
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
"""
def get_dataloader(datapath: str, tokenizer, seq_len: int, batch_size: int):
    """
        Creates a dataloader of the dataset which is tokenized, padded to seq_len and batched
        Returns:
            input_ids: Tensor of shape (batch_size, seq_len)
            attention_mask: Tensor of shape (batch_size, seq_len)
            response_interval: Tensor of shape (batch_size, 2) containing the start and end indices of the response in the input_ids
    """
    ...

def process_batch(
    batch, model, alpha, num_attend_tokens, depth, max_top_k
):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    response_interval = batch["response_interval"]
    # Main Graph has indice 1..S
    main_graph = input_ids
    main_graph_ranks = ... # The ranking in the model prob.
    main_graph_probs = ... # The probabilities according to the model 

    # Extra Graph has indice S+1..S+E
    extra_graph_ranks = ... # The ranking in the model prob.
    extra_graph_probs = ... # The probabilities according to the model
    extra_graph_path_probs = ... # The product of the probabilities along the path from the root in the main_graph to the node
    extra_graph = ...
    extra_graph_position_ids = ...
    extra_graph_parent_indices =  ... # [B, E]
    
    # Use static Cache
    kv_cache, logits = model(input_ids=input_ids, attention_mask=attention_mask)
    # For all positions in the response interval add the children so that (alpha of the prob. space is covered, or max_top_k is reached)
    # Remember that the next position in the respones interval is one of the children
    # Add the children to the extra_graph

    # Select the top num_attend_tokens from extra_graph which have the highest probabilities according to the model and attend to those in the next step.
    branch_off_queue = # [B, num_attend_tokens, 1]

    for i in range(depth):
        # Select the next node for each branch off path from the branch off queue which has the highest path prob
        current_input_ids = ...
        # Remove this node from the branch off queue

        # The mask should be so that each of the current_input_ids attend to:
        # The context tokens in input_ids
        # To it's ancestors in extra
        # TO itself
        # Note that this means that we have a the context attention and then a set of diagonal matrices
        mask = ...
        kv_cache, logits = model(input_ids=current_input_ids, attention_mask=mask, kv_cache=kv_cache)
        # Add the children of this to the extra_graph such that alpha of the prob. space is covered, or max_top_k is reached. 
        # Also add these vertices to the branch_off_queue


def main():
    ...
