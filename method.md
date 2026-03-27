## 1. System Roles

```python
class FrozenTargetModel:
    """
    Large language model used as teacher and verifier.

    Responsibilities:
    - read the true packed sequence,
    - provide hidden-state features from selected layers,
    - provide next-token probabilities used as supervision,
    - verify drafted tree candidates during speculative decoding.

    Important:
    - this model is not optimized by the trainer,
    - its LM head is also reused as the vocabulary projection for the drafter.
    """


class DrafterModel:
    """
    Smaller model that learns to propose future tokens in tree form.

    Inputs:
    - a root token plus masked future tree slots,
    - compressed context features extracted from the frozen target.

    Outputs:
    - one hidden state per tree node,
    - optional confidence estimates (q-head),
    - optional autoregressive refinement scores (AR head).

    Important:
    - this is the model that is trained.
    """


class TreeProcessor:
    """
    Defines the tree shape and the meaning of each tree node.

    It answers questions such as:
    - which node is the parent of which other node,
    - which nodes lie on the same root-to-node path,
    - which absolute sequence position each node corresponds to,
    - which token should be used as the teacher label for each node.
    """
```

---

## 2. What Is Actually Trained

```python
def training_contract():
    """
    Optimized:
    - drafter transformer layers,
    - context projection into drafter space,
    - optional tree-position embeddings,
    - optional relation-based attention bias,
    - optional q-head,
    - optional AR head.

    Frozen:
    - target model,
    - target hidden-state extractor,
    - target LM head used for vocabulary logits.

    Supervision source:
    - tree labels and probabilities come from the frozen target,
      or from precomputed offline tree labels that were generated from it earlier.
    """
```

---

## 3. Logical Batch Format

```python
class PackedBatch:
    """
    A batch row is not necessarily a single original sample.

    Multiple prompt/response pairs can be packed into one fixed-length row.
    This increases utilization, but it creates a new problem:
    tokens from different packed documents must never attend to each other.
    """

    input_ids
    """
    Packed token ids.

    Meaning:
    - contains one or more prompt/response pairs,
    - padding fills any unused space.
    """

    position_ids
    """
    Per-document positions, restarted from zero inside each packed document.

    Meaning:
    - position 0 is the first token of a packed document,
    - position numbers do not continue across document boundaries.
    """

    document_mask
    """
    Document identity for each packed token.

    Meaning:
    - same positive value => same packed document,
    - different value => different packed document,
    - zero is effectively padding / unused area.
    """

    anchors
    """
    Chosen response positions that act as tree roots.

    Meaning:
    - every anchor marks a place where the trainer asks:
      'if we are currently at this token, what future tree should be drafted next?'
    - anchors are sampled only from response regions,
      never from prompts,
      and only where enough future tokens remain.
    """

    sub_trees
    """
    Each response token has a sub tree associated to it.

    Meaning:
    - The sub tree shape has been defined before and it is a alternative continuation tree rooted at this node
    - Each response token has one associated
    - Note that this tree also includes the token from the response as it's root node
    """

    sub_tree_probabilities
    """
    The probability that a token get's sampled by the verifier

    Meaning:
    - Each node in the sub tree has the prob. that a node get's sampled from it
    - The root node of each subtree also get's the probability assigned that it get's sampled by the LLM given the previous tokens in the LLM  
    """
```

```python
def pack_dataset(prompt, response):
    """
    Conceptual packing logic.

    1. Tokenize prompt and response separately.
    2. Concatenate prompt + response to form one logical document.
    3. Pack several documents into one fixed-length training row.
    4. Record which packed positions belong to which document.
    5. Record the response interval of each document.
    6. Later, sample anchor positions only from valid response spans.
    7. Also pack the sub trees and their associated probabilities

    Why this matters:
    - training gets better hardware utilization,
    - document boundaries are still preserved through document_mask.
    """
```

---

## 4. Shared Tree Meaning

```python
class TreeInfo:
    """
    Common logical structure shared by all tree processors.
    """

    tree_mask
    """
    tree_mask[i, j] is True exactly when node j is on the path
    from the root to node i, including node i itself.

    Meaning:
    - row i = the accepted prefix needed to reach node i,
    - this is the ancestor-or-self relation.
    """

    parent_idx
    """
    Direct parent of each node.

    Meaning:
    - root has parent -1,
    - every other node points one step upward in the tree.
    """

    depth
    """
    Distance from the root.

    Meaning:
    - root depth = 0,
    - children of the root have depth 1,
    - deeper nodes represent later sequence positions.
    """

    relation_map
    """
    Encodes relations such as:
    - self,
    - parent-[1-k],
    - child-[1-k],
    - ancestor,
    - descendant,
    - sibling-[1-k]-[1-k].

    Meaning:
    - used to optionally bias attention with structural information,
      without changing the labels themselves.
    - [1-k] here means the top-k output
        - i.e if node j is the top-4 response from node i then j has relation ship 'child-4' to i
    """

    tree_position_ids
    """
    Stable identifier of each node inside the tree layout.

    Meaning:
    - can be used for tree-position embeddings,
    - distinct from absolute sequence position.
    """
```

```python
def root_semantics():
    """
    The root node is not a future prediction.

    It represents the current token at the anchor frontier:
    - in training: the token already present at the chosen anchor,
    - in inference: the last accepted token in the running output.

    Therefore:
    - the root probability is treated as 1,
    - the main prediction loss is applied only to non-root nodes.
    """
```

---

## 5. Tree Processor Variants

```python
def tree_processor_variants():
    """
    BlockTree:
    - a simple linear chain,
    - each node is just the next token after the previous one.

    BranchOffTree:
    - We receive a 2d list of vertex idxs
    - The first index i means that we include sub_tree of anchor_pos + i
    - The second index means which nodes get included from this sub_tree
    - This then constitutes on big tree where all the roots of each sub_tree are connected sequentially
    """
```

---

## 6. How Teacher Labels Are Built

```python
def build_training_extras(batch, tree_processor, frozen_target):
    """
    Produce everything the drafter needs for one training step.

    Outputs:
    - tree_labels:
      the teacher token attached to each tree node.
    - tree_ar_prob:
      the target model's probability for that chosen token at that node.
    - tree_cum_prob:
      probability of the whole path from root to that node.
    - noise_embds:
      input embeddings given to the drafter tree.
    - sequence_position_ids:
      absolute sequence positions represented by each node.
    - target_hidden_states:
      selected hidden-state features from the frozen target context.
    - tree_info:
      the structural meaning of the tree.
    """
```

```python
def path_probability_meaning(node):
    """
    tree_ar_prob[node]:
        'If the verifier is standing at the parent of this node,
         how much probability does it assign to this node's token?'

    tree_cum_prob[node]:
        'How likely is the verifier to walk from the root
         all the way to this node by following this path?'

    Why cumulative probability matters:
    - it says how much real verifier mass reaches a branch,
    - low-mass branches can be down-weighted in the loss.
    """
```

---

## 7. What the Drafter Actually Sees

```python
def build_drafter_inputs(anchor, tree_info, frozen_target_features):
    """
    The drafter does not receive the true future tokens as input.

    It receives:
    - the root token embedding at the anchor,
    - mask-token embeddings for all future tree nodes,
    - projected target context features for the already-known prefix.

    This makes the drafter solve:
    'Given the verified past and the current token, fill in the tree.'
    """
```

```python
class DrafterForwardMeaning:
    """
    Step 1:
    compress selected target hidden states into the drafter hidden size and use them as KV cache

    Step 2:
    let every tree node read that context representation.

    Step 3:
    output one hidden state per node. Note that the backbone hidden states are the second to last layer hidden states 

    Step 4:
    project hidden states through the frozen target LM head
    so the drafter is trained in the target's vocabulary space.
    """
```

```python
class DrafterFlags:
    """
    - use_tree_pos_embds which is a embedding of the tree_position_id and is added to the hidden states right at the beginning
    - use_relative_tree_bias which is a head-wise attention bias based on the relationship map
    """
```

```python
class ArDrafterForwardMeaning:
    """
    - This is essentially just a single layer from the drafter with a causal mask
    Step 1:
    compress input_embds and backbone_hidden_states to hidden size

    Step 3:
    output one hidden state per node. 

    Step 4:
    project hidden states through the frozen target LM head
    so the ar drafter is trained in the target's vocabulary space.
    """
```

---

## 8. Mask Semantics

This is the most important section for understanding the code.

```python
def prefill_mask(query_token, key_token):
    """
    Used when the frozen target reads the packed real sequence.

    Rule:
    allow attention iff
    - query and key belong to the same packed document, and
    - key is not in the future of the query.

    Meaning:
    - each packed document behaves like its own causal sequence,
    - packed neighbors are invisible to each other.
    """
```

```python
def drafter_attention_mask(tree_query, candidate_key):
    """
    Used when the drafter processes training trees.

    Rule for context keys:
    allow attention iff
    - the context token belongs to the same packed document as the anchor, and
    - the context token lies strictly before the anchor position.

    Rule for tree keys:
    allow attention iff
    - the key belongs to the same tree block as the query.

    Important current behavior:
    - this mask isolates documents and isolates tree blocks,
    - but it does NOT enforce ancestor-only visibility within a tree.

    In other words:
    - tree node A can see other nodes from the same drafted tree block,
    - strict path legality is not enforced here.
    """
```
```python
def drafter_ar_attention_mask(tree_query, candidate_key):
    """
    Used when the ar_drafter processes training trees.

    Rule for context keys:
    allow attention iff
    - the context token belongs to the same packed document as the anchor, and
    - the context token lies strictly before the anchor position.

    Rule for tree keys:
    allow attention iff
    - the key belongs to the same tree block as the query.
    - and the key is either a ancestor or self

    Important current behavior:
    - it enforces ancestor-only visibility within a tree.
    """
```

```python
def verifier_tree_mask(candidate_query, candidate_key):
    """
    Used when the frozen target verifies candidate tree tokens.

    Rule:
    allow attention iff
    - the key is in the already-accepted prefix, or
    - the key is an ancestor of the query inside the candidate tree.

    Meaning:
    - the verifier scores each candidate token only under the path that leads to it,
    - siblings and unrelated branches do not leak information into each other.
    """
```

```python
def document_mask_meaning():
    """
    document_mask is not a padding mask in the usual language-model sense.

    It is a segment identity map.

    Practical meaning:
    - same id  => same packed training document,
    - different id => hard attention boundary,
    - no packed document is allowed to explain another one's targets.
    """
```

---

## 9. Main Training Logic

```python
def training_step(batch):
    """
    High-level meaning of one optimization step.
    """

    # 1. Build teacher labels and structural metadata from the frozen target.
    tree_extras = tree_processor.construct_training_extras(batch, frozen_target)

    # 2. Extract selected hidden-state features from the frozen target context.
    target_ctx_features = drafter.extract_ctx_features(tree_extras.target_hidden_states)

    # 3. Build the drafter's attention mask.
    #    This isolates documents and tree blocks.
    drafter_mask = build_drafter_mask(batch.anchors, batch.document_mask, tree_extras)

    # 4. Ask the drafter to fill all masked future tree nodes at once.
    diffusion_hidden_states, backbone_hidden_states = drafter(
        root_plus_masked_tree_embeddings,
        projected_target_context,
        drafter_mask,
        tree_info,
    )

    # 5. Ignore the root, because the root is already known.
    predicted_nodes = all_non_root_nodes(diffusion_hidden_states)
    teacher_nodes   = all_non_root_labels(tree_extras.tree_labels)

    # 6. Score the drafter in the frozen target vocabulary space.
    logits = frozen_target_lm_head(predicted_nodes)

    # 7. Compute losses.
    lm_loss = cross_entropy(logits, teacher_nodes)
    optional_losses = sibling_loss + q_loss 

    if use_ar:
        drafter_ar_mask = build_drafter_mask(batch.anchors, batch.document_mask, tree_extras)
        ar_hidden_states = ar_module(
            backbone_hidden_states,
            embeddings_of_parents_of_tree_parents,
            projected_target_context,
            drafter_ar_mask,
            tree_info,
        )
        ar_predicted_nodes = all_non_root_nodes(ar_hidden_states)
        ar_logits = frozen_target_lm_head(ar_predicted_nodes)
        ar_lm_loss = cross_entropy(ar_logits, teacher_nodes)
        lm_loss += lambda * ar_lm_loss


    # 8. Backpropagate only through the drafter-side parameters.
    optimize(drafter_only)
```

---

## 10. Losses and Their Meaning

```python
def language_model_loss(node_logits, tree_labels, tree_cum_prob=None):
    """
    Primary objective.

    Meaning:
    - teach every non-root node to predict its assigned teacher token.

    Optional weighting:
    - paths with larger verifier mass can receive more weight,
      because they matter more during real generation.
    """
```

```python
def sibling_overlap_loss(top_predictions_for_siblings):
    """
    Optional diversity regularizer.

    Meaning:
    - sibling branches should not all spend tree capacity
      on the same high-probability token.

    It does not teach correctness directly.
    It teaches branch diversity.
    """
```

```python
def q_head_loss(q_logits, drafter_argmax, teacher_labels):
    """
    Optional confidence objective.

    Target meaning:
    - 1 if the drafter's own top prediction matches the teacher token,
    - 0 otherwise.

    Why this exists:
    - later, these scores can help prune the tree down
      to the most promising candidate nodes.
    """
```

---

## 11. What "Accepted Length" Means During Training

```python
def training_acceptance_proxy(drafter_predictions, teacher_tree, tree_mask):
    """
    This is a deterministic proxy metric, not the full stochastic rejection rule.

    A node counts as accepted iff:
    - the drafter prediction matches the teacher token for that node, and
    - every ancestor token on the path also matches.

    Therefore:
    - deeper exact-match paths imply larger speculative usefulness,
    - but this metric is stricter and simpler than real inference-time rejection sampling.
    """
```

---

## 12. Inference / Speculative Decoding Logic

```python
def speculative_generate(prompt):
    """
    Repeat until EOS or length limit:
    1. verify the known prefix with the frozen target,
    2. build a tree rooted at the latest accepted token,
    3. let the drafter propose all tree tokens at once,
    4. optionally prune the tree, 
    5. let the frozen target verify the surviving candidate tree,
    6. accept the deepest valid path,
    7. append one extra verifier token beyond the accepted path.
    """
```

```python
def prefill_inference(prompt):
    """
    The frozen target first reads the true prompt and fills its cache.

    This produces:
    - verifier state for the known prefix,
    - context features that condition the drafter,
    - the first sampled continuation token.
    """
```

```python
def draft_current_tree(last_accepted_token, target_context_features):
    """
    Build a fresh tree:
    - root embedding = last accepted token,
    - all future nodes start as mask tokens,
    - the drafter predicts one token per tree node.

    Root convention:
    - the root token is overwritten to equal the already accepted frontier token,
    - root probability is treated as 1.

    Important current behavior:
    - the main drafter path does not use the verifier's strict ancestor-only mask,
    - structural legality is enforced later by verifier-side tree masking
      and, when enabled, by the AR-head layer.
    """
```

```python
def optional_tree_pruning(candidate_tree, q_values_or_ar_probs):
    """
    If pruning is enabled:
    - score each node by the product of scores along its ancestor path,
    - keep only the top candidate nodes,
    - remap parents and masks into the pruned tree.

    Meaning:
    - the verifier spends compute on the most promising paths,
      not on the full raw draft tree.
    """
```

```python
def verifier_step(candidate_tree):
    """
    The frozen target scores candidate tokens under tree-causal legality.

    Each candidate token is evaluated with access to:
    - the accepted prefix,
    - exactly the ancestors needed to reach that candidate.

    This is the critical step that restores exact target-model semantics.
    """
```

```python
def rejection_sampling_for_each_candidate(candidate_token, p_draft, p_target):
    """
    Acceptance probability:
        min(1, p_target / p_draft)

    Meaning:
    - if the drafter underestimates a token, the token is always safe to keep,
    - if the drafter overestimates a token, it is rejected with the needed probability,
    - this preserves the verifier's distribution exactly.
    """
```

```python
def choose_best_accepted_path(token_accepted, tree_mask, depth):
    """
    A path is valid only if every token on that path is accepted.

    Therefore:
    - the accepted node set is path-closed,
    - the final chosen node is the deepest node whose full ancestor chain survived.

    The generated output then commits:
    - all drafted tokens along that accepted path,
    - plus one extra fresh token sampled by the verifier at the frontier.
    """
```

---

## 13. Cache Meaning During Inference

```python
def verifier_cache_update_after_acceptance():
    """
    After verification, only the accepted path should remain part of the live future.

    So the verifier cache is logically trimmed to:
    - the already accepted prefix,
    - the positions belonging to the newly accepted path.

    Meaning:
    - rejected branches do not pollute later steps,
    - the next draft starts from the verified frontier only.
    """
```

```python
def drafter_cache_update_after_acceptance():
    """
    After drafting we remove all drafted tokens from the cache as we will receive the target hidden states later

    So the verifier cache is logically trimmed to:
    - the already accepted prefix,
    - the positions belonging to the newly accepted path.

    Meaning:
    - rejected branches do not pollute later steps,
    - the next draft starts from the verified frontier only.
    """
```

---

## 14. End-to-End Summary

```python
def end_to_end_summary():
    """
    Training:
    - pack many documents into one row,
    - isolate them with document-aware masking,
    - pick response anchors,
    - let the frozen target define the future tree labels,
    - train the drafter to predict those tree labels from
      masked future nodes plus target context features.

    Inference:
    - use the drafter to propose a tree of future tokens,
    - use the frozen target to verify only legal root-to-node paths,
    - accept the deepest path that survives rejection sampling,
    - continue from the new verified frontier.

    Core idea:
    - the drafter learns to cheaply guess many futures at once,
    - the verifier decides which guessed future is actually valid.
    """
```

---

# Implementation Details
- Use the following:
    - Fabric Lightning
    - Flex Attention
    - Wandb
    - JsonArgparse
    - Huggingface Datasets
- Employ the following trick to save on peak memory usage:
    - Chunked Cross Entropy and Logits
    - Support Chunking the Anchors into multiple forward backward passes while caching the target_ctx_features
- Support DP Training


```python
def reading_notes():
    """
    If you only want the most important meanings, remember these five facts:

    1. The target model is the teacher and verifier; it is frozen.
    2. The drafter is the only model being optimized.
    3. document_mask prevents packed documents from leaking into each other.
    4. tree_mask means 'ancestor-or-self path membership'.
    5. the verifier, not the drafter, enforces strict path legality during candidate verification.
    """
```
