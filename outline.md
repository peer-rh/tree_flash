# Stage 1 - Sequence generation
- Generate synthetic answers to prompts and T=1
- Outputs:
    - prompts B x [S_P]
    - responses B x [S_R]

# Stage 2 - Tree Generation
- Generate continuation trees for responses
- For each input sequence:
    1. Check the most likely branching position anchor points
    2. Generate the continuation trees for the top-k anchor points
    3. Save a vector of shape [respone_length, tree_size] which is (i) the tree continuation tokens, where non chosen anchors are simply a ignore_idx, and another vector which saves the probs of this token being chosen
- Outputs:
    - prompts: B x [S_P]
    - responses: B x [S_R]
    - response_tree_ids: B x [S_R, tree_size]
    - response_probs: B x [S_R, tree_size]

# Stage 3 - Training

```
anchors = random anchor positions # [B, n_anchors]
labels = block[anchors:anchors+tree_depth] # [B, n_anchors, tree_depth, tree_size]

backbone_hs = backbone_drafter(noise_embds, target_ctx_features, ...)
diffusion_preds = diffusion_head(backbone_hs, target_ctx_features, ...)
diff_loss = CE(tree_labels[b, i, j], diffusion_preds[b, i, j], weight=cumprod[b, i, j])

parent_input_embds = embd(parents of tree_labels)
ar_preds[b, i] = ar_head(cat(backbone_hs, parent_input_embds), target_ctx_features, ...)
ar_loss = CE(tree_labels[b, i, j], ar_preds[b, i, j], weight=cumprod[b, i, j])

loss = ar_loss + diff_loss
```
