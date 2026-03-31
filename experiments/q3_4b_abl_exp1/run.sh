#!/bin/bash
#SBATCH --mail-type=NONE # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=outputs/logs/%j.out # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=outputs/logs/%j.err # where to store error messages
#SBATCH --mem=96G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --time=0-24:00:00
#CommentSBATCH --gpus-per-task=1
#CommentSBATCH --nodelist=tikgpu08 # Specify that it should run on this particular node
#CommentSBATCH --account=tik-internal
#CommentSBATCH --constraint='geforce_rtx_3090'
#CommentSBATCH --constraint='rtx_a6000'
#CommentSBATCH --constraint='a100'

# Exit on errors
set -o errexit
echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

# uenv run pytorch/v2.9.1:v2 --view=default bash <<'EOF'
cd ~/tree_flash
# source venv-2.9/bin/activate

checkpoint_path="$OUTPUT_DIR/checkpoints"
mkdir -p $checkpoint_path

TARGET_MODEL="Qwen/Qwen3-4B"
MODEL_JSON=$(cat <<MODELEOF
{
  "architectures": [
    "DFlashDraftModel"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "auto_map": {
    "AutoModel": "dflash.DFlashDraftModel"
  },
  "block_size": 16,
  "bos_token_id": 151643,
  "dflash_config": {
    "mask_token_id": 151669,
    "target_layer_ids": [
      1,
      9,
      17,
      25,
      33
    ]
  },
  "dtype": "bfloat16",
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 2560,
  "initializer_range": 0.02,
  "intermediate_size": 9728,
  "layer_types": [
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention"
  ],
  "max_position_embeddings": 40960,
  "max_window_layers": 5,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 5,
  "num_key_value_heads": 8,
  "num_target_layers": 36,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "transformers_version": "4.57.3",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
  "use_tree_pos_emb": true,
  "use_additive_tree_pos_bias": true,
  "max_tree_size": 128,
  "use_q_head": true
}
MODELEOF
)
TREE_JSON=$(cat <<TREEEOF
{
    "tree_seq_depth": 16,
}
TREEEOF
)
 # TODO: Make prunable and block tree

uv run -m src.trainer \
    --tree_type prunabl --tree_args "$TREE_JSON" \
    --drafter "$MODEL_JSON" \
    --target $TARGET_MODEL \
    --data.path ../dflash_2/datasets/q3_4b_100k_stage2.h5 --data.batch_size 4 --data.num_anchors 512 --data.tree_seq_depth 16   \
    --trainer.save_every 2048 --trainer.precision bf16-true --trainer.grad_accum_steps 8 --trainer.num_epochs 6 \
    --trainer.eval_every 2048 --trainer.wandb_run_name $EXPERIMENT_NAME --trainer.checkpoint_path $OUTPUT_DIR/checkpoints \
    --trainer.anchor_chunk_size null --trainer.ce_chunk_size 32768 \
    --trainer.dev_run false --trainer.verbose false --trainer.compile true 

# EOF