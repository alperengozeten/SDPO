#!/usr/bin/env bash

# Usage: ./run_baseline_grpo.sh [--dry-run]

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry run mode enabled. Commands will be printed but not executed."
fi

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base settings
CONFIG_NAME="baseline_grpo"
BASE_JOB_NAME="rlvr"
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
OUTPUT_DIR="$PROJECT_ROOT/output/SDPO"
mkdir -p "$OUTPUT_DIR"

DATA_PATHS=(
    "datasets/lcb_v6"
)

# Fixed Slurm resources
ACCOUNT="oymak_owned1"
NODES=1
PARTITION="spgpu2"
TIME="4-00:00:00"
NTASKS_PER_NODE=1
GPUS=2
MEM="256G"
CPUS_PER_TASK=16
MAIL_USER="alperen@umich.edu"
MAIL_TYPE="BEGIN,END,FAIL"

# Sweep Parameters
TRAIN_BATCH_SIZES=(32)
ROLLOUT_BATCH_SIZES=(8)
MINI_BATCH_SIZES=(8)

LRS=(1e-6)
MODEL_PATHS=(
    "Qwen/Qwen3-1.7B"
)

# =============================================================================
# JOB SUBMISSION FUNCTION
# =============================================================================

submit_job() {
    local exp_name="$1"
    local script_args="$2"
    local data_path="$3"
    # Define the environment setup and command execution
    # We use the user's home directory dynamically
    local setup_cmds="pip install word2number latex2sympy2 math-verify[antlr4_9_3]==0.8.0; \
pip install -e $PROJECT_ROOT; \
pip install --upgrade wandb; \
export PYTHONPATH=$PROJECT_ROOT:\$PYTHONPATH"

    local run_cmd="bash $PROJECT_ROOT/training/verl_training.sh $exp_name $CONFIG_NAME $data_path $script_args"

    local wrapped_cmd="srun bash -c '$setup_cmds; $run_cmd'"

    local sbatch_cmd=(
        sbatch
        --job-name="$BASE_JOB_NAME"
        --account="$ACCOUNT"
        --nodes="$NODES"
        --partition="$PARTITION"
        --time="$TIME"
        --mail-user="$MAIL_USER"
        --mail-type="$MAIL_TYPE"
        --ntasks-per-node="$NTASKS_PER_NODE"
        --gres="gpu:$GPUS"
        --mem="$MEM"
        --cpus-per-task="$CPUS_PER_TASK"
        --output="$OUTPUT_DIR/%x-%j.log"
        --error="$OUTPUT_DIR/%x-%j.err"
        --wrap="$wrapped_cmd"
    )

    if [ "$DRY_RUN" = true ]; then
        echo "----------------------------------------------------------------"
        echo "Would submit job for: $exp_name"
        echo "${sbatch_cmd[@]}"
    else
        echo "Submitting job for: $exp_name"
        "${sbatch_cmd[@]}"
    fi
}

# =============================================================================
# MAIN SWEEP LOOP
# =============================================================================

for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZES[@]}"; do
    for ROLLOUT_BATCH_SIZE in "${ROLLOUT_BATCH_SIZES[@]}"; do
        for LR in "${LRS[@]}"; do
            for MODEL_PATH in "${MODEL_PATHS[@]}"; do
                for MINI_BATCH_SIZE in "${MINI_BATCH_SIZES[@]}"; do
                    for DATA_PATH in "${DATA_PATHS[@]}"; do
                        # 1. Construct the experiment name (must be unique)
                        MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
                        EXP_NAME="FINAL-GRPO-mbs-${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-model${MODEL_NAME}"

                        # 2. Construct the arguments string to pass to the training script
                        # Format: key=value key2=value2 ...
                        ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=GRPO-rich-feedback \
trainer.nnodes=$NODES \
trainer.n_gpus_per_node=$GPUS \
ray_kwargs.ray_init.num_cpus=$CPUS_PER_TASK \
actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.rollout.val_kwargs.n=4"

                        # 3. Submit
                        submit_job "$EXP_NAME" "$ARGS" "$DATA_PATH"
                    done
                done
            done
        done
    done
done
