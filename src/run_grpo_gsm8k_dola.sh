#!/bin/bash
set -euo pipefail

export NVME_ROOT="/work/nvme/bdhh/deema"
# Use NVMe for caches and temp dirs
mkdir -p "$NVME_ROOT/.cache" "$NVME_ROOT/tmp" \
  "$NVME_ROOT/.cache/pip" "$NVME_ROOT/.cache/huggingface" \
  "$NVME_ROOT/.cache/torch" "$NVME_ROOT/.cache/torch_extensions" \
  "$NVME_ROOT/.cache/triton"
export XDG_CACHE_HOME="$NVME_ROOT/.cache"
export TMPDIR="$NVME_ROOT/tmp"
export PIP_CACHE_DIR="$NVME_ROOT/.cache/pip"
export HF_HOME="$NVME_ROOT/.cache/huggingface"
# Optional explicit caches for HF components
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
# Torch-related caches
export TORCH_HOME="$NVME_ROOT/.cache/torch"
export TORCH_EXTENSIONS_DIR="$NVME_ROOT/.cache/torch_extensions"
export TRITON_CACHE_DIR="$NVME_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$NVME_ROOT/.cache/torch/inductor"

# Workaround: avoid importing torchvision in Transformers (prevents nms op error)
export TRANSFORMERS_NO_TORCHVISION=1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync,max_split_size_mb:64
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1

# ==== Adjust these 2 paths ====
TRAIN_PQ="${HOME}/data/gsm8k/train.parquet"
VAL_PQ="${HOME}/data/gsm8k/test.parquet"

# ==== Pick your model ====
MODEL="Qwen/Qwen2.5-3B-Instruct"   # you chose this

# NOTE: Qwen2.5-3B likely has ~28–32 layers. Set mature/candidates accordingly.
# If you hit a layer-index error, change MATURE and CANDS.
MATURE=27
CANDS="[8,12,16,20,24]"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  +algorithm.use_kl_in_reward=False \
  data.train_files="${TRAIN_PQ}" \
  data.val_files="${VAL_PQ}" \
  data.prompt_key=prompt \
  actor_rollout_ref.actor.strategy=fsdp \
  +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.model.path="${MODEL}" \
  actor_rollout_ref.rollout.name=hf \
  +actor_rollout_ref.rollout.mode=sync \
  +actor_rollout_ref.decoding.name=dola \
  +actor_rollout_ref.decoding.mature_layer=${MATURE} \
  +actor_rollout_ref.decoding.candidate_premature_layers="${CANDS}" \
  +actor_rollout_ref.decoding.relative_top=0.1 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name='verl_grpo_example_gsm8k' \
  trainer.experiment_name='qwen2.5_3b_grpo_dola' \
  trainer.n_gpus_per_node=1 \
  data.val_batch_size=32 \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  actor_rollout_ref.actor.use_kl_loss=True \
  +actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  +actor_rollout_ref.rollout.micro_batch_size=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.response_length=128 \
  data.max_response_length=128 \
  data.max_prompt_length=256 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
  trainer.test_freq=0
