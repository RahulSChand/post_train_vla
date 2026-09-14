#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?usage: run_sample_efficiency.sh pi0|pi05}"
PROJECT_ROOT="/root/post_train_vla"
DATASET_ROOT="/root/libero_spatial_post"
TOKENIZER="$PROJECT_ROOT/assets/paligemma_tokenizer.model"
MANIFEST="$PROJECT_ROOT/outputs/sample_efficiency/trajectory_manifest_seed42.json"
# Runpod's interactive shell keeps Hugging Face credentials on the persistent
# workspace volume. Detached/non-interactive SSH does not inherit that location.
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

case "$MODEL" in
  pi0)
    CHECKPOINT="/root/hf_models/pi0-base-pytorch"
    OUTPUT_DIR="$PROJECT_ROOT/outputs/pi0_sample_efficiency"
    HF_REPO_ID="Chand0320/pi0-libero-spatial-trajectory-efficiency"
    ;;
  pi05)
    CHECKPOINT="/root/hf_models/pi05-base-pytorch"
    OUTPUT_DIR="$PROJECT_ROOT/outputs/pi05_sample_efficiency"
    HF_REPO_ID="Chand0320/pi05-libero-spatial-trajectory-efficiency"
    ;;
  *)
    echo "model must be pi0 or pi05" >&2
    exit 2
    ;;
esac

cd "$PROJECT_ROOT"
exec .venv/bin/python -m post_train_vla.sample_efficiency \
  --model "$MODEL" \
  --checkpoint "$CHECKPOINT" \
  --tokenizer "$TOKENIZER" \
  --dataset-root "$DATASET_ROOT" \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --hf-repo-id "$HF_REPO_ID" \
  --trajectory-budgets 5 10 15 25 50 \
  --seed 42 \
  --batch-size 8 \
  --gradient-accumulation-steps 6 \
  --learning-rate 5e-5 \
  --minimum-epochs 3 \
  --patience 2 \
  --max-epochs 15 \
  --eval-workers 20 \
  --eval-max-batch-size 8 \
  --eval-port 8765
