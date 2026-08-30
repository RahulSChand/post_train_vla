#!/usr/bin/env bash
set -euo pipefail

POST_VLA_ROOT="${POST_VLA_ROOT:-/home/ubuntu/post_train_vla}"
CHECKPOINT="${CHECKPOINT:-/home/ubuntu/.cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch}"
TOKENIZER="${TOKENIZER:-$POST_VLA_ROOT/assets/paligemma_tokenizer.model}"

cd "$POST_VLA_ROOT"
"$POST_VLA_ROOT/.venv/bin/python" -m post_train_vla.serve_torch \
  --checkpoint "$CHECKPOINT" \
  --tokenizer "$TOKENIZER" \
  --model pi0 \
  --device cuda \
  "$@"
