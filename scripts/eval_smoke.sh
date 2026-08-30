#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT="${OPENPI_ROOT:-/home/ubuntu/openpi}"
POST_VLA_ROOT="${POST_VLA_ROOT:-/home/ubuntu/post_train_vla}"

cd "$OPENPI_ROOT"
MUJOCO_GL="${MUJOCO_GL:-egl}" \
PYTHONPATH="$POST_VLA_ROOT/src:$OPENPI_ROOT/third_party/libero${PYTHONPATH:+:$PYTHONPATH}" \
  "$OPENPI_ROOT/examples/libero/.venv/bin/python" -m post_train_vla.eval_libero \
  --suite libero_spatial \
  --task-id 0 \
  --episodes-per-task 1 \
  --output-dir "$POST_VLA_ROOT/outputs/smoke" \
  "$@"

