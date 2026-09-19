#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
POST_TRAIN_REPO="${POST_TRAIN_REPO:-https://github.com/RahulSChand/post_train_vla.git}"
OPENPI_REPO="${OPENPI_REPO:-https://github.com/RahulSChand/openpi_easy.git}"
POST_TRAIN_ROOT="$WORKSPACE_ROOT/post_train_vla"
OPENPI_ROOT="$WORKSPACE_ROOT/openpi_easy"
ARTIFACT_ROOT="$WORKSPACE_ROOT/artifacts"
HF_HOME="${HF_HOME:-$WORKSPACE_ROOT/.hf_home}"
TOKENIZER_URL="https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
TOKENIZER_SHA256="8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
WITH_BASE_WEIGHTS=0

usage() {
  echo "Usage: $0 [--with-base-weights]"
}

while (($#)); do
  case "$1" in
    --with-base-weights) WITH_BASE_WEIGHTS=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ $(id -u) -ne 0 ]]; then
  echo "Run this setup as root on the Vast.ai instance." >&2
  exit 1
fi

export HF_HOME
# Vast hosts can intermittently stall on Hugging Face's Xet transport. The
# regular HTTP path is slower in some regions but is resumable and reliable.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
mkdir -p "$WORKSPACE_ROOT" "$HF_HOME" "$ARTIFACT_ROOT"

clone_or_update() {
  local url="$1"
  local destination="$2"
  if [[ -d "$destination/.git" ]]; then
    git -C "$destination" pull --ff-only
  elif [[ -e "$destination" ]]; then
    echo "Refusing to replace non-git path: $destination" >&2
    exit 1
  else
    git clone "$url" "$destination"
  fi
}

echo "[1/8] Installing system rendering libraries"
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  aria2 libegl1 libegl-mesa0 libosmesa6 libosmesa6-dev

echo "[2/8] Cloning or updating source repositories"
clone_or_update "$POST_TRAIN_REPO" "$POST_TRAIN_ROOT"
clone_or_update "$OPENPI_REPO" "$OPENPI_ROOT"
git -C "$OPENPI_ROOT" submodule update --init --recursive third_party/libero

echo "[3/8] Installing the Python 3.11 model environment"
cd "$POST_TRAIN_ROOT"
uv sync --extra model --extra dev --extra publish
uv pip install --python .venv/bin/python wandb

echo "[4/8] Installing the pinned Python 3.8 LIBERO environment"
cd "$OPENPI_ROOT"
if [[ ! -x examples/libero/.venv/bin/python ]]; then
  uv venv --python 3.8 examples/libero/.venv
fi
uv pip sync \
  examples/libero/requirements.txt \
  third_party/libero/requirements.txt \
  --python examples/libero/.venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match
uv pip install \
  --python examples/libero/.venv/bin/python \
  -e packages/openpi-client \
  -e third_party/libero

echo "[5/8] Configuring LIBERO paths"
mkdir -p /root/.libero
cat >/root/.libero/config.yaml <<EOF
assets: $OPENPI_ROOT/third_party/libero/libero/libero/assets
bddl_files: $OPENPI_ROOT/third_party/libero/libero/libero/bddl_files
benchmark_root: $OPENPI_ROOT/third_party/libero/libero/libero
datasets: $OPENPI_ROOT/third_party/libero/libero/datasets
init_states: $OPENPI_ROOT/third_party/libero/libero/libero/init_files
EOF
chmod 600 /root/.libero/config.yaml

echo "[6/8] Downloading and verifying the PaliGemma tokenizer"
mkdir -p "$POST_TRAIN_ROOT/assets"
TOKENIZER="$POST_TRAIN_ROOT/assets/paligemma_tokenizer.model"
if [[ ! -f "$TOKENIZER" ]] || \
   [[ $(sha256sum "$TOKENIZER" | awk '{print $1}') != "$TOKENIZER_SHA256" ]]; then
  curl -fL --retry 3 --output "$TOKENIZER.tmp" "$TOKENIZER_URL"
  printf '%s  %s\n' "$TOKENIZER_SHA256" "$TOKENIZER.tmp" | sha256sum --check --status
  mv "$TOKENIZER.tmp" "$TOKENIZER"
fi
printf '%s  %s\n' "$TOKENIZER_SHA256" "$TOKENIZER" | sha256sum --check --status

echo "[7/8] Checking credentials and runtime"
git -C "$POST_TRAIN_ROOT" pull --ff-only
git -C "$OPENPI_ROOT" pull --ff-only
"$POST_TRAIN_ROOT/.venv/bin/python" - <<'PY'
from huggingface_hub import HfApi
import torch

identity = HfApi().whoami()
assert torch.cuda.is_available()
print("Hugging Face:", identity.get("name", "authenticated"))
print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))
PY
if [[ -f /root/.netrc ]]; then
  "$POST_TRAIN_ROOT/.venv/bin/wandb" login --verify
else
  echo "W&B: skipped (no /root/.netrc yet)"
fi
PYTHONPATH="$POST_TRAIN_ROOT/src:$OPENPI_ROOT/third_party/libero" \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  "$OPENPI_ROOT/examples/libero/.venv/bin/python" - <<'PY'
import post_train_vla.eval_libero
from libero.libero import benchmark

for name in ("libero_spatial", "libero_10"):
    suite = benchmark.get_benchmark_dict()[name]()
    assert suite.n_tasks == 10
    assert len(suite.get_task_init_states(0)) >= 40
    print(name, "tasks:", suite.n_tasks)
PY

echo "[8/8] Optional converted base weights"
if ((WITH_BASE_WEIGHTS)); then
  download_base_repo() {
    local repo="$1" destination="$2" expected_size="$3" expected_sha256="$4"
    local model="$destination/model.safetensors"
    local input_file partial actual_size

    mkdir -p "$destination"
    "$POST_TRAIN_ROOT/.venv/bin/hf" download "$repo" \
      --exclude model.safetensors --local-dir "$destination"

    actual_size=$(stat -c %s "$model" 2>/dev/null || echo 0)
    if [[ "$actual_size" != "$expected_size" ]]; then
      if [[ -e "$model" ]]; then
        mv "$model" "$model.bad-size.$(date +%s)"
      fi
      partial=$(find "$destination/.cache/huggingface/download" -maxdepth 1 \
        -type f -name "*.$expected_sha256.incomplete" -print -quit 2>/dev/null || true)
      if [[ -n "$partial" ]]; then
        mv "$partial" "$model"
      fi

      input_file=$(mktemp "$WORKSPACE_ROOT/.hf-aria2.XXXXXX")
      chmod 600 "$input_file"
      "$POST_TRAIN_ROOT/.venv/bin/python" - \
        "$repo" "$destination" "$input_file" <<'PY'
import os
import sys
import requests
from huggingface_hub import get_token

repo, destination, output = sys.argv[1:]
token = get_token()
headers = {"Authorization": f"Bearer {token}"} if token else {}
response = requests.get(
    f"https://huggingface.co/{repo}/resolve/main/model.safetensors",
    headers=headers,
    allow_redirects=False,
    timeout=30,
)
response.raise_for_status()
url = response.headers.get("location")
if not url:
    raise RuntimeError("Hugging Face did not return a signed download URL")
with open(output, "w") as stream:
    stream.write(f"{url}\n  dir={destination}\n  out=model.safetensors\n")
PY
      if ! aria2c --continue=true --max-connection-per-server=16 --split=16 \
        --min-split-size=64M --file-allocation=none --summary-interval=10 \
        --input-file="$input_file"; then
        rm -f "$input_file"
        return 1
      fi
      rm -f "$input_file"
    fi

    test "$(stat -c %s "$model")" = "$expected_size"
    printf '%s  %s\n' "$expected_sha256" "$model" | sha256sum --check --status
    echo "$repo verified: $expected_size bytes"
  }

  download_base_repo \
    Chand0320/pi0-base-pytorch "$ARTIFACT_ROOT/pi0-base-pytorch" \
    7002873776 12ddd0836c968eaca3618e51852211cb2cdeeff3e0b58507a08a79f7a03e1efa
  download_base_repo \
    Chand0320/pi05-base-pytorch "$ARTIFACT_ROOT/pi05-base-pytorch" \
    7233650408 3e453e2311a44d9e30c5d6e7101c4f4377e90a91346ddb1351fa256dc9854f83
else
  echo "Base weights skipped. Re-run with --with-base-weights if needed."
fi

echo "Setup complete"
df -h "$WORKSPACE_ROOT"
