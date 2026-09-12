# post_train_vla

Standalone PyTorch pi0/pi0.5 inference and LIBERO evaluation. The model runtime imports neither OpenPI nor JAX and
does not modify the installed Transformers package.

The simulator and model use separate environments because the pinned LIBERO stack requires Python 3.8 while the
Torch model runtime uses Python 3.11. They communicate through a small MessagePack WebSocket protocol.

## Model environment

```bash
cd /home/ubuntu/post_train_vla
uv sync --extra model --extra dev
```

The server requires:

- A converted checkpoint containing `model.safetensors` and `assets/**/norm_stats.json`.
- `paligemma_tokenizer.model`, passed explicitly with `--tokenizer`.

Start the prepared pi0 LIBERO checkpoint with:

```bash
/home/ubuntu/post_train_vla/scripts/serve_pi0_torch.sh
```

Or invoke it explicitly:

```bash
cd /home/ubuntu/post_train_vla
.venv/bin/post-vla-serve-torch \
  --model pi0 \
  --checkpoint /path/to/pi0_libero_pytorch \
  --tokenizer /path/to/paligemma_tokenizer.model \
  --device cuda
```

Pass `--compile` only after ordinary eager inference works. For pi0.5, use a matching converted checkpoint and
`--model pi05`.

### Evaluate the prepared pi0.5 base model on LIBERO Spatial

This machine has the raw `pi05_base` weights converted with the official `pi05_libero` model configuration. The
weights remain the untouched base weights; only the 10-step action horizon, prompt transform, and LIBERO quantile
normalization assets are environment-specific. Start the policy server with:

```bash
cd /root/post_train_vla
.venv/bin/post-vla-serve-torch \
  --model pi05 \
  --checkpoint /workspace/post_train_vla_artifacts/checkpoints/pi05_base_pytorch_libero_eval \
  --tokenizer /root/post_train_vla/assets/paligemma_tokenizer.model \
  --device cuda
```

In another terminal, run the complete 50-episode-per-task LIBERO Spatial evaluation without videos:

```bash
cd /root/openpi_easy
MUJOCO_GL=egl \
PYTHONPATH=/root/post_train_vla/src:/root/openpi_easy/third_party/libero \
  examples/libero/.venv/bin/python -m post_train_vla.eval_libero \
  --suite libero_spatial \
  --episodes-per-task 50 \
  --output-dir /root/post_train_vla/outputs/pi05_base_libero_spatial
```

For a quick pipeline check, add `--task-id 0 --episodes-per-task 1`. The released `pi05_libero` configuration uses
`discrete_state_input=false`; this repository reads that value from the converted checkpoint so it does not insert
proprioceptive state tokens during this evaluation.

For compatibility, the original cache location at
`/root/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch_libero_eval` is a symlink to the checkpoint under
`/workspace`.

The official LIBERO-fine-tuned pi0.5 checkpoint is also prepared at
`/workspace/post_train_vla_artifacts/checkpoints/pi05_libero_pytorch`. Use that path in the same server command to
evaluate the fine-tuned model instead of the untouched base weights.

### Moving to a new instance after the September 10 fixes

Copy this updated source tree (including `src/`, `tests/`, `pyproject.toml`, and `uv.lock`), the tokenizer, and
the required checkpoint/dataset assets. Recreate the model environment with `uv sync --extra model --extra train
--extra dev` and the LIBERO environment as described below; do not copy an existing `.venv` across instances.

The policy loader now initializes image-patch positions and both rotary-attention frequency buffers after meta
allocation. Existing model checkpoints are compatible and do not need conversion or retraining for this fix.
Restart policy servers to load the updated code. Evaluate into a new `--output-dir`; for a checkpoint sweep,
move the existing `CHECKPOINTS_DIR/eval` folder to an unused backup name before rerunning, since the sweep skips
completed summaries even if they were produced by the old loader.

Gradient checkpointing now passes each transformer layer index explicitly into recomputation. Regression tests
compare all parameter and input gradients against ordinary backward, including LoRA, and compare the fast policy
loader against normally initialized models in float32/bfloat16 for pi0/pi0.5:

```bash
.venv/bin/python -m pytest -q tests/test_model_regressions.py
```

For a fresh pi0-base fine-tune, use your converted **pi0_base** checkpoint with matching LIBERO normalization
assets, a new output directory, and omit `--resume`. The example below instead starts from an already
LIBERO-fine-tuned checkpoint. These two code fixes do not establish the cause of the stalled updates observed in
the previous run. Before a long restart, verify actual parameter changes on real training batches and evaluate
an early saved checkpoint. The old step-55988 checkpoint already achieved 14/20 task-0 successes when its position
buffers were restored in an isolated diagnostic; preserve it for comparison.

## Fine-tuning on LIBERO LeRobot data

The minimal trainer uses the same LeRobot field layout as OpenPI's LIBERO converter: `image`, `wrist_image`,
`state`, `action`, and `task_index`. It saves standalone checkpoints which can be served and evaluated by this
repository directly.

```bash
cd /home/ubuntu/post_train_vla
uv sync --extra model --extra train
.venv/bin/post-vla-finetune \
  --checkpoint /home/ubuntu/.cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch \
  --tokenizer assets/paligemma_tokenizer.model \
  --dataset-repo physical-intelligence/libero \
  --output-dir outputs/finetune \
  --steps 1000 --batch-size 32 --lora
```

For a local LeRobot dataset, pass its directory directly to `--dataset-repo`, for example
`--dataset-repo /home/ubuntu/.cache/huggingface/lerobot/libero_spatial`.

Resume a checkpoint containing `optimizer.pt` by passing it as `--checkpoint` with `--resume`. In resume mode,
`--steps` is the target global step, not the number of additional steps. For example, to continue a four-epoch
checkpoint at step 15132 through a fifth 3783-step epoch:

```bash
.venv/bin/post-vla-finetune \
  --checkpoint /home/ubuntu/15132 \
  --tokenizer assets/paligemma_tokenizer.model \
  --dataset-repo /home/ubuntu/libero_spatial_post \
  --output-dir outputs/finetune_bs14_5epochs \
  --steps 18915 --batch-size 14 --learning-rate 5e-5 \
  --save-every 500 --train-only --resume
```

Add `--wandb-run-id 5iwn2igx` to continue logging into the original W&B run; omit it to create a new continuation
run. Resume restores model and optimizer state at the saved global step. Data-loader and RNG state are not saved.
If optimizer state is unavailable or corrupt, add `--reset-optimizer` with `--resume` to preserve the model and
global step while explicitly starting a fresh optimizer.

Training metrics are logged to the `chandrahul0320/post_vla` Weights & Biases project by default. Disable logging
with `--no-wandb`. Checkpoints are saved every 3000 steps by default; override with `--save-every` if needed.

For training without in-process evaluation, pass `--train_only`. It writes compact evaluation checkpoints at
`--save-every` intervals, containing only `model.safetensors`, `config.json`, and normalization stats. It writes
resumable checkpoints with `optimizer.pt` only at completed data epochs and the final step. Evaluate any compact
checkpoint later with `post-vla-serve-torch` and `post-vla-eval-libero`.

To evaluate a saved checkpoint on 20 task-0 episodes every 500 training steps, record every rollout, and log the
success rate to the same W&B run, add:

```bash
--eval-every 500 --eval-episodes 20 --eval-task-id 0 --eval-save-video
```

Each evaluation is stored under `OUTPUT_DIR/eval/step_NNNNNN/`; its `videos/` directory contains the 20 MP4s and
`summary.json` contains the aggregate result. Evaluation pauses training while it runs and requires the prepared
LIBERO Python 3.8 environment under `/home/ubuntu/openpi_easy/examples/libero/.venv`.

`--lora` follows OpenPI's low-memory recipe: it adapts all Q/K/V/O attention and gate/up/down MLP projections,
with rank 16 for PaliGemma and rank 32 for the action expert (alpha equals rank). The converted base model is
frozen; only adapters and the pi0 action/time heads train, which keeps the memory profile appropriate for a
40 GB GPU. Use
`--lora-paligemma-rank` and `--lora-action-expert-rank` to override those defaults. A completed LoRA checkpoint
records its adapter layout in `config.json` and can replace `--checkpoint` for `post-vla-serve-torch` directly.
On the available A100 40 GB GPU, a one-step LIBERO smoke train (forward, backward, optimizer update, and
checkpoint save) succeeded through batch size 96, with a 34.61 GiB PyTorch peak at that size; batch size 97
OOMed during backward. Use 96 as the tested maximum for this exact configuration, or leave headroom for other
GPU workloads.
Start with `--heads-only` when validating a new dataset or using limited GPU memory.
For pi0, the trainer matches OpenPI's legacy LIBERO configuration by converting the first six action dimensions to
state-relative deltas; pass `--no-extra-delta-actions` only if your dataset already stores those deltas.

## LIBERO environment

Install this package without dependencies into the existing LIBERO Python 3.8 environment; its locked LIBERO
requirements already supply the evaluator dependencies:

```bash
cd /home/ubuntu/post_train_vla
uv pip install --python /home/ubuntu/openpi_easy/examples/libero/.venv/bin/python --no-deps -e .
```

Run one smoke episode in another terminal:

```bash
/home/ubuntu/post_train_vla/scripts/eval_smoke.sh
```

Results are written to `outputs/smoke/episodes.jsonl` and `outputs/smoke/summary.json`. Pass `--save-video` to the
script to record a rollout. Remove `--task-id 0` and increase `--episodes-per-task` for complete evaluation.

For fast evaluation of all periodic checkpoints, run 20 LIBERO environments concurrently and batch their policy
requests on the GPU:

```bash
post-vla-eval-checkpoints \
  --checkpoints-dir /home/ubuntu/post_train_vla/outputs/finetune_bs14_4epochs \
  --tokenizer /home/ubuntu/post_train_vla/assets/paligemma_tokenizer.model \
  --every 500 --episodes 20 --workers 20 --max-batch-size 20
```

The sweep is resumable: checkpoints with a complete 20-episode `summary.json` are skipped. Results are written
under `CHECKPOINTS_DIR/eval/step_NNNNNN/`, with an aggregate summary at `CHECKPOINTS_DIR/eval/summary.json`. One
policy server stays alive for the sweep and hot-loads subsequent checkpoints, avoiding repeated model construction.

## Verification

```bash
cd /home/ubuntu/post_train_vla
uv run --extra model --extra dev pytest -q tests
rg '(^| )(from|import) openpi|(^| )(from|import) jax' src
```

The `rg` command must return no matches. OpenPI remains useful as a parity oracle during development, but it is not
a runtime dependency.
