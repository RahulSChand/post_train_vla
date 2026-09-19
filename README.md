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

### Reproducible YAML launches

Both training entrypoints accept `--config PATH.yaml`. The YAML file is a flat mapping whose keys are the long
argument names with hyphens replaced by underscores. Values in the command line override YAML values, which is useful
for an intentional one-off change such as `--learning-rate 5e-5`.

```bash
.venv/bin/post-vla-finetune --config configs/full_dataset.example.yaml
.venv/bin/post-vla-sample-efficiency --config configs/trajectory_efficiency.example.yaml
```

Every run writes the final, fully resolved arguments to `OUTPUT_DIR/run_config.yaml`; this includes defaults and CLI
overrides, not merely the source YAML. Full-dataset checkpoints embed the same record in `metadata.json`. Trajectory
runs also embed it in each checkpoint and publish `run_config.yaml` at the root of their Hugging Face repository.
Start from [`configs/full_dataset.example.yaml`](configs/full_dataset.example.yaml) or
[`configs/trajectory_efficiency.example.yaml`](configs/trajectory_efficiency.example.yaml), copy it to a run-specific
filename, and commit that file with the experiment notes.

### Trajectory-budget sample-efficiency runs

`post-vla-sample-efficiency` runs independent full-model fine-tunes from the converted base checkpoint for nested
trajectory budgets. A seed-backed manifest fixes the exact episode IDs across pi0 and pi0.5. Each completed epoch
is evaluated on 20 rollouts of LIBERO Spatial task 0, uploaded to Hugging Face without optimizer state, verified,
and then deleted locally. Training stops after at least three epochs once three consecutive epochs fail to improve
on the best task-0 success count, with a hard limit of 20 epochs.

The prepared launcher uses budgets `5 10 15 25 50`, seed 42, microbatch 8, gradient accumulation 6 (effective
batch 48), and full-model fine-tuning:

```bash
scripts/run_sample_efficiency.sh pi0
scripts/run_sample_efficiency.sh pi05
```

The checkpoints are uploaded to `Chand0320/pi0-libero-spatial-trajectory-efficiency` and
`Chand0320/pi05-libero-spatial-trajectory-efficiency`, under paths such as
`trajectories-005/epoch-001/`. The shared manifest is uploaded at the repository root and copied into every epoch
checkpoint. Since optimizer state is never saved, an interrupted trajectory-budget run must restart that budget
from the base checkpoint.

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

### Current method: EGL and preprocessing only when replanning

The command structure is unchanged. Checkpoint sweeps and training-time evaluation now default to EGL;
preprocessing runs only when requesting a new action chunk. Explicitly setting both rendering variables below
also overrides any old `osmesa` settings inherited from your shell.

For the existing pi0.5 checkpoint sweep on this machine:

```bash
cd /home/ubuntu/post_train_vla
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  .venv/bin/python -m post_train_vla.eval_checkpoints \
  --checkpoints-dir /home/ubuntu/checkpoints/pi05_libero_spatial_bs40_2epochs \
  --tokenizer /home/ubuntu/post_train_vla/assets/paligemma_tokenizer.model \
  --every 1 --episodes 40 --workers 24 --max-batch-size 24 \
  --suite libero_spatial --pi05
```

`--every 1` selects every numeric checkpoint directory. `--episodes` is per task; omitting `--task-id` evaluates
all ten Spatial tasks (400 episodes per checkpoint above). Add `--task-id 0` for task 0 only. Use your own
checkpoint directory and omit `--pi05` for pi0 checkpoints.

The sweep skips checkpoints whose saved summaries already match the requested task scope and episode count.
Results are written under `CHECKPOINTS_DIR/eval/step_NNNNNN/`, with an aggregate summary at
`CHECKPOINTS_DIR/eval/summary.json`. Changing renderer or preprocessing does **not** invalidate saved summaries.
Use the single-checkpoint commands below with a fresh output directory when comparing methods or rerunning
an already completed checkpoint. One policy server stays alive during a sweep and hot-loads subsequent weights.

On the 40 GB A100, start with 24 workers / max batch 24 for a 40-episode-per-task run. A short concurrency
check sampled 27.5 GiB with these settings, versus 38.6 GiB with 40 workers / max batch 32. These are observed
values, not guaranteed memory limits. For 20 episodes, 20 workers are sufficient.

EGL requires NVIDIA's OpenGL/EGL userspace libraries matching the installed driver, in addition to CUDA. The
matching packages are `libnvidia-gl-580-server` and `libnvidia-common-580-server`, version
`580.105.08-0lambda0.22.04.1` on this instance; these are already installed. Match the actual driver version
when setting up another machine.

Without videos, policy observation preprocessing runs only when requesting a new action chunk (every five
simulation steps by default). Video recording still prepares every frame. Physics and camera observations
continue to update on every simulation step; the action sequence and replanning interval are unchanged.

### One checkpoint, 20 episodes, Spatial task 0

Start the server in one terminal (stop any other server on port 8001 first):

```bash
cd /home/ubuntu/post_train_vla
.venv/bin/python -m post_train_vla.serve_torch \
  --checkpoint /home/ubuntu/checkpoints/pi05_libero_spatial_bs40_2epochs/1750 \
  --tokenizer assets/paligemma_tokenizer.model --model pi05 \
  --device cuda --host 127.0.0.1 --port 8001 \
  --max-batch-size 32 --batch-wait-ms 10
```

Then run the current method in another terminal. Choose an unused `--output-dir` on each rerun:

```bash
cd /home/ubuntu/post_train_vla
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=/home/ubuntu/post_train_vla/src:/home/ubuntu/openpi_easy/third_party/libero \
  /home/ubuntu/openpi_easy/examples/libero/.venv/bin/python -m post_train_vla.eval_libero \
  --policy-url ws://127.0.0.1:8001 --suite libero_spatial --task-id 0 \
  --episodes-per-task 20 --episode-offset 0 --seed 7 --workers 20 --replan-steps 5 \
  --output-dir /home/ubuntu/post_train_vla/outputs/step1750_task0_20ep_egl
```

### Previous method: OSMesa and preprocessing every step

The previous implementation used `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa` and prepared policy observations
on every simulation step, while still requesting inference every five steps. Setting only the rendering
variables on current code selects CPU rendering but **does not** restore the old preprocessing behavior.
Do not change `--replan-steps` to 1: that changes policy behavior and is not the old method.

The pre-optimization evaluator is preserved in commit `53d5c23`. Extract it into a separate directory to
run the old method without changing your working tree. This also works on a fresh clone with that commit's
history available. With the same server command above:

```bash
cd /home/ubuntu/post_train_vla
mkdir -p outputs/legacy_53d5c23
git archive 53d5c23 src/post_train_vla | tar -x -C outputs/legacy_53d5c23
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=/home/ubuntu/post_train_vla/outputs/legacy_53d5c23/src:/home/ubuntu/openpi_easy/third_party/libero \
  /home/ubuntu/openpi_easy/examples/libero/.venv/bin/python -m post_train_vla.eval_libero \
  --policy-url ws://127.0.0.1:8001 --suite libero_spatial --task-id 0 \
  --episodes-per-task 20 --episode-offset 0 --seed 7 --workers 20 --replan-steps 5 \
  --output-dir /home/ubuntu/post_train_vla/outputs/step1750_task0_20ep_osmesa_previous
```

Run comparisons sequentially, using a fresh server for each method. The extracted source is under git-ignored
`outputs/`; it can be recreated from the pinned commit on another machine.

The September 13 paired run took 83.7s with the previous method and 51.2s with the current method: 1.63x
evaluation throughput, or 38.8% less time. Successes were 19/20 and 18/20 respectively; one pair of runs does
not establish success-rate parity. See the [tracked benchmark report](docs/eval_performance_20260913.md).
Full local logs and the runner that seeds both fresh servers with 7 are retained under
`outputs/task0_method_comparison.nUrUBo/`; those generated artifacts are not included in Git.

## GR00T saved-checkpoint evaluation

For full VLM and action-head fine-tuning with local checkpoints at every epoch,
see [GR00T training and checkpoint publication](docs/groot_training.md).
The completed Spatial-50 campaign is archived under
[`experiments/groot_spatial50_20260918`](experiments/groot_spatial50_20260918).

Use `scripts/evaluate_saved_groot.py` for saved GR00T checkpoints and `scripts/report_saved_groot.py` for
validation, JSON/CSV exports, and plots. Run them with the model environment from `/root/minimal-groot`:

```bash
/root/minimal-groot/.venv/bin/python scripts/evaluate_saved_groot.py campaign --out /path/to/evaluation
/root/minimal-groot/.venv/bin/python scripts/report_saved_groot.py --out /path/to/evaluation
```

The evaluation directory must contain `run_manifest.json`, `initial_states.json`, and pinned checkpoint
metadata under `inventory/`. This workflow covers all ten LIBERO Spatial tasks with 40 episodes per task.
The manifest assigns checkpoints to GPUs; each GPU evaluates one checkpoint at a time. Saved configuration,
processor, embodiment mappings, and normalization statistics are required and verified before inference.
Completed checkpoints are skipped when resuming. Use a new output directory when changing the protocol.
Add `--model-epochs-only` to the reporter command to generate one success-versus-epoch chart per model from an
existing `summary.json`. Stars mark each trajectory count's highest measured success rate, with earliest-epoch ties.
Use `--trajectory-epochs-only` for five charts grouped by trajectory count, with one line per model.
Both epoch plotting modes accept `--versions` (for example, `--versions 1.5 1.6 1.7`) to select a subset.

## Verification

```bash
cd /home/ubuntu/post_train_vla
uv run --extra model --extra dev pytest -q tests
rg '(^| )(from|import) openpi|(^| )(from|import) jax' src
```

The `rg` command must return no matches. OpenPI remains useful as a parity oracle during development, but it is not
a runtime dependency.
