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

Training metrics are logged to the `chandrahul0320/post_vla` Weights & Biases project by default. Disable logging
with `--no-wandb`. Checkpoints are saved every 3000 steps by default; override with `--save-every` if needed.

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

## Verification

```bash
cd /home/ubuntu/post_train_vla
uv run --extra model --extra dev pytest -q tests
rg '(^| )(from|import) openpi|(^| )(from|import) jax' src
```

The `rg` command must return no matches. OpenPI remains useful as a parity oracle during development, but it is not
a runtime dependency.
