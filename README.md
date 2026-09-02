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
  --steps 1000 --batch-size 1
```

Start with `--heads-only` when validating a new dataset or using limited GPU memory. A completed checkpoint is
stored at `outputs/finetune/<step>/` and can replace `--checkpoint` for `post-vla-serve-torch`.
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
