# GR00T training, evaluation and publication

Use each model's matching native GR00T checkout and Python environment. Install this project in both the model and LIBERO environments (`pip install -e /path/to/post_train_vla`). Native GR00T, CUDA/PyTorch, FlashAttention and LIBERO dependencies are managed by those environments; they are not installed by this project. Install `pip install -e ".[report,publish]"` for plotting and publication dependencies.

## Training

```sh
python scripts/finetune_groot.py --version n1d7 \
  --checkpoint /path/to/base-model --dataset /path/to/spatial-data \
  --output /path/to/new-run --epochs 8
```

Supported versions: `n1`, `n1d5`, `n1d6`, `n1d7`. This adapter currently supports LIBERO Spatial data. The prepared dataset must contain `trajectory_manifest.json` with `suite`, `trajectory_count`, and `total_frames`, plus native LeRobot data and `meta/` files. The archived campaign contains a source-specific preparation example.

All vision, language and action-head parameters are trainable. Each epoch visits every selected frame once, with action chunks padded within episode boundaries. Gradient accumulation weights partial batches by their sample counts. The first update verifies finite nonzero gradients and actual parameter updates in all three components.

Every epoch writes `epoch-NNN/` locally, including BF16 model weights, processors/typed transforms, normalization statistics, metrics and runtime source. Training uses FP32 weights and optimizer state with BF16 autocast. Checkpoints support inference/evaluation, **not exact optimizer-state resumption**. Training does not upload files, start evaluation or delete saved weights. Use a fresh output directory.

## Native checkpoint evaluation

```sh
python scripts/evaluate_saved_groot.py native model --version n1d7 \
  --checkpoint /path/to/new-run/epoch-008 --output /path/to/fresh-evaluation \
  --sim-python /path/to/libero-env/bin/python --epoch 8 \
  --episodes 50 --tasks 10 --workers 10
```

The `native` mode supports the version-native checkpoint format above; existing campaign/model/sim modes retain their original manifest-driven interface. Native evaluation uses LIBERO Spatial, fixed initial states, seed 7, 220 maximum policy steps, replanning every five actions, and 256×256 rendering. It rejects runtime errors, duplicate episodes and incomplete results. Model and simulator environments must both import `post_train_vla`. Legacy N1 and N1.5 retain their different language-batching conventions.

## Explicit publication

Install `hf` and authenticate separately. Choose the destination explicitly:

```sh
python -m post_train_vla.checkpoint_publication /path/to/new-run/epoch-008 \
  --repo OWNER/NEW-MODEL-REPO --prefix n1d7/epoch-008
```

Publication uploads with `hf`, compares remote sizes and content hashes at a pinned commit, verifies checkpoint download metadata, and writes a local `publication.json` receipt. It never deletes local weights. The repository must already exist.

## Plot exported success rates

```sh
python scripts/report_saved_groot.py --out /path/to/report \
  --success-json /path/to/success_rates.json --trajectory-count 50
```

JSON entries contain `model` (`N1`, `N1.5`, `N1.6`, `N1.7`), `epoch`, `successes`, `episodes`, and `success_rate_percent`. Entries may supply `trajectory_count`; otherwise the CLI value is required. Counts and percentages must agree. Plots use the actual evaluation denominator and observed epochs, and export PNG, PDF and SVG.
