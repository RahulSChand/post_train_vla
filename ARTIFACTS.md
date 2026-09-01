# Runtime artifacts

This file records the external artifacts used for the local LIBERO smoke evaluation.

## Torch checkpoint

- Source checkpoint: `gs://openpi-assets/checkpoints/pi0_libero/`
- Source format: OpenPI JAX/Orbax parameters (downloaded as a conversion intermediate)
- Approximate source size: 11.2 GiB
- Local source cache: `/home/ubuntu/.cache/openpi/openpi-assets/checkpoints/pi0_libero/`
- Converted local checkpoint: `/home/ubuntu/.cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch/`
- Conversion script: `/home/ubuntu/openpi_easy/examples/convert_jax_model_to_pytorch.py`
- Conversion config: `pi0_libero`
- Converted contents include `model.safetensors`, `config.json`, and
  `assets/physical-intelligence/libero/norm_stats.json`.

The runtime uses the converted Torch checkpoint; it does not load the JAX checkpoint.

## PaliGemma tokenizer

- Source: `gs://big_vision/paligemma_tokenizer.model`
- Direct HTTPS source: `https://storage.googleapis.com/big_vision/paligemma_tokenizer.model`
- Local path: `/home/ubuntu/post_train_vla/assets/paligemma_tokenizer.model`
- Approximate size: 4.1 MiB

## Supporting environments

- OpenPI source and LIBERO submodule: `/home/ubuntu/openpi_easy/`
- OpenPI conversion environment: `/home/ubuntu/openpi_easy/.venv/` (Python 3.11)
- LIBERO simulator environment: `/home/ubuntu/openpi_easy/examples/libero/.venv/` (Python 3.8)
- Standalone model environment: `/home/ubuntu/post_train_vla/.venv/` (Python 3.10)

## Verification run

On 2026-09-01, `libero_spatial`, task 0, one episode completed successfully (1/1, 100%).
Results are in `outputs/smoke/` (ignored by Git).
