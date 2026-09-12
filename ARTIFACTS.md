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

## Pi0.5 base prepared for LIBERO Spatial

Prepared and verified on 2026-09-12:

- Source weights: `gs://openpi-assets/checkpoints/pi05_base/`
- Conversion configuration: `pi05_libero` (`action_horizon=10`, `max_token_len=200`,
  `discrete_state_input=false`)
- Converted checkpoint:
  `/workspace/post_train_vla_artifacts/checkpoints/pi05_base_pytorch_libero_eval/`
- Compatibility symlink:
  `/root/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch_libero_eval`
- Converted weights: 7,233,650,408 bytes, SHA-256
  `3e453e2311a44d9e30c5d6e7101c4f4377e90a91346ddb1351fa256dc9854f83`
- LIBERO quantile statistics came from
  `gs://openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json`.
- The 11.6 GiB Orbax/JAX source download was removed after conversion and verification to preserve disk space. It is
  a reproducible intermediate; the converted base checkpoint above is the runtime artifact.

The standalone loader strictly loaded all tensors and produced finite `(10, 7)` actions on the H100. An end-to-end
`libero_spatial` task-0 smoke episode then completed without errors: 0/1 success, 220 simulator steps, 44 policy
calls. Results are in `outputs/pi05_base_libero_spatial_smoke/`.

## Pi0.5 LIBERO fine-tuned checkpoint

Prepared and verified on 2026-09-12:

- Orbax/JAX source:
  `/workspace/post_train_vla_artifacts/openpi-cache/openpi-assets/checkpoints/pi05_libero/`
- Converted checkpoint: `/workspace/post_train_vla_artifacts/checkpoints/pi05_libero_pytorch/`
- Compatibility symlink: `/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch`
- Converted weights: 7,233,650,408 bytes, SHA-256
  `1c7ea3aa5a25ad0f62b7bf5d5ac1fb5537c3c3535767b2ad2bbf14673ee36866`
- Conversion configuration: `pi05_libero` (`action_horizon=10`, `max_token_len=200`,
  `discrete_state_input=false`)

The standalone runtime strictly loaded the converted checkpoint on the H100 and produced finite `(10, 7)` LIBERO
actions. Both the source and converted weights are stored under `/workspace`; the temporary converter environment
was cleared after verification.

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
