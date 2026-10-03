# pi0 LIBERO rollout benchmark, October 3, 2026

On this machine, switching the current evaluator from CPU rendering to GPU rendering
reduced the wall time for 20 Spatial task-0 trajectories from **93.990s to 59.307s**:
**1.585x throughput**, or **36.90% less time**. GPU policy inference was enabled in both runs.
The current branch already supports this configuration; no rollout algorithm change was needed.

| Metric | OSMesa, 20 workers | EGL, 20 workers |
| --- | ---: | ---: |
| Evaluation wall time | 93.990s | 59.307s |
| Fresh policy server startup | 8.828s | 8.425s |
| Episodes/minute | 12.767 | 20.234 |
| Aggregate policy-controlled steps/second | 46.814 | 74.190 |
| Policy-controlled steps | 4,400 | 4,400 |
| Policy requests | 880 | 880 |
| Request-weighted mean inference batch size | 8.068 | 10.077 |
| Sampled mean GPU utilization | 45.9% | 75.3% |
| Sampled peak GPU memory | 13,989 MiB | 25,030 MiB |
| Task successes | 0/20 | 0/20 |
| Runtime errors | 0 | 0 |

Actual OpenGL renderer strings were `llvmpipe (LLVM 15.0.7, 256 bits)` for OSMesa and
`NVIDIA A100-SXM4-40GB/PCIe/SSE2` for EGL. Setting `MUJOCO_GL=egl` is useful only when
the installed graphics libraries actually provide a hardware context; the runner records
the renderer rather than assuming that the environment variable is sufficient.

## Where the time goes

| Profiled component, summed across workers | OSMesa | EGL |
| --- | ---: | ---: |
| Camera rendering, including pixel readback | 869.047s | 38.610s |
| Environment steps, including rendering | 936.624s | 136.695s |
| Policy requests, including queueing and response transfer | 653.829s | 827.595s |
| Policy observation preprocessing | 3.464s | 3.091s |
| Environment construction | 127.383s | 103.769s |
| Environment reset | 82.791s | 48.232s |

These are overlapping, nested measurements accumulated across 20 processes, **not additive
wall-time components**. Rendering includes calls during construction, reset, initial-state
restoration and rollout. Policy requests include waiting for the shared GPU server and
should not be interpreted as pure model compute. The existing server's `infer_ms` metric
is not used to attribute CUDA compute because it does not synchronize the device before
ending its timer.

The rendering total fell by 22.51x and the environment-step total by 6.85x. Once rendering
uses EGL, workers spend most of their episode time waiting for policy responses. The
larger batches and higher GPU utilization indicate that faster observations feed the
policy server more effectively. This is evidence of a rendering bottleneck in the OSMesa
configuration and a shift toward inference after enabling EGL.

MuJoCo physics and the robot controller still execute on the CPU in this LIBERO stack.
The current evaluator prepares policy images only on replanning steps, but the simulator
still renders both cameras on every environment step. Reducing that rendering frequency
would require a separate change and validation; this benchmark does not change it.

## One-worker defaults

The evaluator CLI defaults to one worker and the policy server defaults to maximum batch
size one. A third run used those defaults with EGL and the same 20 initial states, seed,
checkpoint, episode limits and observation settings.

| Metric | EGL, 1 worker / batch 1 | EGL, 20 workers / max batch 20 |
| --- | ---: | ---: |
| Evaluation wall time | 395.692s | 59.307s |
| Episodes/minute | 3.033 | 20.234 |
| Aggregate policy-controlled steps/second | 11.120 | 74.190 |
| Sampled mean GPU utilization | 27.1% | 75.3% |
| Sampled peak GPU memory | 14,541 MiB | 25,030 MiB |
| Policy-controlled steps | 4,400 | 4,400 |
| Task successes | 0/20 | 0/20 |
| Runtime errors | 0 | 0 |

Parallel rollouts and GPU batching together improved throughput **6.67x** over the
one-worker EGL configuration. This comparison does not isolate batching from environment
concurrency. The serial run spent 256.423s in policy requests, 109.622s in environment steps,
and 22.603s in resets. Rendering totaled only 7.093s across all phases. With hardware
rendering already enabled, concurrency and inference batching offer the larger improvement
for this workload. All three configurations used CUDA policy inference.

Reproduce the serial configuration with:

```bash
.venv/bin/python scripts/benchmark_libero.py \
  --backends egl --episodes 20 --workers 1 --max-batch-size 1 \
  --output-dir outputs/pi0_serial_benchmark_rerun
```

Serial run artifacts are under `outputs/pi0_render_benchmark_20261003_serial/` (git-ignored).

## Workload and reproducibility

- Repository revision: `2548ae033b8c11044946879a5d6e77f014baf99e`.
- Checkpoint: [Chand0320/pi0-base-pytorch](https://huggingface.co/Chand0320/pi0-base-pytorch),
  revision `f743a38e7caef2d9e43757c5e567104b5ccf04a5`.
- Weights SHA256: `12ddd0836c968eaca3618e51852211cb2cdeeff3e0b58507a08a79f7a03e1efa`.
- Tokenizer SHA256: `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`.
- One NVIDIA A100-SXM4-40GB, driver 580.105.08, 30 AMD EPYC 7J13 vCPUs.
- Model runtime: PyTorch 2.7.1+cu126, Python 3.10. Simulator runtime: Python 3.8.20,
  robosuite 1.4.1, MuJoCo 3.2.3, NumPy 1.22.4.
- LIBERO revision: `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`.
- Spatial task 0: “pick up the black bowl between the plate and the ramekin and place it on the plate.”
- Initial states 0–19; seed 7; 10 settling steps; full 220-step episode limit;
  five-step replanning; 256px camera images resized to 224px; no videos.
- Eager CUDA inference; checkpoint action horizon 50; maximum inference batch 20;
  10ms batching window. Each run starts a fresh server with `torch.manual_seed(7)`.
- OMP, MKL, OpenBLAS and NumExpr thread counts set to one.

The base checkpoint achieved zero successes in both runs. These are complete policy rollouts,
not dummy-policy tests; the result measures runtime, not a useful task success rate for this
checkpoint. Both runs executed the same number of steps, but asynchronous batching changes
the assignment of policy noise and EGL can change rendered pixels, so the actions were not
identical. This is one measurement per configuration, not a statistical estimate. GPU
telemetry was sampled once per second and may miss peaks.

Evaluation timing includes process startup, environment creation, resets, inference, rollouts,
output aggregation and worker shutdown. It excludes model-server loading and shutdown.
Dependency installation and checkpoint download are also excluded. No synthetic policy warmup
was used; the simulator had been exercised by an EGL smoke test before benchmarking.

Run from the repository with the model environment:

```bash
.venv/bin/python scripts/benchmark_libero.py \
  --checkpoint artifacts/pi0-base-pytorch \
  --tokenizer assets/paligemma_tokenizer.model \
  --episodes 20 --workers 20 --max-batch-size 20 \
  --backends osmesa egl \
  --output-dir outputs/pi0_render_benchmark_rerun
```

Choose a new output directory on each run. The runner starts and stops its own policy servers,
checks the actual OpenGL renderer, records per-episode timing and GPU samples, and fails on
rollout errors. Paths to the simulator checkout and interpreter can be overridden with
`--libero-root` and `--libero-python`. The timing hooks are confined to the benchmark script;
ordinary evaluation is unchanged.

Raw manifests, comparison JSON, episode records and logs from this measurement are under
`outputs/pi0_render_benchmark_20261003/` (git-ignored).

## Machine preparation

The initial checkout had an empty LIBERO submodule, no simulator environment, and no EGL or
OSMesa libraries. The submodule and pinned Python 3.8 environment were restored, LIBERO paths
were configured, and the following graphics packages were installed: `libegl1`, `libosmesa6`,
`libnvidia-gl-580-server` and `libnvidia-common-580-server`. The NVIDIA packages match the
existing 580.105.08 driver; setup did not replace that driver. Downloaded weights are under
git-ignored `artifacts/`, and the tokenizer is under `assets/`.
