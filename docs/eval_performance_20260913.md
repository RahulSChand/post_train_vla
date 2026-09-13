# LIBERO evaluation performance, September 13, 2026

One sequential run of each method on an A100-SXM4-40GB with 30 CPU cores, using pi0.5 checkpoint
`pi05_libero_spatial_bs40_2epochs/1750`, Spatial task 0, initial states 0–19 and 20 workers. Both used a maximum
batch size of 32, a 10ms batch window, seed 7, 256px cameras, 224px policy images, five-step replanning,
ten settling steps, no videos and the full 220-step episode limit. Each run started a fresh policy server
seeded with `torch.manual_seed(7)`; no synthetic inference warmup was used.

The previous method used OSMesa and prepared observations every step. The optimized method used EGL and
prepared observations only when requesting another action chunk. The previous evaluator is available in
commit `53d5c23`; see the README for commands that extract it without reverting the working tree.

| Metric | Previous | Optimized |
| --- | --- | --- |
| Evaluation wall time | 83.711 s | 51.230 s |
| Model server startup | 9.014 s | 9.013 s |
| Startup + evaluation | 92.725 s | 60.243 s |
| Episodes per minute | 14.335 | 23.424 |
| Successes | 19/20 | 18/20 |
| Runtime errors | 0 | 0 |
| Total policy-controlled simulation steps | 1,886 | 2,074 |
| Aggregate simulation steps per second | 22.530 | 40.484 |
| Policy requests | 384 | 422 |
| Request-weighted mean batch size | 6.625 | 8.777 |
| Average sampled GPU utilization | 28.3% | 53.3% |
| Highest sampled VRAM | 14,877 MiB | 25,918 MiB |

Evaluation was **1.634x faster**, requiring **38.80% less time**, with **63.40% higher episode throughput**.
Including model loading, speedup was **1.539x**. Evaluation timing includes worker startup, environment
creation, inference and rollout completion; it excludes model loading and shutdown.

The optimized run executed 10.0% more simulation steps. Aggregate step throughput improved 1.797x, but this
was not an identical-action replay: early success changes the amount and concurrency of work.

This is one paired measurement, not a repeated statistical estimate or proof of success-rate parity.
EGL produces slightly different pixels, and asynchronous batch membership changes how random policy noise
is assigned to episodes even with the same server seed. The success difference cannot be attributed to
one cause from this test. GPU measurements were sampled once per second and may miss shorter peaks.

The matching NVIDIA graphics packages installed on this machine were `libnvidia-gl-580-server` and
`libnvidia-common-580-server`, version `580.105.08-0lambda0.22.04.1`. A fresh machine needs graphics libraries
matching its own driver; cloning the repository does not install them.

The original machine retains full manifests, logs, episode records and comparison scripts under
`outputs/task0_method_comparison.nUrUBo/`. Those generated files and model checkpoints are not tracked.
