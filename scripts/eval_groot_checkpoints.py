#!/usr/bin/env python3
"""Evaluate weight-only GR00T checkpoints using the matching training recipe.

Run with minimal-groot's Python. Reconstructed metadata is explicitly recorded;
the original Hub repositories and dataset are never modified.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GROOT = Path('/root/minimal-groot')
OUT = ROOT / 'outputs/groot_eval'
VERSIONS = {'n1': '1', 'n15': '1.5', 'n16': '1.6', 'n17': '1.7'}
sys.path[:0] = [str(ROOT / 'src'), str(GROOT)]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def prepare_statistics(dataset):
    """Same float32 reductions/quantiles as gr00t.data.stats, reading lowdim only."""
    import numpy as np
    import pyarrow.parquet as pq

    paths = sorted(dataset.glob('data/*/*.parquet'))
    info = json.loads((dataset / 'meta/info.json').read_text())
    if len(paths) != info['total_episodes']:
        raise ValueError(f'Incomplete dataset: {len(paths)} files, expected {info["total_episodes"]}')
    arrays = {key: [] for key in ('state', 'actions')}
    fingerprint = hashlib.sha256()
    for path in paths:
        table = pq.read_table(path, columns=list(arrays))
        fingerprint.update(path.name.encode())
        for key in arrays:
            value = np.asarray(table[key].to_pylist(), dtype=np.float32)
            if not np.isfinite(value).all():
                raise ValueError(f'Nonfinite {key} in {path}')
            fingerprint.update(value.tobytes())
            arrays[key].append(value)
    stats = {}
    for key, chunks in arrays.items():
        value = np.concatenate(chunks)
        expected_dim = 8 if key == 'state' else 7
        if value.shape != (info['total_frames'], expected_dim):
            raise ValueError(f'Unexpected {key} shape: {value.shape}')
        stats[key] = {name: fn(value, axis=0).tolist() for name, fn in
                      [('mean', np.mean), ('std', np.std), ('min', np.min), ('max', np.max)]}
        stats[key].update(q01=np.quantile(value, .01, axis=0).tolist(),
                          q99=np.quantile(value, .99, axis=0).tolist())
    keys = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    nested = {}
    for modality, source in [('state', 'state'), ('action', 'actions')]:
        nested[modality] = {
            key: {stat: values[i: (8 if modality == 'state' and key == 'gripper' else i+1)]
                  for stat, values in stats[source].items()}
            for i, key in enumerate(keys)
        }
    from gr00t.data.types import EmbodimentTag
    write_json(OUT / 'reconstruction/statistics.json', {EmbodimentTag.LIBERO_PANDA.value: nested})
    write_json(OUT / 'reconstruction/dataset_stats.json', stats)
    write_json(OUT / 'reconstruction/provenance.json', {
        'dataset': str(dataset), 'episodes': len(paths), 'frames': info['total_frames'],
        'lowdim_sha256': fingerprint.hexdigest(),
        'training_dataset_confirmed_by_user': True,
        'metadata_reconstructed': True,
        'method': 'Full dataset float32 mean/std/min/max and numpy quantile(.01/.99), matching gr00t/data/stats.py',
        'training_recipe_commit': subprocess.check_output(['git', '-C', str(GROOT), 'rev-parse', 'HEAD'], text=True).strip(),
    })
    print(f'Reconstructed statistics from {len(paths)} episodes / {info["total_frames"]} frames', flush=True)


def prepare_checkpoint(version, checkpoint):
    config = json.loads((OUT / 'base' / version / 'config.json').read_text())
    # Architecture comes from the official generation; training overrides are
    # taken from minimal-groot's sample_efficiency.build_model_and_processor.
    config['state_dropout_prob'] = 0.0
    if version in ('n1', 'n15'):
        config['backbone_cfg'].update(tune_llm=True, tune_visual=True)
        config['action_head_cfg'].update(tune_projector=True, tune_diffusion_model=True)
    else:
        config.update(tune_llm=True, tune_visual=True, tune_projector=True, tune_diffusion_model=True)
    write_json(checkpoint / 'config.json', config)
    weight_map = {}
    total = 0
    for shard in sorted(checkpoint.glob('model-*.safetensors')):
        with shard.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            header = json.loads(stream.read(length))
        for name, meta in header.items():
            if name == '__metadata__':
                continue
            if name in weight_map:
                raise ValueError(f'Duplicate weight {name}')
            weight_map[name] = shard.name
            total += meta['data_offsets'][1] - meta['data_offsets'][0]
    if not weight_map:
        raise ValueError(f'No weight shards at {checkpoint}')
    write_json(checkpoint / 'model.safetensors.index.json', {'metadata': {'total_size': total}, 'weight_map': weight_map})
    return config


def load_policy(version, checkpoint, result_dir):
    import numpy as np
    import torch
    from transformers import AutoConfig
    from gr00t.configs.base_config import Config
    from gr00t.configs.data.libero_spatial import libero_spatial_config
    from gr00t.data.types import EmbodimentTag
    from gr00t.experiment.launch_finetune import select_model_config
    from gr00t.model import MODEL_REGISTRY
    from gr00t.eval.reference_libero import ReferenceLiberoPolicy

    config_dict = prepare_checkpoint(version, checkpoint)
    cfg = Config()
    cfg.model = select_model_config(str(OUT / 'base' / version), VERSIONS[version])
    cfg.model.state_dropout_prob = 0.0
    cfg.model.action_horizon = config_dict['action_horizon']
    for name in ('tune_llm', 'tune_visual', 'tune_projector', 'tune_diffusion_model'):
        setattr(cfg.model, name, True)
    if version in ('n1', 'n15'):
        cfg.model.max_state_dim = config_dict['action_head_cfg']['max_state_dim']
        cfg.model.max_action_dim = config_dict['action_dim']
    elif version == 'n16':
        cfg.model.max_state_dim = config_dict['max_state_dim']
        cfg.model.max_action_dim = config_dict['max_action_dim']
        cfg.model.apply_sincos_state_encoding = config_dict['apply_sincos_state_encoding']
    cfg.training.start_from_checkpoint = str(OUT / 'base' / version)
    cfg.training.transformers_trust_remote_code = True
    cfg.data.modality_configs = {EmbodimentTag.LIBERO_PANDA.value: libero_spatial_config(cfg.model.action_horizon)}
    artifact_dir = result_dir / 'reconstructed_metadata'
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pipeline = MODEL_REGISTRY[type(cfg.model)](cfg, artifact_dir)
    processor = pipeline._create_processor()
    processor.set_statistics(json.loads((OUT / 'reconstruction/statistics.json').read_text()), override=True)
    processor.eval()
    processor.save_pretrained(artifact_dir)
    native = AutoConfig.from_pretrained(str(checkpoint))
    kwargs = {'output_loading_info': True, 'config': native}
    if version in ('n1', 'n15'):
        kwargs.update(tune_llm=True, tune_visual=True, tune_projector=True, tune_diffusion_model=True)
    model, loading = pipeline.model_class.from_pretrained(str(checkpoint), **kwargs)
    write_json(artifact_dir / 'weight_loading.json', loading)
    if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError(f'Weight loading was not exact: {loading}')
    model.to('cuda').eval()
    model.requires_grad_(False)
    model.config.save_pretrained(artifact_dir)
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    return ReferenceLiberoPolicy(model, processor)


async def rollout(policy, args, destination):
    from post_train_vla.policy_server import PolicyServer
    server = PolicyServer(policy, host='127.0.0.1', port=args.port,
                          max_batch_size=args.batch_size, batch_wait_ms=10)
    server_task = asyncio.create_task(server.run())
    await asyncio.sleep(0)
    sim_python = GROOT / 'gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python'
    command = [str(sim_python), '-m', 'post_train_vla.eval_libero',
               '--policy-url', f'ws://127.0.0.1:{args.port}', '--suite', args.suite,
               '--episodes-per-task', str(args.episodes), '--workers', str(args.workers),
               '--seed', '7', '--replan-steps', '5', '--wait-steps', '10',
               '--render-resolution', '256', '--resize-size', '256', '--output-dir', str(destination)]
    if args.smoke:
        command += ['--task-id', '0']
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / 'src')
    process = None
    try:
        with (destination / 'rollout.log').open('w') as log:
            process = await asyncio.create_subprocess_exec(*command, stdout=log, stderr=subprocess.STDOUT, env=env)
            waiter = asyncio.create_task(process.wait())
            done, _ = await asyncio.wait([server_task, waiter], return_when=asyncio.FIRST_COMPLETED)
            if server_task in done:
                await server_task
                raise RuntimeError('Policy server exited early')
            if process.returncode:
                raise RuntimeError(f'Evaluator exited {process.returncode}; see {destination / "rollout.log"}')
    finally:
        if process and process.returncode is None:
            process.terminate()
            await process.wait()
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


def evaluate_one(args, budget):
    from huggingface_hub import HfApi
    repo = f'kaweees/gr00tn{VERSIONS[args.version]}-libero-spatial-trajectory-efficiency'
    prefix = f'{args.version}/trajectories-{budget:03d}'
    destination = OUT / ('smoke' if args.smoke else 'results') / args.suite / args.version / f'trajectories-{budget:03d}'
    if (destination / 'complete.json').exists():
        previous = json.loads((destination / 'complete.json').read_text())
        if previous['episodes_per_task'] == args.episodes:
            print(f'Skipping complete {destination}', flush=True)
            return
        raise ValueError(f'Existing result uses a different protocol: {destination}')
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / 'run_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['suite'] != args.suite or manifest['episodes_per_task'] != args.episodes:
            raise ValueError('Requested protocol differs from the pinned run manifest')
        source = manifest['models'][args.version]
        if source['repo'] != repo:
            raise ValueError('Repository differs from the pinned run manifest')
        revision = source['revision']
    else:
        revision = HfApi().model_info(repo).sha
    staging = OUT / 'staging' / args.version
    checkpoint = staging / prefix
    subprocess.run(['hf', 'download', repo, '--revision', revision, '--include', prefix + '/*', '--local-dir', str(staging)], check=True)
    started = time.time()
    policy = load_policy(args.version, checkpoint, destination)
    asyncio.run(rollout(policy, args, destination))
    records = [json.loads(line) for line in (destination / 'episodes.jsonl').read_text().splitlines() if line]
    expected_tasks = 1 if args.smoke else 10
    counts = collections.Counter(r['task_id'] for r in records)
    identities = {(r['task_id'], r['episode_index']) for r in records}
    if (len(counts) != expected_tasks or any(n != args.episodes for n in counts.values())
            or len(identities) != len(records) or any(r.get('error') for r in records)):
        raise RuntimeError('Rollout incomplete or contains runtime errors; no valid result recorded')
    summary = json.loads((destination / 'summary.json').read_text())
    summary.update(version=args.version, trajectory_count=budget, episodes_per_task=args.episodes,
                   repo=repo, revision=revision, gpu=args.gpu, elapsed_seconds=time.time()-started,
                   metadata_reconstructed=True, training_dataset='/root/libero_long_post')
    summary['per_task'] = {
        str(task): {'task': next(r['task'] for r in records if r['task_id'] == task),
                    'episodes': n, 'successes': sum(r['success'] for r in records if r['task_id'] == task)}
        for task, n in sorted(counts.items())}
    write_json(destination / 'complete.json', summary)
    print(f'COMPLETE {args.version} trajectories={budget}: {summary["successes"]}/{summary["episodes"]}', flush=True)
    if not args.smoke:
        # Only downloaded weight shards are removed; metadata and results stay.
        for shard in checkpoint.glob('model-*.safetensors'):
            shard.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-statistics', type=Path)
    p.add_argument('--version', choices=VERSIONS)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--budget', type=int, choices=[5, 10, 15, 25, 50], default=5)
    p.add_argument('--suite', choices=['libero_spatial', 'libero_10'], default='libero_spatial')
    p.add_argument('--episodes', type=int, default=40)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--port', type=int, default=8800)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), MUJOCO_EGL_DEVICE_ID=str(args.gpu),
                      MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
                      NO_ALBUMENTATIONS_UPDATE='1', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
    if args.prepare_statistics:
        prepare_statistics(args.prepare_statistics)
    elif args.version:
        evaluate_one(args, args.budget)
    else:
        p.error('--version or --prepare-statistics required')


if __name__ == '__main__':
    main()
