#!/usr/bin/env python3
"""Evaluate saved GR00T artifacts, with fixed episode identities and resumable output.

Run `campaign` with minimal-groot's Python; it starts one model process per GPU.
No training, metadata reconstruction, or statistics substitution is performed.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import concurrent.futures
import contextlib
import dataclasses
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
GROOT = Path('/root/minimal-groot')
SIM_PYTHON = GROOT / 'gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python'
sys.path.insert(0, str(ROOT / 'src'))
if len(sys.argv) < 2 or sys.argv[1] != 'native':
    sys.path.insert(1, str(GROOT))

from post_train_vla.groot_noise import seeded_noise_mode
FAILURE_KEYS = ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True) + '\n')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def noise_seed(task, episode, inference_index, seed=7):
    return seed + task * 1_000_000 + episode * 1_000 + inference_index


def environment(gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
               MUJOCO_EGL_DEVICE_ID=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
               NO_ALBUMENTATIONS_UPDATE='1', PYTHONUNBUFFERED='1', PYTHONHASHSEED='7',
               TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               PYTHONPATH=os.pathsep.join([str(ROOT / 'src'), str(GROOT)]))
    return env


def checkpoint_path(out, checkpoint):
    return out / 'inventory' / checkpoint.get('inventory_subdir', 'n' + checkpoint['version']) / checkpoint['prefix']


def verify_checkpoint(out, checkpoint):
    cp = checkpoint_path(out, checkpoint)
    required = ['config.json', 'processor_config.json', 'statistics.json', 'embodiment_id.json',
                'model.safetensors.index.json', 'artifact_checksums.json', 'dataset_metadata/modality.json']
    missing = [p for p in required if not (cp / p).is_file()]
    if missing:
        raise ValueError(f'Missing saved checkpoint metadata: {missing}')
    for name, digest in checkpoint['metadata_hashes'].items():
        if sha256(cp / name) != digest:
            raise ValueError(f'Pinned metadata changed: {name}')
    checks = read_json(cp / 'artifact_checksums.json')
    mismatches = [p for p, h in checks.items() if not (cp / p).is_file() or sha256(cp / p) != h]
    if mismatches:
        raise ValueError(f'Missing or corrupt saved artifacts: {mismatches}')
    differences = [p for p, h in checks.items() if p.startswith('runtime/gr00t/') and
                   (not (GROOT / p.removeprefix('runtime/')).is_file() or
                    sha256(GROOT / p.removeprefix('runtime/')) != h)]
    allowed = {'runtime/gr00t/experiment/checkpoint_publication.py',
               'runtime/gr00t/experiment/sample_efficiency.py'}
    if set(differences) - allowed:
        raise ValueError(f'Saved inference runtime differs from local source: {differences}')
    config = read_json(cp / 'config.json')
    processor = read_json(cp / 'processor_config.json')['processor_kwargs']
    stats, ids = read_json(cp / 'statistics.json'), read_json(cp / 'embodiment_id.json')
    tag = 'libero_sim'
    if tag not in stats or tag not in ids or tag not in processor['modality_configs']:
        raise ValueError('Saved LIBERO statistics, embodiment, or modalities are absent')
    modalities = processor['modality_configs'][tag]
    expected = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    if modalities['state']['modality_keys'] != expected or modalities['action']['modality_keys'] != expected:
        raise ValueError('Saved modalities do not match the LIBERO observation adapter')
    if modalities['video']['modality_keys'] != ['image', 'wrist_image']:
        raise ValueError('Unexpected saved cameras')
    if modalities['action']['delta_indices'] != list(range(config['action_horizon'])):
        raise ValueError('Saved model and processor action horizons differ')
    for modality in ('state', 'action'):
        if set(stats[tag][modality]) != set(expected):
            raise ValueError(f'Incomplete {modality} normalization statistics')
    return dict(files_verified=len(checks), saved_metadata={p: sha256(cp / p) for p in required},
                runtime_inference_matches=True, training_only_runtime_differences=differences,
                metadata_reconstructed=False, model_type=config['model_type'],
                action_horizon=config['action_horizon'], embodiment_tag=tag, embodiment_id=ids[tag],
                verified_at=timestamp())


def load_policy(out, checkpoint, destination):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoProcessor
    import gr00t.model  # Register native architectures and processors.
    from gr00t.eval.reference_libero import ReferenceLiberoPolicy
    from gr00t.utils.determinism import seed_everything
    from gr00t.data.utils import to_json_serializable

    cp = checkpoint_path(out, checkpoint)
    torch.set_num_threads(4)
    seed_everything(7)
    model, loading = AutoModel.from_pretrained(str(cp), output_loading_info=True, local_files_only=True)
    write_json(destination / 'weight_loading.json', loading)
    if any(loading.get(key) for key in FAILURE_KEYS):
        raise ValueError(f'Weights did not load exactly: {loading}')
    processor = AutoProcessor.from_pretrained(str(cp), local_files_only=True)
    processor.eval()
    if to_json_serializable(processor.state_action_processor.statistics) != read_json(cp / 'statistics.json'):
        raise ValueError('Loaded normalization statistics differ from the saved statistics')
    if processor.embodiment_id_mapping != read_json(cp / 'embodiment_id.json'):
        raise ValueError('Loaded embodiment mapping differs from the saved mapping')
    model.requires_grad_(False)
    model.eval().to('cuda')

    class SeededPolicy(ReferenceLiberoPolicy):
        def infer_batch(self, observations):
            with seeded_noise_mode([int(o['_evaluation_noise_seed']) for o in observations]) as mode:
                result = super().infer_batch(observations)
            if mode.draws != 1:
                raise ValueError(f'Expected one action-noise draw, got {mode.draws}')
            return result

    policy = SeededPolicy(model, processor)
    policy.metadata.update(checkpoint=checkpoint['key'], repo=checkpoint['repo_id'],
                           revision=checkpoint['revision'], metadata_reconstructed=False)
    with np.load(cp / 'load_fixture.npz', allow_pickle=False) as fixture:
        observation = {'observation/' + k: fixture[k] for k in ('image', 'wrist_image', 'state')}
        observation['prompt'] = str(fixture['prompt'])
    observations = [{**observation, '_evaluation_noise_seed': noise_seed(0, i, 0)} for i in range(8)]
    started = time.monotonic()
    actions = policy.infer_batch(observations)
    if any(a['actions'].shape != (model.config.action_horizon, 7) for a in actions):
        raise ValueError('Fixture action shape differs from saved model horizon')
    np.savez_compressed(destination / 'fixture_actions.npz', actions=np.stack([a['actions'] for a in actions]))
    write_json(destination / 'load_verification.json', dict(
        strict_weight_load=True, saved_statistics_match=True, saved_embodiment_matches=True,
        offline=True, training=model.training, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        parameters=sum(p.numel() for p in model.parameters()), fixture_batch_size=8,
        fixture_action_shape=list(actions[0]['actions'].shape), fixture_seconds=time.monotonic()-started,
        parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}), inference_autocast='bfloat16',
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda))
    return policy


def sim_worker(policy_url, config, initial_states_path):
    import numpy as np
    from libero.libero import benchmark
    from post_train_vla.evaluator import evaluate
    from post_train_vla.policy import WebsocketPolicy

    saved = read_json(initial_states_path)
    suite = benchmark.get_benchmark_dict()[config.suite]()
    for task in config.task_ids:
        states = suite.get_task_init_states(task)
        definition = saved['tasks'][str(task)]
        if sha256(definition['bddl_file']) != definition['bddl_sha256']:
            raise ValueError('Task definition changed')
        for episode in range(config.episode_offset, config.episode_offset + config.episodes_per_task):
            digest = hashlib.sha256(np.asarray(states[episode], dtype='<f8').tobytes()).hexdigest()
            if digest != definition['states'][str(episode)]:
                raise ValueError(f'Initial state changed: task={task}, episode={episode}')

    class ContextPolicy:
        def __init__(self):
            self.policy = WebsocketPolicy(policy_url, connect_timeout=120)
            self.metadata = self.policy.metadata
            self.contexts = iter((task, episode) for task in config.task_ids for episode in
                                 range(config.episode_offset, config.episode_offset + config.episodes_per_task))

        def reset(self):
            self.task, self.episode = next(self.contexts)
            self.request_index = 0
            # Reconnect after an episode-level transport failure, preserving its record.
            if self.policy._connection is None or self.policy._connection.close_code is not None:
                self.policy.close()
                self.policy._connect()

        def infer(self, observation):
            seed = noise_seed(self.task, self.episode, self.request_index, config.seed)
            self.request_index += 1
            return self.policy.infer({**observation, '_evaluation_noise_seed': seed})

    policy = ContextPolicy()
    try:
        return evaluate(policy, config, policy.metadata)
    finally:
        policy.policy.close()


def simulate(args):
    from post_train_vla.evaluator import EvalConfig
    from post_train_vla.eval_libero import _episode_chunks
    protocol = read_json(args.out / 'run_manifest.json')['protocol']
    base = EvalConfig(suite=protocol['suite'], task_ids=tuple(protocol['task_ids']),
                      episodes_per_task=protocol['episodes_per_task'], episode_offset=protocol['episode_offset'],
                      seed=protocol['seed'], wait_steps=protocol['wait_steps'], replan_steps=protocol['replan_steps'],
                      resize_size=protocol['resize_size'], render_resolution=protocol['render_resolution'])
    if args.retry_file:
        configs = [dataclasses.replace(base, task_ids=(task,), episode_offset=episode, episodes_per_task=1,
                                      output_dir=args.destination / f'worker_{i:03d}')
                   for i, (task, episode) in enumerate(read_json(args.retry_file))]
    else:
        configs = [dataclasses.replace(base, episode_offset=offset, episodes_per_task=count,
                                      output_dir=args.destination / f'worker_{i:03d}')
                   for i, (offset, count) in enumerate(_episode_chunks(base.episodes_per_task, protocol['workers']))]
    errors = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=protocol['workers'],
                                                mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(sim_worker, args.policy_url, config, args.out / 'initial_states.json'): config
                   for config in configs}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception:
                errors.append(dict(worker=str(futures[future].output_dir), traceback=traceback.format_exc()))
                write_json(args.destination / 'worker_errors.json', errors)
    if errors:
        raise RuntimeError(f'{len(errors)} simulator workers failed; see worker_errors.json')


async def rollout(out, checkpoint, destination, policy, attempt, retry_file=None):
    from post_train_vla.policy_server import PolicyServer
    protocol = read_json(out / 'run_manifest.json')['protocol']
    port = 8900 + checkpoint['gpu']
    server = PolicyServer(policy, host='127.0.0.1', port=port,
                          max_batch_size=protocol['max_batch_size'], batch_wait_ms=protocol['batch_wait_ms'])
    server_task = asyncio.create_task(server.run())
    attempt_dir = destination / f'attempt_{attempt:02d}'
    attempt_dir.mkdir(parents=True, exist_ok=False)
    command = [str(SIM_PYTHON), str(Path(__file__).resolve()), 'sim', '--out', str(out),
               '--destination', str(attempt_dir), '--policy-url', f'ws://127.0.0.1:{port}']
    if retry_file:
        command += ['--retry-file', str(retry_file)]
    process = None
    try:
        await asyncio.sleep(.1)
        if server_task.done():
            await server_task
        with (attempt_dir / 'rollout.log').open('w') as log:
            process = await asyncio.create_subprocess_exec(*command, stdout=log, stderr=subprocess.STDOUT,
                                                          start_new_session=True)
            waiter = asyncio.create_task(process.wait())
            done, _ = await asyncio.wait([server_task, waiter], timeout=14400, return_when=asyncio.FIRST_COMPLETED)
            if server_task in done:
                await server_task
                raise RuntimeError('Model server exited during rollout')
            if waiter not in done:
                raise TimeoutError('Checkpoint exceeded four hours')
            if process.returncode:
                raise RuntimeError(f'Simulator exited {process.returncode}; see {attempt_dir}')
    finally:
        if process and process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), 10)
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server_task


def collect_records(destination):
    valid, errors = {}, []
    for path in sorted(destination.glob('attempt_*/worker_*/episodes.jsonl')):
        attempt = int(path.parent.parent.name.removeprefix('attempt_'))
        for line in path.read_text().splitlines():
            if not line:
                continue
            row = {**json.loads(line), 'attempt': attempt, 'source': str(path.relative_to(destination))}
            identity = (row['task_id'], row['episode_index'])
            if row.get('error'):
                row['success'] = None
                errors.append(row)
            elif identity in valid:
                raise ValueError(f'Duplicate completed episode: {identity}')
            else:
                valid[identity] = row
    return valid, errors


def summarize(out, checkpoint, destination, elapsed_seconds=0):
    valid, errors = collect_records(destination)
    protocol = read_json(out / 'run_manifest.json')['protocol']
    expected = {(t, e) for t in protocol['task_ids'] for e in range(protocol['episodes_per_task'])}
    if set(valid) - expected:
        raise ValueError('Unexpected task/episode identity')
    states = read_json(out / 'initial_states.json')['tasks']
    records = []
    for (task, episode), row in sorted(valid.items()):
        if row['seed'] != protocol['seed'] or row['policy_steps'] > protocol['max_policy_steps']:
            raise ValueError('Episode protocol mismatch')
        row.update(checkpoint=checkpoint['key'], version=checkpoint['version'], trajectory_count=checkpoint['trajectory_count'],
                   epoch=checkpoint['epoch'], initial_state_sha256=states[str(task)]['states'][str(episode)],
                   first_action_noise_seed=noise_seed(task, episode, 0), status='success' if row['success'] else 'unsuccessful')
        records.append(row)
    per_task = []
    for task in protocol['task_ids']:
        rows = [r for r in records if r['task_id'] == task]
        successes = sum(r['success'] for r in rows)
        per_task.append(dict(task_id=task, task=states[str(task)]['language'], episodes=len(rows), successes=successes,
                             unsuccessful=len(rows)-successes, success_rate=successes/len(rows) if rows else None,
                             runtime_error_attempts=sum(r['task_id']==task for r in errors)))
    for name, rows in [('episodes.jsonl', records), ('runtime_errors.jsonl', errors)]:
        (destination / name).write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in rows))
    missing = sorted(expected - set(valid))
    successes = sum(r['success'] for r in records)
    summary = dict(checkpoint=checkpoint, status='complete' if not missing else 'incomplete',
                   expected_episodes=len(expected), episodes=len(records), successes=successes,
                   unsuccessful=len(records)-successes, success_rate=successes/len(records) if records else None,
                   runtime_error_attempts=len(errors), missing_episodes=missing, per_task=per_task,
                   metadata_reconstructed=False, elapsed_seconds=elapsed_seconds, updated_at=timestamp(),
                   episodes_sha256=sha256(destination / 'episodes.jsonl'))
    write_json(destination / 'summary.json', summary)
    return summary


def evaluate_one(args):
    checkpoint = next(c for c in read_json(args.out / 'run_manifest.json')['checkpoints'] if c['key'] == args.key)
    if args.gpu is not None:
        checkpoint = {**checkpoint, 'gpu': args.gpu}
    destination = args.out / 'results' / args.key
    destination.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    write_json(destination / 'status.json', dict(state='verifying', time=timestamp()))
    write_json(destination / 'artifact_verification.json', verify_checkpoint(args.out, checkpoint))
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    policy = load_policy(args.out, checkpoint, destination)
    summary = summarize(args.out, checkpoint, destination)
    for _ in range(3):
        if summary['status'] == 'complete':
            break
        attempts = [int(p.name.removeprefix('attempt_')) for p in destination.glob('attempt_*') if p.is_dir()]
        attempt = max(attempts, default=0) + 1
        retry_file = None
        if attempts:
            retry_file = destination / f'retry_{attempt:02d}.json'
            write_json(retry_file, summary['missing_episodes'])
        write_json(destination / 'status.json', dict(state='rolling_out', attempt=attempt, time=timestamp()))
        try:
            asyncio.run(rollout(args.out, checkpoint, destination, policy, attempt, retry_file))
        except Exception:
            append_jsonl(destination / 'process_errors.jsonl', dict(stage='rollout', attempt=attempt,
                          time=timestamp(), traceback=traceback.format_exc()))
        summary = summarize(args.out, checkpoint, destination, time.monotonic()-started)
    write_json(destination / 'status.json', dict(state=summary['status'], time=timestamp()))
    print(json.dumps({k: summary[k] for k in ('status', 'episodes', 'successes', 'runtime_error_attempts')}), flush=True)
    if summary['status'] != 'complete':
        raise RuntimeError(f'Unresolved episodes: {len(summary["missing_episodes"])}')


def evaluate_locked(args):
    checkpoint = next(c for c in read_json(args.out / 'run_manifest.json')['checkpoints'] if c['key'] == args.key)
    gpu = args.gpu if args.gpu is not None else checkpoint['gpu']
    locks = ROOT / 'outputs' / '.groot_gpu_locks'
    locks.mkdir(exist_ok=True)
    with (locks / f'gpu_{gpu}.lock').open('a') as lock:
        write_json(args.out / 'results' / args.key / 'status.json',
                   dict(state='waiting_for_gpu', gpu=gpu, time=timestamp()))
        fcntl.flock(lock, fcntl.LOCK_EX)
        evaluate_one(args)


def download_checkpoint(out, checkpoint):
    cp = checkpoint_path(out, checkpoint)
    checks = read_json(cp / 'artifact_checksums.json')
    if all((cp / name).is_file() for name in checks):
        return
    command = ['hf', 'download', checkpoint['repo_id'], '--revision', checkpoint['revision'],
               '--include', checkpoint['prefix'] + '/*', '--local-dir', str(cp.parents[2]), '--max-workers', '4']
    log_path = out / 'logs' / f'download_{checkpoint["key"]}.log'
    with log_path.open('a') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Hub download failed; see {log_path}')


def campaign_worker(out, checkpoints, gpu_override=None):
    # Consume already-downloaded weights first to minimize disk use.
    checkpoints.sort(key=lambda c: (not any(checkpoint_path(out, c).glob('model-*.safetensors')), c['trajectory_count']))
    failures = []
    for checkpoint in checkpoints:
        if gpu_override is not None:
            checkpoint = {**checkpoint, 'gpu': gpu_override}
        key = checkpoint['key']
        destination = out / 'results' / key
        destination.mkdir(parents=True, exist_ok=True)
        summary_path = destination / 'summary.json'
        if summary_path.exists() and read_json(summary_path)['status'] == 'complete':
            continue
        try:
            write_json(destination / 'status.json', dict(state='downloading', time=timestamp()))
            download_checkpoint(out, checkpoint)
            print(f'START gpu={checkpoint["gpu"]} {key} epoch={checkpoint["epoch"]}', flush=True)
            with (destination / 'model.log').open('a') as log:
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), 'model', '--out', str(out),
                                         '--key', key, '--gpu', str(checkpoint['gpu'])],
                                        stdout=log, stderr=subprocess.STDOUT, env=environment(checkpoint['gpu']))
            if result.returncode:
                raise RuntimeError(f'Model process exited {result.returncode}; see {destination / "model.log"}')
            summary = read_json(summary_path)
            if summary['status'] != 'complete':
                raise RuntimeError('Checkpoint did not complete')
            print(f'COMPLETE {key}: {summary["successes"]}/{summary["episodes"]}', flush=True)
            # Keep locally recovered originals if their contents are unavailable on the Hub.
            availability_path = out / 'artifact_availability.json'
            availability = next((r for r in read_json(availability_path) if r['checkpoint'] == key), {}) \
                if availability_path.exists() else {}
            for shard in checkpoint_path(out, checkpoint).glob('model-*.safetensors'):
                if availability.get('availability_source') != 'verified_local_weights' or shard.is_symlink():
                    shard.unlink()
        except Exception:
            error = dict(checkpoint=key, stage='checkpoint', time=timestamp(), traceback=traceback.format_exc())
            append_jsonl(destination / 'process_errors.jsonl', error)
            write_json(destination / 'status.json', dict(state='error', time=timestamp()))
            failures.append(key)
            print(f'ERROR {key}: {error["traceback"].splitlines()[-1]}', flush=True)
        # Export all completed results after each checkpoint.
        subprocess.run([sys.executable, str(ROOT / 'scripts/report_saved_groot.py'), '--out', str(out)], check=True)
    return failures


def run_campaign(args):
    manifest = read_json(args.out / 'run_manifest.json')
    (args.out / 'logs').mkdir(exist_ok=True)
    availability_path = args.out / 'artifact_availability.json'
    unavailable = {r['checkpoint']: r for r in read_json(availability_path)
                   if not r['all_weight_objects_available']} if availability_path.exists() else {}
    queues = collections.defaultdict(list)
    for checkpoint in manifest['checkpoints']:
        if args.key is None or checkpoint['key'] == args.key:
            if checkpoint['key'] in unavailable:
                write_json(args.out / 'results' / checkpoint['key'] / 'missing_weights.json', unavailable[checkpoint['key']])
            else:
                queues[checkpoint['gpu']].append(checkpoint)
    write_json(args.out / 'campaign_status.json', dict(state='running', pid=os.getpid(), started_at=timestamp()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(campaign_worker, args.out, checkpoints, gpu) for gpu, checkpoints in queues.items()]
        failures = [key for future in futures for key in future.result()]
    write_json(args.out / 'campaign_status.json', dict(state='complete' if not failures and not unavailable else 'incomplete',
               failures=failures, unavailable=list(unavailable), finished_at=timestamp()))
    if failures:
        raise RuntimeError(f'Checkpoints need attention: {failures}')


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'native':
        from post_train_vla.groot_evaluation import main as native_main
        native_main(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__, epilog='Use native --help for version-native checkpoint evaluation.')
    parser.add_argument('mode', choices=['campaign', 'model', 'sim'])
    parser.add_argument('--out', type=Path, required=True, help='Prepared evaluation directory containing run_manifest.json')
    parser.add_argument('--key')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--policy-url')
    parser.add_argument('--retry-file', type=Path)
    args = parser.parse_args()
    args.out = args.out.resolve()
    {'campaign': run_campaign, 'model': evaluate_locked, 'sim': simulate}[args.mode](args)


if __name__ == '__main__':
    main()
