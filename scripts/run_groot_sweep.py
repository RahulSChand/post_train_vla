#!/usr/bin/env python3
"""Run one checkpoint at a time per GPU and export plots after completions."""
import concurrent.futures
import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/groot_eval'
LOCK = threading.Lock()
STATUS = {}
STATUS_FILE = OUT/'status.json'


def update(version, **values):
    with LOCK:
        STATUS[version] = {**STATUS.get(version, {}), **values, 'updated_at': time.time()}
        temp = STATUS_FILE.with_suffix('.tmp')
        temp.write_text(json.dumps(STATUS, indent=2)+'\n')
        temp.replace(STATUS_FILE)


def worker(version, gpu, suite, episodes):
    for budget in [5, 10, 15, 25, 50]:
        update(version, gpu=gpu, budget=budget, suite=suite, episodes_per_task=episodes, state='running')
        logfile = OUT/f'{suite}_{version}_{budget:03d}.log'
        command = [sys.executable, '-u', str(ROOT/'scripts/eval_groot_checkpoints.py'),
                   '--version', version, '--gpu', str(gpu), '--suite', suite,
                   '--budget', str(budget), '--episodes', str(episodes), '--workers', '16',
                   '--batch-size', '8', '--port', str(8800+gpu)]
        print(f'START gpu={gpu} {version} trajectories={budget}', flush=True)
        with logfile.open('a') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        if result.returncode:
            update(version, state='failed', returncode=result.returncode, log=str(logfile))
            print(f'FAILED {version} trajectories={budget}; see {logfile}', flush=True)
            return False
        update(version, state='checkpoint_complete')
        with LOCK:
            subprocess.run([sys.executable, str(ROOT/'scripts/plot_groot_results.py'), '--suite', suite], cwd=ROOT, check=True)
    update(version, state='complete')
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['libero_spatial', 'libero_10'], default='libero_spatial')
    parser.add_argument('--episodes', type=int, default=40)
    parser.add_argument('--versions', nargs='+', choices=['n1','n15','n16','n17'], default=['n1','n15','n16','n17'])
    parser.add_argument('--status-file', type=Path)
    args = parser.parse_args()
    if args.status_file:
        STATUS_FILE = args.status_file
    OUT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, version, gpu, args.suite, args.episodes) for gpu, version in enumerate(['n1','n15','n16','n17']) if version in args.versions]
        completed = [future.result() for future in futures]
    sys.exit(0 if all(completed) else 1)
