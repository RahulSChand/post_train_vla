#!/usr/bin/env python3
"""Audit every rollout and the exact protocol before accepting the full sweep."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]/'outputs/groot_eval'


def validate():
    manifest = json.loads((ROOT/'run_manifest.json').read_text())
    suite = manifest['suite']
    expected_pairs = {(task, episode) for task in manifest['task_ids']
                      for episode in manifest['initial_state_indices']}
    statistics = json.loads((ROOT/'reconstruction/statistics.json').read_text())
    entries = []
    for version, source in manifest['models'].items():
        for budget in manifest['budgets']:
            directory = ROOT/'results'/suite/version/f'trajectories-{budget:03d}'
            summary = json.loads((directory/'complete.json').read_text())
            rows = [json.loads(line) for line in (directory/'episodes.jsonl').read_text().splitlines() if line]
            assert len(rows) == len(expected_pairs), directory
            assert {(r['task_id'], r['episode_index']) for r in rows} == expected_pairs, directory
            assert all(r['suite'] == suite and r['seed'] == manifest['seed'] and not r.get('error') for r in rows), directory
            assert all(0 <= r['policy_steps'] <= manifest['max_policy_steps'] for r in rows), directory
            assert summary['episodes'] == len(rows) and summary['successes'] == sum(r['success'] for r in rows), directory
            assert summary['success_rate'] == summary['successes']/len(rows), directory
            assert summary['repo'] == source['repo'] and summary['revision'] == source['revision'], directory
            assert summary['gpu'] == manifest['gpu_assignments'][version], directory
            assert summary['episodes_per_task'] == manifest['episodes_per_task'], directory
            cfg = summary['config']
            for key in ('seed', 'replan_steps', 'wait_steps', 'render_resolution'):
                assert cfg[key] == manifest[key], (directory, key)
            assert cfg['resize_size'] == manifest['policy_source_resolution'] and cfg['save_video'] is False, directory
            for task in manifest['task_ids']:
                per_task = summary['per_task'][str(task)]
                selected = [r for r in rows if r['task_id'] == task]
                assert per_task['episodes'] == len(selected), directory
                assert per_task['successes'] == sum(r['success'] for r in selected), directory
            metadata = directory/'reconstructed_metadata'
            loading = json.loads((metadata/'weight_loading.json').read_text())
            assert all(not loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')), directory
            saved_statistics = json.loads((metadata/'statistics.json').read_text())
            for tag, value in statistics.items():
                assert saved_statistics[tag] == value, (directory, 'normalization mismatch')
            entries.append({'version':version, 'trajectories':budget, 'successes':summary['successes'],
                            'episodes':len(rows), 'episodes_sha256':hashlib.sha256((directory/'episodes.jsonl').read_bytes()).hexdigest()})
    result = {'complete':True, 'suite':suite, 'checkpoints':len(entries),
              'episodes':sum(r['episodes'] for r in entries), 'runtime_errors':0, 'results':entries,
              'metadata_reconstructed':True, 'original_preprocessing_verified':False,
              'run_manifest_sha256':hashlib.sha256((ROOT/'run_manifest.json').read_bytes()).hexdigest()}
    (ROOT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'Validated {result["checkpoints"]} checkpoints / {result["episodes"]} episodes; zero runtime errors')


if __name__ == '__main__':
    validate()
