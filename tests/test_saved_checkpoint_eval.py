"""Protect episode accounting and batching-independent random noise."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location('saved_eval', Path(__file__).parents[1] / 'scripts/evaluate_saved_groot.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_noise_is_independent_of_batch_order_and_size():
    with module.seeded_noise_mode([12, 34]):
        batch = torch.randn(size=(2, 16, 7))
    with module.seeded_noise_mode([34]):
        single = torch.randn(size=(1, 16, 7))
    with module.seeded_noise_mode([34, 12]):
        reordered = torch.randn(size=(2, 16, 7))
    assert torch.equal(batch[1], single[0])
    assert torch.equal(batch[1], reordered[0])
    assert torch.equal(batch[0], reordered[1])


def test_error_attempt_is_separate_from_completed_retry(tmp_path):
    for attempt, error in [(1, 'ConnectionClosed'), (2, None)]:
        path = tmp_path / f'attempt_{attempt:02d}' / 'worker_000' / 'episodes.jsonl'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(task_id=0, episode_index=0, error=error, success=False))+'\n')
    valid, errors = module.collect_records(tmp_path)
    assert len(valid) == len(errors) == 1
    assert valid[(0, 0)]['attempt'] == 2
    assert valid[(0, 0)]['success'] is False
    assert errors[0]['success'] is None


def test_duplicate_completed_episode_is_rejected(tmp_path):
    for attempt in [1, 2]:
        path = tmp_path / f'attempt_{attempt:02d}' / 'worker_000' / 'episodes.jsonl'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(task_id=0, episode_index=0, error=None, success=True))+'\n')
    with pytest.raises(ValueError, match='Duplicate'):
        module.collect_records(tmp_path)
