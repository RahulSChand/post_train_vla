import json
import random

import pytest

from post_train_vla.sample_efficiency import (
    EarlyStoppingState,
    create_or_validate_manifest,
    should_stop_early,
    update_early_stopping,
)


def _write_metadata(root, count=8):
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"total_episodes": count}))
    (meta / "tasks.jsonl").write_text(
        "\n".join(json.dumps({"task_index": index, "task": f"task {index}"}) for index in range(2)) + "\n"
    )
    (meta / "episodes.jsonl").write_text(
        "\n".join(
            json.dumps({"episode_index": index, "tasks": [f"task {index % 2}"], "length": 10 + index})
            for index in range(count)
        )
        + "\n"
    )


def test_manifest_is_deterministic_and_nested(tmp_path):
    _write_metadata(tmp_path)
    manifest_path = tmp_path / "manifest.json"

    manifest = create_or_validate_manifest(tmp_path, manifest_path, seed=42)

    expected = list(range(8))
    random.Random(42).shuffle(expected)
    assert manifest["ordered_episode_indices"] == expected
    assert manifest["ordered_episode_indices"][:5] == manifest["ordered_episode_indices"][:7][:5]
    assert create_or_validate_manifest(tmp_path, manifest_path, seed=42) == manifest


def test_manifest_rejects_different_seed(tmp_path):
    _write_metadata(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    create_or_validate_manifest(tmp_path, manifest_path, seed=42)

    with pytest.raises(ValueError, match="does not match"):
        create_or_validate_manifest(tmp_path, manifest_path, seed=43)


def test_early_stopping_tracks_best_not_previous_epoch():
    state = None
    for epoch, successes in enumerate((9, 11, 10, 11, 10), start=1):
        state = update_early_stopping(state, epoch=epoch, successes=successes)

    assert state == EarlyStoppingState(best_successes=11, best_epoch=2, epochs_without_improvement=3)
    assert should_stop_early(state, epoch=5, minimum_epochs=3, patience=3)


def test_early_stopping_honors_minimum_epochs():
    state = EarlyStoppingState(best_successes=10, best_epoch=1, epochs_without_improvement=3)

    assert not should_stop_early(state, epoch=2, minimum_epochs=3, patience=3)
