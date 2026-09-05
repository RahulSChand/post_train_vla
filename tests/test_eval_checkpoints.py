from post_train_vla.eval_checkpoints import _checkpoint_steps, _summary_is_complete


def test_checkpoint_steps_selects_only_periodic_model_checkpoints(tmp_path):
    for step in (500, 1000, 3783):
        checkpoint = tmp_path / str(step)
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").touch()
    incomplete = tmp_path / "1500"
    incomplete.mkdir()

    assert _checkpoint_steps(tmp_path, 500) == [500, 1000]


def test_summary_is_complete_checks_eval_identity():
    summary = {
        "episodes": 20,
        "suite": "libero_spatial",
        "config": {"task_ids": [0], "episodes_per_task": 20, "episode_offset": 0},
    }

    assert _summary_is_complete(summary, episodes=20, suite="libero_spatial", task_id=0)
    assert not _summary_is_complete(summary, episodes=20, suite="libero_spatial", task_id=1)
