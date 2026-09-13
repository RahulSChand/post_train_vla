import json
import pathlib

import numpy as np
import pytest
import torch

import post_train_vla.finetune as finetune
from post_train_vla.finetune import read_resume_step, restore_optimizer, save_checkpoint
from post_train_vla.models.configuration import Pi0Config
from post_train_vla.transforms import LiberoTransforms


class _Tokenizer:
    def tokenize(self, prompt, state=None):
        del prompt, state
        return np.zeros(48, dtype=np.int64), np.ones(48, dtype=bool)


def test_encode_training_batch_pads_libero_actions():
    transforms = object.__new__(LiberoTransforms)
    transforms.config = Pi0Config()
    transforms.tokenizer = _Tokenizer()
    transforms.stats = {
        "state": {"mean": [0.0] * 8, "std": [1.0] * 8},
        "actions": {"mean": [0.0] * 7, "std": [1.0] * 7},
    }
    batch = {
        "image": np.zeros((2, 16, 16, 3), dtype=np.uint8),
        "wrist_image": np.zeros((2, 16, 16, 3), dtype=np.uint8),
        "state": np.zeros((2, 8), dtype=np.float32),
        "action": np.zeros((2, 50, 7), dtype=np.float32),
        "prompt": ["task one", "task two"],
    }

    observation, actions = transforms.encode_training_batch(batch, "cpu")

    assert observation.images["base_0_rgb"].shape == (2, 16, 16, 3)
    assert observation.tokenized_prompt.shape == (2, 48)
    assert actions.shape == (2, 50, 32)
    assert torch.count_nonzero(actions[..., 7:]) == 0


def test_encode_batch_combines_independent_inference_requests():
    transforms = object.__new__(LiberoTransforms)
    transforms.config = Pi0Config()
    transforms.tokenizer = _Tokenizer()
    transforms.stats = {
        "state": {"mean": [0.0] * 8, "std": [1.0] * 8},
        "actions": {"mean": [0.0] * 7, "std": [1.0] * 7},
    }
    raw = {
        "observation/image": np.zeros((16, 16, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((16, 16, 3), dtype=np.uint8),
        "observation/state": np.zeros(8, dtype=np.float32),
        "prompt": "task",
    }

    observation, states = transforms.encode_batch([raw, raw], "cpu")

    assert observation.images["base_0_rgb"].shape == (2, 16, 16, 3)
    assert observation.state.shape == (2, 32)
    assert len(states) == 2


def test_save_checkpoint_preserves_standalone_runtime_assets(tmp_path):
    source = tmp_path / "source"
    stats = source / "assets" / "physical-intelligence" / "libero"
    stats.mkdir(parents=True)
    (source / "config.json").write_text(json.dumps({"action_dim": 32}))
    (stats / "norm_stats.json").write_text(json.dumps({"state": {}, "actions": {}}))
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())

    checkpoint = save_checkpoint(model, optimizer, 3, tmp_path / "output", source)

    assert (checkpoint / "model.safetensors").is_file()
    assert (checkpoint / "optimizer.pt").is_file()
    assert (checkpoint / "config.json").is_file()
    assert (checkpoint / "assets" / "physical-intelligence" / "libero" / "norm_stats.json").is_file()


def test_save_checkpoint_can_omit_optimizer_for_evaluation(tmp_path):
    source = tmp_path / "source"
    stats = source / "assets" / "physical-intelligence" / "libero"
    stats.mkdir(parents=True)
    (source / "config.json").write_text(json.dumps({"action_dim": 32}))
    (stats / "norm_stats.json").write_text(json.dumps({"state": {}, "actions": {}}))
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())

    checkpoint = save_checkpoint(model, optimizer, 3, tmp_path / "output", source, include_optimizer=False)

    assert (checkpoint / "model.safetensors").is_file()
    assert not (checkpoint / "optimizer.pt").exists()
    assert json.loads((checkpoint / "metadata.json").read_text())["resumable"] is False


def test_resume_restores_step_and_optimizer_state(tmp_path):
    source = tmp_path / "source"
    stats = source / "assets" / "physical-intelligence" / "libero"
    stats.mkdir(parents=True)
    (source / "config.json").write_text(json.dumps({"action_dim": 32}))
    (stats / "norm_stats.json").write_text(json.dumps({"state": {}, "actions": {}}))

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    checkpoint = save_checkpoint(model, optimizer, 12, tmp_path / "output", source)

    resumed_model = torch.nn.Linear(2, 2)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=5e-5)
    restore_optimizer(resumed_optimizer, checkpoint)

    assert read_resume_step(checkpoint) == 12
    assert resumed_optimizer.state_dict()["state"]
    assert resumed_optimizer.state_dict()["param_groups"] == optimizer.state_dict()["param_groups"]


def test_resume_rejects_non_resumable_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text(json.dumps({"step": 10, "resumable": False}))

    with pytest.raises(ValueError, match="non-resumable"):
        read_resume_step(checkpoint)


def test_warm_resume_does_not_require_optimizer_file(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text(json.dumps({"step": 15132, "resumable": True}))

    assert read_resume_step(checkpoint, require_optimizer=False) == 15132


def test_gradient_accumulation_cli_defaults_to_one():
    args = finetune.build_parser().parse_args(
        ["--checkpoint", "model", "--tokenizer", "tokenizer", "--output-dir", "output", "--steps", "1"]
    )

    assert args.gradient_accumulation_steps == 1


def test_gradient_accumulation_cli_accepts_override():
    args = finetune.build_parser().parse_args(
        [
            "--checkpoint",
            "model",
            "--tokenizer",
            "tokenizer",
            "--output-dir",
            "output",
            "--steps",
            "1",
            "--gradient-accumulation-steps",
            "4",
        ]
    )

    assert args.gradient_accumulation_steps == 4


def test_optimizer_checkpoint_cli_can_be_disabled():
    args = finetune.build_parser().parse_args(
        [
            "--checkpoint",
            "model",
            "--tokenizer",
            "tokenizer",
            "--output-dir",
            "output",
            "--steps",
            "1",
            "--no-save-optimizer",
        ]
    )

    assert args.save_optimizer is False


def test_evaluate_checkpoint_runs_separate_libero_environment(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.touch()
    eval_python = tmp_path / "libero-python"
    eval_python.touch()
    openpi_root = tmp_path / "openpi"
    openpi_root.mkdir()
    calls = {}

    class FakeServer:
        def terminate(self):
            calls["terminated"] = True

        def wait(self, timeout=None):
            calls["wait_timeout"] = timeout
            return 0

    def fake_popen(command, **kwargs):
        calls["server_command"] = command
        calls["server_kwargs"] = kwargs
        return FakeServer()

    def fake_run(command, **kwargs):
        calls["evaluator_command"] = command
        calls["evaluator_kwargs"] = kwargs
        evaluation_dir = pathlib.Path(command[command.index("--output-dir") + 1])
        evaluation_dir.mkdir(parents=True)
        (evaluation_dir / "summary.json").write_text(
            json.dumps({"success_rate": 0.75, "successes": 15, "episodes": 20})
        )

    monkeypatch.setattr(finetune.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(finetune.subprocess, "run", fake_run)
    monkeypatch.setattr(finetune, "_wait_for_server", lambda *args: None)

    summary = finetune.evaluate_checkpoint(
        checkpoint,
        tokenizer,
        500,
        tmp_path / "training",
        eval_python=eval_python,
        openpi_root=openpi_root,
        suite="libero_spatial",
        task_id=0,
        episodes=20,
        save_video=True,
        port=8123,
        server_timeout=60,
        device="cuda",
        pi05=False,
        eval_workers=20,
        max_batch_size=20,
        batch_wait_ms=10,
    )

    assert summary["success_rate"] == 0.75
    assert calls["terminated"]
    assert calls["evaluator_command"][0] == str(eval_python)
    assert "--save-video" in calls["evaluator_command"]
    evaluation_dir = calls["evaluator_command"][calls["evaluator_command"].index("--output-dir") + 1]
    assert pathlib.Path(evaluation_dir).is_absolute()
    assert calls["evaluator_command"][calls["evaluator_command"].index("--episodes-per-task") + 1] == "20"
    assert calls["server_command"][calls["server_command"].index("--port") + 1] == "8123"
    assert calls["server_command"][calls["server_command"].index("--max-batch-size") + 1] == "20"
    assert calls["evaluator_command"][calls["evaluator_command"].index("--workers") + 1] == "20"
