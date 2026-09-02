import json

import numpy as np
import torch

from post_train_vla.finetune import save_checkpoint
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
