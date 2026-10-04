import json

import pytest

from post_train_vla.checkpoint_publication import publication_files


def test_publication_files_uses_manifest_allowlist(tmp_path):
    checkpoint = tmp_path / "epoch-001"
    checkpoint.mkdir()
    for name in (
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "processor_config.json",
        "statistics.json",
        "embodiment_id.json",
        "epoch.json",
    ):
        (checkpoint / name).write_text(name)
    (checkpoint / "runtime").mkdir()
    (checkpoint / "runtime/gr00t.py").write_text("must not upload")
    (checkpoint / "campaign_code").mkdir()
    (checkpoint / "campaign_code/train.py").write_text("must not upload")
    allowed = sorted(
        path.name for path in checkpoint.iterdir() if path.is_file()
    )
    (checkpoint / "checkpoint_manifest.json").write_text(
        json.dumps({"format": "groot-inference-checkpoint-v1", "files": allowed})
    )

    selected = {str(path.relative_to(checkpoint)) for path in publication_files(checkpoint)}

    assert "checkpoint_manifest.json" in selected
    assert "runtime/gr00t.py" not in selected
    assert "campaign_code/train.py" not in selected
    assert len(selected) == 9


def test_publication_files_rejects_path_traversal(tmp_path):
    checkpoint = tmp_path / "epoch-001"
    checkpoint.mkdir()
    (checkpoint / "checkpoint_manifest.json").write_text(
        json.dumps({"format": "groot-inference-checkpoint-v1", "files": ["../secret"]})
    )

    with pytest.raises(ValueError, match="Unsafe checkpoint path"):
        publication_files(checkpoint)
