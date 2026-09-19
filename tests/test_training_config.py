import argparse
import pathlib

import pytest
import yaml

from post_train_vla.training_config import add_config_argument, parse_args_with_config, write_resolved_config


def _parser():
    parser = argparse.ArgumentParser()
    add_config_argument(parser)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--models", nargs="+", choices=("pi0", "pi05"), default=("pi0",))
    parser.add_argument("--enabled", action=argparse.BooleanOptionalAction, default=False)
    return parser


def test_yaml_config_supplies_required_values_and_cli_overrides(tmp_path):
    config = tmp_path / "run.yaml"
    config.write_text(
        "checkpoint: /tmp/base\nbatch_size: 32\nmodels: [pi0, pi05]\nenabled: true\n"
    )

    args = parse_args_with_config(_parser(), ["--config", str(config), "--batch-size", "8"])

    assert args.checkpoint == pathlib.Path("/tmp/base")
    assert args.batch_size == 8
    assert args.models == ["pi0", "pi05"]
    assert args.enabled is True


def test_yaml_config_rejects_unknown_keys(tmp_path):
    config = tmp_path / "run.yaml"
    config.write_text("checkpoint: /tmp/base\nunknown_hparam: 1\n")

    with pytest.raises(ValueError, match="Unknown training config keys"):
        parse_args_with_config(_parser(), ["--config", str(config)])


def test_resolved_config_records_paths_and_defaults(tmp_path):
    args = parse_args_with_config(_parser(), ["--checkpoint", "/tmp/base"])

    resolved = write_resolved_config(tmp_path, args)
    saved = yaml.safe_load((tmp_path / "run_config.yaml").read_text())

    assert resolved == saved
    assert saved["arguments"]["checkpoint"] == "/tmp/base"
    assert saved["arguments"]["batch_size"] == 1
