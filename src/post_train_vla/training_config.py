"""YAML launch configuration and provenance helpers for training entrypoints."""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    """Add the shared YAML config option to a training parser."""
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        help="YAML file whose top-level keys match CLI argument names; explicit CLI options take precedence",
    )


def parse_args_with_config(parser: argparse.ArgumentParser, argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse arguments after applying values from an optional flat YAML mapping."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=pathlib.Path)
    config_args, _ = config_parser.parse_known_args(argv)
    values: dict[str, Any] = {}
    if config_args.config is not None:
        values = _load_yaml_mapping(config_args.config)
        _validate_and_normalize_config(parser, values)
        parser.set_defaults(**values)
        for action in parser._actions:
            if action.required and action.dest in values:
                action.required = False

    args = parser.parse_args(argv)
    if args.config is not None:
        args.config = args.config.expanduser().resolve()
    return args


def write_resolved_config(output_dir: pathlib.Path, args: argparse.Namespace) -> dict[str, Any]:
    """Persist the resolved launch arguments, including defaults and CLI overrides."""
    import yaml

    output_dir.mkdir(parents=True, exist_ok=True)
    arguments = {
        key: _json_safe(value)
        for key, value in vars(args).items()
        if key != "config"
    }
    resolved = {
        "schema_version": 1,
        "config_source": str(args.config) if getattr(args, "config", None) else None,
        "command_line": list(sys.argv[1:]),
        "arguments": arguments,
    }
    (output_dir / "run_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True))
    return resolved


def _load_yaml_mapping(path: pathlib.Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("YAML configuration requires PyYAML; install the training extra") from exc

    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Training config not found: {path}")
    contents = yaml.safe_load(path.read_text())
    if contents is None:
        return {}
    if not isinstance(contents, Mapping) or not all(isinstance(key, str) for key in contents):
        raise ValueError("Training config must be a YAML mapping with string argument names")
    return dict(contents)


def _validate_and_normalize_config(parser: argparse.ArgumentParser, values: dict[str, Any]) -> None:
    actions = {action.dest: action for action in parser._actions}
    unknown = sorted(set(values).difference(actions))
    if unknown:
        raise ValueError(f"Unknown training config keys: {unknown}")
    if "config" in values:
        raise ValueError("Do not set 'config' inside a training config")

    for key, value in list(values.items()):
        action = actions[key]
        if action.type is not None and value is not None:
            if action.nargs in ("+", "*") or isinstance(action.nargs, int):
                if not isinstance(value, list):
                    raise ValueError(f"Training config key '{key}' must be a YAML list")
                value = [action.type(item) for item in value]
            else:
                value = action.type(value)
        if action.choices is not None:
            choices = set(action.choices)
            items = value if isinstance(value, list) else [value]
            invalid = [item for item in items if item not in choices]
            if invalid:
                raise ValueError(f"Invalid value for training config key '{key}': {invalid}")
        values[key] = value


def _json_safe(value: Any) -> Any:
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value
