#!/usr/bin/env python3
"""Create success-rate plots from one or more JSON summaries and a plot config.

The input summaries can use a root list, a ``results`` list, an ``evaluations``
list, or a combined ``suites`` object.  The script normalizes the common field
names used by the experiment exports before plotting them.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator, PercentFormatter


DEFAULT_COLORS = ["#2563EB", "#F97316", "#16A34A", "#DC2626", "#7C3AED", "#0891B2"]
DEFAULT_MODEL_STYLES = {
    "pi0": {"label": "π₀", "linestyle": "--"},
    "pi05": {"label": "π₀.₅", "linestyle": "-"},
}
DEFAULT_VARIANT_STYLES = {
    "reference": {"label": "Reference", "marker": "o", "linewidth": 2.4},
    "seed726": {"label": "Training seed 726", "marker": "D", "linewidth": 3.1},
}
SUITE_NAMES = {
    "libero_spatial": "LIBERO-Spatial",
    "libero_goal": "LIBERO-Goal",
    "libero_object": "LIBERO-Object",
    "libero_10": "LIBERO-Long",
}


def nested_value(value: dict[str, Any], path: str) -> Any:
    """Return a dotted-path value, or ``None`` when a path is absent."""
    current: Any = value
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def first_value(value: dict[str, Any], paths: tuple[str, ...]) -> Any:
    for path in paths:
        candidate = nested_value(value, path)
        if candidate is not None:
            return candidate
    return None


def normalize_model(model: Any) -> str:
    text = str(model).strip().lower().replace("_", "")
    if text in {"pi0", "π0", "π₀"}:
        return "pi0"
    if text in {"pi05", "pi0.5", "π0.5", "π₀.₅"}:
        return "pi05"
    return str(model).strip()


def result_lists(payload: Any) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return ``(root_metadata, record)`` pairs from supported JSON layouts."""
    if isinstance(payload, list):
        return [({}, record) for record in payload]
    if not isinstance(payload, dict):
        raise ValueError("Expected the JSON root to be an object or list")
    if isinstance(payload.get("evaluations"), list):
        return [(payload, record) for record in payload["evaluations"]]
    if isinstance(payload.get("results"), list):
        return [(payload, record) for record in payload["results"]]
    if isinstance(payload.get("checkpoint_summaries"), list):
        return [(payload, record) for record in payload["checkpoint_summaries"]]
    if isinstance(payload.get("checkpoints"), dict):
        return [
            (payload, record)
            for record in payload["checkpoints"].values()
            if isinstance(record, dict)
        ]
    if isinstance(payload.get("suites"), dict):
        pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for suite, suite_payload in payload["suites"].items():
            if not isinstance(suite_payload, dict) or not isinstance(suite_payload.get("results"), list):
                continue
            root = {**payload, **suite_payload, "suite": suite}
            pairs.extend((root, record) for record in suite_payload["results"])
        return pairs
    raise ValueError("Could not find a results, evaluations, or suites result list")


def resolve_path(config_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def extract_input(
    config_path: Path, input_config: dict[str, Any]
) -> tuple[list[dict[str, Any]], set[str]]:
    path = resolve_path(config_path, input_config["path"])
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    extracted: list[dict[str, Any]] = []
    suites: set[str] = set()
    for root, result in result_lists(payload):
        if not isinstance(result, dict):
            continue
        model = input_config["model"] if "model" in input_config else first_value(
            result, ("model", "policy_metadata.model")
        ) or first_value(root, ("model",))
        trajectory = input_config["trajectory"] if "trajectory" in input_config else first_value(
            result,
            ("trajectory_count", "trajectory_budget", "trajectories", "source.trajectory_count"),
        )
        x_path = input_config.get("x_path")
        x_value = input_config["x"] if "x" in input_config else first_value(
            result, (x_path,) if x_path else ("epoch", "source.epoch")
        )
        rate_percent = first_value(result, ("success_rate_percent",))
        if rate_percent is None:
            rate = first_value(result, ("success_rate", "overall.success_rate"))
            rate_percent = None if rate is None else float(rate) * 100
        episodes = first_value(result, ("episodes", "overall.episodes"))
        suite = input_config.get("suite") or first_value(result, ("suite",)) or first_value(root, ("suite",))

        missing = [
            label
            for label, value in (("model", model), ("trajectory", trajectory), ("x value", x_value), ("success rate", rate_percent))
            if value is None
        ]
        if missing:
            raise ValueError(f"{path.name}: could not infer {', '.join(missing)}; add an input override in the config")

        x_number = float(x_value)
        if "min_x" in input_config and x_number < float(input_config["min_x"]):
            continue
        if "max_x" in input_config and x_number > float(input_config["max_x"]):
            continue

        extracted.append(
            {
                "model": normalize_model(model),
                "trajectory": int(trajectory),
                "x": x_number,
                "rate": float(rate_percent),
                "episodes": int(episodes) if episodes is not None else None,
                "variant": input_config.get("variant", "reference"),
            }
        )
        if suite is not None:
            suites.add(str(suite))
    return extracted, suites


def model_style(model: str, config: dict[str, Any]) -> dict[str, Any]:
    custom = config.get("models", {}).get(model, {})
    default = DEFAULT_MODEL_STYLES.get(model, {"label": model, "linestyle": "-"})
    return {**default, **custom}


def variant_style(variant: str, config: dict[str, Any]) -> dict[str, Any]:
    custom = config.get("variants", {}).get(variant, {})
    default = DEFAULT_VARIANT_STYLES.get(
        variant, {"label": variant, "marker": "o", "linewidth": 2.4}
    )
    return {**default, **custom}


def default_title(suites: set[str]) -> str:
    if not suites:
        return "Success Rate by Training Epoch"
    names = [SUITE_NAMES.get(suite, suite) for suite in sorted(suites)]
    return f"{' + '.join(names)}: Success Rate by Training Epoch"


def default_subtitle(records: list[dict[str, Any]]) -> str:
    models = {record["model"] for record in records}
    variants = {record["variant"] for record in records}
    pieces = ["Color = number of training trajectories"]
    if {"pi0", "pi05"}.issubset(models):
        pieces.append("solid = π₀.₅ · dashed = π₀")
    if "seed726" in variants:
        pieces.append("diamonds = seed 726")
    return " · ".join(pieces)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to a plot configuration JSON")
    parser.add_argument("--output", type=Path, help="Override the config's output path")
    args = parser.parse_args()

    config_path = args.config.resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config.get("inputs"), list) or not config["inputs"]:
        raise ValueError("Config needs a non-empty inputs list")

    records: list[dict[str, Any]] = []
    suites: set[str] = set()
    for input_config in config["inputs"]:
        input_records, input_suites = extract_input(config_path, input_config)
        records.extend(input_records)
        suites.update(input_suites)
    if not records:
        raise ValueError("No plot records were extracted")

    grouped: dict[tuple[str, int, str], list[tuple[float, float]]] = defaultdict(list)
    for record in records:
        grouped[(record["model"], record["trajectory"], record["variant"])].append(
            (record["x"], record["rate"])
        )
    trajectory_counts = sorted({record["trajectory"] for record in records})
    custom_colors = config.get("colors", config.get("trajectory_colors", {}))
    color_by = config.get("color_by", "trajectory")
    if color_by not in {"trajectory", "model"}:
        raise ValueError("color_by must be trajectory or model")
    color_values: list[Any] = trajectory_counts if color_by == "trajectory" else sorted(
        {record["model"] for record in records}
    )
    colors = {
        value: custom_colors.get(str(value), DEFAULT_COLORS[index % len(DEFAULT_COLORS)])
        for index, value in enumerate(color_values)
    }
    x_values = [record["x"] for record in records]
    min_x, max_x = min(x_values), max(x_values)
    y_max = float(config.get("y_max", 100))
    models = sorted({record["model"] for record in records})
    variants = sorted({record["variant"] for record in records})

    figure_size = config.get("figure_size", [11.5, 7])
    fig, ax = plt.subplots(figsize=figure_size, dpi=int(config.get("dpi", 200)))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FAFAFA")
    mark_best_points = config.get("mark_best_points", False)

    for (model, trajectory, variant), points in sorted(
        grouped.items(), key=lambda item: (item[0][2] != "reference", item[0][0], item[0][1])
    ):
        points.sort()
        model_props = model_style(model, config)
        variant_props = variant_style(variant, config)
        ax.plot(
            [x_value for x_value, _ in points],
            [rate for _, rate in points],
            color=colors[trajectory if color_by == "trajectory" else model],
            linestyle=model_props.get("linestyle", "-"),
            linewidth=float(variant_props.get("linewidth", 2.4)),
            marker=variant_props.get("marker", "o"),
            markersize=float(variant_props.get("markersize", 6)),
            markeredgecolor="white",
            markeredgewidth=1.05,
            zorder=4 if variant != "reference" else 2,
        )
        if mark_best_points:
            best_x, best_rate = max(points, key=lambda point: point[1])
            best_color = colors[trajectory if color_by == "trajectory" else model]
            ax.scatter(
                best_x,
                best_rate,
                marker="*",
                s=float(config.get("best_point_size", 180)),
                color=best_color,
                edgecolor="white",
                linewidth=1.1,
                zorder=6,
            )
            ax.annotate(
                f"{best_rate:g}%",
                (best_x, best_rate),
                xytext=(0, float(config.get("best_point_label_offset", 11))),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=float(config.get("best_point_label_size", 9.5)),
                fontweight="bold",
                color=best_color,
                zorder=7,
            )

    x_axis = config.get("x_axis", "epoch")
    if "x_limits" in config:
        ax.set_xlim(*config["x_limits"])
    else:
        default_padding = 0.25 if x_axis == "epoch" else max((max_x - min_x) * 0.05, 1)
        ax.set_xlim(min_x - float(config.get("x_padding", default_padding)), max_x + float(config.get("x_padding", default_padding)))
    ax.set_ylim(0, y_max)
    if "x_ticks" in config:
        ax.set_xticks(config["x_ticks"])
    elif x_axis == "epoch":
        ax.set_xticks(range(1, int(max_x) + 1))
    elif "x_tick_step" in config:
        ax.xaxis.set_major_locator(MultipleLocator(float(config["x_tick_step"])))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.yaxis.set_major_locator(MultipleLocator(float(config.get("y_tick_step", 10))))
    y_decimals = 1 if y_max <= 1 else 0
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=y_decimals))
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#94A3B8")
    ax.tick_params(colors="#475569")
    ax.set_xlabel(config.get("x_label", "Training epoch"), fontsize=11, labelpad=10)
    ax.set_ylabel(config.get("y_label", "Success rate"), fontsize=11, labelpad=10)

    color_legend = ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=colors[value],
                linewidth=3,
                label=str(value) if color_by == "trajectory" else model_style(str(value), config).get("label", str(value)),
            )
            for value in color_values
        ],
        title=config.get(
            "color_legend_title",
            "Training trajectories" if color_by == "trajectory" else "Model",
        ),
        frameon=False,
        ncols=len(color_values),
        loc="upper left",
        bbox_to_anchor=(0, 1.005),
        handlelength=2.2,
        columnspacing=1.4,
    )
    ax.add_artist(color_legend)
    show_model_legend = config.get("show_model_legend", color_by != "model")
    if len(models) > 1 and show_model_legend:
        model_legend = ax.legend(
            handles=[
                Line2D(
                    [0], [0], color="#334155", linestyle=model_style(model, config).get("linestyle", "-"),
                    linewidth=2.6, label=model_style(model, config).get("label", model),
                )
                for model in models
            ],
            title="Model",
            frameon=False,
            ncols=len(models),
            loc="upper right",
            bbox_to_anchor=(1, 1.005),
            handlelength=2.5,
            columnspacing=1.5,
        )
        ax.add_artist(model_legend)
    if len(variants) > 1:
        ax.legend(
            handles=[
                Line2D(
                    [0], [0], color="#334155", marker=variant_style(variant, config).get("marker", "o"),
                    linewidth=0, markersize=7, label=variant_style(variant, config).get("label", variant),
                )
                for variant in variants
            ],
            title="Run configuration",
            frameon=False,
            ncols=len(variants),
            loc="upper right",
            bbox_to_anchor=(1, 0.91 if len(models) > 1 else 1.005),
            columnspacing=1.4,
        )

    fig.suptitle(
        config.get("title", default_title(suites)),
        x=0.08,
        y=0.985,
        ha="left",
        fontsize=16.5,
        fontweight="bold",
        color="#0F172A",
    )
    ax.set_title(
        config.get("subtitle", default_subtitle(records)),
        loc="left",
        fontsize=10.5,
        color="#64748B",
        pad=58,
    )
    if "footnote" in config:
        footnote = config["footnote"]
    else:
        episode_counts = sorted({record["episodes"] for record in records if record["episodes"] is not None})
        footnote = " / ".join(map(str, episode_counts)) + " evaluation episodes per checkpoint"
    fig.text(0.985, 0.02, footnote, ha="right", fontsize=8.5, color="#64748B")

    output = args.output or config.get("output")
    if output is None:
        raise ValueError("Provide output in the config or use --output")
    output_path = resolve_path(config_path, str(output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.78)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved {output_path}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
