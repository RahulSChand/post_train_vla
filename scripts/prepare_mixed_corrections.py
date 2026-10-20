#!/usr/bin/env python3
"""Create a symlinked correction dataset with episode/frame provenance."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        nargs=2,
        metavar=("NAME", "CORRECTION_DIR"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    correction_dir = args.output_dir / "corrections"
    correction_dir.mkdir(parents=True, exist_ok=True)
    records = []
    seen_names = set()
    for source_name, source_value in args.source:
        source = Path(source_value).resolve()
        for path in sorted(source.glob("*.npz")):
            if path.name in seen_names:
                raise ValueError(f"Duplicate correction filename: {path.name}")
            seen_names.add(path.name)
            with np.load(path, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata_json"]))
                frames = len(payload["expert_actions"])
            destination = correction_dir / path.name
            if destination.is_symlink() or destination.exists():
                if destination.resolve() != path:
                    raise ValueError(f"Existing destination points elsewhere: {destination}")
            else:
                os.symlink(path, destination)
            records.append(
                {
                    "source": source_name,
                    "path": str(path),
                    "filename": path.name,
                    "frames": frames,
                    "rollout_id": metadata.get("rollout_id"),
                    "source_policy": metadata.get("source_policy"),
                    "canonical_task_id": metadata.get("canonical_task_id"),
                }
            )

    summary = {
        "episodes": len(records),
        "frames": sum(record["frames"] for record in records),
        "episodes_by_source": dict(sorted(Counter(record["source"] for record in records).items())),
        "frames_by_source": dict(
            sorted(
                (source, sum(record["frames"] for record in records if record["source"] == source))
                for source, _ in args.source
            )
        ),
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps({"summary": summary, "episodes": records}, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
