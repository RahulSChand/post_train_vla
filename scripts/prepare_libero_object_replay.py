#!/usr/bin/env python3
"""Download a reproducible LIBERO-Object-only replay subset from the Hub."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


SOURCE_REPO = "physical-intelligence/libero"
SOURCE_REVISION = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"
OBJECT_TASKS = {
    f"pick up the {name} and place it in the basket"
    for name in (
        "alphabet soup",
        "cream cheese",
        "salad dressing",
        "bbq sauce",
        "ketchup",
        "tomato sauce",
        "butter",
        "milk",
        "chocolate pudding",
        "orange juice",
    )
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hf", type=Path, default=Path("hf"))
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    metadata = args.metadata_root / "meta"
    info = json.loads((metadata / "info.json").read_text())
    episodes = [json.loads(line) for line in (metadata / "episodes.jsonl").read_text().splitlines()]
    selected = [episode for episode in episodes if episode["tasks"][0] in OBJECT_TASKS]
    represented = {episode["tasks"][0] for episode in selected}
    if represented != OBJECT_TASKS:
        raise ValueError(f"Missing object tasks: {sorted(OBJECT_TASKS - represented)}")

    data_files = [
        info["data_path"].format(
            episode_chunk=int(episode["episode_index"]) // int(info["chunks_size"]),
            episode_index=int(episode["episode_index"]),
        )
        for episode in selected
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(args.hf),
            "download",
            SOURCE_REPO,
            *data_files,
            "--repo-type",
            "dataset",
            "--revision",
            SOURCE_REVISION,
            "--local-dir",
            str(args.output_dir),
            "--max-workers",
            str(args.max_workers),
        ],
        check=True,
    )

    output_meta = args.output_dir / "meta"
    output_meta.mkdir(exist_ok=True)
    subset_info = dict(info)
    subset_info.update(
        total_episodes=len(selected),
        total_frames=sum(int(episode["length"]) for episode in selected),
        splits={"train": f"0:{len(selected)}"},
    )
    (output_meta / "info.json").write_text(json.dumps(subset_info, indent=2, sort_keys=True) + "\n")
    (output_meta / "episodes.jsonl").write_text(
        "".join(json.dumps(episode, sort_keys=True) + "\n" for episode in selected)
    )
    for name in ("tasks.jsonl", "stats.json"):
        (output_meta / name).write_bytes((metadata / name).read_bytes())

    missing = [relative for relative in data_files if not (args.output_dir / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Subset download is missing {len(missing)} files")
    manifest = {
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "selection": "all episodes for the ten canonical LIBERO-Object tasks",
        "tasks": sorted(OBJECT_TASKS),
        "episodes": len(selected),
        "frames": subset_info["total_frames"],
        "data_files": data_files,
        "episodes_jsonl_sha256": hashlib.sha256(
            (output_meta / "episodes.jsonl").read_bytes()
        ).hexdigest(),
    }
    (args.output_dir / "subset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: manifest[key] for key in ("episodes", "frames", "source_revision")}, indent=2))


if __name__ == "__main__":
    main()
