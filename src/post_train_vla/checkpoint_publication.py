"""Publish immutable epoch artifacts and verify their remote content identities."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import time



def digest(path):
    with path.open('rb') as f:
        hasher = hashlib.sha256()
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            hasher.update(chunk)
        return hasher.hexdigest()


def publication_files(folder):
    """Return only files explicitly emitted by the inference-checkpoint saver."""
    folder = Path(folder)
    manifest_path = folder / 'checkpoint_manifest.json'
    if not manifest_path.is_file():
        raise ValueError(f"Missing checkpoint manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('format') != 'groot-inference-checkpoint-v1':
        raise ValueError(f"Unsupported checkpoint manifest format: {manifest.get('format')!r}")
    names = manifest.get('files')
    if not isinstance(names, list) or not names:
        raise ValueError("Checkpoint manifest must contain a nonempty files list")
    if len(names) != len(set(names)):
        raise ValueError("Checkpoint manifest contains duplicate paths")
    files = []
    for name in names:
        if not isinstance(name, str):
            raise ValueError(f"Unsafe checkpoint path: {name!r}")
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f"Unsafe checkpoint path: {name!r}")
        path = folder.joinpath(*relative.parts)
        if not path.is_file():
            raise ValueError(f"Manifest file is missing: {name}")
        files.append(path)
    if not any(path.name == 'config.json' for path in files):
        raise ValueError("Checkpoint manifest is missing config.json")
    if not any(path.suffix == '.safetensors' for path in files):
        raise ValueError("Checkpoint manifest contains no safetensors weights")
    return sorted(files + [manifest_path])


def publish(folder, *, repo, prefix):
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
    if not repo or not prefix or prefix.startswith("/") or ".." in prefix.split("/"):
        raise ValueError("An explicit repository and relative destination prefix are required")
    prefix = prefix.rstrip("/")
    if prefix in {".", ""}:
        raise ValueError("Use an explicit subdirectory prefix")
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {folder}")
    files = publication_files(folder)
    checks = {str(p.relative_to(folder)): digest(p) for p in files}
    (folder / 'artifact_checksums.json').write_text(json.dumps(checks, indent=2)+'\n')
    files = sorted(files + [folder / 'artifact_checksums.json'])
    relative_files = [str(path.relative_to(folder)) for path in files]
    for attempt in range(5):
        try:
            subprocess.run(['hf', 'upload', repo, str(folder), prefix, '--include', *relative_files,
                            '--commit-message', f'Save {prefix}'], check=True)
            api = HfApi()
            revision = api.model_info(repo).sha
            for start in range(0, len(files), 50):
                group = files[start:start+50]
                paths = [prefix+'/'+str(p.relative_to(folder)) for p in group]
                remote = {r.path: r for r in api.get_paths_info(repo, paths, revision=revision)}
                for path, local in zip(paths, group):
                    r = remote[path]
                    if r.size != local.stat().st_size:
                        raise ValueError(f"Remote size mismatch: {path}")
                    if r.lfs:
                        if r.lfs.sha256 != digest(local):
                            raise ValueError(f"Remote SHA-256 mismatch: {path}")
                    else:
                        blob = local.read_bytes()
                        if r.blob_id != hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest():
                            raise ValueError(f"Remote Git hash mismatch: {path}")
                    if local.suffix == '.safetensors':
                        meta = get_hf_file_metadata(hf_hub_url(repo, path, revision=revision))
                        if meta.size != local.stat().st_size:
                            raise ValueError(f"Download size mismatch: {path}")
            receipt = dict(repo=repo, prefix=prefix, revision=revision, verified_files=len(files),
                           sha256=checks, verified_at=time.time())
            (folder/'publication.json').write_text(json.dumps(receipt,indent=2)+'\n')
            return receipt
        except Exception:
            if attempt == 4:
                raise
            time.sleep(10)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.folder, repo=args.repo, prefix=args.prefix)))


if __name__ == '__main__':
    main()
