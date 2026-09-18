"""Publish immutable epoch artifacts and verify their remote content identities."""
import hashlib
import json
from pathlib import Path
import subprocess
import time



def digest(path):
    with path.open('rb') as f:
        hasher = hashlib.sha256()
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            hasher.update(chunk)
        return hasher.hexdigest()


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
    files = sorted(p for p in folder.rglob('*') if p.is_file() and '.cache' not in p.parts
                   and p.name != 'publication.json')
    checks = {str(p.relative_to(folder)): digest(p) for p in files if p.name != 'artifact_checksums.json'}
    (folder / 'artifact_checksums.json').write_text(json.dumps(checks, indent=2)+'\n')
    files = sorted(set(files + [folder / 'artifact_checksums.json']))
    for attempt in range(5):
        try:
            subprocess.run(['hf', 'upload', repo, str(folder), prefix, '--exclude', '.cache/**',
                            '--exclude', 'publication.json', '--commit-message', f'Save {prefix}'], check=True)
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
