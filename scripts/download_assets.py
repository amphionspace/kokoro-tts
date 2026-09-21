"""Fetch pinned training assets, or verify existing local files without network."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    assets = json.loads((ROOT / 'provenance/model_assets.json').read_text())
    for asset in assets:
        target = ROOT / asset['path']
        if target.exists():
            if sha256(target) != asset['sha256']:
                raise ValueError(f'Hash mismatch: {target}; existing file left unchanged')
            print('verified', asset['path'])
            continue
        if args.verify_only:
            raise FileNotFoundError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + '.part')
        try:
            with urllib.request.urlopen(asset['url'], timeout=60) as source, partial.open('wb') as dest:
                while chunk := source.read(1024 * 1024):
                    dest.write(chunk)
            if partial.stat().st_size != asset['bytes'] or sha256(partial) != asset['sha256']:
                raise ValueError(f'Download does not match manifest: {target}')
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
        print('downloaded', asset['path'])


if __name__ == '__main__':
    main()
