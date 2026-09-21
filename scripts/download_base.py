"""Download the pinned general Kokoro checkpoint and verify upstream SHA256."""
import hashlib
import json
from pathlib import Path
import urllib.request

root = Path(__file__).resolve().parents[1]
base = json.loads((root / 'configs/project.json').read_text())['base_model']
assert base['repo_id'] == 'hexgrad/Kokoro-82M'
expected = '496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4'
folder = root / 'models' / 'Kokoro-82M'
folder.mkdir(parents=True, exist_ok=True)
for name in ('config.json', base['checkpoint']):
    target = folder / name
    temporary = target.with_suffix(target.suffix + '.part')
    url = f"https://huggingface.co/{base['repo_id']}/resolve/{base['revision']}/{name}"
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
            digest.update(chunk)
    if name.endswith('.pth') and digest.hexdigest() != expected:
        raise ValueError('Checkpoint SHA256 does not match official model card')
    temporary.replace(target)
    print(name, digest.hexdigest(), flush=True)
