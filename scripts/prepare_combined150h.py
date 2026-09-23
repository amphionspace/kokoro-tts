"""Freeze audited long synthesis plus the existing prepared 100h for training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unicodedata
import wave

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def norm(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).lower()
                   if unicodedata.category(c)[0] in ('L', 'N'))


def stats(rows):
    return {'rows': len(rows), 'hours': sum(r['duration'] for r in rows) / 3600,
            'by_language': {lang: {'rows': sum(r['language'] == lang for r in rows),
                'hours': sum(r['duration'] for r in rows if r['language'] == lang) / 3600}
                for lang in ('zh', 'en', 'mixed')}}


def audio_check(row):
    with wave.open(row['audio']) as wav:
        if wav.getframerate() != 24000 or wav.getnchannels() != 1:
            raise ValueError(f"Not mono 24k: {row['audio']}")
        duration = wav.getnframes() / 24000
        if abs(duration - row['duration']) > .001:
            raise ValueError(f"Duration mismatch: {row['audio']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--long-root', type=Path, default=ROOT / 'data/majestic_long50h')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/prepared_combined150h_20260923')
    args = parser.parse_args()
    old = ROOT / 'data/prepared'
    long = args.long_root.resolve()
    audit_path = long / 'reports/final_audit.json'
    audit = json.loads(audit_path.read_text())
    status = json.loads((long / 'collection_status.json').read_text())
    if not audit.get('passed') or status.get('status') != 'complete':
        raise ValueError('Long synthesis must finish its final audit first')
    if args.output.exists():
        raise FileExistsError(args.output)
    base = {split: read(old / f'{split}.jsonl') for split in ('train', 'val', 'test')}
    new = read(long / 'accepted/train.jsonl')
    for lang, target in audit['targets'].items():
        selected = [r for r in new if r['language'] == lang]
        hours = sum(r['duration'] for r in selected) / 3600
        if len(selected) != audit['counts'][lang] or abs(hours - audit['hours'][lang]) > 1e-6 or hours < target:
            raise ValueError(f'Final audit/manifest mismatch: {lang}')
    vocab = json.loads((ROOT / 'models/Kokoro-82M/config.json').read_text())['vocab']
    seen = {key: {} for key in ('audio', 'text', 'phones', 'source_group')}
    for split, rows in base.items():
        for row in rows:
            for key, value in [('audio', str(Path(row['audio']).resolve())), ('text', norm(row['text'])),
                               ('phones', tuple(row['token_ids'])), ('source_group', row.get('source_group'))]:
                if value is not None:
                    seen[key].setdefault(value, set()).add(split)
    for row in new:
        if row['split'] != 'train' or not row.get('synthetic') or not row.get('quality'):
            raise ValueError('Invalid synthesis provenance')
        if not 15 <= row['duration'] <= 45 or row['sentence_count'] < 3:
            raise ValueError('Invalid long sample')
        for key, value in [('audio', str(Path(row['audio']).resolve())), ('text', norm(row['text'])),
                           ('phones', tuple(row['token_ids'])), ('source_group', row.get('source_group'))]:
            if value is None:
                raise ValueError(f'Missing {key}')
            previous = seen[key].get(value, set())
            if previous and (key != 'source_group' or previous - {'train'}):
                raise ValueError(f'New sample overlaps {key}: {row["id"]} / {previous}')
            seen[key].setdefault(value, set()).add('train')
        row['speaker'] = 'majestic'
        row['training_collection'] = long.name
    combined = base['train'] + new
    for row in combined + base['val'] + base['test']:
        if not 0 < len(row['token_ids']) <= 510 or [vocab[p] for p in row['phonemes']] != row['token_ids']:
            raise ValueError(f"Invalid frontend IDs: {row['audio']}")
    print('Validating all audio headers', len(combined) + len(base['val']) + len(base['test']), flush=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(audio_check, combined + base['val'] + base['test']))
    references = read(old / 'voicepack_references.jsonl')
    if not {r['audio'] for r in references} <= {r['audio'] for r in base['train']}:
        raise ValueError('Voice references must belong to training')
    staging = Path(tempfile.mkdtemp(prefix=args.output.name + '.tmp.', dir=args.output.parent))
    try:
        with (staging / 'train.jsonl').open('w') as out:
            for row in combined:
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
        for name in ('val.jsonl', 'test.jsonl', 'eval_manifest.jsonl', 'eval_protocol.json', 'voicepack_references.jsonl'):
            shutil.copy2(old / name, staging / name)
        report = {'passed': True, 'original': stats(base['train']), 'added': stats(new),
                  'train': stats(combined), 'val': stats(base['val']), 'test': stats(base['test']),
                  'audio_headers_checked': len(combined) + len(base['val']) + len(base['test']),
                  'new_data_overlap': False, 'heldout_and_evaluation_unchanged': True,
                  'long_audit': str(audit_path), 'long_audit_sha256': digest(audit_path),
                  'input_sha256': {str(p): digest(p) for p in [old / 'train.jsonl', long / 'accepted/train.jsonl']},
                  'manifest_sha256': {p.name: digest(p) for p in staging.iterdir()}}
        (staging / 'preparation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        staging.rename(args.output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
