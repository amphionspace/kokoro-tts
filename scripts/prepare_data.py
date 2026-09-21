"""Export LITs MajesticVoice text/audio pairs without reusing LITs phoneme IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import unicodedata
import wave
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = Path('/119010446/tts-assets/training_runs/ljs_majestic_100h_backbone21k_20260914/data')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def validate_audio(row):
    audio = Path(row['audio'])
    with wave.open(str(audio)) as wav:
        if wav.getframerate() != 24000 or wav.getnchannels() != 1:
            raise ValueError(f'Expected mono 24k: {audio}')
        duration = wav.getnframes() / wav.getframerate()
    if abs(duration - row['duration']) > 0.05:
        raise ValueError(f'Duration mismatch: {audio}')
    return duration

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT)
    args = parser.parse_args()
    output = ROOT / 'data'
    output.mkdir(exist_ok=True)
    report = {'source': str(args.source), 'selection': 'speaker=1; Chinese, English, Mixed', 'splits': {}}
    seen_audio, seen_text, seen_group = {}, {}, {}
    prepared = {}
    for split in ('train', 'val', 'test'):
        source = args.source / f'{split}.jsonl'
        records, seconds = [], 0.0
        with source.open() as stream:
            for line_no, line in enumerate(stream, 1):
                row = json.loads(line)
                if row['speaker'] != 1 or row['language'] not in ('Chinese', 'English', 'Mixed'):
                    continue
                if row['split'] != split or not row['synthetic']:
                    raise ValueError(f'Unexpected provenance at {source}:{line_no}')
                audio = Path(row['audio'])
                if not audio.is_absolute():
                    raise ValueError('Audio path must be absolute')
                text = row['text'].strip()
                if not text:
                    raise ValueError('Empty transcript')
                text_key = ''.join(unicodedata.normalize('NFKC', text).split())
                for key, seen, label in ((str(audio), seen_audio, 'audio'), (text_key, seen_text, 'text'), (row['source_group'], seen_group, 'source_group')):
                    if key in seen and seen[key] != split:
                        raise ValueError(f'Cross-split {label}: {key}')
                    seen[key] = split
                duration = row['duration']
                records.append({'audio': str(audio), 'text': text, 'speaker': 'majestic', 'language': {'Chinese': 'zh', 'English': 'en', 'Mixed': 'mixed'}[row['language']], 'duration': duration, 'split': split, 'synthetic': True, 'source': row['source'], 'source_group': row['source_group'], 'source_manifest_line': line_no})
                seconds += duration
        with ThreadPoolExecutor(max_workers=16) as pool:
            actual_durations = list(pool.map(validate_audio, records))
        for row, duration in zip(records, actual_durations):
            row['duration'] = duration
        seconds = sum(actual_durations)
        prepared[split] = records
        print(f'Validated {split}: {len(records)} rows', flush=True)
        languages = {}
        for item in records:
            group = languages.setdefault(item['language'], {'rows': 0, 'hours': 0.0})
            group['rows'] += 1
            group['hours'] += item['duration'] / 3600
        report['splits'][split] = {'languages': languages, 'rows': len(records), 'hours': seconds / 3600, 'source_sha256': sha(source)}
    for split, records in prepared.items():
        target = output / f'{split}.jsonl'
        temporary = target.with_suffix('.jsonl.tmp')
        temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records))
        temporary.replace(target)
        report['splits'][split]['manifest_sha256'] = sha(target)
    protocol = json.loads((args.source / 'eval_protocol.json').read_text())
    report['reference_audio'] = protocol['group_speaker_references']['majestic_zh']
    report['reference_sha256'] = sha(Path(report['reference_audio']))
    if report['reference_sha256'] != protocol['group_reference_sha256']['majestic_zh']:
        raise ValueError('Reference checksum mismatch')
    report['group_references'] = {k: {'audio': v, 'sha256': sha(Path(v))} for k, v in protocol['group_speaker_references'].items() if k.startswith('majestic_')}
    for key, value in report['group_references'].items():
        if value['sha256'] != protocol['group_reference_sha256'][key]:
            raise ValueError(f'Reference checksum mismatch: {key}')
    report['checks'] = ['all selected WAV headers: mono 24k, duration', 'cross-split audio/text/source-group disjoint', 'reference SHA256']
    report['frontend_status'] = 'Raw text only; Kokoro/Misaki phonemization still required. LITs IDs intentionally excluded.'
    (ROOT / 'reports' / 'data_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
