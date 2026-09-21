"""Prepare auditable phonemes without changing source splits or transcripts."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.common import digest, records, write_json


def initialize():
    import torch
    torch.set_num_threads(1)
    from training.frontend import Frontend
    global frontend
    frontend = Frontend()


def convert(row):
    row = dict(row)
    try:
        row['phonemes'], row['token_ids'] = frontend(row['text'], row['language'])
        if len(row['token_ids']) > 510:
            row['rejection'] = 'more_than_510_tokens_no_audio_truncation'
        elif row.get('duration', 2) < 1.05:
            row['rejection'] = 'less_than_1.05_seconds_style_encoder_crop'
    except Exception as exc:
        row['rejection'] = f'{type(exc).__name__}: {exc}'
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=12)
    args = p.parse_args()
    initialize()  # Ensure shared resources are installed before starting workers.
    report = {}
    output = ROOT / 'data/prepared'
    output.mkdir(exist_ok=True)
    with ProcessPoolExecutor(args.workers, initializer=initialize) as pool:
        for split in ('train', 'val', 'test'):
            source = ROOT / 'data' / f'{split}.jsonl'
            good, rejected = [], []
            for index, row in enumerate(pool.map(convert, records(source), chunksize=32)):
                (rejected if 'rejection' in row else good).append(row)
                if (index + 1) % 5000 == 0:
                    print(split, index + 1, flush=True)
            for suffix, rows in [('', good), ('.rejected', rejected)]:
                (output / f'{split}{suffix}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
            report[split] = {'accepted': len(good), 'rejected': len(rejected),
                             'hours': sum(r['duration'] for r in good)/3600,
                             'by_language': {lang: {'rows': sum(r['language']==lang for r in good),
                                                   'hours': sum(r['duration'] for r in good if r['language']==lang)/3600} for lang in ('zh','en','mixed')},
                             'reasons': dict(Counter(r['rejection'] for r in rejected)),
                             'source_sha256': digest(source), 'prepared_sha256': digest(output/f'{split}.jsonl')}
            print(json.dumps({split: report[split]},ensure_ascii=False),flush=True)
    # Keep LITs' evaluation texts and matching speaker references frozen.
    lits = Path('/119010446/tts-assets/training_runs/ljs_majestic_100h_backbone21k_20260914/data')
    evaluation = []
    for row in records(lits/'eval_manifest.jsonl'):
        if not row['group'].startswith('majestic_'):
            continue
        lang = {'majestic_zh':'zh','majestic_en':'en','majestic_mixed':'mixed'}[row['group']]
        ps, ids = frontend(row['ref_text'],lang)
        result = {k:v for k,v in row.items() if k not in ('token_ids','tone_ids')}
        result.update(phonemes=ps,token_ids=ids,language=lang)
        evaluation.append(result)
    (output/'eval_manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in evaluation))
    (output/'eval_protocol.json').write_bytes((lits/'eval_protocol.json').read_bytes())
    report['evaluation']={'rows':len(evaluation),'source_sha256':digest(lits/'eval_manifest.jsonl')}
    write_json(ROOT/'reports/frontend_audit.json',report)
    if any('Unknown phoneme' in key for v in report.values() for key in v.get('reasons',{})):
        raise RuntimeError('Unknown phonemes quarantined; audit and resolve before training')


if __name__ == '__main__':
    main()
