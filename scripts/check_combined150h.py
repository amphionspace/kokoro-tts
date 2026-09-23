"""Exercise both DDP stages on the longest and most token-dense merged samples."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = ROOT / 'data/prepared_combined150h_20260923'
    run = ROOT / 'runs/majestic_combined150h_preflight_20260923'
    run.mkdir(parents=True, exist_ok=True)
    smoke = run / 'data'
    smoke.mkdir(exist_ok=True)
    rows = [json.loads(line) for line in (data / 'train.jsonl').read_text().splitlines()]
    selected = sorted(rows, key=lambda r: r['duration'], reverse=True)[:64]
    selected += sorted(rows, key=lambda r: len(r['token_ids']), reverse=True)[:64]
    (smoke / 'train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
    for name in ('val.jsonl', 'voicepack_references.jsonl'):
        shutil.copy2(data / name, smoke / name)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
               KOKORO_PROJECT_ROOT=str(ROOT), PYTHONPATH=str(ROOT),
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    report = {'passed': False, 'samples': len(selected), 'max_seconds': max(r['duration'] for r in selected),
              'max_tokens': max(len(r['token_ids']) for r in selected), 'stages': {}}
    for stage, name in [(1, 'train_combined150h.json'), (2, 'stage2_combined150h.json')]:
        config = json.loads((ROOT / 'configs' / name).read_text())
        config.update(prepared_directory=str(smoke), require_data_audit=False, log_every=1)
        if stage == 2:
            config.update(stage2_joint_step=0, voicepack_start_step=0)
        path = run / f'config_stage{stage}.json'
        path.write_text(json.dumps(config, indent=2) + '\n')
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                   '-m', 'training.train', '--config', str(path), '--run-dir', str(run),
                   '--stage', str(stage), '--max-steps', '2', '--skip-validation']
        if stage == 2:
            command += ['--initialize', str(run / 'checkpoints/stage1_step_00000002.pth')]
        print('Checking stage', stage, flush=True)
        with (run / f'stage{stage}.log').open('w') as log:
            subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        report['stages'][str(stage)] = json.loads((run / 'status.json').read_text())
    report['passed'] = True
    (ROOT / 'reports/combined150h_preflight.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
