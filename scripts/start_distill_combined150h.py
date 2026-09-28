"""Preflight long inputs, then build fresh 150h teacher targets and start distillation."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/majestic_student7m_combined150h_20260924'
CACHE = ROOT / 'data/distill7m_teacher_combined150h_20260924'
SMOKE_DATA = ROOT / 'data/distill7m_combined150h_preflight'
PYTHON = str(ROOT / '.venv/bin/python')


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def status(state, **extra):
    write(RUN / 'preparation_status.json', dict(status=state, pid=os.getpid(), updated_at=time.time(), **extra))
    print(state, flush=True)


def launch_workers(cache, prepared, teacher, env):
    cache.mkdir(parents=True, exist_ok=True)
    children = []
    for rank in range(4):
        command = [PYTHON, '-u', '-m', 'distillation.dump_teacher', '--output', str(cache),
                   '--teacher', str(teacher), '--prepared-dir', str(prepared),
                   '--rank', str(rank), '--world-size', '4']
        with (cache / f'rank{rank}.log').open('a') as log:
            child = subprocess.Popen(command, cwd=RUN / 'source', env=dict(env, CUDA_VISIBLE_DEVICES=str(rank)),
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
    write(cache / 'processes.json', {'processes': [{'rank': rank, 'pid': p.pid} for rank, p in enumerate(children)]})
    return children


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN / 'preparation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (RUN / 'launch_receipt.json').exists():
        raise RuntimeError('Already launched; inspect existing supervisor before restarting')
    children = []
    try:
        config = json.loads((ROOT / 'configs/distill_7m_combined150h.json').read_text())
        prepared = ROOT / config['prepared_directory']
        teacher = ROOT / config['teacher_export']
        audit = json.loads((prepared / 'preparation_report.json').read_text())
        if not audit['passed']:
            raise ValueError('Source data audit failed')
        for name, expected in audit['manifest_sha256'].items():
            if hashlib.sha256((prepared / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Source data changed: {name}')
        source = RUN / 'source'
        if not source.exists():
            source.mkdir()
            for folder in ['distillation', 'training', 'vendor', 'configs']:
                shutil.copytree(ROOT / folder, source / folder, ignore=shutil.ignore_patterns('__pycache__'))
            write(RUN / 'config.json', config)
            shutil.copy2(ROOT / 'docs/distillation_7m_plan.md', RUN / 'plan.md')
            write(RUN / 'source_manifest.json', {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in source.rglob('*') if p.is_file()})
            with (RUN / 'environment.txt').open('w') as out:
                subprocess.run([PYTHON, '-m', 'pip', 'freeze'], stdout=out, check=True)
        elif json.loads((RUN / 'config.json').read_text()) != config:
            raise ValueError('Frozen configuration differs')
        env = dict(os.environ, KOKORO_PROJECT_ROOT=str(ROOT), PYTHONPATH=str(source),
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                   TOKENIZERS_PARALLELISM='false', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        smoke = RUN / 'preflight'
        smoke.mkdir(exist_ok=True)
        status('preflight_teacher_generation')
        children = launch_workers(smoke / 'cache', SMOKE_DATA, teacher, env)
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError('Preflight teacher worker failed')
            time.sleep(5)
        if any(p.returncode != 0 for p in children):
            raise RuntimeError('Preflight teacher worker failed')
        children = []
        with (RUN / 'preflight.log').open('a') as log:
            subprocess.run([PYTHON, '-m', 'distillation.prepare_cache', '--cache', str(smoke / 'cache'),
                '--prepared-dir', str(SMOKE_DATA), '--teacher', str(teacher)], cwd=source, env=env,
                stdout=log, stderr=subprocess.STDOUT, check=True)
            smoke_config = dict(config, gan_warmup=0, log_every=1)
            write(smoke / 'config.json', smoke_config)
            status('preflight_student_four_gpu')
            subprocess.run([PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                '-m', 'distillation.train', '--config', str(smoke / 'config.json'), '--run-dir', str(smoke),
                '--cache', str(smoke / 'cache'), '--max-steps', '2', '--skip-validation'],
                cwd=source, env=dict(env, CUDA_VISIBLE_DEVICES='0,1,2,3'), stdout=log,
                stderr=subprocess.STDOUT, check=True)
        report = {'passed': True, 'scope': '192 duration/token extremes, 12 validation targets, 4 GPUs batch16; two student steps with GAN active',
                  'status': json.loads((smoke / 'status.json').read_text()),
                  'gradients': json.loads((smoke / 'first_step_gradients.json').read_text()),
                  'teacher_cache': json.loads((smoke / 'cache/ready.json').read_text())}
        write(ROOT / 'reports/distill7m_combined150h_preflight.json', report)
        shutil.rmtree(smoke)
        shutil.rmtree(SMOKE_DATA)
        status('starting_full_teacher_cache')
        children = launch_workers(CACHE, prepared, teacher, env)
        with (RUN / 'supervisor.log').open('a') as log:
            supervisor = subprocess.Popen([PYTHON, '-u', '-m', 'distillation.run', '--run-dir', str(RUN),
                '--cache', str(CACHE), '--config', str(RUN / 'config.json')], cwd=source, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(RUN / 'launch_receipt.json', {'supervisor_pid': supervisor.pid,
            'cache_worker_pids': [p.pid for p in children], 'cache': str(CACHE), 'started_at': time.time()})
        children = []  # The distillation supervisor now checks the recorded worker PIDs.
        status('teacher_cache_started', supervisor_pid=supervisor.pid, cache=str(CACHE))
    except BaseException as exc:
        for child in children:
            if child.poll() is None:
                child.terminate()
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
