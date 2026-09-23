"""Wait for the audited merge, run full-batch preflight, and launch both stages."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/majestic_combined150h_20260923'


def status(state, **extra):
    value = dict(status=state, pid=os.getpid(), updated_at=time.time(), **extra)
    path = RUN / 'preparation_status.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)
    print(json.dumps(value), flush=True)


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN / 'preparation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (RUN / 'launch_receipt.json').exists():
        raise RuntimeError('Launch already recorded; inspect existing run before retrying')
    python = str(ROOT / '.venv/bin/python')
    try:
        deadline = time.monotonic() + 7200
        report_path = ROOT / 'data/prepared_combined150h_20260923/preparation_report.json'
        while not report_path.exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Data audit/merge did not finish within two hours')
            status('waiting_for_audited_merge')
            time.sleep(15)
        report = json.loads(report_path.read_text())
        if not report.get('passed'):
            raise ValueError('Merged data audit failed')
        # The collection supervisor stops its own servers immediately after audit.
        servers = json.loads((ROOT / 'data/majestic_long50h/servers.json').read_text())
        while any(Path(f'/proc/{record["pid"]}/cmdline').exists()
                  and b'vllm' in Path(f'/proc/{record["pid"]}/cmdline').read_bytes()
                  for record in servers.values()):
            if time.monotonic() > deadline:
                raise TimeoutError('Synthesis servers did not exit')
            status('waiting_for_synthesis_shutdown')
            time.sleep(5)
        status('checking_both_stages', training_data=report['train'])
        with (RUN / 'preflight.log').open('a') as log:
            subprocess.run([python, '-u', str(ROOT / 'scripts/check_combined150h.py')],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        preflight = json.loads((ROOT / 'reports/combined150h_preflight.json').read_text())
        if not preflight.get('passed'):
            raise ValueError('Training preflight failed')
        command = [python, '-u', str(ROOT / 'scripts/run_training.py'), '--run-dir', str(RUN),
                   '--config', str(ROOT / 'configs/train_combined150h.json'),
                   '--stage2-config', str(ROOT / 'configs/stage2_combined150h.json')]
        receipt = RUN / 'launch_receipt.json'
        receipt.write_text(json.dumps({'status': 'launching', 'command': command, 'time': time.time()}, indent=2) + '\n')
        with (RUN / 'supervisor.log').open('a') as log:
            train = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        # Wait until source_evaluation has been copied before starting its watcher.
        while not (RUN / 'supervisor_status.json').exists():
            if train.poll() is not None:
                raise RuntimeError(f'Training supervisor exited {train.returncode}')
            time.sleep(1)
        with (RUN / 'evaluation.log').open('a') as log:
            evaluator = subprocess.Popen([python, '-u', str(ROOT / 'scripts/watch_evaluation.py'),
                '--run-dir', str(RUN), '--gpu', '3'], cwd=ROOT, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        receipt.write_text(json.dumps({'status': 'launched', 'command': command, 'time': time.time(),
            'training_pid': train.pid, 'evaluation_pid': evaluator.pid}, indent=2) + '\n')
        status('training_launched', training_pid=train.pid, evaluation_pid=evaluator.pid)
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
