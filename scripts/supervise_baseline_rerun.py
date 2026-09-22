"""Poll an existing run, then launch a frozen full baseline rerun exactly once."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def locked(path):
    if not path.exists():
        return False
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def ready(run):
    return (all((run / name).exists() for name in
                ('training_complete.json', 'stage2_complete.json'))
            and read(run / 'supervisor_status.json').get('status') == 'complete'
            and not locked(run / 'training.lock')
            and not locked(run / 'supervisor.lock'))


def launch(command, log, cwd):
    with log.open('a') as stream:
        return subprocess.Popen(command, cwd=cwd, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True).pid


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    plan = read(args.plan)
    current, target = Path(plan['current_run']), Path(plan['target_run'])
    root = Path(plan['project_root'])
    if args.check_only:
        print(json.dumps({'ready': ready(current), 'current': read(current / 'status.json')}))
        return
    lock = (target / 'handoff.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        state = {'checked_at': time.time(), 'pid': os.getpid(),
                 'current_progress': read(current / 'status.json'),
                 'current_supervisor': read(current / 'supervisor_status.json'),
                 'interval_seconds': plan['interval_seconds']}
        launched = (target / 'launch_receipt.json').exists()
        if launched:
            state['rerun_progress'] = read(target / 'status.json')
            state['rerun_supervisor'] = read(target / 'supervisor_status.json')
            state['status'] = 'rerun_complete' if (target / 'training_complete.json').exists() else 'rerun_running'
            if state['rerun_supervisor'].get('status') == 'failed':
                state['status'] = 'rerun_failed'
        elif ready(current):
            # Persist launch intent before spawning: restarting this monitor cannot
            # accidentally start another full run if interrupted during launch.
            write(target / 'launch_receipt.json', {'status': 'launching', 'time': time.time()})
            python = str(root / '.venv/bin/python')
            train_pid = launch([python, '-u', str(target / 'runner.py'),
                                '--run-dir', str(target), '--config', str(target / 'config.json'),
                                '--stage2-config', str(target / 'config_stage2.json')],
                               target / 'supervisor.log', root)
            eval_pid = launch([python, '-u', str(target / 'evaluator.py'), '--run-dir', str(target), '--gpu', '3'],
                              target / 'evaluation.log', root)
            write(target / 'launch_receipt.json', {'status': 'launched', 'time': time.time(),
                                                  'training_pid': train_pid, 'evaluation_pid': eval_pid})
            state['status'] = 'rerun_launched'
        else:
            state['status'] = 'waiting_for_current_completion'
            if state['current_supervisor'].get('status') == 'failed':
                state['status'] = 'current_failed_waiting'
        write(target / 'handoff_status.json', state)
        with (target / 'progress_checks.jsonl').open('a') as stream:
            stream.write(json.dumps(state) + '\n')
        print(json.dumps(state), flush=True)
        if state['status'] == 'rerun_complete':
            return
        time.sleep(plan['interval_seconds'])


if __name__ == '__main__':
    main()
