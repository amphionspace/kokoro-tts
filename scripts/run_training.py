"""Supervise the two DDP stages using a frozen source/config snapshot."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def write(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/train.json')
    p.add_argument('--stage2-config',type=Path,default=ROOT/'configs/stage2.json')
    p.add_argument('--stage2-resume-from',type=Path,help='Resume Stage 2 before LR decay with an extended epoch budget in a new run directory')
    args=p.parse_args()
    run=args.run_dir.resolve();run.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(run/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=run/'source'
    if not source.exists() and not args.stage2_resume_from:
        source.mkdir()
        for directory in ('training','vendor','scripts'):
            shutil.copytree(ROOT/directory,source/directory,ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(args.config,run/'config.json')
    if not (run/'environment.txt').exists():
        with (run/'environment.txt').open('w') as stream:
            subprocess.run([str(ROOT/'.venv/bin/python'),'-m','pip','freeze'],stdout=stream,check=True)
    stage2_source=run/'source_stage2'
    if not stage2_source.exists():
        stage2_source.mkdir()
        for directory in ('training','vendor','scripts'):
            shutil.copytree(ROOT/directory,stage2_source/directory,ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(args.stage2_config,run/'config_stage2.json')
    evaluation_source=run/'source_evaluation'
    if not evaluation_source.exists():
        evaluation_source.mkdir()
        for directory in ('training','vendor','scripts'):
            shutil.copytree(ROOT/directory,evaluation_source/directory,ignore=shutil.ignore_patterns('__pycache__'))
    env=dict(os.environ,KOKORO_PROJECT_ROOT=str(ROOT),PYTHONPATH=str(source),
             CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
             OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    python=str(ROOT/'.venv/bin/python')
    write(run/'supervisor_status.json',{'status':'running','pid':os.getpid(),'started_at':time.time()})
    try:
        for stage in ((2,) if args.stage2_resume_from else (1,2)):
            if (run/f'stage{stage}_complete.json').exists():continue
            active_source=source if stage==1 else stage2_source
            active_config=run/('config.json' if stage==1 else 'config_stage2.json')
            stage_env=dict(env,PYTHONPATH=str(active_source))
            command=[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                     '-m','training.train','--config',str(active_config),'--run-dir',str(run),'--stage',str(stage)]
            checkpoints=sorted((run/'checkpoints').glob(f'stage{stage}_step_*.pth'))
            if checkpoints:command+=['--resume',str(checkpoints[-1])]
            elif stage==2 and args.stage2_resume_from:
                command+=['--resume',str(args.stage2_resume_from.resolve()),'--extend-stage2-budget']
            elif stage==2:command+=['--initialize',str(run/'checkpoints/stage1_final.pth')]
            with (run/f'stage{stage}.log').open('a',buffering=1) as stream:
                process=subprocess.Popen(command,cwd=active_source,env=stage_env,stdout=stream,stderr=subprocess.STDOUT)
                write(run/'supervisor_status.json',{'status':'running','pid':os.getpid(),'child_pid':process.pid,'stage':stage,'updated_at':time.time()})
                code=process.wait()
            if code:raise RuntimeError(f'Stage {stage} exited with code {code}; inspect stage{stage}.log')
        write(run/'training_complete.json',{'completed_at':time.time()})
        write(run/'supervisor_status.json',{'status':'complete','updated_at':time.time()})
    except Exception as exc:
        write(run/'supervisor_status.json',{'status':'failed','error':repr(exc),'updated_at':time.time()})
        raise

if __name__=='__main__':main()
