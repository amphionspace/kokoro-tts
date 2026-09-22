"""Freeze a distillation run, wait for a complete teacher cache, and train/evaluate."""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--config',type=Path,default=ROOT/'configs/distill_7m.json');p.add_argument('--tensorboard-port',type=int,default=0,help='0: use existing shared TensorBoard on 32003; positive: launch standalone');a=p.parse_args()
    run=a.run_dir.resolve();cache=a.cache.resolve();run.mkdir(parents=True,exist_ok=True)
    lock=(run/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=run/'source';python=str(ROOT/'.venv/bin/python')
    if not source.exists():
        source.mkdir()
        for folder in ['distillation','training','vendor','configs']:
            shutil.copytree(ROOT/folder,source/folder,ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(a.config,run/'config.json')
        shutil.copy2(ROOT/'docs/distillation_7m_plan.md',run/'plan.md')
        hashes={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
        write(run/'source_manifest.json',hashes)
        with (run/'environment.txt').open('w') as f:subprocess.run([python,'-m','pip','freeze'],stdout=f,check=True)
    env=dict(os.environ,KOKORO_PROJECT_ROOT=str(ROOT),PYTHONPATH=str(source),CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    try:
        while not all((cache/f'complete_rank{r}.json').exists() for r in range(4)):
            progress={}
            for rank in range(4):
                path=cache/f'status_rank{rank}.json'
                if path.exists():progress[rank]=json.loads(path.read_text())
            write(run/'supervisor_status.json',{'status':'waiting_for_teacher_cache','pid':os.getpid(),'progress':progress,'updated_at':time.time()})
            processes=ROOT/'runs/distill7m_teacher_cache/processes.json'
            if processes.exists():
                for record in json.loads(processes.read_text())['processes']:
                    if (cache/f"complete_rank{record['rank']}.json").exists():continue
                    proc=Path(f"/proc/{record['pid']}/cmdline")
                    if not proc.exists() or b'distillation.dump_teacher' not in proc.read_bytes():raise RuntimeError(f"Teacher cache worker {record['rank']} exited before completion")
            time.sleep(15)
        with (run/'cache_audit.log').open('w') as f:subprocess.run([python,'-m','distillation.prepare_cache','--cache',str(cache)],cwd=source,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        shutil.copy2(cache/'ready.json',run/'teacher_cache_audit.json')
        processes={'supervisor_pid':os.getpid(),'started_at':time.time()}
        processes.update(tensorboard_pid=None,tensorboard_port=32003,tensorboard_mode='shared')
        if a.tensorboard_port > 0:
            with (run/'tensorboard.log').open('a') as f:
                tb=subprocess.Popen([python,'-m','tensorboard.main','--logdir',str(run/'tensorboard'),'--host','0.0.0.0','--port',str(a.tensorboard_port)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            processes.update(tensorboard_pid=tb.pid,tensorboard_port=a.tensorboard_port)
        with (run/'evaluation.log').open('a') as f:
            evaluator=subprocess.Popen([python,'-u','-m','distillation.watch_evaluation','--run-dir',str(run),'--gpu','3'],cwd=source,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        processes['evaluation_pid']=evaluator.pid
        if not (run/'training_complete.json').exists():
            command=[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=4','-m','distillation.train','--config',str(run/'config.json'),'--run-dir',str(run),'--cache',str(cache)]
            latest=run/'latest_checkpoint.json'
            if latest.exists():command+=['--resume',json.loads(latest.read_text())['path']]
            with (run/'train.log').open('a') as log:
                train=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
                processes['training_pid']=train.pid;write(run/'processes.json',processes)
                write(run/'supervisor_status.json',{'status':'training','pid':os.getpid(),'child_pid':train.pid,'updated_at':time.time()})
                code=train.wait()
            if code:raise RuntimeError(f'Training exited {code}; inspect train.log')
        write(run/'supervisor_status.json',{'status':'training_complete','pid':os.getpid(),'evaluation_pid':evaluator.pid,'updated_at':time.time()})
    except Exception as e:
        write(run/'supervisor_status.json',{'status':'failed','error':repr(e),'updated_at':time.time()});raise

if __name__=='__main__':main()
