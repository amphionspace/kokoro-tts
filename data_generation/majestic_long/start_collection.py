"""Own synthesis/QC processes until quotas pass; never launch model training."""
import fcntl,json,os,signal,subprocess,time,urllib.request
from pathlib import Path
from common import ROOT,ASSETS,CODE,REPO,config,atomic_json,connection,totals,goals


def main():
    lock=(ROOT/'collection.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=config();env=dict(os.environ,MAJESTIC_LONG_ROOT=str(ROOT),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    workers={};logs=[]
    def stop(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        while not (ROOT/'text/import_complete.json').exists():
            atomic_json(ROOT/'collection_status.json',dict(status='waiting_for_text_preparation',pid=os.getpid(),updated_at=time.time()));time.sleep(5)
        for url in cfg['tts_endpoints']:
            with urllib.request.urlopen(url+'/health',timeout=10) as response:assert response.status==200
        db=connection()
        # Refuse to reset leases when another worker still owns the collection.
        previous=ROOT/'workers.json'
        if previous.exists():
            for rec in json.loads(previous.read_text()).values():
                p=Path(f"/proc/{rec['pid']}/cmdline")
                if p.exists() and str(CODE/'run_workers.py').encode() in p.read_bytes():raise RuntimeError('Previous worker alive')
        for busy,ready in [('synthesizing','queued'),('asr_running','synthesized'),('metrics_running','asr_done')]:db.execute('update candidates set state=?,lease_until=0 where state=?',(ready,busy))
        for kind,gpu in [('feeder',None),('synth',None),('asr','2'),('metrics','3,0')]:
            python=ASSETS/'.venv-voxcpm2/bin/python' if kind!='metrics' else Path('/119010446/UltraEval-Audio/envs/metrics/bin/python')
            childenv=dict(env)
            if gpu is not None:childenv['CUDA_VISIBLE_DEVICES']=gpu
            if kind=='metrics':childenv['PYTHONPATH']=cfg['dnsmos_runtime_path']+':'+env.get('PYTHONPATH','')
            log=(ROOT/'logs'/f'{kind}.log').open('a');logs.append(log)
            workers[kind]=subprocess.Popen([str(python),'-u',str(CODE/'run_workers.py'),kind],cwd=REPO,env=childenv,stdout=log,stderr=subprocess.STDOUT)
        atomic_json(ROOT/'workers.json',{k:{'pid':p.pid} for k,p in workers.items()})
        while True:
            for kind,p in workers.items():
                if p.poll() is not None:raise RuntimeError(f'{kind} exited {p.returncode}')
            atomic_json(ROOT/'collection_status.json',dict(status='synthesizing_and_filtering',pid=os.getpid(),updated_at=time.time(),workers={k:p.pid for k,p in workers.items()}))
            if (ROOT/'collection_ready.json').exists():break
            time.sleep(10)
        atomic_json(ROOT/'collection_status.json',dict(status='quotas_met_pending_final_audit',updated_at=time.time()))
    except BaseException as exc:
        atomic_json(ROOT/'collection_status.json',dict(status='stopped',error=repr(exc),updated_at=time.time()));raise
    finally:
        for p in workers.values():
            if p.poll() is None:p.terminate()
        for p in workers.values():
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for log in logs:log.close()
    subprocess.run([str(REPO/'.venv/bin/python'),str(CODE/'audit.py')],env=env,cwd=REPO,check=True)
    atomic_json(ROOT/'collection_status.json',dict(status='complete',updated_at=time.time(),report=str(ROOT/'reports/final_audit.json')))
    # Stop only services owned by this collection and recorded with a matching command.
    for record in json.loads((ROOT/'servers.json').read_text()).values():
        pid=record['pid'];p=Path(f'/proc/{pid}/cmdline')
        if p.exists() and b'824' in p.read_bytes() and os.getpgid(pid)==pid:os.killpg(pid,signal.SIGTERM)

if __name__=='__main__':main()
