"""Wait for current student completion and prepared texts, then use four GPUs."""
import fcntl,hashlib,json,os,signal,socket,subprocess,time,urllib.request
from pathlib import Path
from common import ROOT,REPO,CODE,atomic_json,config

TRAIN=REPO/'runs/majestic_student7m_boundary_20260922'


def active_training():
    meta=json.loads((TRAIN/'processes.json').read_text())
    pid=meta['training_pid'];path=Path(f'/proc/{pid}/cmdline')
    return path.exists() and b'torch.distributed.run' in path.read_bytes() and str(TRAIN).encode() in path.read_bytes()


def main():
    lock=(ROOT/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'reports/final_audit.json').exists():return
    cfg=config();env=dict(os.environ,MAJESTIC_LONG_ROOT=str(ROOT),MAJESTIC_VOICE_DATA_ROOT=str(ROOT),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
    servers={};logs=[]
    try:
        while True:
            training_done=(TRAIN/'training_complete.json').exists() and not active_training()
            texts_done=(ROOT/'text/import_complete.json').exists() and (ROOT/'reports/text_preflight.json').exists()
            atomic_json(ROOT/'launch_status.json',dict(status='waiting',training_complete=training_done,texts_complete=texts_done,pid=os.getpid(),updated_at=time.time()))
            if training_done and texts_done:break
            time.sleep(10)
        # Freeze the exact operational source/config before starting services.
        snapshot=ROOT/'source';snapshot.mkdir(exist_ok=True)
        import shutil
        for p in CODE.iterdir():
            if p.is_file():shutil.copy2(p,snapshot/p.name)
        atomic_json(ROOT/'source_manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.iterdir() if p.is_file()})
        atomic_json(ROOT/'launch_status.json',dict(status='starting_services',pid=os.getpid(),updated_at=time.time()))
        owned=json.loads((ROOT/'servers.json').read_text()) if (ROOT/'servers.json').exists() else {}
        for gpu,url in enumerate(cfg['tts_endpoints']):
            port=int(url.rsplit(':',1)[1])
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1',port))==0:
                    record=owned.get(str(gpu),{});cmd=Path(f"/proc/{record.get('pid',0)}/cmdline")
                    if record.get('port')!=port or not cmd.exists() or b'VoxCPM2' not in cmd.read_bytes() or str(port).encode() not in cmd.read_bytes():
                        raise RuntimeError(f'Port {port} occupied by unowned service')
                    servers[str(gpu)]=record
                    continue
            log=(ROOT/'logs'/f'tts_gpu{gpu}.log').open('a');logs.append(log)
            p=subprocess.Popen(['bash',str(CODE/'serve.sh'),str(gpu),str(port)],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            servers[str(gpu)]={'pid':p.pid,'port':port,'command':['bash',str(CODE/'serve.sh'),str(gpu),str(port)]}
        atomic_json(ROOT/'servers.json',servers)
        deadline=time.monotonic()+1200;pending=set(cfg['tts_endpoints'])
        while pending:
            for url in list(pending):
                try:
                    with urllib.request.urlopen(url+'/health',timeout=3) as response:
                        if response.status==200:pending.remove(url)
                except Exception:pass
            if time.monotonic()>deadline:raise RuntimeError(f'TTS services not healthy: {pending}')
            if pending:time.sleep(5)
        subprocess.run([str(REPO/'.venv/bin/python'),str(CODE/'pilot.py')],cwd=REPO,env=env,check=True)
        atomic_json(ROOT/'launch_status.json',dict(status='collecting',pid=os.getpid(),updated_at=time.time()))
        for attempt in range(4):
            with (ROOT/'logs/supervisor.log').open('a') as log:
                p=subprocess.Popen([str(REPO/'.venv/bin/python'),'-u',str(CODE/'start_collection.py')],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
                atomic_json(ROOT/'supervisor_process.json',dict(pid=p.pid,attempt=attempt,started_at=time.time()))
                code=p.wait()
            if code==0:break
            if attempt==3:raise RuntimeError('Collection failed after three recovery attempts; inspect logs')
            time.sleep(30)
        atomic_json(ROOT/'launch_status.json',dict(status='complete',updated_at=time.time()))
    except BaseException as exc:
        atomic_json(ROOT/'launch_status.json',dict(status='failed',error=repr(exc),updated_at=time.time()));raise
    finally:
        for f in logs:f.close()

if __name__=='__main__':main()
