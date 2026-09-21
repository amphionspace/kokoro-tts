"""Drain checkpoint evaluations on a GPU shared with DDP, with explicit failures."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.common import write_json
from torch.utils.tensorboard import SummaryWriter

PYTHONS={'synthesize':ROOT/'.venv/bin/python',
         'asr':Path('/119010446/tts-assets/.venv-voxcpm2/bin/python'),
         'metrics':Path('/119010446/UltraEval-Audio/envs/metrics/bin/python'),
         'summarize':ROOT/'.venv/bin/python'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--gpu',type=int,default=3)
    args=p.parse_args();run=args.run_dir.resolve()
    import fcntl
    lock=(run/'evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=run/'source_evaluation' if (run/'source_evaluation').exists() else ROOT
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(source),KOKORO_PROJECT_ROOT=str(ROOT))
    writer=SummaryWriter(str(run/'tensorboard/evaluation'))
    while True:
        jobs=sorted((run/'eval_queue').glob('*.json'),key=lambda path:(json.loads(path.read_text())['global_step'],path.name))
        pending=[]
        for job in jobs:
            out=run/'eval'/job.stem
            if (out/'complete.json').exists():continue
            pending.append(job)
            if (out/'failed.json').exists():continue
            out.mkdir(parents=True,exist_ok=True)
            item=json.loads(job.read_text())
            try:
                write_json(run/'evaluation_status.json',{'status':'running','job':str(job),'updated_at':time.time()})
                for stage,python in PYTHONS.items():
                    command=[str(python),'-m','training.evaluate','--stage',stage,'--checkpoint',item['checkpoint'],'--output',str(out)]
                    if stage=='synthesize' and item['smoke']:command+=['--per-group-limit','8']
                    with (out/f'{stage}.log').open('w') as log:
                        subprocess.run(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200)
                summary=json.loads((out/'summary.json').read_text())
                failures=sum(value['evaluation_failures'] for value in summary['groups'].values())
                for group,metrics in summary['groups'].items():
                    for key,value in metrics.items():
                        if isinstance(value,(float,int)):writer.add_scalar(f'{group}/{key}',value,item['global_step'])
                import soundfile as sf
                synthesis=[json.loads(line) for line in (out/'synthesis.jsonl').read_text().splitlines()]
                seen=set()
                for row in synthesis:
                    if row['group'] in seen or 'audio' not in row:continue
                    audio,sr=sf.read(row['audio']);writer.add_audio('audio/'+row['group'],audio,item['global_step'],sample_rate=sr)
                    seen.add(row['group'])
                writer.flush()
                if failures:raise RuntimeError(f'{failures} evaluation items failed; see per-stage logs')
                write_json(out/'complete.json',{'global_step':item['global_step'],'completed_at':time.time()})
                write_json(run/'evaluation_status.json',{'status':'complete','output':str(out),'global_step':item['global_step'],'updated_at':time.time()})
            except Exception as exc:
                error={'status':'failed','job':str(job),'error':repr(exc),'updated_at':time.time()}
                write_json(out/'failed.json',error);write_json(run/'evaluation_status.json',error)
                print(json.dumps(error),flush=True)
        if (run/'training_complete.json').exists() and not pending:return
        time.sleep(20)


if __name__=='__main__':main()
