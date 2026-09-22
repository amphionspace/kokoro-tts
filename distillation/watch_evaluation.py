"""Drain student checkpoints and select by full-set intelligibility, then CAMP."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
from training.common import ROOT,records,write_json
from torch.utils.tensorboard import SummaryWriter
PYTHONS={'synthesize':ROOT/'.venv/bin/python','asr':Path('/119010446/tts-assets/.venv-voxcpm2/bin/python'),'metrics':Path('/119010446/UltraEval-Audio/envs/metrics/bin/python'),'summarize':ROOT/'.venv/bin/python'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--gpu',default='3');a=p.parse_args();run=a.run_dir.resolve()
    lock=(run/'evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=run/'source';source=source if source.exists() else ROOT
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,KOKORO_PROJECT_ROOT=str(ROOT),PYTHONPATH=str(source),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    writer=SummaryWriter(str(run/'tensorboard/evaluation'))
    while True:
        pending=[]
        for job in sorted((run/'eval_queue').glob('*.json'),key=lambda p:(json.loads(p.read_text())['step'],p.name)):
            item=json.loads(job.read_text());out=run/'eval'/job.stem
            if (out/'complete.json').exists():continue
            pending.append(job)
            if (out/'failed.json').exists():continue
            out.mkdir(parents=True,exist_ok=True)
            try:
                write_json(run/'evaluation_status.json',{'status':'running','job':str(job),'updated_at':time.time()})
                for stage,python in PYTHONS.items():
                    command=[str(python),'-m','distillation.evaluate','--stage',stage,'--checkpoint',item['checkpoint'],'--output',str(out)]
                    if item['smoke'] and stage=='synthesize':command+=['--per-group-limit','8']
                    with (out/f'{stage}.log').open('w') as log:subprocess.run(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200)
                summary=json.loads((out/'summary.json').read_text());failures=sum(v['evaluation_failures'] for v in summary['groups'].values())
                for group,metrics in summary['groups'].items():
                    for key,value in metrics.items():
                        if isinstance(value,(float,int)):writer.add_scalar(f'{group}/{key}',value,item['step'])
                import soundfile as sf
                seen=set()
                for row in records(out/'synthesis.jsonl'):
                    if row['group'] in seen or 'audio' not in row:continue
                    audio,sr=sf.read(row['audio']);writer.add_audio('audio/'+row['group'],audio,item['step'],sample_rate=sr);seen.add(row['group'])
                writer.flush()
                if failures:raise RuntimeError(f'{failures} evaluation items failed')
                if not item['smoke']:
                    groups=summary['groups'];asr=(groups['majestic_zh']['cer_micro']+groups['majestic_en']['wer_micro']+groups['majestic_mixed']['cer_micro'])/3
                    camp=sum(v['camp_similarity_mean'] for v in groups.values())/3
                    best_path=run/'best.json';old=json.loads(best_path.read_text()) if best_path.exists() else None
                    candidate={'criterion':'min equal-language mean of zh CER, en WER, mixed CER; tie-break max mean CAMP; full 450 only','asr_score':asr,'mean_camp':camp,'step':item['step'],'checkpoint':item['checkpoint'],'export':str(out),'summary':summary}
                    if old is None or (asr,-camp)<(old['asr_score'],-old['mean_camp']):write_json(best_path,candidate)
                write_json(out/'complete.json',{'step':item['step'],'completed_at':time.time()})
                write_json(run/'evaluation_status.json',{'status':'complete','step':item['step'],'output':str(out),'updated_at':time.time()})
            except Exception as e:
                failure={'status':'failed','error':repr(e),'job':str(job),'updated_at':time.time()};write_json(out/'failed.json',failure);write_json(run/'evaluation_status.json',failure);print(json.dumps(failure),flush=True)
        if (run/'training_complete.json').exists() and not pending:writer.close();return
        time.sleep(20)

if __name__=='__main__':main()
