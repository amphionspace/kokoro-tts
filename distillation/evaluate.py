"""Export the 7M student and reuse the unchanged LITs evaluation protocol."""
import argparse,json,time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from training.common import ROOT,records,write_json,digest
from distillation.model import Student


def synthesize(a):
    torch.set_num_threads(2);torch.manual_seed(20260922)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=Student(checkpoint['architecture']).cuda().eval();model.load_state_dict(checkpoint['model'],strict=True)
    voice=checkpoint['voice'].cuda()
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    model.export(out/'kokoro.pth');torch.save(voice.cpu().unsqueeze(0).expand(510,1,256).clone(),out/'majestic.pt')
    write_json(out/'config.json',checkpoint['architecture'])
    write_json(out/'export.json',{'parameters':sum(p.numel() for p in model.parameters()),'step':checkpoint['step'],'checkpoint_sha256':digest(a.checkpoint),'conditioning':'frozen_teacher_majestic_voice','voice_sha256':checkpoint['identity']['voice'],'loader':'distillation.model.Student; strict nested loading','student_initialization':checkpoint['config']['initial_weights'] or 'random'})
    counts={}
    with (out/'synthesis.jsonl').open('w',buffering=1) as stream,torch.inference_mode():
        for source in records(ROOT/'data/prepared/eval_manifest.jsonl'):
            group=source['group'];counts.setdefault(group,0)
            if a.per_group_limit and counts[group]>=a.per_group_limit:continue
            counts[group]+=1;row={k:v for k,v in source.items() if k not in ['phonemes','token_ids']};started=time.time()
            try:
                audio,duration=model.synthesize(source['token_ids'],voice)
                wave=audio.float().cpu().numpy()
                if not np.isfinite(wave).all():raise ValueError('Nonfinite student audio')
                target=out/'wavs'/(source['id']+'.wav');target.parent.mkdir(parents=True,exist_ok=True)
                sf.write(target,np.clip(wave,-1,1),24000,subtype='PCM_16')
                row.update(audio=str(target),duration=len(wave)/24000,duration_frames=duration.cpu().tolist(),peak=float(np.abs(wave).max()),clipped_fraction=float(np.mean(np.abs(wave)>1)),synthesis_seconds=time.time()-started)
            except Exception as e:row['synthesis_error']=repr(e)
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps({'counts':counts}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['synthesize','asr','metrics','summarize'],required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--per-group-limit',type=int,default=0)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.stage=='synthesize':synthesize(a)
    else:
        import importlib.util
        path=Path('/119010446/LITs/training/common/evaluate_checkpoint.py');spec=importlib.util.spec_from_file_location('lits_distill_evaluation',path);common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common);common.DATA=ROOT/'data/prepared';getattr(common,a.stage)(a)

if __name__=='__main__':main()
