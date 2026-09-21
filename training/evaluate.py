"""Kokoro synthesis with LITs' unchanged ASR and voice-quality scoring."""
import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch

from training.common import ROOT, BASE_KEYS, Features, load_exact, records, write_json, digest


def export_and_synthesize(args):
    from kokoro import KModel
    from models import StyleEncoder
    torch.set_num_threads(2)
    torch.manual_seed(20260920)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    net=ckpt['net']
    weights=out/'kokoro.pth'
    torch.save({key:net[key] for key in BASE_KEYS},weights)
    model=KModel(repo_id='hexgrad/Kokoro-82M',config=str(ROOT/'models/Kokoro-82M/config.json'),model=str(weights)).cuda().eval()
    for key in BASE_KEYS:load_exact(getattr(model,key),net[key])
    acoustic=StyleEncoder(64,128,512).cuda().eval()
    acoustic.load_state_dict(net['style_encoder'],strict=True)
    prosodic=None
    if ckpt['stage']==2:
        prosodic=StyleEncoder(64,128,512).cuda().eval()
        prosodic.load_state_dict(net['predictor_encoder'],strict=True)
    features=Features().cuda()
    refs=records(ROOT/'data/prepared/voicepack_references.jsonl')
    acoustic_vectors=[];prosodic_vectors=[]
    with torch.inference_mode():
        for row in refs:
            audio,sr=sf.read(row['audio'],dtype='float32')
            if sr!=24000 or audio.ndim!=1:raise ValueError(row['audio'])
            wave=torch.nn.functional.pad(torch.from_numpy(audio).cuda(),(5000,5000))
            mel=features(wave)[None,None]
            acoustic_vectors.append(acoustic(mel))
            if prosodic is not None:prosodic_vectors.append(prosodic(mel))
        timbre=torch.cat(acoustic_vectors).mean(0,keepdim=True)
        if prosodic is not None:
            style=torch.cat([timbre,torch.cat(prosodic_vectors).mean(0,keepdim=True)],dim=-1)
            torch.save(style.cpu().unsqueeze(0).expand(510,1,256).clone(),out/'encoder_mean.pt')
            if 'voicepack' in net and bool(net['voicepack']['initialized']):
                style=net['voicepack']['vector'].cuda()
                if style.shape!=(1,256) or not torch.isfinite(style).all():
                    raise ValueError('Invalid learned voicepack')
            torch.save(style.cpu().unsqueeze(0).expand(510,1,256).clone(),out/'majestic.pt')
        else:
            # Stage 1 does not train predictor_encoder. Never guess that a random
            # encoder is trained from its norm. Use native base prosody explicitly.
            native=torch.load(ROOT/'models/Kokoro-82M/zf_xiaobei.pt',map_location='cpu',weights_only=True)
            style=torch.cat([timbre,native[100,:,128:].cuda()],dim=-1)
            torch.save(style.cpu().unsqueeze(0).expand(510,1,256).clone(),out/'stage1_diagnostic.pt')
    write_json(out/'export.json',{'stage':ckpt['stage'],'global_step':ckpt['global_step'],
                                'conditioning':('directly_optimized_voicepack' if 'voicepack' in net and bool(net['voicepack']['initialized']) else 'fixed_training_reference_mean') if prosodic is not None else 'stage1_acoustic_plus_native_base_prosody_diagnostic',
                                'reference_manifest_sha256':digest(ROOT/'data/prepared/voicepack_references.jsonl'),
                                'checkpoint_sha256':digest(args.checkpoint)})
    sources=records(ROOT/'data/prepared/eval_manifest.jsonl')
    counts={}
    with (out/'synthesis.jsonl').open('w',buffering=1) as stream, torch.inference_mode():
        for row in sources:
            group=row['group'];counts.setdefault(group,0)
            if args.per_group_limit and counts[group]>=args.per_group_limit:continue
            counts[group]+=1
            result={k:v for k,v in row.items() if k not in ('token_ids','phonemes')}
            started=time.monotonic()
            try:
                ids=row['token_ids']
                if len(ids)>510:raise ValueError('Evaluation text too long; no silent truncation')
                audio,durations=model.forward_with_tokens(torch.tensor([[0,*ids,0]],device='cuda'),style)
                wave=audio.float().cpu().numpy().reshape(-1)
                if not np.isfinite(wave).all() or not 0<len(wave)<=24000*120:raise ValueError('Invalid generated waveform')
                target=out/'wavs'/(row['id']+'.wav');target.parent.mkdir(parents=True,exist_ok=True)
                sf.write(target,np.clip(wave,-1,1),24000,subtype='PCM_16')
                result.update(audio=str(target),duration=len(wave)/24000,synthesis_seconds=time.monotonic()-started,
                              duration_frames=durations.cpu().tolist())
            except Exception as exc:
                result['synthesis_error']=repr(exc)
            stream.write(json.dumps(result,ensure_ascii=False)+'\n')
    print(json.dumps({'synthesized_groups':counts}),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['synthesize','asr','metrics','summarize'],required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--per-group-limit',type=int,default=0)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.stage=='synthesize':export_and_synthesize(args)
    else:
        # Import by file path to avoid this project's training package shadowing LITs.
        import importlib.util
        path=Path('/119010446/LITs/training/common/evaluate_checkpoint.py')
        spec=importlib.util.spec_from_file_location('lits_common_evaluation',path)
        common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
        common.DATA=ROOT/'data/prepared'
        getattr(common,args.stage)(args)


if __name__=='__main__':main()
