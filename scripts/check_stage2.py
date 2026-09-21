"""Bounded Stage 2 forward/backward and evaluator checks, without optimizer steps."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from training.common import make_model,load_exact,seed_all,write_json,digest
from training.data import SpeechDataset,collate
from training.network import Generator,Discriminators
from training.train import move,make_optimizers
from losses import MultiResolutionSTFTLoss,WavLMLoss


def parameter_hash(model):
    result=hashlib.sha256()
    for key,module in model.items():
        for name,value in module.named_parameters():result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);seed_all(20260920)
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False);assert ckpt['stage']==1
    config=ckpt['config'];step=ckpt['global_step'];source_hash=digest(args.checkpoint)
    model,_=make_model('cuda',base=False)
    for key,module in model.items():load_exact(module,ckpt['net'][key])
    del ckpt
    # Match the actual Stage 1 -> 2 initialization in training.train exactly.
    model.predictor_encoder.load_state_dict(model.style_encoder.state_dict(),strict=True)
    native=torch.load(ROOT/'models/Kokoro-82M/zf_xiaobei.pt',map_location='cuda',weights_only=True)[100,0,128:]
    with torch.no_grad():
        model.predictor_encoder.unshared.weight.mul_(.01);model.predictor_encoder.unshared.bias.copy_(native)
    optimizers,keys=make_optimizers(model,2,config)
    generator=Generator(model,2,config['max_mel_frames']);discriminator=Discriminators(model)
    spectral=MultiResolutionSTFTLoss().cuda()
    perceptual=WavLMLoss(str(ROOT/'models/wavlm-base-plus'),None,24000,16000).cuda().eval().requires_grad_(False)
    data=SpeechDataset(ROOT/'data/prepared/val.jsonl')
    chosen=[]
    for lang in ('zh','en','mixed'):
        indices=sorted([i for i,r in enumerate(data.rows) if r['language']==lang],key=lambda i:data.rows[i]['duration'])
        chosen.append(indices[len(indices)//2])
    batch=move(collate([data[i] for i in chosen]),'cuda')
    before_hash=parameter_hash(model)
    report={'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':source_hash,
            'scope':'implementation audit of Stage 2 initialized from step 2000; NOT a trained Stage 2 quality result',
            'optimizer_steps':0,'formal_training_resumed':False,'world_size':1,
            'validation_indices':chosen,'languages':[data.rows[i]['language'] for i in chosen],
            'optimizer_keys':keys,'phases':{}}
    for joint in (False,True):
        for module in model.values():module.zero_grad(set_to_none=True)
        active=generator.set_mode(True,joint=joint);discriminator.requires_grad_(False)
        seed_all(20260920)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['bf16']):
            generated,target,sup=generator(batch)
        generated=generated.float();target=target.float()
        mel=spectral(generated,target)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['bf16']):
            wavlm=perceptual(target,generated)
            gan=discriminator.generator_loss(target,generated) if joint else generated.new_zeros(())
        loss=5*mel+wavlm+gan+sup['duration']+20*sup['duration_ce']+sup['f0']+sup['energy']
        assert torch.isfinite(loss);loss.backward()
        gradients={}
        for key,module in generator.nets.items():
            vals=[v.grad.detach().float().square().sum() for v in module.parameters() if v.grad is not None]
            norm=float(torch.stack(vals).sum().sqrt()) if vals else 0.
            gradients[key]={'gradient_norm':norm,'trainable':any(v.requires_grad for v in module.parameters())}
            assert torch.isfinite(torch.tensor(norm)),key
            if key in active:assert norm>0,key
            else:assert norm==0,key
        report['phases']['joint' if joint else 'warmup']={'active':active,'loss':float(loss),'gradients':gradients,
                         'supervision':{k:float(v) for k,v in sup.items()},'decoder_receives_parameter_gradient':gradients['decoder']['gradient_norm']>0}
        print(json.dumps({'phase':'joint' if joint else 'warmup','active':active,'loss':float(loss)}),flush=True)
        del generated,target,sup,loss,mel,wavlm,gan
    report['parameters_unchanged_without_optimizer_step']=before_hash==parameter_hash(model)
    assert report['parameters_unchanged_without_optimizer_step']
    # Save only a temporary initialized checkpoint to exercise Stage 2 export.
    with tempfile.TemporaryDirectory(prefix='kokoro_stage2_check_') as temp:
        temp=Path(temp);temporary_checkpoint=temp/'stage2_untrained_audit.pth'
        torch.save({'net':{k:m.state_dict() for k,m in model.items()},'stage':2,'global_step':step,'config':config,
                    'implementation_check_only':True},temporary_checkpoint)
        del generator,discriminator,model,optimizers,perceptual,spectral,batch
        gc.collect();torch.cuda.empty_cache()
        python=str(ROOT/'.venv/bin/python');deployment=temp/'export'
        command=[python,'-m','training.evaluate','--stage','synthesize','--checkpoint',str(temporary_checkpoint),'--output',str(deployment),'--per-group-limit','1']
        with (out/'stage2_export.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        synthesized=[json.loads(x) for x in (deployment/'synthesis.jsonl').read_text().splitlines()]
        assert len(synthesized)==3 and all('audio' in x and 'synthesis_error' not in x for x in synthesized)
        report['export']={'strict_load_and_three_language_synthesis_passed':True,'voicepack_shape':list(torch.load(deployment/'majestic.pt',weights_only=True).shape)}
    assert digest(args.checkpoint)==source_hash
    report['checkpoint_unchanged']=True;report['completed_at']=time.time();report['status']='passed'
    write_json(out/'stage2_audit.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
