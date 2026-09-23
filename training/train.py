"""Four-GPU Kokoro adaptation with strict checkpoints and fixed evaluation hooks."""
import argparse
import fcntl
from collections import Counter
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from training.common import (ROOT, make_model, write_json, digest, load_exact, seed_all,
                             rng_state, restore_rng, save_checkpoint, assert_optimizer_parameters)
from training.data import SpeechDataset, BucketBatches, collate
from training.curriculum import validate_long_sampling, validate_budget_resume
from training.network import Generator, Discriminators
from losses import MultiResolutionSTFTLoss, WavLMLoss


def move(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}


def finite(value, name):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f'Nonfinite {name}')


def lr_for(base, step, total, warmup):
    warm = min(1.0, (step + 1) / max(1,warmup))
    decay = max(0.1, 1 - 0.9 * max(0,step-total*0.8)/max(1,total*0.2))
    return base * warm * decay


def make_optimizers(model, stage, config):
    keys = (['text_encoder','style_encoder','decoder','text_aligner'] if stage == 1 else
            ['bert','bert_encoder','predictor','predictor_encoder','style_encoder','decoder'])
    if stage == 2 and 'voicepack' in model:
        keys.append('voicepack')
    groups=[]
    for key in keys:
        if key == 'voicepack':
            rate=config['voicepack_lr']
        elif key == 'bert':
            rate=config['bert_lr']
        elif key in ('style_encoder','predictor_encoder','text_aligner'):
            rate=config['auxiliary_lr']
        else:
            rate=config['backbone_lr']
        groups.append({'params':list(model[key].parameters()),'lr':rate,'base_lr':rate,'name':key,'weight_decay':0.0 if key=='voicepack' else 1e-4})
    g=torch.optim.AdamW(groups,betas=(0.0,0.99),eps=1e-9,weight_decay=1e-4)
    d=torch.optim.AdamW([{'params':list(model.mpd.parameters())+list(model.msd.parameters()),
                         'lr':config['discriminator_lr'],'base_lr':config['discriminator_lr'],'name':'discriminators'}],
                        betas=(0.0,0.99),eps=1e-9,weight_decay=1e-4)
    assert_optimizer_parameters(model,g,keys)
    return {'generator':g,'discriminator':d},keys


def validation(generator, dataset, device, rank, world, config):
    generator.set_mode(False,joint=True)
    loader=DataLoader(Subset(dataset,list(range(rank,len(dataset),world))),batch_size=1,
                      num_workers=0,collate_fn=collate)
    criterion=MultiResolutionSTFTLoss().to(device)
    totals={lang:Counter() for lang in ('zh','en','mixed')}
    with torch.inference_mode():
        for batch in loader:
            batch=move(batch,device)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['bf16']):
                wave,target,sup=generator(batch,deterministic=True)
            scores={'mel':criterion(wave.float(),target.float()),**sup}
            language=batch['languages'][0]
            totals[language]['count']+=1
            for key,value in scores.items():
                finite(value,'validation/'+key)
                totals[language][key]+=float(value)
    gathered=[None]*world
    if world>1:dist.all_gather_object(gathered,totals)
    else:gathered=[totals]
    result={}
    for language in totals:
        combined=Counter()
        for part in gathered:combined.update(part[language])
        count=combined.pop('count')
        result[language]={'count':count,**{k:v/count for k,v in combined.items()}}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--stage',type=int,choices=[1,2],required=True)
    parser.add_argument('--initialize',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--extend-stage2-budget',action='store_true',help='Resume before LR decay while extending only Stage 2 epochs and the future sampling plan')
    parser.add_argument('--max-steps',type=int,default=0,help='Bounded smoke/benchmark only; zero uses configured epochs')
    parser.add_argument('--batch-size',type=int)
    parser.add_argument('--skip-validation',action='store_true')
    args=parser.parse_args()
    if args.initialize and args.resume:raise ValueError('Initialize and resume are mutually exclusive')
    if args.extend_stage2_budget and (args.stage!=2 or not args.resume):
        raise ValueError('--extend-stage2-budget requires --stage 2 and --resume')
    config=json.loads(args.config.read_text())
    if args.batch_size:config['batch_size']=args.batch_size
    if config.get('long_sampling'):
        validate_long_sampling(config['long_sampling'],config['stage2_epochs'])
    if args.stage==2 and config.get('learn_voicepack',False):
        if config['voicepack_start_step']<config['stage2_joint_step']:
            raise ValueError('Voicepack must start with or after joint decoder training')
        if not 0<config['voicepack_fraction']<1 or config['batch_size']<2:
            raise ValueError('Mixed voice training requires fraction in (0,1) and batch size >=2')
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));local=int(os.environ.get('LOCAL_RANK','0'))
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    torch.set_num_threads(config.get('cpu_threads',2))
    if world>1:dist.init_process_group('nccl',device_id=device)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    if os.environ.get('KOKORO_ANOMALY')=='1':torch.autograd.set_detect_anomaly(True)
    seed_all(config['seed'])
    run=args.run_dir;run.mkdir(parents=True,exist_ok=True)
    run_lock=None
    if rank==0:
        run_lock=(run/'training.lock').open('a')
        fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepared=ROOT/config.get('prepared_directory','data/prepared')
    if config.get('require_data_audit',False):
        report=json.loads((prepared/'preparation_report.json').read_text())
        if not report.get('passed'):raise ValueError('Data preparation audit failed')
        for name,expected in report['manifest_sha256'].items():
            if digest(prepared/name)!=expected:raise ValueError(f'Data changed after audit: {name}')
    train=SpeechDataset(prepared/'train.jsonl');val=SpeechDataset(prepared/'val.jsonl')
    data_hash=digest(prepared/'train.jsonl')
    model,audit=make_model(device,base=not bool(args.resume),
                           with_voicepack=args.stage==2 and config.get('learn_voicepack',False))
    checkpoint=None
    if args.initialize or args.resume:
        checkpoint=torch.load(args.resume or args.initialize,map_location='cpu',weights_only=False)
        for key,module in model.items():
            if key=='voicepack' and args.initialize and checkpoint['stage']==1 and key not in checkpoint['net']:
                continue
            load_exact(module,checkpoint['net'][key])
    if args.stage==2 and not args.resume:
        if not checkpoint or checkpoint['stage']!=1 or not checkpoint.get('stage_complete'):
            # Smoke tests may explicitly use a bounded Stage 1 checkpoint.
            if not args.max_steps:raise ValueError('Stage 2 requires a completed Stage 1 checkpoint')
        # In-place initialization before optimizer creation preserves identity.
        model.predictor_encoder.load_state_dict(model.style_encoder.state_dict(),strict=True)
        # Reuse learned acoustic features, but start the distinct prosody space
        # around native pretrained prosody rather than acoustic coordinates.
        native=torch.load(ROOT/'models/Kokoro-82M/zf_xiaobei.pt',map_location=device,weights_only=True)[100,0,128:]
        with torch.no_grad():
            model.predictor_encoder.unshared.weight.mul_(0.01)
            model.predictor_encoder.unshared.bias.copy_(native)
    optimizers,optimizer_keys=make_optimizers(model,args.stage,config)
    generator=Generator(model,args.stage,config['max_mel_frames'],config.get('voicepack_fraction',0.5))
    discriminator=Discriminators(model)
    # Register all potential stage parameters before any warmup freezing.
    gen=DDP(generator,device_ids=[local],find_unused_parameters=True,broadcast_buffers=False) if world>1 else generator
    dis=DDP(discriminator,device_ids=[local],broadcast_buffers=False) if world>1 else discriminator
    perceptual=WavLMLoss(str(ROOT/'models/wavlm-base-plus'),None,24000,16000).to(device)
    perceptual.wavlm.eval().requires_grad_(False)
    spectral=MultiResolutionSTFTLoss().to(device)
    start_epoch=next_batch=stage_step=global_step=0
    if checkpoint and args.initialize:global_step=checkpoint['global_step']
    if args.resume:
        if checkpoint['stage']!=args.stage or checkpoint['data_sha256']!=data_hash:
            raise ValueError('Resume stage/data mismatch')
        if args.extend_stage2_budget:
            validate_budget_resume(config,checkpoint,len(train),world,lr_for)
        elif checkpoint['world_size']!=world or checkpoint['config']!=config:
            raise ValueError('Exact resume requires identical world size and configuration')
        for key,opt in optimizers.items():opt.load_state_dict(checkpoint['optimizers'][key])
        start_epoch,next_batch=checkpoint['next_epoch'],checkpoint['next_batch']
        stage_step,global_step=checkpoint['stage_step'],checkpoint['global_step']
        for key,module in model.items():
            for name,buffer in module.named_buffers():
                buffer.copy_(checkpoint['rank_state'][rank]['buffers'][key][name].to(buffer.device))
        restore_rng(checkpoint['rank_state'][rank]['rng'])
    else:seed_all(config['seed']+rank)
    del checkpoint
    if rank==0:
        write_json(run/f'stage{args.stage}_initialization.json',{'base_load':audit,'optimizer_keys':optimizer_keys,'resumed':bool(args.resume),
                   'extended_stage2_budget':args.extend_stage2_budget,
                   'resume_checkpoint':str(args.resume.resolve()) if args.resume else None,
                   'resume_checkpoint_sha256':digest(args.resume) if args.resume else None,
                   'world_size':world,'batch_size_per_gpu':config['batch_size']})
        writer=SummaryWriter(str(run/'tensorboard'/f'stage{args.stage}'))
        writer.add_text('recipe',json.dumps(config,ensure_ascii=False,indent=2),global_step)
    else:writer=None
    epochs=config[f'stage{args.stage}_epochs']
    steps_per_epoch=math.ceil(len(train)/(config['batch_size']*world))
    total_steps=steps_per_epoch*epochs
    long_sampling=config.get('long_sampling') if args.stage==2 else None
    weights={'s2s':1.,'mono':1.,'duration':1.,'duration_ce':20.,'f0':1.,'energy':1.}
    last_validation={}
    wall=time.monotonic();last_log=wall;last_log_step=stage_step
    gradient_report={}

    def save(path,epoch,batch,stage_complete=False):
        local_state={'rng':rng_state(),'buffers':{k:{n:b.detach().cpu().clone() for n,b in m.named_buffers()} for k,m in model.items()}}
        states=[None]*world
        if world>1:dist.all_gather_object(states,local_state)
        else:states=[local_state]
        if rank==0:
            save_checkpoint(path,model,optimizers,stage=args.stage,global_step=global_step,stage_step=stage_step,
                            next_epoch=epoch,next_batch=batch,stage_complete=stage_complete,world_size=world,
                            config=config,data_sha256=data_hash,rank_state=states,validation=last_validation)
        if world>1:dist.barrier()

    stopped=False
    for epoch in range(start_epoch,epochs):
        start=next_batch if epoch==start_epoch else 0
        batches=BucketBatches(train.rows,config['batch_size'],rank,world,epoch,start,config['seed'],
                              long_sampling=long_sampling)
        if long_sampling and rank==0:
            batches.batches()
            write_json(run/'sampling'/f'epoch_{epoch+1:04d}.json',batches.sampling_report)
            writer.add_scalar('sampling/long_weight',batches.sampling_report['long_weight'],global_step)
            for lang,stats in batches.sampling_report['languages'].items():
                writer.add_scalar(f'sampling/{lang}/long_fraction',stats['sampled_long_fraction'],global_step)
        # Own generator prevents DataLoader seeding from changing model/dropout RNG.
        loader=DataLoader(train,batch_sampler=batches,num_workers=config['workers_per_gpu'],collate_fn=collate,
                          pin_memory=True,persistent_workers=config['workers_per_gpu']>0,
                          generator=torch.Generator().manual_seed(config['seed']+epoch))
        for batch_index,batch in enumerate(loader,start):
            iteration_start=time.monotonic()
            joint=args.stage==1 or stage_step>=config['stage2_joint_step']
            if args.stage==2 and 'voicepack' in model and stage_step>=config['voicepack_start_step']:
                from training.voicepack import initialize_voicepack
                initialized=initialize_voicepack(model,prepared/'voicepack_references.jsonl')
                if initialized and rank==0:
                    write_json(run/'voicepack_initialization.json',{'stage_step':stage_step,'global_step':global_step,
                               'references_sha256':digest(prepared/'voicepack_references.jsonl'),
                               'vector_norm':float(model.voicepack.vector.norm())})
            active=generator.set_mode(True,joint=joint)
            for opt in optimizers.values():
                for group in opt.param_groups:
                    group['lr']=lr_for(group['base_lr'],stage_step,total_steps,config['warmup_steps'])
            batch=move(batch,device)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['bf16']):
                generated,target,sup=gen(batch)
            generated,target=generated.float(),target.float()
            discriminator.requires_grad_(True)
            optimizers['discriminator'].zero_grad(set_to_none=True)
            if joint:
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['bf16']):
                    d_loss=dis(target.detach(),generated.detach())
                finite(d_loss,'discriminator');d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(),config['grad_clip'],error_if_nonfinite=True)
                optimizers['discriminator'].step()
            else:d_loss=generated.new_zeros(())
            discriminator.requires_grad_(False)
            optimizers['generator'].zero_grad(set_to_none=True)
            mel_loss=spectral(generated,target)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['bf16']):
                perceptual_loss=perceptual(target,generated)
                adversarial=discriminator.generator_loss(target,generated) if joint else generated.new_zeros(())
            loss=config['mel_weight']*mel_loss + config['wavlm_weight']*perceptual_loss + adversarial
            for key,value in sup.items():loss=loss+weights[key]*value
            finite(loss,'generator');loss.backward()
            grad_norm=torch.nn.utils.clip_grad_norm_(generator.parameters(),config['grad_clip'],error_if_nonfinite=True)
            if stage_step in (0,99,config.get('voicepack_start_step',-1)):
                for key in active:
                    values=[p.grad.detach().float().square().sum() for p in model[key].parameters() if p.grad is not None]
                    norm=float(torch.stack(values).sum().sqrt()) if values else 0.
                    gradient_report[f'{stage_step+1}/{key}']=norm
                    if norm==0 or not math.isfinite(norm):raise RuntimeError(f'No valid gradient in active module {key}')
            optimizers['generator'].step()
            stage_step+=1;global_step+=1
            next_epoch,next_position=(epoch+1,0) if batch_index+1==steps_per_epoch else (epoch,batch_index+1)
            if stage_step%config['log_every']==0 or stage_step==1:
                torch.cuda.synchronize()
                now=time.monotonic()
                values={'generated_peak':float(generated.detach().abs().max()),'target_peak':float(target.abs().max()),'loss':float(loss),'mel':float(mel_loss),'wavlm':float(perceptual_loss),'gan':float(adversarial),
                        'discriminator':float(d_loss),'grad_norm':float(grad_norm),**{k:float(v) for k,v in sup.items()}}
                if 'voicepack' in model and bool(model.voicepack.initialized):
                    values['voicepack_delta_norm']=float((model.voicepack.vector-model.voicepack.initial_vector).norm())
                    values['voicepack_grad_norm']=float(model.voicepack.vector.grad.norm()) if model.voicepack.vector.grad is not None else 0.
                keys=list(values);vector=torch.tensor([values[k] for k in keys],device=device)
                if world>1:dist.all_reduce(vector);vector/=world
                if rank==0:
                    values=dict(zip(keys,vector.tolist()))
                    throughput=(stage_step-last_log_step)*config['batch_size']*world/(now-last_log)
                    for key,value in values.items():writer.add_scalar('train/'+key,value,global_step)
                    writer.add_scalar('performance/samples_per_second',throughput,global_step)
                    writer.add_scalar('performance/gpu_peak_gib',torch.cuda.max_memory_allocated()/2**30,global_step)
                    for group in optimizers['generator'].param_groups:writer.add_scalar('lr/'+group['name'],group['lr'],global_step)
                    for lang,count in Counter(batch['languages']).items():writer.add_scalar('batch/'+lang,count,global_step)
                    status={'status':'training','stage':args.stage,'epoch':epoch,'global_step':global_step,'stage_step':stage_step,
                            'stage_budget_steps':total_steps,'samples_per_second':throughput,'world_size':world,
                            'batch_size_per_gpu':config['batch_size'],'losses':values,'updated_at':time.time()}
                    if long_sampling:status['long_sample_weight']=batches.sampling_report['long_weight']
                    write_json(run/'status.json',status);write_json(run/f'stage{args.stage}_gradients.json',gradient_report)
                    writer.flush();print(json.dumps(status),flush=True)
                last_log,last_log_step=now,stage_step
            need_save=(stage_step%config['checkpoint_every']==0 or stage_step in config['early_checkpoints'])
            stop=bool(args.max_steps and stage_step>=args.max_steps)
            if need_save or stop:
                if not args.skip_validation and (stage_step%config['validate_every']==0 or stop):
                    saved_rng=rng_state()
                    last_validation=validation(generator,val,device,rank,world,config)
                    restore_rng(saved_rng)
                    if rank==0:
                        for lang,metrics in last_validation.items():
                            for key,value in metrics.items():writer.add_scalar(f'val/{lang}/{key}',value,global_step)
                path=run/'checkpoints'/f'stage{args.stage}_step_{global_step:08d}.pth'
                save(path,next_epoch,next_position)
                if rank==0:
                    write_json(run/'latest_checkpoint.json',{'path':str(path),'global_step':global_step,'stage':args.stage})
                    if need_save:write_json(run/'eval_queue'/f'step_{global_step:08d}.json',{'checkpoint':str(path),'stage':args.stage,'global_step':global_step,'smoke':stage_step in config['early_checkpoints']})
            if stop:stopped=True;break
        if stopped:break
    if not stopped:
        saved_rng=rng_state()
        if not args.skip_validation:last_validation=validation(generator,val,device,rank,world,config)
        restore_rng(saved_rng)
        path=run/'checkpoints'/f'stage{args.stage}_final.pth'
        save(path,epochs,0,stage_complete=True)
        if rank==0:
            write_json(run/'eval_queue'/f'stage{args.stage}_final.json',{'checkpoint':str(path),'stage':args.stage,'global_step':global_step,'smoke':False})
            write_json(run/f'stage{args.stage}_complete.json',{'checkpoint':str(path),'global_step':global_step,'time':time.time()})
    if writer:writer.close()
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
