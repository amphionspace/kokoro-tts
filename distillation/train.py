"""Four-GPU student training, resumable checkpoints, validation and eval queue."""
import argparse,fcntl,json,math,os,time
from pathlib import Path
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from training.common import ROOT,records,write_json,digest,seed_all,rng_state,restore_rng
from training.data import BucketBatches
from distillation.model import Student
from distillation.data import CacheDataset,collate,move
from distillation.losses import AcousticLoss
from Modules.discriminators import MultiPeriodDiscriminator,MultiResSpecDiscriminator
from losses import DiscriminatorLoss,WavLMLoss,feature_loss,generator_loss,generator_TPRLS_loss


class Critics(nn.Module):
    def __init__(self):
        super().__init__();self.mpd=MultiPeriodDiscriminator();self.msd=MultiResSpecDiscriminator()
    def forward(self,target,pred):
        return DiscriminatorLoss(self.mpd,self.msd)(target[:,None],pred[:,None])
    def generator(self,target,pred):
        adversarial=pred.new_zeros(());features=pred.new_zeros(());relative=pred.new_zeros(())
        for critic in [self.mpd,self.msd]:
            real,fake,rf,ff=critic(target[:,None],pred[:,None])
            adversarial=adversarial+generator_loss(fake)[0]
            features=features+feature_loss(rf,ff)
            relative=relative+generator_TPRLS_loss(real,fake)
        return {'gan_adversarial':adversarial,'gan_features':features,'gan_relative':relative}


def cpu_buffers(module):return {k:v.detach().cpu().clone() for k,v in module.named_buffers()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--resume',type=Path);p.add_argument('--max-steps',type=int,default=0)
    p.add_argument('--batch-size',type=int);p.add_argument('--workers',type=int);p.add_argument('--skip-validation',action='store_true')
    a=p.parse_args();c=json.loads(a.config.read_text())
    if a.batch_size:c['batch_size']=a.batch_size
    if a.workers is not None:c['workers']=a.workers
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(c['cpu_threads'])
    if world>1:dist.init_process_group('nccl',device_id=device)
    seed_all(c['seed']);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    run=a.run_dir.resolve();run.mkdir(parents=True,exist_ok=True)
    lock=None
    if rank==0:
        lock=(run/'training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    data=CacheDataset(a.cache/'train.jsonl');val=CacheDataset(a.cache/'val.jsonl')
    identity={key:digest(a.cache/f'{key}.jsonl') for key in ['train','val']}
    identity['teacher_protocol']=digest(a.cache/'protocol_rank0.json');identity['architecture']=digest(ROOT/'configs/student_7m.json')
    student=Student().to(device);critics=Critics().to(device)
    init=(student.load_nested(ROOT/c['initial_weights']) if c['initial_weights'] else {'parameters':sum(p.numel() for p in student.parameters()),'loaded_parameters':0,'initialization':'random_student_own_teacher'})
    if not c['initial_weights'] and c.get('duration_bias_from_train',False):
        mean_frames=sum(r['frames'] for r in data.rows)/sum(len(r['token_ids'])+2 for r in data.rows)
        probability=min(max(mean_frames/student.config['max_dur'],0.01),0.99)
        nn.init.constant_(student.predictor.duration_proj.linear_layer.bias,math.log(probability/(1-probability)))
        init['duration_initial_mean_frames']=mean_frames
    voice=torch.load(ROOT/c['teacher_export']/'majestic.pt',map_location=device,weights_only=True)[0]
    identity['voice']=digest(ROOT/c['teacher_export']/'majestic.pt')
    for p_ in student.parameters():p_.requires_grad_(True)
    model=DDP(student,device_ids=[local],find_unused_parameters=True,broadcast_buffers=False) if world>1 else student
    discriminator=DDP(critics,device_ids=[local],broadcast_buffers=False) if world>1 else critics
    opt_g=torch.optim.AdamW(student.parameters(),lr=c['lr'],betas=(0.8,0.99),weight_decay=0.01)
    opt_d=torch.optim.AdamW(critics.parameters(),lr=c['discriminator_lr'],betas=(0.8,0.99),weight_decay=0.01)
    acoustic=AcousticLoss().to(device).eval()
    perceptual=WavLMLoss(str(ROOT/'models/wavlm-base-plus'),None,24000,16000).to(device).eval().requires_grad_(False)
    step=epoch=next_batch=0
    if a.resume:
        saved=torch.load(a.resume,map_location='cpu',weights_only=False)
        if saved['config']!=c or saved['identity']!=identity or saved['world_size']!=world:raise ValueError('Resume config/data/world mismatch')
        student.load_state_dict(saved['model'],strict=True);critics.load_state_dict(saved['critics'],strict=True)
        opt_g.load_state_dict(saved['opt_g']);opt_d.load_state_dict(saved['opt_d'])
        step,epoch,next_batch=saved['step'],saved['next_epoch'],saved['next_batch']
        for module,key in [(student,'model_buffers'),(critics,'critic_buffers')]:
            for name,b in module.named_buffers():b.copy_(saved['rank_states'][rank][key][name].to(device))
        restore_rng(saved['rank_states'][rank]['rng']);del saved
    else:seed_all(c['seed']+rank)
    writer=SummaryWriter(str(run/'tensorboard/train')) if rank==0 else None
    if rank==0:write_json(run/'initialization.json',{'initial_weights':c['initial_weights'],'load':init,'config':c,'identity':identity,'world_size':world,'resumed':str(a.resume) if a.resume else None})
    weights={'stft':c['stft_weight'],'mel':c['mel_weight'],'silence':c['silence_weight'],'duration':c['duration_weight'],'f0':c['f0_weight'],'energy':c['energy_weight'],'wavlm':c['wavlm_weight'],'gan_adversarial':1,'gan_features':1,'gan_relative':1}
    def save(next_epoch,next_position,final=False):
        state={'rng':rng_state(),'model_buffers':cpu_buffers(student),'critic_buffers':cpu_buffers(critics)}
        states=[None]*world if rank==0 else None
        if world>1:dist.gather_object(state,states,dst=0)
        else:states=[state]
        if rank==0:
            path=run/'checkpoints'/('final.pt' if final else f'step_{step:08d}.pt');path.parent.mkdir(exist_ok=True)
            payload={'model':student.state_dict(),'critics':critics.state_dict(),'opt_g':opt_g.state_dict(),'opt_d':opt_d.state_dict(),'step':step,'next_epoch':next_epoch,'next_batch':next_position,'config':c,'architecture':student.config,'identity':identity,'world_size':world,'rank_states':states,'voice':voice.cpu(),'complete':final}
            torch.save(payload,path.with_suffix('.tmp'));path.with_suffix('.tmp').replace(path)
            write_json(run/'latest_checkpoint.json',{'path':str(path),'step':step})
            if final or step%c['evaluate_every']==0 or step==c['early_evaluation_step']:
                write_json(run/'eval_queue'/('final.json' if final else f'step_{step:08d}.json'),{'checkpoint':str(path),'step':step,'smoke':not final and step==c['early_evaluation_step']})
        if world>1:dist.barrier()
    def validate():
        old=rng_state();student.eval();totals={lang:torch.zeros(6,device=device) for lang in ['zh','en','mixed']}
        try:
            with torch.inference_mode():
                for index in range(rank,len(val),world):
                    seed_all(c['seed']+index)
                    batch=move(collate([val[index]]),device)
                    generated,target,loss=student(batch,voice,c['crop_frames'],c['context_frames'],True,c['bf16'])
                    loss.update(acoustic(generated,target));lang=batch['languages'][0]
                    totals[lang]+=torch.tensor([float(loss[k]) for k in ['stft','mel','duration','f0','energy']]+[1],device=device)
            report={}
            for lang,values in totals.items():
                if world>1:dist.all_reduce(values)
                report[lang]={k:float(values[i]/values[-1]) for i,k in enumerate(['stft','mel','duration','f0','energy'])}
                report[lang]['count']=int(values[-1])
            if rank==0:
                write_json(run/'validation'/f'step_{step:08d}.json',report)
                for lang,metrics in report.items():
                    for key,value in metrics.items():writer.add_scalar(f'validation/{lang}/{key}',value,step)
        finally:student.train();restore_rng(old)
    wall=time.monotonic();last_step=step;last_wall=wall
    stop=c['steps'] if not a.max_steps else min(c['steps'],step+a.max_steps)
    while step<stop:
        sampler=BucketBatches(data.rows,c['batch_size'],rank,world,epoch,next_batch,c['seed'])
        loader=DataLoader(data,batch_sampler=sampler,num_workers=c['workers'],collate_fn=collate,pin_memory=True,persistent_workers=c['workers']>0,generator=torch.Generator().manual_seed(c['seed']+epoch))
        base=next_batch
        for batch_index,batch in enumerate(loader,start=base):
            factor=min((step+1)/c['warmup_steps'],1)*(0.1+0.9*0.5*(1+math.cos(math.pi*min(step,c['steps'])/c['steps'])))
            for group in opt_g.param_groups:group['lr']=c['lr']*factor
            for group in opt_d.param_groups:group['lr']=c['discriminator_lr']*factor
            batch=move(batch,device);student.train();critics.train()
            generated,target,values=model(batch,voice,c['crop_frames'],c['context_frames'],False,c['bf16'])
            if not torch.isfinite(generated).all():raise RuntimeError('Nonfinite student waveform')
            critics.requires_grad_(True);opt_d.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=c['bf16']):d_loss=discriminator(target,generated.detach())
            d_loss.backward();d_grad=nn.utils.clip_grad_norm_(critics.parameters(),5,error_if_nonfinite=True);opt_d.step()
            critics.requires_grad_(False);opt_g.zero_grad(set_to_none=True)
            values.update(acoustic(generated,target))
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=c['bf16']):
                values['wavlm']=perceptual(target,generated)
                if step>=c['gan_warmup']:values.update(critics.generator(target,generated))
            loss=sum(values[key]*weight for key,weight in weights.items() if key in values)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite student loss')
            loss.backward();gradient=nn.utils.clip_grad_norm_(student.parameters(),5,error_if_nonfinite=True)
            if step==0:
                norms={key:sum(float(p_.grad.detach().float().square().sum()) for p_ in getattr(student,key).parameters() if p_.grad is not None)**0.5 for key in ['bert','bert_encoder','text_encoder','predictor','decoder']}
                if not all(v>0 and math.isfinite(v) for v in norms.values()):raise RuntimeError(f'Missing block gradients: {norms}')
                if rank==0:write_json(run/'first_step_gradients.json',norms)
            opt_g.step();step+=1
            batches_per_epoch=math.ceil(len(data)/(c['batch_size']*world))
            next_epoch,next_position=(epoch+1,0) if batch_index+1==batches_per_epoch else (epoch,batch_index+1)
            if step%c['log_every']==0 or step==1:
                scalars={k:float(v.detach()) for k,v in values.items()};scalars.update(loss=float(loss.detach()),discriminator=float(d_loss.detach()),grad_norm=float(gradient),discriminator_grad_norm=float(d_grad),generated_peak=float(generated.detach().abs().max()),lr=opt_g.param_groups[0]['lr'])
                names=list(scalars);numbers=torch.tensor([scalars[k] for k in names],device=device)
                if world>1:dist.all_reduce(numbers);numbers/=world
                if rank==0:
                    now=time.monotonic();metrics={k:float(v) for k,v in zip(names,numbers)}
                    status={'status':'training','step':step,'budget_steps':c['steps'],'epoch':epoch+1,'world_size':world,'batch_size_per_gpu':c['batch_size'],'samples_per_second':(step-last_step)*c['batch_size']*world/(now-last_wall),'peak_memory_gb':torch.cuda.max_memory_allocated()/1024**3,'losses':metrics,'updated_at':time.time()}
                    write_json(run/'status.json',status);print(json.dumps(status),flush=True)
                    for key,value in metrics.items():writer.add_scalar('train/'+key,value,step)
                    writer.add_scalar('performance/samples_per_second',status['samples_per_second'],step);writer.add_scalar('performance/peak_memory_gb',status['peak_memory_gb'],step);writer.flush()
                    last_step=step;last_wall=now
            if not a.skip_validation and (step%c['validate_every']==0 or step==c['early_evaluation_step']):validate()
            if step%c['checkpoint_every']==0 or step==c['early_evaluation_step']:save(next_epoch,next_position)
            if step>=stop:break
        epoch,next_batch=next_epoch,next_position
    if not a.max_steps:
        if not a.skip_validation:validate()
        save(epoch,next_batch,True)
        if rank==0:write_json(run/'training_complete.json',{'step':step,'completed_at':time.time()})
    else:save(epoch,next_batch)
    if writer:writer.close()
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
