"""Cache teacher waveforms and the exact durations/F0/energy that generated them."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from training.common import ROOT,records,digest,write_json,seed_all
from kokoro import KModel


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--teacher',type=Path,default=ROOT/'runs/majestic_s2_20ep_long_resume6k_20260921/eval/stage2_final')
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1)
    p.add_argument('--limit-per-language',type=int,default=0)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);seed_all(20260922)
    teacher=args.teacher.resolve()
    meta={'teacher':str(teacher),'weights_sha256':digest(teacher/'kokoro.pth'),'voice_sha256':digest(teacher/'majestic.pt'),'sample_rate':24000,'samples_per_frame':600,'speed':1,'seed':20260922,'source_sha256':{s:digest(ROOT/f'data/prepared/{s}.jsonl') for s in ['train','val']}}
    meta['contract_sha256']=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    existing=out/f'protocol_rank{args.rank}.json'
    if existing.exists() and json.loads(existing.read_text())!=meta:raise ValueError('Teacher cache protocol changed')
    write_json(existing,meta)
    model=KModel(repo_id='hexgrad/Kokoro-82M',config=str(ROOT/'models/Kokoro-82M/config.json'),model=str(teacher/'kokoro.pth')).cuda().eval()
    style=torch.load(teacher/'majestic.pt',map_location='cuda',weights_only=True)[0]
    captured={}
    def capture(module,inputs):
        captured['f0']=inputs[1].detach();captured['energy']=inputs[2].detach()
    hook=model.decoder.register_forward_pre_hook(capture)
    total=0;wall=time.time()
    for split in ['val','train']:
        rows=records(ROOT/f'data/prepared/{split}.jsonl');counts={};selected=[]
        for i,row in enumerate(rows):
            lang=row['language'];counts.setdefault(lang,0)
            if args.limit_per_language and counts[lang]>=args.limit_per_language:continue
            counts[lang]+=1;selected.append((i,row))
        dest=out/split;dest.mkdir(exist_ok=True)
        with (out/f'{split}_rank{args.rank}.jsonl').open('w',buffering=1) as log,torch.inference_mode():
            for position,(i,row) in enumerate(selected):
                if position%args.world_size!=args.rank:continue
                path=dest/f'{i:06d}.npz'
                if not path.exists():
                    seed_all(20260922+i+(1000000 if split=='val' else 0))
                    tokens=[0,*row['token_ids'],0]
                    wave,durations=model.forward_with_tokens(torch.tensor([tokens],device='cuda'),style)
                    wave=wave.float().cpu().numpy().reshape(-1);dur=durations.cpu().numpy()
                    f0=captured['f0'].float().cpu().numpy().reshape(-1);energy=captured['energy'].float().cpu().numpy().reshape(-1)
                    assert len(tokens)==len(dur) and len(wave)==int(dur.sum())*600
                    assert len(f0)==len(energy)==int(dur.sum())*2
                    if not np.isfinite(wave).all() or not np.isfinite(f0).all() or not np.isfinite(energy).all():raise ValueError('Nonfinite teacher output')
                    if len(wave)>24000*120 or np.max(np.abs(wave))>1.05:raise ValueError('Invalid teacher length/amplitude')
                    audio=np.round(np.clip(wave,-1,1)*32767).astype(np.int16)
                    with path.with_suffix('.tmp').open('wb') as f:
                        np.savez(f,audio=audio,ids=np.asarray(tokens,dtype=np.int16),dur=dur.astype(np.int16),f0=f0,energy=energy,contract=np.array(meta['contract_sha256']),source_index=np.array(i),clipped_fraction=np.array(np.mean(np.abs(wave)>1)))
                    path.with_suffix('.tmp').replace(path)
                with np.load(path,allow_pickle=False) as cache:
                    assert str(cache['contract'])==meta['contract_sha256']
                    length=int(cache['dur'].sum());assert len(cache['audio'])==length*600
                    item={'id':f'{split}/{i:06d}','cache':str(path),'language':row['language'],'text':row['text'],'token_ids':row['token_ids'],'frames':length,'duration':length/40,'source_index':i,'source_split':split,'clipped_fraction':float(cache['clipped_fraction'])}
                log.write(json.dumps(item,ensure_ascii=False)+'\n');total+=1
                if total%25==0:
                    write_json(out/f'status_rank{args.rank}.json',{'status':'generating','completed':total,'split':split,'items_per_second':total/(time.time()-wall),'updated_at':time.time()})
                    print(json.dumps({'rank':args.rank,'completed':total,'split':split,'items_per_second':round(total/(time.time()-wall),2)}),flush=True)
    hook.remove();write_json(out/f'complete_rank{args.rank}.json',{'completed':total,'protocol':meta,'completed_at':time.time()})

if __name__=='__main__':main()
