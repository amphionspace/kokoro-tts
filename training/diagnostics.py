"""Condition-controlled checkpoint evaluation; never updates model parameters.

Matched crops reuse Generator.forward verbatim. ASR is applied only to complete
utterances with matching transcripts, not the five-second reconstruction crops.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from training.common import ROOT, make_model, load_exact, records, write_json, digest, seed_all
from training.data import SpeechDataset, collate
from training.network import Generator
from training.train import move
from losses import MultiResolutionSTFTLoss, WavLMLoss
from utils import length_to_mask, mask_from_lens, maximum_path, log_norm

LANGS=('zh','en','mixed')


def finite(x, label):
    if not torch.isfinite(x).all():raise RuntimeError(f'Nonfinite {label}')


def save_audio(path, wave):
    array=wave.detach().float().cpu().numpy().reshape(-1) if torch.is_tensor(wave) else np.asarray(wave).reshape(-1)
    if not np.isfinite(array).all() or not 0<len(array)<=24000*120:raise ValueError(f'Invalid audio: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    # Float WAV preserves raw amplitudes; no clipping or per-file normalization.
    sf.write(path,array,24000,subtype='FLOAT')
    return {'audio':str(path),'duration':len(array)/24000,'peak':float(np.abs(array).max()),
            'rms':float(np.sqrt(np.mean(array**2))),'over_one_fraction':float(np.mean(np.abs(array)>1))}


def selected_indices(rows, per_language):
    selected=[]
    for lang in LANGS:
        indices=sorted([i for i,r in enumerate(rows) if r['language']==lang],key=lambda i:(rows[i]['duration'],i))
        selected.extend(indices[int(j)] for j in np.linspace(0,len(indices)-1,min(per_language,len(indices))))
    return sorted(set(selected))


def aggregate(rows):
    result={}
    for lang in (*LANGS,'all'):
        part=[r for r in rows if lang=='all' or r['language']==lang]
        keys=sorted({k for r in part for k,v in r.items() if isinstance(v,(int,float)) and k not in ('index',)})
        result[lang]={'samples':len(part),**{k:float(np.mean([r[k] for r in part if k in r])) for k in keys}}
    return result


class Capture:
    def __init__(self, model):
        self.values={}
        self.handles=[model.decoder.register_forward_pre_hook(self.decoder),
                      model.text_aligner.register_forward_hook(self.aligner),
                      model.text_encoder.register_forward_hook(self.text)]
    def decoder(self,module,args):
        self.values['decoder_inputs']=tuple(x.detach().clone() for x in args)
        self.values['decoder_cuda_rng']=torch.cuda.get_rng_state().clone()
    def aligner(self,module,args,result):self.values['raw_attention']=result[2].detach().clone()
    def text(self,module,args,result):self.values['text_features']=result.detach().clone()


def alignment(batch, raw):
    lengths=batch['token_lengths'];mel_lengths=batch['mel_lengths']
    text_mask=length_to_mask(lengths);mel_mask=length_to_mask(mel_lengths//2)
    attn=raw.transpose(-1,-2)[...,1:].transpose(-1,-2)
    valid=(~text_mask).unsqueeze(-1)&(~mel_mask).unsqueeze(1)
    attn=attn.masked_fill(~valid,0)
    mono=maximum_path(attn.float(),mask_from_lens(attn,lengths,mel_lengths//2))
    a=attn[0,:lengths[0],:mel_lengths[0]//2].float()
    m=mono[0,:lengths[0],:mel_lengths[0]//2]
    durations=m.sum(-1);inside=durations[1:-1]
    normalized=a/a.sum(-1,keepdim=True).clamp_min(1e-8)
    peaks=a.argmax(-1)
    stats={'token_count':int(len(durations)), 'alignment_frames':int(m.shape[1]),
           'unassigned_frame_fraction':float((m.sum(0)!=1).float().mean()),
           'zero_duration_token_fraction':float((inside==0).float().mean()),
           'interior_duration_max_frames':float(inside.max()),
           'interior_duration_over_50_fraction':float((inside>50).float().mean()),
           'boundary_frame_fraction':float((durations[0]+durations[-1])/m.shape[1]),
           'soft_attention_mass_inside_mas':float((normalized*m).sum(-1).mean()),
           'soft_peak_backward_fraction':float((peaks[1:]<peaks[:-1]).float().mean())}
    return mono,a,m,durations,stats


def plot_alignment(path,a,m,tokens):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
    for ax,value,title in zip(axes,(a,m),('Soft attention (before monotonic search)','Maximum monotonic path (constrained)')):
        ax.imshow(value.float().cpu().numpy(),origin='lower',aspect='auto',interpolation='nearest',extent=(0,value.shape[1]*.025,0,value.shape[0]))
        ax.set_ylabel('Token position');ax.set_title(title)
    axes[-1].set_xlabel('Seconds in padded waveform')
    fig.tight_layout();path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=130);plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--deployment-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tensorboard',type=Path,required=True)
    p.add_argument('--per-language',type=int,default=8)
    p.add_argument('--limit',type=int,default=0,help='Bounded implementation smoke only; production uses all validation rows')
    args=p.parse_args();args.output=args.output.resolve();out=args.output;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);seed_all(20260920)
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    stage=ckpt['stage'];step=ckpt['global_step'];config=ckpt['config']
    meta=json.loads((args.deployment_dir/'export.json').read_text())
    checkpoint_hash=digest(args.checkpoint)
    assert meta['checkpoint_sha256']==checkpoint_hash and meta['stage']==stage
    model,_=make_model('cuda',base=False,with_voicepack='voicepack' in ckpt['net'])
    for key,module in model.items():load_exact(module,ckpt['net'][key]);module.eval()
    del ckpt
    generator=Generator(model,stage,config['max_mel_frames']);generator.set_mode(False,joint=True)
    captured=Capture(model)
    spectral=MultiResolutionSTFTLoss().cuda().eval()
    perceptual=WavLMLoss(str(ROOT/'models/wavlm-base-plus'),None,24000,16000).cuda().eval().requires_grad_(False)
    from kokoro import KModel
    inference=KModel(repo_id='hexgrad/Kokoro-82M',config=str(ROOT/'models/Kokoro-82M/config.json'),model=str(args.deployment_dir/'kokoro.pth')).cuda().eval()
    voice_name='majestic.pt' if stage==2 else 'stage1_diagnostic.pt'
    voice=torch.load(args.deployment_dir/voice_name,map_location='cuda',weights_only=True)[100]
    fixed_acoustic=voice[:,:128]
    data=SpeechDataset(ROOT/'data/prepared/val.jsonl')
    selection=selected_indices(data.rows,args.per_language)
    indices=list(range(len(data)))[:args.limit or len(data)]
    selection=indices if args.limit else [i for i in selection if i in indices]
    writer=SummaryWriter(str(args.tensorboard))
    protocol={'stage':stage,'global_step':step,'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':checkpoint_hash,
              'validation_sha256':digest(ROOT/'data/prepared/val.jsonl'),'voice_reference_sha256':meta['reference_manifest_sha256'],
              'diagnostic_source_sha256':digest(Path(__file__)), 'precision':'same BF16 setting as training for matched crops; FP32 deployment model',
              'selection':'fixed per-language duration quantiles, independent of model scores','selected_validation_indices':selection,
              'matched_crop_samples':len(indices),'full_utterance_samples':len(selection),
              'wave_storage':'float32 WAV; no clipping or normalization',
              'asr_scope':'full utterances only; never compare full transcripts against cropped waveforms',
              'fixed_style_source':meta['conditioning'],
              'stage2_note':'matched uses reference prosody encoder; oracle_fixed_style changes only acoustic style; full free_fixed_style uses fixed acoustic AND prosodic styles'}
    write_json(out/'protocol.json',protocol)
    write_json(out/'status.json',{'status':'running','completed':0,'total':len(indices)})
    rows=[];alignment_rows=[];full_rows=[]
    stream=(out/'matched_crops.jsonl').open('w',buffering=1)
    align_stream=(out/'alignments.jsonl').open('w',buffering=1)
    full_dir=out/'full_utterance';full_dir.mkdir(exist_ok=True)
    (full_dir/'README.md').write_text('# 整句音频对照\n\n同名 WAV 是同一句话；`zh/en/mixed` 是语言目录。每种条件 24 条（每语言 8 条），按固定时长分位点选择。\n\n| wavs 下的目录 | 对齐/时长 | F0、能量 | style |\n| --- | --- | --- | --- |\n| target | 原始音频 | 原始音频 | 原始音频 |\n| oracle_reference_style | 从真实音频估计的 MAS 对齐 | 从真实音频提取 | 该句真实音频的声学编码 |\n| oracle_fixed_style | 同上 | 同上 | 固定部署 voice 的声学部分 |\n| free_fixed_style | 模型从文本预测 | 模型预测 | 固定部署 voice 的声学、韵律两部分 |\n| aligned_predicted_reference（仅 Stage 2） | 真实音频的 MAS 对齐 | 模型预测 | 该句音频的声学、韵律编码 |\n| aligned_predicted_fixed（仅 Stage 2） | 真实音频的 MAS 对齐 | 模型预测 | 固定部署 voice 的声学、韵律两部分 |\n\n`oracle` 表示使用真实音频提供条件，不表示人工标注。`oracle_reference_style` 与 `oracle_fixed_style` 用于判断固定声学 style 带来的变化；`free_fixed_style` 才是只输入文本的部署路径。Stage 1 的固定 voice 使用声学参考均值 + 原生韵律向量。Stage 2 voice 参数初始化后使用直接优化的 voice；具体来源见上一级 `protocol.json`。\n\n评分见 `../summary.json`；试听和对齐图见 `../report.html`。完整句子才参与 ASR；短片段重建在另一目录，不与整句文本评分。\n')
    full_stream=(full_dir/'synthesis.jsonl').open('w',buffering=1)
    with torch.inference_mode():
        for number,index in enumerate(indices,1):
            source=data.rows[index];lang=source['language'];sample=f'{lang}/val_{index:04d}'
            batch=move(collate([data[index]]),'cuda');seed_all(20260920+index)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['bf16']):
                original,target,sup=generator(batch,deterministic=True)
            decoder_inputs=captured.values['decoder_inputs'];decoder_rng=captured.values['decoder_cuda_rng']
            rng_after=torch.cuda.get_rng_state().clone()
            mono,a,m,durations,align_stats=alignment(batch,captured.values['raw_attention'])
            text_features=captured.values['text_features']
            aligned={'index':index,'language':lang,'text':source['text'],'phonemes':source['phonemes'],
                     'duration_frames':durations.cpu().tolist(),**align_stats}
            alignment_rows.append(aligned);align_stream.write(json.dumps(aligned,ensure_ascii=False)+'\n')
            crop_frames=original.shape[-1]//600
            start=(int(batch['mel_lengths'][0])//2-crop_frames)//2
            crop_mel=batch['mels'][:,:,start*2:(start+crop_frames)*2]
            variants={'matched_reference_style':original.float()}
            torch.cuda.set_rng_state(decoder_rng)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['bf16']):
                variants['matched_fixed_acoustic']=model.decoder(*decoder_inputs[:3],fixed_acoustic).squeeze(1).float()
            if stage==2:
                # Additional control: ground-truth F0/energy isolates acoustic reconstruction capacity.
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['bf16']):
                    true_f0,_,_=model.pitch_extractor(crop_mel.unsqueeze(1));true_energy=log_norm(crop_mel.unsqueeze(1)).squeeze(1)
                    torch.cuda.set_rng_state(decoder_rng)
                    variants['oracle_f0_energy']=model.decoder(decoder_inputs[0],true_f0,true_energy,decoder_inputs[3]).squeeze(1).float()
            torch.cuda.set_rng_state(rng_after)
            row={'index':index,'language':lang,'crop_seconds':original.shape[-1]/24000,
                 **{f'supervision/{key}':float(value) for key,value in sup.items()},
                 'style_distance_to_fixed':float((decoder_inputs[3].float()-fixed_acoustic).norm())}
            for name,wave in variants.items():
                finite(wave,name)
                row[f'{name}/mel']=float(spectral(wave,target.float()))
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['bf16']):
                    row[f'{name}/wavlm']=float(perceptual(target.float(),wave))
                row[f'{name}/peak']=float(wave.abs().max())
                row[f'{name}/over_one_fraction']=float((wave.abs()>1).float().mean())
            row['fixed_minus_reference/mel']=row['matched_fixed_acoustic/mel']-row['matched_reference_style/mel']
            row['fixed_minus_reference/wavlm']=row['matched_fixed_acoustic/wavlm']-row['matched_reference_style/wavlm']
            rows.append(row);stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            if index in selection:
                plot_alignment(out/'alignment_plots'/f'{lang}_val_{index:04d}.png',a,m,source['phonemes'])
                save_audio(out/'matched_audio'/'target'/f'{sample}.wav',target)
                for name,wave in variants.items():save_audio(out/'matched_audio'/name/f'{sample}.wav',wave)
                writer.add_audio(f'crop/{sample}/target',target.float().cpu().reshape(1,-1),step,sample_rate=24000)
                for name,wave in variants.items():writer.add_audio(f'crop/{sample}/{name}',wave.cpu().reshape(1,-1),step,sample_rate=24000)
                # Full-utterance controls use full reference style and ground-truth alignment/F0/N.
                mel=batch['mels'];length=int(batch['mel_lengths'][0]);raw_wave=batch['waves'][0][5000:-5000]
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['bf16']):
                    contents=text_features@mono
                    full_f0,_,_=model.pitch_extractor(mel.unsqueeze(1));full_energy=log_norm(mel.unsqueeze(1)).squeeze(1)
                    ref_style=model.style_encoder(mel.unsqueeze(1))
                    seed_all(20260920+index)
                    reference_wave=model.decoder(contents,full_f0,full_energy,ref_style).squeeze(1).float()
                    seed_all(20260920+index)
                    fixed_wave=model.decoder(contents,full_f0,full_energy,fixed_acoustic).squeeze(1).float()
                # Strip only the known training padding from reconstructions, preserving full transcript coverage.
                reference_wave=reference_wave[:,5000:5000+len(raw_wave)]
                fixed_wave=fixed_wave[:,5000:5000+len(raw_wave)]
                seed_all(20260920+index)
                free_wave,pred_durations=inference.forward_with_tokens(batch['tokens'],voice)
                full_variants={'target':raw_wave,'oracle_reference_style':reference_wave,
                               'oracle_fixed_style':fixed_wave,'free_fixed_style':free_wave}
                for name,wave in full_variants.items():
                    saved=save_audio(full_dir/'wavs'/name/f'{sample}.wav',wave)
                    item={'id':f'{name}/{sample}','variant':name,'validation_index':index,'group':f'majestic_{lang}',
                          'speaker':1,'language':lang,'ref_text':source['text'],'text':source['text'],**saved}
                    full_rows.append(item);full_stream.write(json.dumps(item,ensure_ascii=False)+'\n')
                    writer.add_audio(f'full/{sample}/{name}',torch.as_tensor(wave).float().cpu().reshape(1,-1),step,sample_rate=24000)
                if stage==2:
                    # Full-sentence teacher-aligned prediction bridge for the learned prosody path.
                    text_mask=length_to_mask(batch['token_lengths'])
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['bf16']):
                        contextual=model.bert_encoder(model.bert(batch['tokens'],attention_mask=(~text_mask).int())).transpose(-1,-2)
                        for name,ps,ac in [('aligned_predicted_reference',model.predictor_encoder(mel.unsqueeze(1)),ref_style),
                                           ('aligned_predicted_fixed',voice[:,128:],fixed_acoustic)]:
                            dl,pros=model.predictor(contextual,ps,batch['token_lengths'],mono,text_mask)
                            f0,en=model.predictor.F0Ntrain(pros,ps)
                            seed_all(20260920+index)
                            wave=model.decoder(contents,f0,en,ac).squeeze(1).float()[:,5000:5000+len(raw_wave)]
                            saved=save_audio(full_dir/'wavs'/name/f'{sample}.wav',wave)
                            item={'id':f'{name}/{sample}','variant':name,'validation_index':index,'group':f'majestic_{lang}',
                                  'speaker':1,'language':lang,'ref_text':source['text'],'text':source['text'],**saved}
                            full_rows.append(item);full_stream.write(json.dumps(item,ensure_ascii=False)+'\n')
                            writer.add_audio(f'full/{sample}/{name}',wave.cpu().reshape(1,-1),step,sample_rate=24000)
            if number%25==0 or number==len(indices):
                write_json(out/'status.json',{'status':'running','completed':number,'total':len(indices),'updated_at':time.time()})
                print(json.dumps({'completed':number,'total':len(indices)}),flush=True);writer.flush()
    stream.close();align_stream.close();full_stream.close()
    result={'stage':stage,'global_step':step,'matched_crops':aggregate(rows),'alignment':aggregate(alignment_rows),
            'full_utterance_rows':len(full_rows),'full_utterance_status':'generated; ASR and quality scoring pending',
            'alignment_caveat':'Monotonicity of MAS is enforced, not evidence of correct boundaries. Inspect soft attention, durations and reconstructions.'}
    write_json(out/'reconstruction_summary.json',result)
    for category in ('matched_crops','alignment'):
        for lang,values in result[category].items():
            for key,value in values.items():writer.add_scalar(f'{category}/{lang}/{key}',value,step)
    writer.flush();writer.close()
    write_json(out/'status.json',{'status':'reconstruction_complete','completed':len(indices),'total':len(indices),'updated_at':time.time()})
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
