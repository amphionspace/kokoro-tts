"""Paired long-input evaluation; no sentence chunking or token truncation."""
import argparse, json, re, shutil, hashlib
from pathlib import Path
import torch, soundfile as sf
from training.common import ROOT, records, write_json, digest
from training.frontend import Frontend
from kokoro import KModel

OUT=ROOT/'runs/long_text_comparison_20260922'
EXPORTS={'baseline':None,
         'long_weighted':ROOT/'runs/majestic_s2_20ep_long_resume6k_20260921/eval/stage2_final'}
CUSTOM={
'zh':'傍晚的时候，我们沿着河边慢慢往前走。远处的路灯一盏接着一盏亮了起来，水面上倒映着暖黄色的光。走到桥下时，朋友忽然停下来，说他想起了小时候住过的地方。我们没有急着回家，而是在附近找了一家小店，坐下来聊了很久。',
'en':'We walked slowly along the river as the evening light began to fade. Across the water, the windows of the old houses glowed in the gathering darkness. When we reached the bridge, my friend stopped and told me about the town where she grew up. Instead of going straight home, we found a quiet cafe and talked about the places we hoped to visit together.',
'mixed':'今天我们要介绍新的training流程。首先检查每个batch中的文本和音频，确认它们的内容一致，然后再开始训练。完成第一轮之后，我们会打开TensorBoard，查看loss曲线和生成的语音。最后把baseline和新模型放在一起，使用相同的测试文本，比较语速、停顿和音色。'}

def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline-export',type=Path,required=True,help='Explicit baseline: original export was deleted; rerun is a different model')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();OUT=args.output.resolve();EXPORTS['baseline']=args.baseline_export.resolve()
    for export in EXPORTS.values():
        for name in ['kokoro.pth','majestic.pt']:
            if not (export/name).is_file():raise FileNotFoundError(export/name)
    torch.set_num_threads(2);torch.manual_seed(20260922)
    OUT.mkdir(exist_ok=False)
    frontend=Frontend(); selected=[]
    val=records(ROOT/'data/prepared/val.jsonl')
    for lang in ('zh','en','mixed'):
        rows=sorted((r for r in val if r['language']==lang),key=lambda r:(-len(r['token_ids']),r['audio']))[:4]
        for i,row in enumerate(rows):
            selected.append(dict(id=f'{lang}_val_{i+1}',language=lang,ref_text=row['text'],token_ids=row['token_ids'],phonemes=row['phonemes'],reference_audio=row['audio'],selection='longest validation texts per language'))
        phonemes,ids=frontend(CUSTOM[lang],lang)
        if len(ids)>510:raise ValueError((lang,len(ids)))
        selected.append(dict(id=f'{lang}_context',language=lang,ref_text=CUSTOM[lang],token_ids=ids,phonemes=phonemes,selection='authored four-sentence probe'))
    for row in selected:
        assert len(row['token_ids'])<=510
        # Preserve exact token identity to isolate context, avoiding G2P changes.
        end=next((i+1 for i,c in enumerate(row['phonemes']) if c in '.!?'),None)
        if end and end<len(row['token_ids'])-8:
            row['prefix_ids']=row['token_ids'][:end]
            row['prefix_text']=re.split(r'(?<=[。！？.!?])',row['ref_text'],maxsplit=1)[0]
    (OUT/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    provenance={'speed':1,'sentence_chunking':False,'selection':'four longest validation texts per language plus one authored four-sentence probe per language','model_hashes':{},'validation_manifest_sha256':digest(ROOT/'data/prepared/val.jsonl')}
    for label,export in EXPORTS.items():
        dest=OUT/label;dest.mkdir();(dest/'wavs').mkdir();(dest/'prefix').mkdir()
        provenance['model_hashes'][label]={name:digest(export/name) for name in ['kokoro.pth','majestic.pt']}
        model=KModel(repo_id='hexgrad/Kokoro-82M',config=str(ROOT/'models/Kokoro-82M/config.json'),model=str(export/'kokoro.pth')).cuda().eval()
        voice=torch.load(export/'majestic.pt',map_location='cuda',weights_only=True)[0]
        with torch.inference_mode(),(dest/'synthesis.jsonl').open('w') as stream:
            for row in selected:
                tokens=torch.tensor([[0,*row['token_ids'],0]],device='cuda')
                wave,duration=model.forward_with_tokens(tokens,voice,speed=1)
                assert torch.isfinite(wave).all()
                path=dest/'wavs'/(row['id']+'.wav');sf.write(path,wave.cpu().numpy().reshape(-1),24000,subtype='PCM_16')
                result={k:v for k,v in row.items() if k not in ('prefix_ids','token_ids','phonemes')}
                result.update(audio=str(path),duration=wave.numel()/24000,token_count=len(row['token_ids']),duration_frames=duration.cpu().tolist(),group='majestic_'+row['language'],speaker=1)
                if 'prefix_ids' in row:
                    prefix,sd=model.forward_with_tokens(torch.tensor([[0,*row['prefix_ids'],0]],device='cuda'),voice,speed=1)
                    p=dest/'prefix'/(row['id']+'.wav');sf.write(p,prefix.cpu().numpy().reshape(-1),24000,subtype='PCM_16')
                    short_frames=int(sd[1:-1].sum());context_frames=int(duration[1:1+len(row['prefix_ids'])].sum())
                    result.update(prefix_audio=str(p),prefix_frames_alone=short_frames,prefix_frames_in_context=context_frames,prefix_change_percent=100*(context_frames/short_frames-1))
                stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush()
                print(label,row['id'],len(row['token_ids']),round(result['duration'],2),flush=True)
        del model;torch.cuda.empty_cache()
    write_json(OUT/'protocol.json',provenance)

if __name__=='__main__':main()
