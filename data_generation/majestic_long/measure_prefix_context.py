"""Estimate the shared first-sentence span by acoustic DTW and pause boundaries."""
import json,html,collections
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from scipy.spatial.distance import cdist
from common import ROOT,atomic_json

OUT=ROOT/'reports/prefix_context_probe'
HOP=160


def active_bounds(y):
    n=160;pad=(-len(y))%n
    rms=np.sqrt(np.mean(np.pad(y,(0,pad)).reshape(-1,n)**2,axis=1))
    active=rms>max(.002,float(rms.max())*.02)
    indices=np.flatnonzero(active)
    if not len(indices):raise ValueError('Silent sample')
    return int(indices[0]*n),int(min(len(y),(indices[-1]+1)*n)),active


def features(y):
    return librosa.feature.mfcc(y=y,sr=16000,n_mfcc=20,n_fft=400,hop_length=HOP)[1:]


def measure(single,long):
    a,_=librosa.load(single,sr=16000);b,_=librosa.load(long,sr=16000)
    a0,a1,_=active_bounds(a);b0,_,active=active_bounds(b)
    x=features(a[a0:a1]);limit=min(len(b),b0+int((a1-a0)*1.8)+8000);y=features(b[b0:limit])
    joined=np.concatenate([x,y],axis=1);mean=joined.mean(1,keepdims=True);std=joined.std(1,keepdims=True).clip(.1)
    x=(x-mean)/std;y=(y-mean)/std
    cost=cdist(x.T,y.T,'cosine')
    D,wp=librosa.sequence.dtw(C=cost,subseq=True,backtrack=True,step_sizes_sigma=np.array([[1,1],[1,2],[2,1]]),weights_mul=np.array([1.,1.5,1.5]))
    dtw_start=b0+int(wp[-1,1])*HOP;dtw_end=b0+int(wp[0,1]+1)*HOP
    assert abs(dtw_start-b0)/16000<.25,('alignment starts too late',dtw_start,b0)
    # Refine to the adjacent silence onset; never count the sentence-final pause.
    intervals=[];start=None
    for i,on in enumerate(active):
        if not on and start is None:start=i
        if on and start is not None:
            if i-start>=6 and start*HOP>b0+int((a1-a0)*.5):intervals.append((start*HOP,i*HOP))
            start=None
    candidates=[p for p in intervals if abs(p[0]-dtw_end)/16000<=.35]
    end=min(candidates,key=lambda p:abs(p[0]-dtw_end))[0] if candidates else dtw_end
    duration=(end-b0)/16000;alone=(a1-a0)/16000
    return dict(single_speech_start=a0/16000,single_speech_end=a1/16000,single_speech_seconds=alone,long_speech_start=b0/16000,long_first_sentence_end=end/16000,long_first_sentence_seconds=duration,dtw_endpoint=dtw_end/16000,pause_refined=bool(candidates),change_percent=100*(duration/alone-1),alignment_mean_cost=float(np.mean(cost[wp[:,0],wp[:,1]])))


def main():
    rows=[json.loads(x) for x in (OUT/'manifest.jsonl').read_text().splitlines()];byid={r['id']:r for r in rows};results=[]
    for lang in ['zh','en','mixed']:
        for repeat in range(3):
            short=byid[f'{lang}_{repeat}_single'];long=byid[f'{lang}_{repeat}_long'];r=measure(short['audio'],long['audio']);r.update(language=lang,repeat=repeat,first_sentence=short['text'],single_audio=short['audio'],long_audio=long['audio'],single_file_seconds=short['duration'],long_file_seconds=long['duration'])
            wave,sr=sf.read(long['audio']);path=OUT/f'{lang}_{repeat}_context_prefix.wav';sf.write(path,wave[round(r['long_speech_start']*sr):round(r['long_first_sentence_end']*sr)],sr,subtype='PCM_16');r['context_prefix_audio']=str(path)
            wave,sr=sf.read(short['audio']);path=OUT/f'{lang}_{repeat}_single_trimmed.wav';sf.write(path,wave[round(r['single_speech_start']*sr):round(r['single_speech_end']*sr)],sr,subtype='PCM_16');r['single_trimmed_audio']=str(path);results.append(r)
    summary={lang:{key:float(np.median([r[key] for r in results if r['language']==lang])) for key in ['single_speech_seconds','long_first_sentence_seconds','change_percent','alignment_mean_cost']} for lang in ['zh','en','mixed']}
    atomic_json(OUT/'timing.json',dict(method='MFCC subsequence DTW, then nearby RMS silence-onset refinement; approximate automatic boundaries, not forced word alignment',frame_seconds=.01,results=results,summary=summary))
    page=['<!doctype html><meta charset="utf-8"><title>首句上下文对照</title><style>body{max-width:1000px;margin:30px auto;font:16px system-ui;line-height:1.6}table{width:100%}td{padding:12px}audio{width:100%}</style><h1>同一句话：单独合成与加长上下文</h1><p>VoxCPM2，同一参考音色、同种子；每种语言重复3次。首句时长通过声学DTW和静音边界估计，去除首尾静音，不含句后停顿。试听片段仅裁剪，不变速。诊断音频不进入训练配额。</p>']
    for r in results:
        page.append(f'<h2>{r["language"]} · seed {20260923+r["repeat"]}</h2><p>{html.escape(r["first_sentence"])}</p><p>单独 {r["single_speech_seconds"]:.2f}s；长段落首句 {r["long_first_sentence_seconds"]:.2f}s；变化 {r["change_percent"]:+.1f}%</p><table><tr>')
        for label,key in [('单句（去首尾静音）','single_trimmed_audio'),('长段落首句','context_prefix_audio'),('完整长段落','long_audio')]:page.append(f'<td>{label}<audio controls preload="none" src="{Path(r[key]).name}"></audio></td>')
        page.append('</tr></table>')
    (OUT/'index.html').write_text('\n'.join(page));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
