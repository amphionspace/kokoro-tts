"""Same first sentence, alone versus longer context; diagnostic audio only."""
import concurrent.futures,json,urllib.request,sys,time
from pathlib import Path
from common import ROOT,REPO,config,atomic_json
from run_workers import synthesis_reference,reference_encoding
from synthesize import save_audio
TEXTS={
'zh':['傍晚的时候我们沿着河边慢慢往前走。','远处的路灯一盏接着一盏亮了起来。','走到桥下时朋友忽然停下来聊起小时候的事情。','我们在附近找了一家小店坐下来休息。'],
'en':['We walked slowly along the river as the evening light began to fade.','Across the water the windows of the old houses began to glow.','My friend stopped at the bridge and told me about her childhood.','We found a quiet cafe nearby and stayed there for a while.'],
'mixed':['今天我们一起检查新的 training 流程。','首先确认每个 batch 里面的文本和音频是否一致。','接下来打开 TensorBoard 查看模型的训练曲线。','最后把两个模型的 sample 放在一起比较语速和音色。']}


def main():
    sys.path.insert(0,str(REPO));from training.frontend import Frontend
    frontend=Frontend();cfg=config();out=ROOT/'reports/prefix_context_probe';out.mkdir(exist_ok=True)
    jobs=[]
    for language,parts in TEXTS.items():
        for repeat in range(3):
            for mode,text in [('single',parts[0]),('long',' '.join(parts))]:
                _,ids=frontend(text,language);assert len(ids)<=510
                jobs.append(dict(id=f'{language}_{repeat}_{mode}',language=language,repeat=repeat,mode=mode,text=text,first_sentence=parts[0],token_count=len(ids),seed=20260923+repeat))
    atomic_json(out/'protocol.json',dict(model=cfg['tts_model'],repeats=3,texts=TEXTS,same_reference=True,same_seed_within_pair=True,sentence_chunking=False,quota_credit=False,created_at=time.time()))
    def generate(row):
        path=out/f"{row['id']}.wav"
        if path.exists():
            import soundfile as sf
            return dict(row,audio=str(path),duration=sf.info(path).duration)
        ref=synthesis_reference(cfg,row['language']);uri,sha=reference_encoding(ref['reference_audio'])
        body=dict(model=cfg['tts_model'],input=row['text'],ref_audio=uri,ref_text=ref['prompt_text'],response_format='wav',stream=False,seed=row['seed'],max_new_tokens=1000)
        # Both versions of a pair use the same service.
        endpoint=cfg['tts_endpoints'][row['repeat']%4]
        request=urllib.request.Request(endpoint+'/v1/audio/speech',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=1800) as response:data=response.read()
        info=save_audio(data,dict(text=row['text'],language='English' if row['language']=='en' else 'Chinese',output_24k=str(path),output_48k=str(out/f"{row['id']}_48k.wav")))
        return dict(row,audio=str(path),reference_sha256=sha,**info)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        with (out/'manifest.jsonl').open('w') as f:
            for row in pool.map(generate,jobs):f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();print(row['id'],row['duration'],flush=True)

if __name__=='__main__':main()
