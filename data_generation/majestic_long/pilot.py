"""Small whole-utterance generation check before admitting quota work."""
import concurrent.futures,json,urllib.request,time
from pathlib import Path
from common import ROOT,config,connection,atomic_json
from run_workers import synthesis_reference,reference_encoding,wave_checks
from synthesize import save_audio


def main():
    cfg=config();db=connection();out=ROOT/'pilot';out.mkdir(exist_ok=True);rows=[]
    for language in ('zh','en','mixed'):
        selected=db.execute("select * from pool where language=? and state='available' order by rank_key limit 4",(language,)).fetchall()
        assert len(selected)==4,f'Insufficient {language} pilot texts'
        rows.extend(dict(r) for r in selected)
    def generate(index,row):
        payload=json.loads(row['payload']);ref=synthesis_reference(cfg,row['language']);uri,sha=reference_encoding(ref['reference_audio'])
        body=dict(model=cfg['tts_model'],input=payload['text'],ref_audio=uri,ref_text=ref['prompt_text'],response_format='wav',stream=False,seed=cfg['seed']+index,max_new_tokens=cfg['synthesis_max_tokens'])
        request=urllib.request.Request(cfg['tts_endpoints'][index%4]+'/v1/audio/speech',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=1800) as response:data=response.read()
        paths={f'output_{rate}k':str(out/f'{index}_{rate}k.wav') for rate in (24,48)}
        result=save_audio(data,dict(text=payload['text'],language='English' if row['language']=='en' else 'Chinese',**paths))
        return dict(language=row['language'],text=payload['text'],sentence_count=payload['sentence_count'],tokens=payload['kokoro_tokens_with_boundaries'],reference_sha256=sha,**result,**wave_checks(paths['output_24k']))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(lambda pair:generate(*pair),enumerate(rows)))
    atomic_json(ROOT/'reports/pilot.json',dict(results=results,quota_credit=False,scope='Generation/length check; production applies complete ASR/speaker/DNSMOS gates'))
    for lang in ('zh','en','mixed'):
        assert any(r['language']==lang and 15<=r['duration']<=cfg['max_audio_seconds'] for r in results),f'No usable-duration pilot for {lang}'

if __name__=='__main__':main()
