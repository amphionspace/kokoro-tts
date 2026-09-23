"""Exact Kokoro phonemization, VoxCPM2 context budgeting and old/new dedup."""
import collections,hashlib,json,math,multiprocessing as mp,os,sqlite3,sys,time
from pathlib import Path
from common import ROOT,REPO,ASSETS,config,connection,initialize,norm,atomic_json
from text_rules import check,sentences


def init_worker():
    sys.path.insert(0,str(REPO))
    import torch
    torch.set_num_threads(1)
    from training.frontend import Frontend
    from transformers import AutoTokenizer
    import soundfile as sf
    global frontend,tokenizer,splitmap,reference_cost,cfg
    frontend=Frontend();cfg=config();tokenizer=AutoTokenizer.from_pretrained(str(ASSETS/'VoxCPM2'),local_files_only=True,trust_remote_code=True)
    splitmap={}
    for token,tid in tokenizer.get_vocab().items():
        clean=token.replace('▁','')
        if len(clean)>=2 and all(0x4e00<=ord(c)<=0x9fff or 0x3400<=ord(c)<=0x4dbf or 0xf900<=ord(c)<=0xfaff or 0x20000<=ord(c)<=0x2fa1f for c in clean):
            ids=tokenizer.convert_tokens_to_ids(list(clean))
            if all(i!=tokenizer.unk_token_id for i in ids):splitmap[tid]=ids
    model=json.loads((ASSETS/'VoxCPM2/config.json').read_text());patch=model['patch_size']*math.prod(model['audio_vae_config']['encoder_rates']);sr=model['audio_vae_config']['sample_rate']
    reference_cost={}
    for language in ['zh','en','mixed']:
        ref=cfg.get('synthesis_references',{}).get(language,{'reference_audio':cfg['reference_audio'],'prompt_text':cfg['prompt_text']})
        info=sf.info(ref['reference_audio']);audio_tokens=math.ceil(math.ceil(info.frames*sr/info.samplerate)/patch)
        reference_cost[language]=audio_tokens+len(vox_ids(ref['prompt_text']))+1


def vox_ids(text):
    ids=[j for i in tokenizer.encode(text,add_special_tokens=True) for j in splitmap.get(i,[i])]
    return ids[1:] if ids and ids[0]==tokenizer.bos_token_id else ids


def process(r):
    try:
        checked,reason=check(r['text'])
        if reason:return None,reason
        phones,ids=frontend(r['text'],r['language'])
        if len(ids)>cfg['kokoro_max_tokens']:return None,'kokoro_over_510'
        prefill=len(vox_ids(r['text']))+reference_cost[r['language']]
        if prefill+cfg['synthesis_max_tokens']>cfg['tts_max_model_len']:return None,'voxcpm_context'
        r.update(ids=ids,token_ids=ids,phonemes=phones,kokoro_tokens_with_boundaries=len(ids)+2,voxcpm_prefill_tokens=prefill,voxcpm_total_token_budget=prefill+cfg['synthesis_max_tokens'],split='train',speaker=1,synthetic=True)
        return r,None
    except Exception as exc:return None,type(exc).__name__+':'+str(exc)[:160]


def main():
    initialize();source=ROOT/'text/raw_candidates.jsonl'
    while not (ROOT/'text/scan_complete.json').exists():time.sleep(5)
    db=connection();db.executescript('''CREATE TABLE IF NOT EXISTS pool(id INTEGER PRIMARY KEY,language TEXT,split TEXT,source_group TEXT,length_bin TEXT,norm TEXT UNIQUE,phone_key TEXT UNIQUE,payload TEXT,estimated_seconds REAL,source_seconds REAL,rank_key TEXT,state TEXT DEFAULT 'available');CREATE INDEX IF NOT EXISTS pool_select ON pool(state,split,language,length_bin,rank_key);''')
    ex=sqlite3.connect(f'file:{ROOT}/text/exclusions.sqlite?mode=ro',uri=True);oldtext={x[0] for x in ex.execute('select norm from old')};oldphones={x[0] for x in ex.execute('select key from phones')};oldsentences={norm(s) for (text,) in ex.execute('select text from old') for s in sentences(text) if len(norm(s))>=20};ex.close()
    counts=collections.Counter();hours=collections.Counter();started=time.time();last=time.time()
    rows=(json.loads(line) for line in source.open())
    with mp.get_context('spawn').Pool(int(os.environ.get('TEXT_WORKERS','8')),initializer=init_worker) as pool:
        for r,reason in pool.imap_unordered(process,rows,chunksize=16):
            if reason:counts[reason]+=1;continue
            key=norm(r['text']);phone=hashlib.sha256(bytes(r['ids'])).hexdigest()
            if key in oldtext or phone in oldphones or any(norm(part) in oldsentences for part in sentences(r['text'])):counts['old_text_or_phone']+=1;continue
            rid=int(hashlib.sha256((r['source']+r['source_id']).encode()).hexdigest()[:15],16)
            cur=db.execute('insert or ignore into pool(id,language,split,source_group,length_bin,norm,phone_key,payload,estimated_seconds,source_seconds,rank_key) values(?,?,?,?,?,?,?,?,?,?,?)',(rid,r['language'],'train',r['source_group'],'long',key,phone,json.dumps(r,ensure_ascii=False),r['estimated_seconds'],r['source_seconds'],hashlib.sha256(('20260922:'+str(rid)).encode()).hexdigest()))
            if cur.rowcount:counts['imported_'+r['language']]+=1;hours[r['language']]+=r['estimated_seconds']/3600
            else:counts['new_duplicate']+=1
            if time.time()-last>10:
                atomic_json(ROOT/'text/import_status.json',dict(counts=counts,estimated_hours=hours,elapsed=time.time()-started));last=time.time()
    report=dict(counts=counts,estimated_hours=hours,elapsed=time.time()-started,source=str(source),complete=True)
    atomic_json(ROOT/'text/import_complete.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
