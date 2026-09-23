"""Scan local real transcripts and exclusions without loading source audio."""
import collections,json,re,sqlite3,time,hashlib
from pathlib import Path
from common import ROOT,ASSETS,REPO,norm,atomic_json,initialize
from text_rules import check


def main():
    initialize();started=time.time();oldroot=Path('/ai_sds_wuzz/DATA_TTS/MajesticVoice_200h_20260913')
    ex=ROOT/'text/exclusions.sqlite'
    if not ex.exists():
        source=sqlite3.connect(f'file:{oldroot}/text/exclusions.sqlite?mode=ro',uri=True);dest=sqlite3.connect(ex);source.backup(dest);source.close();dest.close()
    db=sqlite3.connect(ex);counts=collections.Counter();heldout_groups=set()
    old=sqlite3.connect(f'file:{oldroot}/pipeline.sqlite?mode=ro',uri=True)
    def add(text,source):
        key=norm(text)
        cur=db.execute('insert or ignore into old(norm,text,source) values(?,?,?)',(key,text,source))
        if cur.rowcount:db.execute('insert into old_fts(rowid,norm) values(?,?)',(cur.lastrowid,key))
    for (text,) in old.execute('select text from texts'):add(text,'previous_collection')
    heldout_groups.update(x[0] for x in old.execute("select distinct source_group from pool where split!='train'"));old.close()
    base=Path('/ai_sds_wuzz/DATA_TTS/Emilia2_TTS_prepared/LM-TTS-Training')
    for p in sorted(base.glob('*/val.jsonl'))+sorted(base.glob('*/test.jsonl')):
        for line in p.open():
            r=json.loads(line);add(r['text'],str(p));meta=r.get('source',{})
            if isinstance(meta,dict) and meta.get('recording_id'):heldout_groups.add('Emilia2/'+meta['recording_id'])
    for p in (REPO/'data/prepared').glob('*.jsonl'):
        for line in p.open():
            r=json.loads(line);add(r['text'],str(p))
            if 'token_ids' in r:db.execute('insert or ignore into phones values(?)',(hashlib.sha256(bytes(r['token_ids'])).hexdigest(),))
    db.commit();excluded={x[0] for x in db.execute('select norm from old')};db.close()
    seen=set();hours=collections.Counter();limits={'zh':250,'en':150,'mixed':150};out=ROOT/'text/raw_candidates.jsonl'
    dest=out.with_suffix('.tmp').open('w');last=time.time()
    def admit(text,meta):
        nonlocal last
        counts['scanned']+=1;r,reason=check(text)
        if reason:counts[reason]+=1;return
        lang=r['language']
        if hours[lang]>=limits[lang]:return
        group=meta['source_group'];key=norm(r['text'])
        if key in seen or key in excluded or group in heldout_groups:counts['duplicate_or_heldout']+=1;return
        r.update(meta);dest.write(json.dumps(r,ensure_ascii=False)+'\n');seen.add(key);hours[lang]+=r['estimated_seconds']/3600;counts['eligible_'+lang]+=1
        if time.time()-last>10:
            dest.flush();atomic_json(ROOT/'text/scan_status.json',dict(counts=counts,estimated_hours=hours,elapsed=time.time()-started));last=time.time()
    foundation=ASSETS/'data_24k/foundation/dataset.sqlite';fd=sqlite3.connect(f'file:{foundation}?mode=ro',uri=True)
    for (payload,) in fd.execute("select payload from samples where split='train'"):
        r=json.loads(payload);p=Path(r['audio']);group='HiFiTTS/'+p.parent.name if r['source']=='HiFiTTS' else 'Premium/'+p.stem.split('_S')[0]
        admit(r['text'],dict(source=r['source'],source_id=str(r.get('id',r['audio'])),source_group=group,source_seconds=r['duration'],source_manifest=str(foundation)))
    fd.close();dest.flush();atomic_json(ROOT/'text/scan_status.json',dict(stage='emilia',counts=counts,estimated_hours=hours,elapsed=time.time()-started))
    source=base/'emilia-full-en-zh/train.jsonl';pattern=re.compile(rb'"text"\s*:\s*("(?:[^"\\]|\\.)*")')
    with source.open('rb') as f:
        for index,line in enumerate(f,1):
            match=pattern.search(line)
            if not match:continue
            text=json.loads(match[1]);pre,reason=check(text)
            if reason:counts['emilia_'+reason]+=1;continue
            if hours[pre['language']]>=limits[pre['language']]:continue
            r=json.loads(line);group=r.get('source',{}).get('recording_id')
            if not group:continue
            admit(text,dict(source='Emilia2',source_id=r['id'],source_group='Emilia2/'+group,source_seconds=r['duration'],source_manifest=str(source),source_line=index))
            if all(hours[k]>=v for k,v in limits.items()):break
    dest.close();out.with_suffix('.tmp').replace(out)
    atomic_json(ROOT/'text/scan_complete.json',dict(counts=counts,estimated_hours=hours,excluded_texts=len(excluded),heldout_groups=len(heldout_groups),elapsed=time.time()-started,source=str(source),source_size=source.stat().st_size,note='Text estimates only, not accepted audio hours; exact Kokoro and VoxCPM2 limits checked in prepare_pool.py'))

if __name__=='__main__':main()
