"""Descriptive batch pace comparison; length matched, no synthesis changes."""
import collections,hashlib,json,re,sqlite3,statistics,time
from pathlib import Path
import numpy as np
from common import ROOT,REPO,norm,atomic_json,config


def stats(rows):
    result={'n':len(rows),'hours':sum(r['duration'] for r in rows)/3600}
    for name in ['duration','tokens','token_rate','content_token_rate','unit_rate','active_fraction','active_token_rate','internal_silence','edge_silence']:
        values=[r[name] for r in rows if r.get(name) is not None]
        result[name]={'median':float(np.median(values)),'p10':float(np.percentile(values,10)),'p90':float(np.percentile(values,90)),'mean':float(np.mean(values))} if values else None
    return result


def make(text,language,duration,ids,meta,source,reference):
    han=len(re.findall('[\u4e00-\u9fff]',text));words=len(re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?",text))
    units=han if language=='zh' else words if language=='en' else han+2*words
    content=sum(t not in {0,1,2,3,4,5,6,9,10,11,12,13,14,15,16} for t in ids)
    active=meta.get('active_fraction')
    return dict(text=text,language=language,duration=duration,tokens=len(ids),token_rate=len(ids)/duration,content_token_rate=content/duration,unit_rate=units/duration,active_fraction=active,active_token_rate=content/(duration*active) if active else None,internal_silence=meta.get('internal_silence'),edge_silence=meta.get('leading_silence',0)+meta.get('trailing_silence',0),source=source,reference=reference,mix_fraction=2*words/max(1,han+2*words))


def matched(old,new,match_reference=False,match_source=False):
    groups=collections.defaultdict(list)
    def key(r):
        return (r['tokens']//50,int(r['mix_fraction']*5) if r['language']=='mixed' else 0,r['reference'] if match_reference else '',r['source'] if match_source else '')
    for r in old:groups[key(r)].append(r)
    pairs=[]
    for r in new:
        base=groups.get(key(r),[])
        if len(base)>=20:pairs.append((r,base))
    out={'covered_new':len(pairs),'total_new':len(new),'min_old_per_stratum':20}
    for metric in ['content_token_rate','unit_rate','active_fraction','active_token_rate']:
        usable=[(a,b) for a,b in pairs if a.get(metric) is not None and all(x.get(metric) is not None for x in b)]
        if not usable:continue
        new_mean=np.mean([a[metric] for a,b in usable]);old_mean=np.mean([np.mean([x[metric] for x in b]) for a,b in usable])
        out[metric]={'new_mean':float(new_mean),'old_length_weighted_mean':float(old_mean),'relative_change_percent':float((new_mean/old_mean-1)*100)}
    return out


def main():
    oldroot=Path('/ai_sds_wuzz/DATA_TTS/MajesticVoice_200h_20260913')
    olddb=sqlite3.connect(f'file:{oldroot}/pipeline.sqlite?mode=ro',uri=True)
    metadata={str(tid):(json.loads(raw or '{}'),json.loads(payload).get('source','unknown')) for tid,raw,payload in olddb.execute('select t.id,c.synthesis,t.payload from texts t join candidates c on c.id=t.accepted_candidate')}
    old=[];seen=set();zhref=hashlib.sha256((ROOT/'reference/zh.wav').read_bytes()).hexdigest()
    for path in (REPO/'data/prepared').glob('*.jsonl'):
        for line in path.open():
            r=json.loads(line)
            if 'token_ids' not in r or 'audio' not in r or 'rejection' in r:continue
            key=norm(r['text'])
            if key in seen:continue
            seen.add(key);meta,source=metadata.get(Path(r['audio']).stem,({},r.get('source','unknown')))
            old.append(make(r['text'],r['language'],r['duration'],r['token_ids'],meta,source,meta.get('reference_sha256',zhref)))
    db=sqlite3.connect(f'file:{ROOT}/pipeline.sqlite?mode=ro',uri=True);new=[]
    for text,lang,payload,duration,raw in db.execute('select t.text,t.language,t.payload,c.duration,c.synthesis from texts t join candidates c on c.id=t.accepted_candidate'):
        r=json.loads(payload);meta=json.loads(raw);new.append(make(text,lang,duration,r['token_ids'],meta,r.get('source','unknown'),meta.get('reference_sha256')))
    result={'created_at':time.time(),'old_scope':'Existing Kokoro prepared train/val/test unique texts from original collection','new_scope':'All accepted new long utterances in one SQLite read snapshot','units':{'zh':'Han characters/sec','en':'English words/sec','mixed':'(Han characters + 2*English words)/sec, proxy only'},'notes':['Rates include pauses unless explicitly active_token_rate.','Same PCM RMS threshold-based activity detector used in old/new collection.','Matched by 50-token bins and mixed English-content fraction; require >=20 old rows per stratum.','Descriptive comparison: sources/content differ; not a same-text controlled experiment.'],'languages':{}}
    for lang in ['zh','en','mixed']:
        a=[r for r in old if r['language']==lang];b=[r for r in new if r['language']==lang]
        result['languages'][lang]={'old_all':stats(a),'old_at_least_15s':stats([r for r in a if r['duration']>=15]),'new':stats(b),'length_matched':matched(a,b),'length_and_reference_matched':matched(a,b,True),'length_reference_source_matched':matched(a,b,True,True),'old_sources':dict(collections.Counter(r['source'] for r in a)),'new_sources':dict(collections.Counter(r['source'] for r in b))}
    atomic_json(ROOT/'reports/pace_comparison.json',result)
    for lang,r in result['languages'].items():
        print(lang,json.dumps({'old_n':r['old_all']['n'],'new_n':r['new']['n'],'old_unit_rate':r['old_all']['unit_rate'],'new_unit_rate':r['new']['unit_rate'],'old_long_unit_rate':r['old_at_least_15s']['unit_rate'],'old_activity':r['old_all']['active_fraction'],'new_activity':r['new']['active_fraction'],'length_matched':r['length_matched'],'source_matched':r['length_reference_source_matched'],'new_sources':r['new_sources']},ensure_ascii=False))

if __name__=='__main__':main()
