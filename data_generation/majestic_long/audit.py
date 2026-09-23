"""Validate real accepted hours, whole-text limits, QC, headers, and disjointness."""
import collections,json,sqlite3,wave
from pathlib import Path
from common import ROOT,config,connection,totals,goals,norm,atomic_json,near_match
from text_rules import check,sentences
from run_workers import judge


def main():
    cfg=config();db=connection();ex=sqlite3.connect(f'file:{ROOT}/text/exclusions.sqlite?mode=ro',uri=True);ex.row_factory=sqlite3.Row
    old={r[0] for r in ex.execute('select norm from old')};oldphones={r[0] for r in ex.execute('select key from phones')}
    seen=set();hours=collections.Counter();counts=collections.Counter();mins={};maxs={}
    for row in db.execute('select t.text,t.language,t.phone_key,t.payload,c.* from texts t join candidates c on c.id=t.accepted_candidate'):
        row=dict(row);p=json.loads(row['payload']);meta,reason=check(row['text']);assert reason is None,reason
        key=norm(row['text']);assert key not in old and key not in seen and row['phone_key'] not in oldphones;seen.add(key)
        assert near_match(ex,'old_fts',row['text']) is None
        assert len(p['token_ids'])<=510 and p['kokoro_tokens_with_boundaries']==len(p['token_ids'])+2
        assert p['voxcpm_total_token_budget']<=cfg['tts_max_model_len']
        assert not judge(row,json.loads(row['metrics']),cfg['thresholds'])
        with wave.open(row['audio_24k']) as w:
            assert w.getframerate()==24000 and w.getnchannels()==1
            duration=w.getnframes()/24000
        assert 15<=duration<=cfg['max_audio_seconds'] and abs(duration-row['duration'])<.001
        lang=row['language'];hours[lang]+=duration/3600;counts[lang]+=1;mins[lang]=min(mins.get(lang,duration),duration);maxs[lang]=max(maxs.get(lang,duration),duration)
    assert all(hours[l]>=v for l,v in cfg['targets_train_hours'].items()),dict(hours)
    atomic_json(ROOT/'reports/final_audit.json',dict(passed=True,hours=hours,counts=counts,min_seconds=mins,max_seconds=maxs,targets=cfg['targets_train_hours'],min_sentences=3,max_kokoro_tokens=510,previous_collection_overlap=False,training_started=False))

if __name__=='__main__':main()
