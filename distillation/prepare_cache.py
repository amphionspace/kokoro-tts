"""Finalize teacher shards only when all workers completed and source splits match."""
import argparse,json,collections
from pathlib import Path
from training.common import ROOT,records,write_json,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--world-size',type=int,default=4);a=p.parse_args();cache=a.cache.resolve()
    protocols=[]
    for rank in range(a.world_size):
        complete=json.loads((cache/f'complete_rank{rank}.json').read_text());protocols.append(complete['protocol'])
    if any(p!=protocols[0] for p in protocols):raise ValueError('Teacher worker protocol mismatch')
    report={'teacher':protocols[0],'splits':{}}
    for split in ['train','val']:
        source=records(ROOT/f'data/prepared/{split}.jsonl')
        rows=[]
        for rank in range(a.world_size):rows+=records(cache/f'{split}_rank{rank}.jsonl')
        rows.sort(key=lambda r:r['source_index'])
        if len(rows)!=len(source) or [r['source_index'] for r in rows]!=list(range(len(source))):raise ValueError(f'{split}: missing/duplicate rows')
        for row,original in zip(rows,source):
            if row['token_ids']!=original['token_ids'] or row['text']!=original['text'] or row['language']!=original['language']:raise ValueError('Source row mismatch')
            if row['source_split']!=split or not Path(row['cache']).is_file():raise ValueError('Cache missing or wrong split')
        target=cache/f'{split}.jsonl';temp=target.with_suffix('.tmp');temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));temp.replace(target)
        report['splits'][split]={'count':len(rows),'hours':sum(r['duration'] for r in rows)/3600,'languages':dict(collections.Counter(r['language'] for r in rows)),'max_seconds':max(r['duration'] for r in rows),'clipped_rows':sum(r['clipped_fraction']>0 for r in rows),'manifest_sha256':digest(target)}
    write_json(cache/'ready.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
