"""Check deterministic language-balanced sampling and DDP resume slicing."""
import json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.common import records,write_json
from training.data import BucketBatches
from training.curriculum import validate_long_sampling

def main():
 config=json.loads((ROOT/'configs/stage2_20ep_long.json').read_text())
 sampling=config['long_sampling'];validate_long_sampling(sampling,config['stage2_epochs'])
 rows=records(ROOT/'data/prepared/train.jsonl');report={}
 rng=random.getstate()
 for epoch in [0,10,11,12,13,14,15,16,19]:
  sampler=BucketBatches(rows,16,world_size=4,epoch=epoch,long_sampling=sampling)
  batches=sampler.batches()
  assert len(batches)==840
  assert batches==BucketBatches(rows,16,world_size=4,epoch=epoch,long_sampling=sampling).batches()
  if epoch<12:assert batches==BucketBatches(rows,16,world_size=4,epoch=epoch).batches()
  stats=sampler.sampling_report
  assert sum(v['samples'] for v in stats['languages'].values())==len(rows)
  for lang,v in stats['languages'].items():
   assert v['samples']==sum(r['language']==lang for r in rows)
   if epoch>=16:assert abs(v['sampled_long_fraction']-v['expected_long_fraction'])<.025
  shards=[list(BucketBatches(rows,16,rank=rank,world_size=4,epoch=epoch,start=7,long_sampling=sampling)) for rank in range(4)]
  assert [[i for rank in range(4) for i in shards[rank][j]] for j in range(len(shards[0]))]==batches[7:]
  report[str(epoch+1)]=stats
 assert random.getstate()==rng
 write_json(ROOT/'reports/long_sampling_check.json',report)
 print('Sampler checks passed: unchanged early epochs, deterministic replay, DDP shards, language counts and long fraction.')
if __name__=='__main__':main()
