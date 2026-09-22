import numpy as np
import torch
from torch.utils.data import Dataset
from training.common import records

class CacheDataset(Dataset):
    def __init__(self,manifest):self.rows=records(manifest)
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        with np.load(row['cache'],allow_pickle=False) as data:
            ids=torch.from_numpy(data['ids'].astype(np.int64));dur=torch.from_numpy(data['dur'].astype(np.float32))
            if len(ids)!=len(dur) or len(data['audio'])!=int(dur.sum())*600:raise ValueError(row['cache'])
            return {'ids':ids,'durations':dur,'f0':torch.from_numpy(data['f0'].copy()),'energy':torch.from_numpy(data['energy'].copy()),'audio':torch.from_numpy(data['audio'].astype(np.float32)/32767),'language':row['language'],'index':index}

def collate(items):
    items=sorted(items,key=lambda x:len(x['ids']),reverse=True)
    result={'lengths':torch.tensor([len(i['ids']) for i in items]),'audio':[i['audio'] for i in items],'languages':[i['language'] for i in items],'indices':[i['index'] for i in items]}
    for key in ['ids','durations','f0','energy']:
        result[key]=torch.nn.utils.rnn.pad_sequence([i[key] for i in items],batch_first=True)
    return result

def move(batch,device):
    return {k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
