import math
import random

import soundfile as sf
import torch
from torch.utils.data import Dataset, Sampler

from training.common import Features, records


class SpeechDataset(Dataset):
    def __init__(self, manifest):
        self.rows = records(manifest)
        self.features = Features()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        audio, sr = sf.read(row['audio'], dtype='float32')
        if sr != 24000 or audio.ndim != 1:
            raise ValueError(row['audio'])
        wave = torch.nn.functional.pad(torch.from_numpy(audio), (5000, 5000))
        mel = self.features(wave)
        mel = mel[:, :mel.shape[-1] // 2 * 2]
        return {'wave': wave, 'mel': mel, 'tokens': torch.tensor([0, *row['token_ids'], 0]),
                'language': row['language'], 'index': index}


def collate(items):
    items = sorted(items, key=lambda item: len(item['tokens']), reverse=True)
    token_lengths = torch.tensor([len(item['tokens']) for item in items])
    mel_lengths = torch.tensor([item['mel'].shape[-1] for item in items])
    tokens = torch.zeros(len(items), int(token_lengths.max()), dtype=torch.long)
    mels = torch.zeros(len(items), 80, int(mel_lengths.max()))
    for index, item in enumerate(items):
        tokens[index, :token_lengths[index]] = item['tokens']
        mels[index, :, :mel_lengths[index]] = item['mel']
    return {'tokens': tokens, 'token_lengths': token_lengths, 'mels': mels,
            'mel_lengths': mel_lengths, 'waves': [i['wave'] for i in items],
            'languages': [i['language'] for i in items], 'indices': [i['index'] for i in items]}


class BucketBatches(Sampler):
    """Deterministic global duration buckets, evenly sharded across DDP ranks."""
    def __init__(self, rows, batch_size, rank=0, world_size=1, epoch=0, start=0, seed=20260920,
                 long_sampling=None):
        self.rows, self.batch_size = rows, batch_size
        self.rank, self.world_size = rank, world_size
        self.epoch, self.start, self.seed = epoch, start, seed
        self.long_sampling = long_sampling
        self.sampling_report = None
        self._batches = None

    def batches(self):
        if self._batches is not None:
            return self._batches
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.rows)))
        if self.long_sampling:
            from training.curriculum import sample_epoch
            indices, self.sampling_report = sample_epoch(self.rows, self.epoch, self.long_sampling, rng)
        rng.shuffle(indices)
        global_batch = self.batch_size * self.world_size
        ordered = []
        for offset in range(0, len(indices), global_batch * 32):
            bucket = indices[offset:offset + global_batch * 32]
            ordered.extend(sorted(bucket, key=lambda i: self.rows[i]['duration']))
        padding = (-len(ordered)) % global_batch
        ordered.extend(ordered[:padding])
        batches = [ordered[i:i+global_batch] for i in range(0,len(ordered),global_batch)]
        rng.shuffle(batches)
        self._batches = batches
        return batches

    def __iter__(self):
        for batch in self.batches()[self.start:]:
            yield batch[self.rank*self.batch_size:(self.rank+1)*self.batch_size]

    def __len__(self):
        return math.ceil(len(self.rows)/(self.batch_size*self.world_size)) - self.start
