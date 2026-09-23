"""A directly optimized deployment voice, initialized only from training references."""
import torch
import torch.distributed as dist
from torch import nn

from training.common import ROOT, Features, records, rng_state, restore_rng


class TrainableVoice(nn.Module):
    def __init__(self):
        super().__init__()
        self.vector = nn.Parameter(torch.zeros(1, 256))
        self.register_buffer('initialized', torch.tensor(False))
        self.register_buffer('initial_vector', torch.zeros(1, 256))


@torch.no_grad()
def initialize_voicepack(model,reference_manifest=None):
    """Initialize in place; preserve optimizer identity, RNG and encoder modes."""
    import soundfile as sf
    voice = model.voicepack
    if bool(voice.initialized):
        return False
    state = rng_state()
    encoders = [model.style_encoder, model.predictor_encoder]
    modes = [(m, m.training) for encoder in encoders for m in encoder.modules()]
    try:
        for encoder in encoders:
            encoder.eval()
        device = voice.vector.device
        if not dist.is_initialized() or dist.get_rank() == 0:
            features = Features().to(device)
            vectors = []
            with torch.autocast('cuda', enabled=False):
                for row in records(reference_manifest or ROOT/'data/prepared/voicepack_references.jsonl'):
                    audio, sr = sf.read(row['audio'], dtype='float32')
                    if sr != 24000 or audio.ndim != 1:
                        raise ValueError(row['audio'])
                    wave = torch.nn.functional.pad(torch.from_numpy(audio).to(device), (5000, 5000))
                    mel = features(wave)[None, None]
                    vectors.append(torch.cat([encoder(mel) for encoder in encoders], dim=-1))
                mean = torch.cat(vectors).mean(0, keepdim=True)
                if not torch.isfinite(mean).all():
                    raise FloatingPointError('Invalid initial voicepack')
                voice.vector.copy_(mean)
        if dist.is_initialized():
            dist.broadcast(voice.vector, src=0)
        voice.initial_vector.copy_(voice.vector)
        voice.initialized.fill_(True)
    finally:
        for module, mode in modes:
            module.training = mode
        restore_rng(state)
    return True
