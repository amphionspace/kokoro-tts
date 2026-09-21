import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('KOKORO_PROJECT_ROOT',str(CODE_ROOT)))
STYLE_SRC = CODE_ROOT / 'vendor/styletts2'
sys.path[:0] = [str(STYLE_SRC), str(CODE_ROOT / 'vendor')]
BASE_KEYS = ('bert', 'bert_encoder', 'predictor', 'text_encoder', 'decoder')


def records(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class Features(torch.nn.Module):
    """Preserve the auxiliary checkpoints' legacy filterbank everywhere.

    Waveforms are 24 kHz. The original StyleTTS2 ASR/JDC path used the
    default 16 kHz Mel filterbank on these samples. This is an explicit
    compatibility convention, NOT waveform resampling. Training and voice
    extraction share this exact transform and normalization.
    """
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=2048, win_length=1200,
            hop_length=300, n_mels=80, f_min=0, f_max=8000,
        )

    def forward(self, wave):
        return (torch.log(self.mel(wave.float()) + 1e-5) + 4) / 4


def canonical_state(state):
    result = {}
    for key, value in state.items():
        while key.startswith('module.'):
            key = key[7:]
        if key in result:
            raise ValueError(f'Checkpoint key collision: {key}')
        result[key] = value
    return result


def load_exact(module, state):
    while hasattr(module, 'module'):
        module = module.module
    state = canonical_state(state)
    module.load_state_dict(state, strict=True)
    return {'matched_tensors': len(state), 'missing': [], 'unexpected': []}


def make_model(device='cpu', base=True, with_voicepack=False):
    from munch import Munch
    from transformers import AlbertConfig
    from kokoro.modules import CustomAlbert
    from models import TextEncoder, ProsodyPredictor, StyleEncoder
    from Utils.ASR.models import ASRCNN
    from Utils.JDC.model import JDCNet
    import yaml
    from Modules.istftnet import Decoder
    from Modules.discriminators import MultiPeriodDiscriminator, MultiResSpecDiscriminator, WavLMDiscriminator
    cfg = json.loads((ROOT / 'models/Kokoro-82M/config.json').read_text())
    model = Munch(
        bert=CustomAlbert(AlbertConfig(vocab_size=cfg['n_token'], **cfg['plbert'])),
        bert_encoder=torch.nn.Linear(cfg['plbert']['hidden_size'], cfg['hidden_dim']),
        text_encoder=TextEncoder(cfg['hidden_dim'], cfg['text_encoder_kernel_size'], cfg['n_layer'], cfg['n_token']),
        predictor=ProsodyPredictor(cfg['style_dim'], cfg['hidden_dim'], cfg['n_layer'], cfg['max_dur'], cfg['dropout']),
        decoder=Decoder(dim_in=cfg['hidden_dim'], style_dim=cfg['style_dim'], dim_out=cfg['n_mels'], **cfg['istftnet']),
        style_encoder=StyleEncoder(cfg['dim_in'], cfg['style_dim'], cfg['max_conv_dim']),
        predictor_encoder=StyleEncoder(cfg['dim_in'], cfg['style_dim'], cfg['max_conv_dim']),
        text_aligner=ASRCNN(**yaml.safe_load((STYLE_SRC / 'Utils/ASR/config.yml').read_text())['model_params']),
        pitch_extractor=JDCNet(num_class=1,seq_len=192),
        mpd=MultiPeriodDiscriminator(), msd=MultiResSpecDiscriminator(),
        wd=WavLMDiscriminator(768, 13, 64),
    )
    # These two version-pinned upstream auxiliary checkpoints include training
    # metadata requiring legacy pickle loading. Never apply this to user files.
    model.text_aligner.load_state_dict(torch.load(ROOT/'models/auxiliary/asr.pth',map_location='cpu',weights_only=False)['model'],strict=True)
    model.pitch_extractor.load_state_dict(torch.load(ROOT/'models/auxiliary/jdc.t7',map_location='cpu',weights_only=False)['net'],strict=True)
    audit = {}
    if base:
        weights = torch.load(ROOT / 'models/Kokoro-82M/kokoro-v1_0.pth', map_location='cpu', weights_only=True)
        if set(weights) != set(BASE_KEYS):
            raise ValueError(f'Unexpected base modules: {set(weights)}')
        for key in BASE_KEYS:
            audit[key] = load_exact(model[key], weights[key])
        del weights
        # Keep new auxiliary encoders near a valid pretrained conditioning vector.
        # Random near-zero styles drive the pretrained ISTFT exponential out of range.
        native = torch.load(ROOT/'models/Kokoro-82M/zf_xiaobei.pt',map_location='cpu',weights_only=True)[100,0]
        with torch.no_grad():
            for key, offset in [('style_encoder',0),('predictor_encoder',128)]:
                model[key].unshared.weight.mul_(0.01)
                model[key].unshared.bias.copy_(native[offset:offset+128])
    if with_voicepack:
        from training.voicepack import TrainableVoice
        model.voicepack = TrainableVoice()
    for module in model.values():
        module.to(device)
    return model, audit


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def save_checkpoint(path, model, optimizers, **metadata):
    state = {'net': {key: canonical_state(module.state_dict()) for key, module in model.items()},
             'optimizers': {key: opt.state_dict() for key, opt in optimizers.items()},
             'rng': rng_state(), **metadata}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    tmp.replace(path)


def assert_optimizer_parameters(model, optimizer, keys):
    expected = {id(p) for key in keys for p in model[key].parameters()}
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    if set(actual) != expected or len(actual) != len(set(actual)):
        raise ValueError('Optimizer parameters do not match the live modules exactly')
