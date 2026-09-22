"""Bounded CPU reproductions for the pinned community implementation review."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import torch
import torchaudio
import soundfile as sf

torch.set_num_threads(2)
torch.manual_seed(42)
root = Path(__file__).resolve().parents[1]
up = root / 'scripts/review_sources'
report = {'scope': 'Mechanism probes, not end-to-end training or quality measurements', 'torch': torch.__version__, 'torchaudio': torchaudio.__version__}
# Execute exactly the checkpoint loader from the reviewed source without importing its training dependencies.
tree = ast.parse((up / 'load_checkpoint.py').read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_checkpoint')
namespace = {'torch': torch}
exec(compile(ast.Module(body=[function], type_ignores=[]), 'reviewed_load_checkpoint', 'exec'), namespace)
with tempfile.TemporaryDirectory() as folder:
    model = torch.nn.Linear(3, 2)
    before = {k: v.clone() for k,v in model.state_dict().items()}
    target = {f'module.{k}': torch.full_like(v, 7) for k,v in before.items()}
    path = Path(folder) / 'wrapped.pth'
    torch.save({'net': {'decoder': target}}, path)
    namespace['load_checkpoint']({'decoder': model}, None, path)
    unchanged = all(torch.equal(v, model.state_dict()[k]) for k,v in before.items())
    report['prefixed_load'] = {'loader_returned_without_error': True, 'every_parameter_unchanged': unchanged}
    assert unchanged
# Replacing a module after optimizer construction leaves optimizer pointing to old Parameters.
model = torch.nn.Linear(3, 2)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
model = copy.deepcopy(model)
owned = {id(p) for g in optimizer.param_groups for p in g['params']}
overlap = sum(id(p) in owned for p in model.parameters())
before = [p.clone() for p in model.parameters()]
model(torch.ones(1, 3)).sum().backward()
optimizer.step()
report['optimizer_replacement'] = {'current_parameter_objects_owned': overlap, 'parameters_have_gradients': all(p.grad is not None for p in model.parameters()), 'current_parameters_unchanged_after_step': all(torch.equal(a,b) for a,b in zip(before,model.parameters()))}
assert overlap == 0
# Compare training and extraction transforms on identical samples (no padding confound).
row = json.loads((root / 'data/train.jsonl').open().readline())
audio, sr = sf.read(row['audio'],dtype='float32')
wave = torch.from_numpy(audio)
kw = dict(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
train_transform = torchaudio.transforms.MelSpectrogram(**kw)
export_transform = torchaudio.transforms.MelSpectrogram(sample_rate=24000, **kw)
train_mel = (torch.log(1e-5+train_transform(wave))+4)/4
export_mel = (torch.log(1e-5+export_transform(wave))+4)/4
report['mel_mismatch'] = {'audio':row['audio'],'actual_sample_rate':sr,'training_transform_sample_rate':train_transform.sample_rate,'export_transform_sample_rate':export_transform.sample_rate,'shape':list(train_mel.shape),'normalized_mel_mean_abs_difference':(train_mel-export_mel).abs().mean().item(),'filterbanks_equal':torch.equal(train_transform.mel_scale.fb,export_transform.mel_scale.fb)}
assert not report['mel_mismatch']['filterbanks_equal']
# The extraction script's threshold cannot establish whether an encoder was trained.
spec = importlib.util.spec_from_file_location('review_extract',up/'extract_voicepack.py')
module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
encoder = module.StyleEncoder(dim_in=64,style_dim=128,max_conv_dim=512)
with torch.no_grad(): norm = encoder(torch.randn(1,1,80,200)).norm().item()
report['untrained_encoder_detection'] = {'random_untrained_encoder_norm':norm,'passes_script_norm_threshold':norm <= 1000}
(root/'reports/community_review_probes.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
