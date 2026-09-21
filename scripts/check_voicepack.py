"""Bounded four-rank voicepack update/resume/export audit, outside the formal run."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from training.common import digest, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reuse-training', type=Path, help='Verify existing continuous/resumed bounded runs')
    args = p.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    source_hash = digest(args.checkpoint)
    python = str(ROOT/'.venv/bin/python')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    with tempfile.TemporaryDirectory(prefix='kokoro_voicepack_check_') as temp:
        work = args.reuse_training.resolve() if args.reuse_training else Path(temp)
        if not args.reuse_training:
            config = json.loads((ROOT/'configs/stage2.json').read_text())
            config.update(batch_size=4, workers_per_gpu=0, log_every=1,
                          stage2_joint_step=1, voicepack_start_step=1, early_checkpoints=[2])
            write_json(work/'config.json', config)
            base = [python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                    '-m', 'training.train', '--config', str(work/'config.json'), '--stage', '2',
                    '--max-steps', '3', '--skip-validation']
            source = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            origin_step = source['global_step']; del source
            for name, option, checkpoint in (
                ('continuous', '--initialize', args.checkpoint.resolve()),
                ('resumed', '--resume', work/f'continuous/checkpoints/stage2_step_{origin_step+2:08d}.pth'),
            ):
                with (out/f'{name}.log').open('w') as log:
                    subprocess.run(base+['--run-dir', str(work/name), option, str(checkpoint)],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
        paths = sorted((work/'continuous/checkpoints').glob('stage2_step_*.pth'))
        assert len(paths)==2
        first = torch.load(paths[0], map_location='cpu', weights_only=False)
        final = torch.load(paths[1], map_location='cpu', weights_only=False)
        resumed_path = work/'resumed/checkpoints'/paths[1].name
        resumed = torch.load(resumed_path, map_location='cpu', weights_only=False)
        assert first['stage_step']==2 and final['stage_step']==resumed['stage_step']==3
        voice = final['net']['voicepack']
        assert bool(voice['initialized'])
        deltas = [(voice['vector'][:,i:i+128]-voice['initial_vector'][:,i:i+128]).norm().item() for i in (0,128)]
        assert min(deltas)>0
        assert torch.equal(first['net']['voicepack']['initial_vector'], voice['initial_vector'])
        assert not (work/'resumed/voicepack_initialization.json').exists()
        mismatches = []; tensor_count = 0; max_abs = 0.; optimizer_max_abs = 0.
        def compare(a,b,path):
            nonlocal tensor_count,max_abs,optimizer_max_abs
            if torch.is_tensor(a):
                tensor_count += 1
                if not torch.equal(a,b):
                    mismatches.append(path)
                    if path.startswith('optimizers') and a.is_floating_point():
                        assert a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
                        optimizer_max_abs=max(optimizer_max_abs,float((a-b).abs().max()))
                    elif a.is_floating_point():
                        max_abs=max(max_abs,float((a-b).abs().max()))
                        torch.testing.assert_close(a,b,rtol=0,atol=1e-5)
                    else:
                        raise AssertionError('Non-floating state differs: '+path)
            elif isinstance(a,dict):
                assert a.keys()==b.keys(),path
                for key in a:compare(a[key],b[key],path+'/'+str(key))
            elif isinstance(a,(tuple,list)):
                assert len(a)==len(b)
                for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
            else:assert a==b,path
        compare(final['net'],resumed['net'],'net')
        compare(final['optimizers'],resumed['optimizers'],'optimizers')
        for rank in range(4):
            compare(final['rank_state'][rank]['buffers'],resumed['rank_state'][rank]['buffers'],f'rank{rank}/buffers')
            for key in ('torch','cuda'):
                compare(final['rank_state'][rank]['rng'][key],resumed['rank_state'][rank]['rng'][key],f'rank{rank}/rng/{key}')
        continuous_status=json.loads((work/'continuous/status.json').read_text())
        resumed_status=json.loads((work/'resumed/status.json').read_text())
        forward_keys=('loss','mel','wavlm','gan','discriminator','duration','duration_ce','f0','energy')
        forward_differences={key:abs(continuous_status['losses'][key]-resumed_status['losses'][key]) for key in forward_keys}
        assert max(forward_differences.values())<1e-6,forward_differences
        expected_voice = voice['vector'].clone()
        report = {'status':'passed','formal_training_resumed':False,'checkpoint_sha256':source_hash,
                  'world_size':4,'scope':'3 temporary Stage 2 steps; threshold lowered to 1 to exercise the transition',
                  'acoustic_delta_norm':deltas[0],'prosody_delta_norm':deltas[1],
                  'resume_compared_tensors':tensor_count,'resume_bitwise_equal':not mismatches,
                  'resume_model_max_abs_difference':max_abs,'resume_optimizer_moment_max_abs_difference':optimizer_max_abs,
                  'resume_cuda_and_cpu_rng_exact':True,'resume_reinitialized_voice':False,
                  'numerical_note':'CUDA backward reductions are not bitwise deterministic; forward losses and RNG must agree, model tensors checked at absolute tolerance 1e-5; post-update moment drift is reported separately',
                  'resume_forward_loss_differences':forward_differences,
                  'gradient_report':json.loads((work/'continuous/stage2_gradients.json').read_text())}
        del first,final,resumed,voice;gc.collect()
        deployment=Path(temp)/'export'
        single_env=dict(env,CUDA_VISIBLE_DEVICES='0')
        with (out/'export.log').open('w') as log:
            subprocess.run([python,'-m','training.evaluate','--stage','synthesize','--checkpoint',str(paths[1]),
                            '--output',str(deployment),'--per-group-limit','1'],cwd=ROOT,env=single_env,
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        exported=torch.load(deployment/'majestic.pt',weights_only=True)
        assert torch.equal(exported,expected_voice.unsqueeze(0).expand(510,1,256))
        assert json.loads((deployment/'export.json').read_text())['conditioning']=='directly_optimized_voicepack'
        synthesis=[json.loads(x) for x in (deployment/'synthesis.jsonl').read_text().splitlines()]
        assert len(synthesis)==3 and all('synthesis_error' not in x for x in synthesis)
        report.update(export_exactly_matches_learned_voice=True,three_language_synthesis_passed=True,
                      encoder_mean_control_saved=(deployment/'encoder_mean.pt').exists())
        assert digest(args.checkpoint)==source_hash
        write_json(out/'voicepack_audit.json',report)
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
