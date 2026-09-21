"""Run paired checkpoint diagnostics, LITs scoring and a listening report."""
import argparse
from collections import defaultdict
import html
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CODE_ROOT=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('KOKORO_PROJECT_ROOT',str(CODE_ROOT)))
sys.path.insert(0,str(CODE_ROOT))
from training.common import write_json,records,digest
from torch.utils.tensorboard import SummaryWriter


def main():
    p=argparse.ArgumentParser()
    for name in ('checkpoint','deployment-dir','output','tensorboard'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--asr-gpu',type=int);p.add_argument('--metrics-gpu',type=int)
    p.add_argument('--per-language',type=int,default=8)
    args=p.parse_args()
    for name in ('checkpoint','deployment_dir','output','tensorboard'):setattr(args,name,getattr(args,name).resolve())
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        assert json.loads((out/'protocol.json').read_text())['checkpoint_sha256']==digest(args.checkpoint)
        print('Diagnostics already complete');return
    env=dict(os.environ,PYTHONPATH=str(CODE_ROOT),KOKORO_PROJECT_ROOT=str(ROOT),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    python=str(ROOT/'.venv/bin/python')
    def command(stage,interpreter,output):
        return [str(interpreter),'-m','training.evaluate','--stage',stage,'--checkpoint',str(args.checkpoint),'--output',str(output)]
    try:
        if not (out/'reconstruction_summary.json').exists():
            cmd=[python,'-m','training.diagnostics','--checkpoint',str(args.checkpoint),'--deployment-dir',str(args.deployment_dir),
                 '--output',str(out),'--tensorboard',str(args.tensorboard),'--per-language',str(args.per_language)]
            with (out/'reconstruction.log').open('w') as log:subprocess.run(cmd,cwd=CODE_ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(args.gpu)),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200)
        protocol=json.loads((out/'protocol.json').read_text());assert protocol['checkpoint_sha256']==digest(args.checkpoint)
        full=out/'full_utterance';rows=records(full/'synthesis.jsonl');ids={r['id'] for r in rows}
        assert len(ids)==len(rows)
        write_json(out/'status.json',{'status':'scoring_full_utterances','samples':len(rows),'updated_at':time.time()})
        processes=[];logs=[]
        configurations=[('asr',ROOT.parent/'tts-assets/.venv-voxcpm2/bin/python',args.asr_gpu if args.asr_gpu is not None else args.gpu),
                        ('metrics',ROOT.parent/'UltraEval-Audio/envs/metrics/bin/python',args.metrics_gpu if args.metrics_gpu is not None else args.gpu)]
        parallel=configurations[0][2]!=configurations[1][2]
        for stage,interpreter,gpu in configurations:
            target=full/f'{stage}.jsonl'
            existing=records(target) if target.exists() else []
            if len(existing)==len(rows) and {r['id'] for r in existing}==ids:continue
            log=(full/f'{stage}.log').open('w');logs.append(log)
            proc=subprocess.Popen(command(stage,interpreter,full),cwd=CODE_ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
            processes.append((stage,proc))
            if not parallel:
                code=proc.wait(timeout=7200)
                if code:raise RuntimeError(f'{stage} exited {code}')
        for stage,proc in processes:
            code=proc.wait(timeout=7200)
            if code:raise RuntimeError(f'{stage} exited {code}')
        for log in logs:log.close()
        by_stage={stage:{r['id']:r for r in records(full/f'{stage}.jsonl')} for stage in ('asr','metrics')}
        assert all(set(value)==ids for value in by_stage.values())
        path=ROOT.parent/'LITs/training/common/evaluate_checkpoint.py'
        spec=importlib.util.spec_from_file_location('lits_diagnostic_summary',path);common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
        by_variant=defaultdict(list)
        for row in rows:by_variant[row['variant']].append(row)
        summaries={}
        for variant,part in by_variant.items():
            folder=full/variant;folder.mkdir(exist_ok=True)
            (folder/'synthesis.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in part))
            for stage in ('asr','metrics'):
                (folder/f'{stage}.jsonl').write_text(''.join(json.dumps(by_stage[stage][r['id']],ensure_ascii=False)+'\n' for r in part))
            common.summarize(argparse.Namespace(output=folder,checkpoint=args.checkpoint))
            summaries[variant]=json.loads((folder/'summary.json').read_text())
        failures=sum(g['evaluation_failures'] for s in summaries.values() for g in s['groups'].values())
        if failures:raise RuntimeError(f'{failures} paired evaluation failures')
        reconstruction=json.loads((out/'reconstruction_summary.json').read_text())
        summary={'stage':protocol['stage'],'global_step':protocol['global_step'],'checkpoint_sha256':protocol['checkpoint_sha256'],
                 'matched_crops':reconstruction['matched_crops'],'alignment':reconstruction['alignment'],
                 'paired_full_utterances':summaries,'evaluation_failures':failures,
                 'selection':protocol['selection'],'per_condition_samples':len(rows)//len(by_variant),
                 'caveats':['Full-utterance ASR uses complete matching transcripts; crop reconstructions are never ASR-scored.',
                            'Oracle reference style uses the target audio and measures conditional reconstruction, not zero-shot generation.',
                            'Duration-quantile cohort is a diagnostic set, not an unbiased estimate over the full corpus.',
                            'MAS monotonicity is enforced; inspect soft alignments and durations for alignment quality.']}
        write_json(out/'summary.json',summary)
        writer=SummaryWriter(str(args.tensorboard));step=protocol['global_step']
        for variant,report in summaries.items():
            for group,metrics in report['groups'].items():
                for key,value in metrics.items():
                    if isinstance(value,(int,float)):writer.add_scalar(f'full_scoring/{variant}/{group}/{key}',value,step)
        import matplotlib.image as mpimg
        for plot in sorted((out/'alignment_plots').glob('*.png')):
            writer.add_image('alignment/'+plot.stem,mpimg.imread(plot),step,dataformats='HWC')
        writer.flush();writer.close()
        # Standalone static listening page, with relative links to preserved artifacts.
        labels={'target':'目标音频','oracle_reference_style':'真实对齐/F0/能量 + 逐句 style',
                'oracle_fixed_style':'真实对齐/F0/能量 + 固定声学 style','free_fixed_style':'固定 voicepack 自由合成',
                'aligned_predicted_reference':'真实对齐 + 参考 style + 预测 F0/能量',
                'aligned_predicted_fixed':'真实对齐 + 固定 style + 预测 F0/能量'}
        content=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Checkpoint paired evaluation</title>',
                 '<style>body{font-family:system-ui;max-width:1400px;margin:32px auto;padding:0 20px;background:#fafafa;color:#172033}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #ccd3de;padding:9px;text-align:left}section{background:white;padding:20px;margin:20px 0;border:1px solid #ddd;border-radius:8px}.players{display:flex;flex-wrap:wrap;gap:16px}.player{width:310px}audio{width:300px}img{max-width:100%}small{color:#526078}</style>',
                 f'<h1>Stage {protocol["stage"]} · step {step} 对照评价</h1>',
                 '<p>主要结果是 400 条验证集的匹配条件重建。下面的完整句子按语言和时长分位点固定选择；目标音频也接受相同 ASR/音质评分。真实参考 style 为有目标音频的条件重建，不代表自由生成能力。</p>',
                 '<p>评分音频保留原始幅值；裁剪重建不参与全文 ASR。<a href="summary.json">完整指标 JSON</a> · <a href="protocol.json">协议与来源</a></p>',
                 '<h2>完整句子评分</h2><table><tr><th>条件</th><th>语言</th><th>样本数</th><th>CER micro</th><th>WER micro</th><th>WavLM similarity</th><th>CAMP similarity</th><th>DNSMOS</th></tr>']
        def fmt(v):return '—' if v is None else f'{v:.4f}'
        for variant,report in summaries.items():
            for group,metrics in report['groups'].items():
                values=[labels[variant],group,str(metrics['samples']),*[fmt(metrics.get(k)) for k in ('cer_micro','wer_micro','wavlm_similarity_mean','camp_similarity_mean','dnsmos_ovrl_mean')]]
                content.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
        content.append('</table>')
        by_index=defaultdict(dict)
        for row in rows:by_index[row['validation_index']][row['variant']]=row
        for index,parts in sorted(by_index.items()):
            first=next(iter(parts.values()));lang=first['language']
            content.append(f'<section><h2>{lang} · val_{index:04d}</h2><p>{html.escape(first["ref_text"])}</p><div class="players">')
            for variant,row in parts.items():
                src=os.path.relpath(row['audio'],out);asr=by_stage['asr'][row['id']]
                content.append(f'<div class="player"><b>{labels[variant]}</b><audio controls preload="none" src="{html.escape(src)}"></audio><small>ASR: {html.escape(asr.get("asr_text",""))}<br>Peak {row["peak"]:.3f} · duration {row["duration"]:.2f}s</small></div>')
            content.append(f'</div><details><summary>软对齐与单调路径</summary><img src="alignment_plots/{lang}_val_{index:04d}.png"></details></section>')
        content.append('</html>');(out/'report.html').write_text('\n'.join(content))
        write_json(out/'complete.json',{'status':'complete','global_step':step,'evaluation_failures':0,'completed_at':time.time()})
        write_json(out/'status.json',{'status':'complete','global_step':step,'updated_at':time.time()})
        print(json.dumps({'status':'complete','output':str(out),'failures':0}),flush=True)
    except Exception as exc:
        write_json(out/'failed.json',{'error':repr(exc),'updated_at':time.time()});raise


if __name__=='__main__':main()
