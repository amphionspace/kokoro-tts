"""Aggregate the paired long-text comparison and write a portable listening page."""
import argparse,html,json,statistics,shutil
from pathlib import Path
from training.common import ROOT,records,write_json
OUT=ROOT/'runs/long_text_comparison_20260922'
def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    OUT=parser.parse_args().output.resolve()
    protocol=json.loads((OUT/'protocol.json').read_text())
    labels=list(protocol['model_hashes'])
    if len(labels)!=2 or labels[0]!='baseline':raise ValueError('Expected baseline and one candidate')
    candidate=labels[1]
    names=protocol.get('model_labels',{'baseline':'旧 baseline · 10轮','long_weighted':'延长至20轮 + 长样本加权'})
    data={label:{r['id']:r for r in records(OUT/label/'details.jsonl')} for label in labels}
    if set(data['baseline'])!=set(data[candidate]):raise ValueError('Unpaired sample IDs')
    summary={}
    for lang in ('zh','en','mixed'):
        summary[lang]={}
        for label,rows in data.items():
            rs=[r for r in rows.values() if r['language']==lang]
            metric,denom=('wer','reference_words') if lang=='en' else ('cer','reference_characters')
            prefix=[r['prefix_change_percent'] for r in rs if 'prefix_change_percent' in r]
            summary[lang][label]={'samples':len(rs),'asr_metric':metric,'asr_micro':sum(r[metric]*r[denom] for r in rs)/sum(r[denom] for r in rs),'mean_seconds':statistics.mean(r['duration'] for r in rs),'prefix_pairs':len(prefix),'prefix_change_median_percent':statistics.median(prefix) if prefix else None,'camp_mean':statistics.mean(r['camp_similarity'] for r in rs),'dnsmos_mean':statistics.mean(r['dnsmos_ovrl'] for r in rs)}
        summary[lang]['median_paired_duration_change_percent']=statistics.median(100*(data[candidate][r['id']]['duration']/r['duration']-1) for r in data['baseline'].values() if r['language']==lang)
    write_json(OUT/'comparison.json',summary)
    title='长文本：'+names['baseline']+' vs '+names[candidate]
    parts=['<!doctype html><meta charset="utf-8"><title>长文本 A/B 对比</title><style>body{max-width:1100px;margin:30px auto;font:16px system-ui;line-height:1.6}table{width:100%;border-collapse:collapse}td,th{padding:12px;border:1px solid #ddd;vertical-align:top}audio{width:100%}pre{white-space:pre-wrap}section{margin:40px 0}</style><h1>'+html.escape(title)+'</h1><p>两模型各自训练的 voice，speed=1，整段生成，不分句、不截断。每种语言选验证集 token 最长的4条，另加1条四句探针，共15组。仅为小样本比较，不能单独归因于某一个训练因素。ASR是自动转写，不等于人工听写；时长不直接等同于发音速度。首句比较使用完全相同的前缀音素，排除 BOS/EOS。</p>']
    parts.append('<h2>汇总</h2><p>箭头表示旧模型 → 新模型。首句变化越接近0，表示长上下文对该句预测时长的影响越小。</p><table><tr><th>指标</th><th>中文</th><th>英文</th><th>混读</th></tr>')
    for key,title,pattern in [('mean_seconds','整段平均时长（秒）','.3f'),('prefix_change_median_percent','首句预测时长变化中位数（%）','+.2f'),('asr_micro','ASR错误率：中文/混读CER，英文WER','.3%'),('camp_mean','CAMP音色相似度','.3f'),('dnsmos_mean','DNSMOS总体评分','.3f')]:
        cells=[]
        for lang in ('zh','en','mixed'):
            values=[summary[lang][label][key] for label in labels]
            cells.append('<td>'+' → '.join(format(v,pattern) if v is not None else '—' for v in values)+'</td>')
        parts.append('<tr><th>'+title+'</th>'+''.join(cells)+'</tr>')
    parts.append('</table><p>首句配对数量：中文5组、英文1组、混读5组。英文的首句对照仅有一个四句探针。</p><h2>逐条试听</h2>')
    (OUT/'references').mkdir(exist_ok=True)
    for ident,row in data['baseline'].items():
        parts.append('<section><h2>'+html.escape(ident)+'</h2><p>'+html.escape(row['ref_text'])+'</p>')
        if row.get('reference_audio'):
            ref=OUT/'references'/(ident+'.wav');shutil.copy2(row['reference_audio'],ref)
            parts.append('<p>原始验证录音</p><audio controls preload="none" src="references/'+ident+'.wav"></audio>')
        parts.append('<table><tr>'+''.join('<th>'+html.escape(names[label])+'</th>' for label in labels)+'</tr><tr>')
        for label in data:
            r=data[label][ident];parts.append('<td><audio controls preload="none" src="'+label+'/wavs/'+ident+'.wav"></audio>')
            parts.append(f'<p>时长 {r["duration"]:.2f}s；CER {r["cer"]:.2%}；WER {r["wer"]:.2%}；CAMP {r["camp_similarity"]:.3f}</p><p>ASR：'+html.escape(r['asr_text'])+'</p>')
            if 'prefix_audio' in r:
                parts.append(f'<p>首句放入长上下文后的预测时长变化：{r["prefix_change_percent"]:+.1f}%</p><p>单独生成首句：</p><audio controls preload="none" src="'+label+'/prefix/'+ident+'.wav"></audio>')
            parts.append('</td>')
        parts.append('</tr></table></section>')
    (OUT/'index.html').write_text('\n'.join(parts))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
