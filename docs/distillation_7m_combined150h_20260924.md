# 新150h教师的7.48M学生蒸馏

2026-09-24按用户要求启动。长文本A/B临时目录`runs/long_text_comparison_20260924`已删除，数值结论保留于[150h训练报告](combined150h_training_report.md)。

## 教师、数据与学生

- 教师为本次150.7982h两阶段训练的82M最终导出：`runs/majestic_combined150h_20260923/eval/stage2_final`，使用其`kokoro.pth`和学习得到的`majestic.pt`。
- 文本来自`data/prepared_combined150h_20260923`：训练63,226条，验证400条，覆盖中文、英文、中英混读。保留原测试集与450条固定评价协议。
- 对每条完整文本，由新教师重新生成波形及生成该波形的精确时长、F0和能量。缓存逐rank记录教师权重、voice和源清单哈希；合并前核对完整性、划分、文本和哈希。
- **150.80h是源数据集的时长**；蒸馏目标是新教师重新生成的音频，其总时长以缓存完成后的`ready.json`为准。没有混用旧教师的缓存。
- 学生沿用7,477,702参数结构，从随机初始化开始，不恢复旧学生或其优化器。初始化时长bias仅用训练缓存统计。

## 训练配方

沿用之前已修正的完整窗口监督：均匀随机截取120帧（约3秒），`context_frames=0`，整段裁剪窗口参与声学损失，不额外裁掉首尾，也不额外提高首尾抽样比例。完整文本与教师时长进入学生，声学监督采用随机窗口。

四卡DDP、每卡batch16、全局64，BF16，20,000步；500步学习率warmup及500步GAN warmup。损失沿用STFT、Mel、时长、静音、WavLM、F0/能量和GAN。完整参数见[配置](../configs/distill_7m_combined150h.json)，学生结构见[结构配置](../configs/student_7m.json)。

阶段第100步进行小样本评估，每1,000步和最终checkpoint进行450条评估。完整评价按三类语言的平均识别错误率选best，同分时比较CAMP；本轮已完成20,000步，best与final均完成450条评价，结果见下文。

## 启动与状态

启动器先对192条时长/音素极端训练样本和12条验证样本生成小缓存，执行四卡batch16、GAN开启的两步预检。成功后保留预检结论并删除临时权重与缓存，再启动四卡全量教师生成。监督进程等待所有rank完成并审计缓存，然后自动启动正式训练和评价。

- 运行：`runs/majestic_student7m_combined150h_20260924`
- 缓存：`data/distill7m_teacher_combined150h_20260924`
- 启动/预检：运行目录`preparation_status.json`、`preparation.log`
- 当前阶段：运行目录`supervisor_status.json`
- 缓存进度：缓存目录`status_rank0.json`至`status_rank3.json`
- 正式训练：运行目录`status.json`、`train.log`
- 自动评价：运行目录`evaluation_status.json`
- 预检汇总：`reports/distill7m_combined150h_preflight.json`
- TensorBoard：32003，新增`student7m_150h_train`与`student7m_150h_eval`

源码与配置冻结到本轮`source/`和`config.json`；教师生成也使用该快照。监督入口支持同目录恢复，已有进程时不要重复启动：

```bash
.venv/bin/python -m distillation.run \
  --run-dir "$PWD/runs/majestic_student7m_combined150h_20260924" \
  --cache "$PWD/data/distill7m_teacher_combined150h_20260924" \
  --config "$PWD/configs/distill_7m_combined150h.json"
```

## 完成结果（2026-09-28核查）

训练于2026-09-24 06:07 UTC完成20,000步。教师训练缓存为63,226条、150.6289小时，验证400条、0.7714小时，无削波样本；这是教师生成时长，与源数据150.7982小时不同。

| 最终20,000步 | 中文 | 英文 | 混读 |
|---|---:|---:|---:|
| 主要识别指标 | CER 0.212% | WER 0.773% | CER 0.000% |
| CAMP | 0.746 | 0.811 | 0.716 |
| DNSMOS OVRL | 3.466 | 3.302 | 3.433 |

最终450条评价全部成功。按既定识别错误率优先规则选出的best是7,000步；它的识别综合分略优，但CAMP和DNSMOS低于final，因此best不代表综合听感最佳。这一轮学生尚未进行新的长文本试听对比。

原始证据：[final](evaluation/distillation_7m_combined150h/student_final_summary.json)、[best](evaluation/distillation_7m_combined150h/best.json)、[教师缓存审计](evaluation/distillation_7m_combined150h/teacher_cache_audit.json)、[训练完成记录](evaluation/distillation_7m_combined150h/training_complete.json)。
