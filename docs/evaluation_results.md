# 大气女声训练与评估结果

整理日期：2026-09-21。运行：`majestic_v1_20260920`。Stage 1 已完成 4 epochs / 3,360 步；Stage 2 已完成 10 epochs / 8,400 步，累计 11,760 步。两阶段分别于 2026-09-20 10:22、14:36 UTC 完成。

最终固定文本评估完成 450/450 条，无合成、ASR 或评分失败。Stage 2 最终 checkpoint 的 SHA256 为 `b890ef9a66ac9ec7e0ccf588ff1b10b29e1a3a9cd8a1c399213348b2c8c5b561`；导出条件为直接优化的 256 维 voice。此处没有宣称最终 checkpoint 是所有指标最优的一份。

## 450 条文本自由合成

固定中文200、英文200、混读50条；只给文本和固定 voice，输出24k单声道。ASR使用Qwen3-ASR-1.7B；CER/WER下表为micro，按参考字符数/词数加权。纯中文看CER，英文看WER；混读同时给CER和英文/数字词WER。相似度与DNSMOS是逐句均值。

| 阶段 | 语言 | 条数 | CER micro (%) | WER micro (%) | WavLM ↑ | CAMP ↑ | DNSMOS OVRL ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| Stage 1 final | 中文 | 200 | 0.238 | — | 0.7777 | 0.5929 | 3.4105 |
| Stage 1 final | 英文 | 200 | 0.304 | 0.821 | 0.7688 | 0.7744 | 3.3608 |
| Stage 1 final | 混读 | 50 | 1.684 | 5.208 | 0.7229 | 0.5781 | 3.4259 |
| Stage 2 final | 中文 | 200 | 0.185 | — | 0.8301 | 0.7474 | 3.4754 |
| Stage 2 final | 英文 | 200 | 0.325 | 0.821 | 0.8167 | 0.8218 | 3.3741 |
| Stage 2 final | 混读 | 50 | 0.089 | 4.167 | 0.7762 | 0.7208 | 3.4494 |

Stage 2 相比 Stage 1 末尾，中文 CAMP 从0.5929到0.7474，混读从0.5781到0.7208；混读CER从1.684%到0.089%，英文WER均为0.821%。这里比较的是两阶段完整导出方案：Stage 1 是声学参考均值＋基座原生韵律，Stage 2 是学到的固定voice，不能据此把改善全部归因于某一个模块。

混读WER为4.167%，只统计正则提取的英文/数字词，不能读作整句中文词错误率。固定450条用于周期观察，不等于独立608条保留测试集的最终成绩；本次未发现该测试集的独立最终评估产物。DNSMOS是模型预测分数，不是人工MOS。

原始汇总：[Stage 1](evaluation/stage1_final/summary.json)、[Stage 2](evaluation/stage2_final/summary.json)；全部15次周期/阶段末评估见[CSV](evaluation/history.csv)，其中两次24条冒烟评估不能与450条全量结果直接比较。每份汇总及导出元信息都已随仓库提交，来源路径和SHA256见[清单](evaluation/sources.json)。

## 400 条配对重建与24条整句诊断

这部分使用验证音频提供条件，独立于上述450条自由合成。Stage 2 final覆盖400条配对裁剪，每条至多约5秒；整句诊断按语言和时长分位固定选择24条，每个条件24条，共六种条件。没有把裁剪音频与整句文本送入ASR比较。

| Stage 2 final配对裁剪条件 | Mel重建损失 ↓ |
|---|---:|
| 预测F0/能量＋逐句声学style | 0.343729 |
| 预测F0/能量＋固定声学style | 0.355637 |
| 真实F0/能量＋逐句声学style | 0.281940 |

固定声学style使Mel损失增加约0.011909。这里两条预测分支的韵律仍使用逐句编码，因此不是固定部署voice全部条件的对照。真实F0/能量条件更好，说明预测条件与重建之间仍存在差距，不能从这一项推出具体因果归属。

对齐检查中未分配帧、零时长token和内部token时长超过50帧的比例均为0；软注意力落在MAS路径内的平均质量为0.721924。MAS单调性本身是算法约束。完整数值见[最终诊断](evaluation/diagnostics_stage2_final_summary.json)和[协议](evaluation/diagnostics_stage2_final_protocol.json)。

### 整句试听条件

| 目录 | 对齐/时长 | F0、能量 | style |
|---|---|---|---|
| `target` | 原始音频 | 原始音频 | 原始音频 |
| `oracle_reference_style` | 真实音频MAS | 真实音频提取 | 逐句声学style |
| `oracle_fixed_style` | 真实音频MAS | 真实音频提取 | 固定声学style |
| `free_fixed_style` | 文本预测 | 模型预测 | 固定声学＋韵律style |
| `aligned_predicted_reference`（Stage 2） | 真实音频MAS | 模型预测 | 逐句声学＋韵律style |
| `aligned_predicted_fixed`（Stage 2） | 真实音频MAS | 模型预测 | 固定声学＋韵律style |

同名WAV对应同一句，`zh/en/mixed`是语言目录。`oracle`使用真实音频条件，测量条件重建；`free_fixed_style`才是部署路径。24条子集用于定位问题，不是总体质量的无偏估计，也不能替代450条评价。

本机试听与对齐图：`runs/majestic_v1_20260920/diagnostics/stage2_final/report.html`；整句音频在该目录的`full_utterance/wavs/`。这些WAV、HTML附属媒体和checkpoint体积较大，未上传Git。Stage 1第2000步的历史诊断保存在[evaluation](evaluation/diagnostics_step_00002000_summary.json)，不再作为当前训练状态。

## 尚未覆盖的结论

没有正式人工听测、独立608条测试集最终评分、其他原生voice退化评估，亦没有固定学习voice相对encoder_mean的完整自由合成消融。不能用当前指标宣称这些项目已验证。
