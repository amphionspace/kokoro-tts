# Kokoro TTS ZH · 大气女声

基座为官方通用 **hexgrad/Kokoro-82M v1.0**，不使用 v1.1-zh。以 LITs 的 MajesticVoice 中文、英文、中英混合合成语音做单音色适配。

文档：[完整训练报告](docs/training_report.md)、[训练技术参考](docs/training_reference.md)、[训练与评估结果](docs/evaluation_results.md)、[依赖与模型准备](docs/dependencies.md)。两阶段训练和最终450条固定文本评价已完成。评估文档已补充与LITs直接训练IMF最终170k的同文本对比。

20轮长样本加权实验已完成，最终450条常规评估无失败；原baseline的完整重跑（Stage 1四轮、Stage 2十轮）及最终评估也已完成。实验方案见[长文问题与采样策略](docs/long_text_duration.md)，最新结果见[长文本A/B对比](docs/long_text_comparison_20260922.md)。小样本中中文/混读的首句缩短略有缓解，尚未解决长上下文加速问题。

完整baseline重跑位于`runs/majestic_baseline_full_rerun_20260921`，保留全部checkpoint、评估及日志，不进行自动清理。后台交接记录为`handoff_status.json`和`progress_checks.jsonl`，监督脚本为`scripts/supervise_baseline_rerun.py`。TensorBoard：旧baseline与加权实验对比使用**32003**，完整重跑使用**32004**。本次长文本对比使用旧baseline导出，未包含重跑模型。

## 网页 Demo

2026-09-22本机服务已启动，地址为`http://服务器IP:32002`，加载**20轮长样本加权实验最终模型**。默认开启逐句合成；关闭“每句一块”可试听模型直接处理长上下文的效果。启动相同模型的命令（从项目根目录执行，已有服务时不重复启动）：

```bash
.venv/bin/python -m pip install -r demo/requirements.txt
CUDA_VISIBLE_DEVICES=0 .venv/bin/python demo/server.py \
  --host 0.0.0.0 --port 32002 --device cuda:0 \
  --export-dir "$PWD/runs/majestic_s2_20ep_long_resume6k_20260921/eval/stage2_final"
```

省略`--export-dir`仍会加载旧10轮baseline。模型路径、后台启动方法、日志和接口见[Demo说明](demo/README.md)；固定文本的逐条A/B试听包见[长文本对比报告](docs/long_text_comparison_20260922.md)。

## 数据与基座

- 官方 revision：`f3ff3571791e39611d31c381e3a41a3af07b4987`。
- `models/Kokoro-82M/kokoro-v1_0.pth` 的 SHA256 已完整校验；五组基座权重逐项严格加载，禁止静默忽略缺失参数。
- 原始训练划分 53,732 条约 100h（中文 50h、英文 25h、混合 25h）；前端转换后接受 53,723 条约 99.98h，9 条含不支持的外语字符被隔离。验证 400 条、测试 608 条保持原划分。
- 数据为 VoxCPM2 生成的 MajesticVoice 音色库，音频路径复用 `/119010446/tts-assets`，不复制音频、不复用 LITs 音素 ID。
- 中文使用匹配通用 v1.0 的 Misaki 前端；英文片段单独转换，所有音素严格检查基座词表。见 `reports/frontend_audit.json`。
- 训练、验证、测试按音频路径、规范化文字和来源分组检查互斥；test 保留最终评价。

## 训练实现

入口 `training/train.py`，Stage 1 配置 `configs/train.json`，Stage 2 配置 `configs/stage2.json`。四卡 DDP：Stage 1 训练声学解码、音色编码、文本编码与对齐；Stage 2 训练时长、F0、能量及韵律编码，再联合微调音色与解码器。保留波形判别器、谱重建和冻结 WavLM 感知损失，不使用扩散模型或 SLM OOD 对抗分支。

训练架构从固定版本的社区代码整理到 `vendor/styletts2`，推理实现位于 `vendor/kokoro`。保留权重归一化结构、许可证及来源，修正参数加载和推理 AdaIN 参数不兼容问题。详情见 `provenance/` 和 `reports/community_flow_review.md`；该报告中的原始路径是调研时的历史位置。

新音色编码器以原生 voice 向量附近的小扰动初始化，避免零附近的随机 style 令预训练 ISTFT 解码器输出极大幅值。Stage 2 复制已训练编码器特征后，以原生韵律向量初始化韵律输出头。文本编码器采用 FP32，避免 packed LSTM 的 BF16 反向异常；其余路径使用 BF16。

特征提取沿用 ASR/JDC 预训练辅助模型的历史约定：24k 波形上使用 16k 参数的 Mel 滤波器。**这不是重采样**；训练与 voicepack 提取共用完全相同的变换、归一化及首尾补零。音频重建损失使用实际 24k 参数。

优化器绑定实际参与前向的参数；检查所有活跃模块梯度。checkpoint 保存优化器、逐 rank RNG、buffer、数据哈希、配置及下一批位置，恢复时严格检查数据、配置与卡数。Stage 1 使用原 `source/` 和 `config.json`，Stage 2 使用独立 `source_stage2/` 和 `config_stage2.json`，评价使用 `source_evaluation/`。正式运行会冻结源码和配置快照，依次执行两个阶段，失败时停止并记录原因。

```bash
.venv/bin/python scripts/run_training.py \
  --run-dir "$PWD/runs/majestic_s2_20ep_long_resume6k_20260921" \
  --stage2-config "$PWD/configs/stage2_20ep_long.json" \
  --stage2-resume-from "$PWD/runs/majestic_s2_20ep_long_resume6k_20260921/origin/stage2_step_00009360.pth"
```

旧实验完成Stage 1的3,360步与Stage 2的8,400步。上面命令用于新20轮Stage 2实验的启动或恢复；同目录有文件锁，已有进程时不要重复启动。

## TensorBoard 与 LITs 评价

TensorBoard 端口 **32003**，只展示旧/新Stage 2的训练与常规评估，排除Stage 1和诊断；对比视图配置保存在本机`runs/tensorboard_stage2_compare/server.json`。记录损失、学习率、梯度范数、吞吐、显存、分语言验证指标，以及评价音频和评分。

```bash
.venv/bin/python scripts/watch_evaluation.py --run-dir "$PWD/runs/majestic_s2_20ep_long_resume6k_20260921" --gpu 3
# TensorBoard 的四个相关目录见 runs/tensorboard_stage2_compare/server.json
```

评价使用 LITs 原有 `training/common/evaluate_checkpoint.py` 的 ASR、metrics 和 summarize 实现：Qwen3-ASR、CER/WER、WavLM/CAMP 相似度、DNSMOS。固定中文 200、英文 200、混合 50 条文本及对应参考；第 100 步先做每类 8 条冒烟评价，后续每 1,000 步和阶段结束做全量评价。GPU 3 同时承担训练和异步评价，因此需要显存余量。

voicepack 从固定的 96 条**训练集**参考中提取均值，不用 val/test 拟合音色。Stage 1 的韵律编码器尚未训练，评价明确标记为“已学声学 style + 原生韵律”的诊断输出；Stage 2 完成前 1000 步后初始化并直接优化独立 256 维 voice，每批一半使用此 voice，另一半使用参考编码器。`majestic.pt` 导出学习到的参数，编码器均值另存 `encoder_mean.pt` 对照。早期评分用于验证链路，不能视作最终音质结论。

专项诊断模块及自动调用已从当前代码移除。旧运行只保留必要基线权重、曲线及仓库评估汇总，后续仅执行常规验证和固定文本评估；历史汇总见[评估结果](docs/evaluation_results.md)。

## 查看本机运行记录

```bash
cat runs/majestic_s2_20ep_long_resume6k_20260921/status.json
cat runs/majestic_s2_20ep_long_resume6k_20260921/supervisor_status.json
cat runs/majestic_s2_20ep_long_resume6k_20260921/evaluation_status.json
tail -n 5 runs/majestic_s2_20ep_long_resume6k_20260921/stage2.log
nvidia-smi
```

运行 checkpoint 在 `checkpoints/`；自动评价在 `eval/`；固定源码在 `source/`。训练环境为独立 `.venv`（复用现有 LITs 环境的 PyTorch），ASR 和音质评价分别复用现有独立环境。不要删除共享音频、LITs 评价代码或这些环境。

## 清理后的目录

`runs/`保留新Stage 2实验、旧Stage 2必要基线、网页服务及TensorBoard对比配置，旧目录仅保留Stage 2基线部署模型、结果及相关曲线。一次性长文分析脚本、数据和试听WAV已清理，结论统一保存在[长文说明](docs/long_text_duration.md)。新实验的冻结源码快照保留；旧诊断和冻结源码已清理。

## 仓库内容

提交源码、训练及模型结构配置、依赖版本与来源校验值、许可证、评估JSON/CSV。大权重、训练数据、checkpoint、WAV、虚拟环境和凭据不入Git。在其他机器使用前按[依赖准备](docs/dependencies.md)配置外部数据及评价环境。
