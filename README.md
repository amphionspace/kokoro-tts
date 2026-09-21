# Kokoro TTS ZH · 大气女声

基座为官方通用 **hexgrad/Kokoro-82M v1.0**，不使用 v1.1-zh。以 LITs 的 MajesticVoice 中文、英文、中英混合合成语音做单音色适配。

文档：[训练技术参考](docs/training_reference.md)、[训练与评估结果](docs/evaluation_results.md)、[依赖与模型准备](docs/dependencies.md)。两阶段训练和最终450条固定文本评价已完成。

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
.venv/bin/python scripts/run_training.py --run-dir "$PWD/runs/majestic_v1_20260920"
```

本轮完成Stage 1的3,360步与Stage 2的8,400步，累计11,760步。上面命令用于备齐本机数据和依赖后的运行或恢复；同目录有文件锁，已有进程时不要重复启动。

## TensorBoard 与 LITs 评价

TensorBoard 端口 **32003**，日志目录 `runs/majestic_v1_20260920/tensorboard`。记录损失、学习率、梯度范数、吞吐、显存、分语言验证指标，以及评价音频和评分。

```bash
.venv/bin/python scripts/watch_evaluation.py --run-dir "$PWD/runs/majestic_v1_20260920" --gpu 3
.venv/bin/python -m tensorboard.main --logdir runs/majestic_v1_20260920/tensorboard --host 0.0.0.0 --port 32003
```

评价使用 LITs 原有 `training/common/evaluate_checkpoint.py` 的 ASR、metrics 和 summarize 实现：Qwen3-ASR、CER/WER、WavLM/CAMP 相似度、DNSMOS。固定中文 200、英文 200、混合 50 条文本及对应参考；第 100 步先做每类 8 条冒烟评价，后续每 1,000 步和阶段结束做全量评价。GPU 3 同时承担训练和异步评价，因此需要显存余量。

voicepack 从固定的 96 条**训练集**参考中提取均值，不用 val/test 拟合音色。Stage 1 的韵律编码器尚未训练，评价明确标记为“已学声学 style + 原生韵律”的诊断输出；Stage 2 完成前 1000 步后初始化并直接优化独立 256 维 voice，每批一半使用此 voice，另一半使用参考编码器。`majestic.pt` 导出学习到的参数，编码器均值另存 `encoder_mean.pt` 对照。早期评分用于验证链路，不能视作最终音质结论。

补充配对评价位于运行目录`diagnostics/<checkpoint>/report.html`：400条重建、24条整句多条件试听及LITs评分，已覆盖两个阶段末尾。可直接在GitHub查阅的汇总见[评估结果](docs/evaluation_results.md)。

## 查看本机运行记录

```bash
cat runs/majestic_v1_20260920/status.json
cat runs/majestic_v1_20260920/supervisor_status.json
cat runs/majestic_v1_20260920/evaluation_status.json
tail -n 5 runs/majestic_v1_20260920/stage1.log
nvidia-smi
```

运行 checkpoint 在 `checkpoints/`；自动评价在 `eval/`；固定源码在 `source/`。训练环境为独立 `.venv`（复用现有 LITs 环境的 PyTorch），ASR 和音质评价分别复用现有独立环境。不要删除共享音频、LITs 评价代码或这些环境。

## 清理后的目录

`runs/` 仅保留正式运行 `majestic_v1_20260920`。旧预检 checkpoint、样例音频、调试脚本/配置、重复转换权重和临时日志已清理；检查结论保留在 `reports/`。当前运行的冻结源码快照保持原样，其中历史调试脚本属于启动时快照，不作为新的运行入口。

## 仓库内容

提交源码、训练及模型结构配置、依赖版本与来源校验值、许可证、评估JSON/CSV。大权重、训练数据、checkpoint、WAV、虚拟环境和凭据不入Git。在其他机器使用前按[依赖准备](docs/dependencies.md)配置外部数据及评价环境。
