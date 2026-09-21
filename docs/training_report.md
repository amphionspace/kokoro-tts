# Kokoro 大气女声：完整训练报告

更新于2026-09-21。运行名为`majestic_v1_20260920`。本轮已完成预定4个Stage 1 epoch和10个Stage 2 epoch，以及阶段末评价；**这是固定预算训练结束，不是loss已收敛或最终质量验收通过**。长文本语速偏快已在有限诊断中复现，见第7节。

本文串起模型、数据、训练操作与结果。逐项实现和损失公式见[技术参考](training_reference.md)，依赖、模型来源和校验值见[依赖准备](dependencies.md)。

## 1. 模型及训练目标

使用官方通用`hexgrad/Kokoro-82M v1.0`，revision为`f3ff3571791e39611d31c381e3a41a3af07b4987`。没有采用v1.1-zh中文权重，也没有把LITs的21k或IMF权重作为Kokoro初始化。目标是一个固定大气女声，支持中文、英文与混读，输出24kHz单声道语音。

部署模型保留五组基座参数及独立voice：

| 模块 | 功能 |
|---|---|
| `bert`＋`bert_encoder` | 将音素上下文编码为韵律预测所需表示；768维映射到512维 |
| `text_encoder` | 卷积与双向LSTM生成声学内容表示 |
| `predictor` | 预测音素时长、F0及能量 |
| `decoder` | 内容、F0、能量与声学style经ISTFT解码器直接生成波形 |
| 固定voice | 256维，前128维声学style、后128维韵律style，三种语言共用 |

推理顺序为：文字规范化与G2P → 音素编码 → 时长预测与展开 → F0/能量预测 → 波形解码。没有额外接入Vocos，也不加载训练用的参考录音、对齐器和判别器。

训练另外引入声学/韵律编码器、ASRCNN对齐器、冻结JDC音调提取器、MPD/多分辨率谱判别器及冻结WavLM感知网络。它们用于提供条件、监督与损失，不都属于82M推理网络。没有训练diffusion或SLM OOD对抗分支。

架构整理自固定版本的kikiri-tts及其StyleTTS2/Kokoro子模块，本项目自行适配训练驱动。来源与许可证保留在`provenance/`和`vendor/`。官方基座五组权重严格加载，不忽略缺失参数。辅助编码器输出围绕原生`zf_xiaobei`向量初始化，避免随机近零style导致的解码幅值异常；该原生音色只提供稳定起点。

## 2. 训练数据

复用LITs已冻结的MajesticVoice数据清单，只选`speaker=1`，不包含LJSpeech音频。波形由VoxCPM2生成，属于合成数据；本轮Kokoro训练没有重新生成或复制全部音频。

上游文本来自Foundation train完整转写、Emilia2完整转写和获准使用的既有DOTA-ME-CS脚本。中文/混读参考为`ts004_cn_000001.wav`，英文使用选定英文候选`72003.wav`，英文合成及音色评分均以英文参考为准。上游记录实施ASR、音色、DNSMOS、时长和静音/削波筛选；Kokoro接入时另外核对音频头、时长、划分和前端，没有重跑上游全量声学QC。

| 语言 | 训练条数 | 训练小时 | 验证条数 | 保留测试条数 |
|---|---:|---:|---:|---:|
| 中文 | 25,323 | 50.0004 | 192 | 205 |
| 英文 | 16,515 | 25.0002 | 115 | 294 |
| 混读 | 11,885 | 24.9749 | 93 | 109 |
| 合计 | 53,723 | 99.9755 | 400 | 608 |

原始训练清单为53,732条；9条包含当前前端不支持的外语拉丁变音字符，显式隔离，未静默删除字符后继续使用原音频。划分审计检查跨split音频路径、规范化文本与来源组互斥。训练清单SHA256为`f700caca1a40f37b7e47085893f5dfa6448b5a73c5ca4a8a7e0e8072d1447b19`。细项见[数据审计](../reports/data_audit.json)和[前端审计](../reports/frontend_audit.json)。

Misaki 0.9.4与通用v1.0词表配套；中文/混读里的拉丁文本显式路由英语G2P，不复用LITs音素或tone ID。有效音素最多510个，加首尾ID 0。未知符号和超长输入直接拒绝。

音频保持24k；辅助特征沿用ASR/JDC历史约定，在24k波形上使用16k参数构建80-bin Mel滤波器。这一步不是重采样。FFT/窗长/hop为2048/1200/300；训练与voice提取采用同一归一化。谱重建损失按24k构建，WavLM感知输入才真正重采样为16k。

## 3. 两阶段如何训练

### Stage 1：对齐与声学重建

4 epochs，共3,360次更新。更新对齐器、文本编码器、声学style编码器、decoder及波形判别器。BERT、韵律预测器和韵律编码器暂不更新，JDC始终冻结。

整句Mel和音素经对齐器产生软对齐，再求最大单调路径MAS。训练约一半使用软对齐、一半使用MAS展开内容；验证使用MAS。从对齐结果中截取至多5秒，真实音频提供F0、能量和声学style，decoder重建对应波形。

生成器损失为`5×Mel + WavLM + GAN + 文本交叉熵 + 单调对齐损失`，单调项内部含10倍系数。这个阶段主要建立目标音色的重建能力；自由合成评价仍借用基座原生韵律，不能称作完整目标韵律已学好。

### Stage 2：预测韵律及联合适配

10 epochs，共8,400次更新，累计步数从3,360继续至11,760。加载Stage 1结果，初始化韵律编码器并重新创建优化器。

前1,000步更新BERT、bert_encoder、predictor和predictor_encoder；decoder冻结但保留输入梯度，GAN暂不启用。之后加入声学编码器、decoder、固定voice及波形判别器联合训练。对齐器、文本编码器、JDC冻结。

时长用MAS目标监督：50维logits的sigmoid之和接受L1监督，同时用逐帧占用目标训练BCE；F0、能量接受真实音频条件监督并进入波形生成。损失为`5×Mel + WavLM + 联合阶段GAN + duration + 20×duration_BCE + F0 + energy`，F0项内部除以10。

训练波形路径按真实MAS时长展开，推理按预测时长取整展开。这种条件差异仍存在，不能假定波形loss已直接优化了推理时的整数时长路径。

### 固定voice如何得到

Stage 2第1,001次更新前，从固定96条**训练集**参考提取声学与韵律编码均值，初始化独立`[1,256]`参数。此后每卡16条中随机8条使用固定voice，另8条使用逐句参考编码器；分支选择在时长、F0/能量和decoder处一致。

voice两部分都反向更新，学习率`1e-4`、weight decay为0。checkpoint保存其参数及初始化标志，恢复时不重复初始化。导出`majestic.pt`将同一向量复制为`[510,1,256]`；这些位置内容相同，并未学习长度相关voice。`encoder_mean.pt`仅作对照，部署不覆盖学到的voice。

## 4. 运行配置、保存与复现

四张A800 80GB，DDP每卡batch16，全局64，每epoch840步。长度分桶减少padding，没有每batch固定语言配额。整句用于对齐与时长监督，波形重建随机裁剪最多400 Mel帧约5秒，验证居中裁剪。

BF16 autocast，文本编码器FP32；AdamW的betas为`(0,0.99)`、eps为`1e-9`，常规weight decay为`1e-4`。BERT学习率`1e-5`，主干`5e-5`，辅助/判别器`1e-4`。每阶段独立500步warmup，预算前80%后线性下降到基础学习率的0.1；梯度裁剪上限5。

先按[依赖准备](dependencies.md)准备环境、基座和辅助权重；提供本机数据及外部评价依赖后运行：

```bash
.venv/bin/python scripts/download_assets.py --verify-only
.venv/bin/python scripts/prepare_data.py --source /path/to/lits/data
.venv/bin/python scripts/prepare_frontend.py
.venv/bin/python scripts/run_training.py --run-dir "$PWD/runs/new_run"
```

训练参数见`configs/train.json`与`configs/stage2.json`。supervisor固定四卡，冻结源码及配置快照，依次启动两阶段；训练失败则停止。另起评价worker与TensorBoard：

```bash
.venv/bin/python scripts/watch_evaluation.py --run-dir "$PWD/runs/new_run" --gpu 3
.venv/bin/python -m tensorboard.main --logdir runs/new_run/tensorboard \
  --host 0.0.0.0 --port 32003
```

checkpoint保存模型、优化器、逐rank随机状态、buffer、数据哈希和下一批位置。恢复要求数据、配置与卡数匹配；使用相同run目录会恢复最近checkpoint，已完成阶段会跳过。新训练应使用新目录，不修改已有运行冻结快照。已有进程时不重复启动。

本轮Stage 1于2026-09-20 10:22 UTC结束，Stage 2于14:36 UTC结束。曾在Stage 1约2k处暂停复核并从2k恢复，因此墙钟跨度不等于纯训练耗时。`status.json`保存最后一次batch状态，仍可能写着training；完成状态应同时检查`stage*_complete.json`、`training_complete.json`及supervisor状态。

## 5. 评估协议与最终结果

三套证据分开：400条验证重建、固定450条文本自由合成、24条多条件整句诊断。保留608条测试集尚未做独立最终评分。阶段第100步为24条冒烟；每1,000步及阶段末全量评价。ASR使用Qwen3-ASR，音色使用WavLM+ECAPA和CAMPPlus，DNSMOS为预测音质分数。CER/WER主要报告micro。

| Stage 2 final | 条数 | CER % | WER % | WavLM | CAMP | DNSMOS OVRL |
|---|---:|---:|---:|---:|---:|---:|
| 中文 | 200 | 0.185 | — | 0.8301 | 0.7474 | 3.4754 |
| 英文 | 200 | 0.325 | 0.821 | 0.8167 | 0.8218 | 3.3741 |
| 混读 | 50 | 0.089 | 4.167 | 0.7762 | 0.7208 | 3.4494 |

450条均完成评分。中文200条只有115种文本、英文200条有193种，均保留原始重复权重。混读WER只统计英文/数字词；DNSMOS不等于人工MOS。

同文本对比LITs直接训练IMF最终170k：中文CER为0.185%对0.582%，英文WER为0.821%对0.869%，混读CER相同。450条加权OVRL为3.4275对3.3032。Kokoro三组相似度和OVRL均更高，但基座、前端、预算、声码器和数据构成不同，不是纯架构或同算力对照。全部数值、协议核对和历史曲线数据见[评估结果](evaluation_results.md)。

## 6. loss尚不能认定收敛

本轮按固定epoch退出，没有early stopping或收敛判据。Stage 2后期中文验证损失仍下降：

| 累计步数 | Mel | duration L1 | F0 |
|---|---:|---:|---:|
| 9,360 | 0.33531 | 0.45932 | 2.14170 |
| 10,360 | 0.33527 | 0.45757 | 2.13966 |
| 11,360 | 0.33391 | 0.45209 | 2.08879 |

来源为本轮Stage 2 TensorBoard验证标量，摘录见[验证记录](evaluation/stage2_validation.json)。训练总loss含对抗项且batch组成变化，后期并非单调下降，不能仅凭总loss平台或单个末尾batch判断收敛。当前证据支持“仍有继续观察/优化空间”，不保证增加epoch必然改善听感或长文语速。

## 7. 长文本语速与后续实验

相同首句放入长上下文后预测时长缩短；数据覆盖、句尾韵律、固定voice与duration总体校准存在耦合，尚不能归因于单一机制。分析结论、分块试听反馈和从Stage 2学习率下降前checkpoint恢复并延长到20轮的方案统一见[长文本语速与重训实验](long_text_duration.md)。一次性分析代码和试听文件已按要求清理。

## 8. 导出、网页与下一步

最终部署使用`eval/stage2_final/kokoro.pth`、`majestic.pt`及基座config。模型与WAV留在本机，不入Git；最终checkpoint哈希见[导出元信息](evaluation/stage2_final/export.json)。网页命令：

```bash
.venv/bin/python demo/server.py --host 0.0.0.0 --port 32002
```

网页支持中英文/混读、语速和下载；详见[Demo文档](../demo/README.md)。网页默认按句末标点及换行分块，可关闭对比；中英文逗号不切。语速滑块不是长度效应修复。后续质量判断还需长文配对诊断、独立保留集评价和人工试听；本轮固定文本低CER不能替代这些检查。
