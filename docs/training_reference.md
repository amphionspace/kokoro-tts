# Kokoro 训练技术参考

这里记录结构、参数、损失和代码入口；环境见[依赖准备](dependencies.md)，已完成训练的结果见[评估结果](evaluation_results.md)。

运行`majestic_v1_20260920`的两阶段和最终450条评价已完成。Stage 1使用本机运行目录的`config.json`和`source/`，Stage 2使用`config_stage2.json`和`source_stage2/`，评价使用`source_evaluation/`。运行产物不入Git，仓库保留配置、实现和评估汇总。

## 1. 当前运行配置

| 项目 | 实际设置 |
| --- | --- |
| 基座 | `hexgrad/Kokoro-82M` 通用 **v1.0** |
| 基座 revision | `f3ff3571791e39611d31c381e3a41a3af07b4987` |
| 官方 checkpoint | `models/Kokoro-82M/kokoro-v1_0.pth` |
| 基座 SHA256 | `496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4` |
| 输出音频 | 24,000 Hz、单声道 |
| GPU | 4 × NVIDIA A800 80GB，DDP |
| 每卡 batch | 16 条 |
| 全局 batch | 64 条，无梯度累积 |
| 每卡 DataLoader workers | 4 |
| 精度 | BF16 autocast；文本编码器固定 FP32；模型参数和 AdamW 状态保留 FP32 |
| 随机种子 | 20260920，训练随机状态按 rank 分开 |
| Stage 1 | 4 epochs，约 3,360 optimizer steps |
| Stage 2 | 10 epochs，约 8,400 optimizer steps |
| Stage 2 联合训练起点 | 完成该阶段前 1,000 步后解冻音色编码器和解码器，并初始化、优化固定 voice |
| 单次波形重建长度 | 最长 400 Mel 帧，即约 5 秒；短样本 batch 相应缩短 |
| TensorBoard | `http://服务器地址:32003/` |

每个 epoch 的步数是 `ceil(53723 / 64) = 840`。为了让四张卡完成相同步数，sampler 每轮补齐 37 个样本位置；这是有限的重复采样，不会永久改变原始清单。本轮两阶段已完成累计11,760次生成器更新。阶段转换时 `stage_step` 归零，`global_step` 连续累计。

训练预算是固定 epoch，目前没有自动 early stopping 或自动选择“最优音色”checkpoint 的逻辑。最终质量需要结合评价曲线及试听判断。

## 2. 数据从哪里来，如何进入模型

### 2.1 数据来源和划分

原始清单来自：

```text
/119010446/tts-assets/training_runs/ljs_majestic_100h_backbone21k_20260914/data
```

仅使用 `speaker=1` 的 MajesticVoice。音频由 VoxCPM2 生成，属于合成语音；这点会影响最终模型的音质上限、韵律和潜在伪影。训练直接读取共享音频绝对路径，不另复制一份 WAV。

前端处理后实际进入训练的数据：

| 语言 | 训练条数 | 时长 |
| --- | ---: | ---: |
| 中文 | 25,323 | 50.0004 h |
| 英文 | 16,515 | 25.0002 h |
| 中英混合 | 11,885 | 24.9749 h |
| 合计 | **53,723** | **99.9755 h** |

验证集为 400 条：中文 192、英文 115、混合 93。测试集为 608 条：中文 205、英文 294、混合 109。测试集不参与参数训练、voicepack 拟合和周期性模型选择。

原训练清单有 53,732 条，其中 9 条混合文本带有当前英文前端不支持的外语拉丁变音字符，被显式隔离到 `data/prepared/train.rejected.jsonl`。没有静默删除无法识别的音素。

数据审计包含 WAV 头、24k 单声道、时长，以及跨划分音频路径、规范化文字、来源分组的重叠检查。这些检查不等于重新对全部音频执行 ASR 质检或内容相似去重。

### 2.2 文本前端

实现：[`training/frontend.py`](../training/frontend.py)，批量准备入口：[`scripts/prepare_frontend.py`](../scripts/prepare_frontend.py)。固定使用 Misaki 0.9.4。

```text
文本 → 规范化 → 中英分段G2P → 严格词表检查 → 音素ID与边界token
```

具体规则：

1. 对文本做 NFKC，合并连续空白；`·` 映射为空格，`–` 映射为 `—`。
2. 英文文本走美式英语 G2P。
3. 中文及混合文本显式切出拉丁字母片段，交给英文 G2P；其余片段走 `ZHG2P(version=None)`。这是为了处理旧版中文前端会保留原始英文单词的问题。
4. 得到音素字符串后，逐字符检查是否在基座 `config.json` 的词表中。未知音素直接报错或隔离。
5. 有效音素最多 510 个，再在两端加入 ID 0，满足 512 长度的上下文约束。超过长度的音频/文字对被拒绝，不截断文字后继续使用完整音频。

配置中的 embedding 容量是 `n_token=178`，但显式可用符号映射有 114 个；两者含义不同。LITs 原有音素 ID 和 tone ID 不直接输入 Kokoro。

训练清单哈希：

```text
f700caca1a40f37b7e47085893f5dfa6448b5a73c5ca4a8a7e0e8072d1447b19
```

### 2.3 波形与特征

实现：[`training/data.py`](../training/data.py)、[`training/common.py`](../training/common.py) 的 `Features`。

每条音频保持 24k，在首尾各补 5,000 个零采样，然后提取 Mel。Mel 帧数裁到偶数，便于和半速率对齐特征匹配。

| 特征参数 | 设置 |
| --- | --- |
| FFT | 2048 |
| 窗长 | 1200 samples |
| hop | 300 samples |
| Mel 通道 | 80 |
| 滤波器构造参数 | `sample_rate=16000, f_min=0, f_max=8000` |
| 归一化 | `(log(Mel + 1e-5) + 4) / 4` |

这里有一个必须明确的历史兼容约定：**波形是 24k，但 ASR/JDC 辅助模型对应的旧训练实现使用 16k 参数构造 Mel 滤波器。当前保留这一约定，不进行这一步的波形重采样。** 如果只在 voicepack 提取时改成 24k 滤波器，就会让训练与导出看到不同的特征分布。因此训练、验证、voicepack 提取共用同一个 `Features`。

这与另外两处处理需要区分：谱重建损失按实际 24k 构造；WavLM 感知损失会把 24k 波形真正重采样到 16k。

### 2.4 batch 组织和时间分辨率

sampler 每轮先按固定种子打乱样本，在局部大桶内按时长排序，再打乱全局 batch。每个全局 batch 切成四份，每卡 16 条，减少 padding 浪费；没有另外设置每个 batch 的固定语言比例。

- Mel 每帧对应 300 samples，即 12.5 ms。
- 对齐后的文本/韵律特征位于半速率时间轴，每帧对应 600 samples，即 25 ms。
- 200 个半速率帧对应 400 Mel 帧和 120,000 samples，即 5 秒。
- 整句用于文本—音频对齐和时长监督；从对齐结果中随机截取最长 5 秒做波形重建，控制显存。
- 验证用确定性的居中截取，降低随机裁剪对指标的影响。

## 3. 模型由哪些部分组成

| 模块 | 作用 | 初始化来源 | 最终文本推理是否需要 |
| --- | --- | --- | --- |
| `bert` | 音素上下文表示，服务于时长和韵律预测 | Kokoro v1.0 | 需要 |
| `bert_encoder` | 将 BERT 的 768 维表示映射为 512 维 | Kokoro v1.0 | 需要 |
| `text_encoder` | 卷积 + 双向 LSTM，把音素变为声学内容表示 | Kokoro v1.0 | 需要 |
| `predictor` | 时长预测、韵律特征、F0 和能量预测 | Kokoro v1.0 | 需要 |
| `decoder` | 根据内容、F0、能量和声学 style 直接生成 24k 波形 | Kokoro v1.0 | 需要 |
| `style_encoder` | 从参考 Mel 提取 128 维声学/音色 style | 新建辅助编码器，输出头围绕原生 style 初始化 | 导出 voicepack 时需要 |
| `predictor_encoder` | 从参考 Mel 提取 128 维韵律 style | 新建；Stage 2 使用 Stage 1 编码器特征初始化 | 导出 voicepack 时需要 |
| `text_aligner` | 根据音频和文字估计软对齐，提供单调对齐及时长目标 | 预训练 ASRCNN 辅助权重 | 不需要 |
| `pitch_extractor` | 从真实 Mel 提取 F0 监督 | 预训练 JDC | 不需要 |
| `mpd` / `msd` | 多周期波形判别器 / 多分辨率谱判别器 | 随机初始化 | 不需要 |
| 冻结 WavLM | 计算真实与生成语音的感知特征距离 | `microsoft/wavlm-base-plus` | 不需要 |

这里的 `msd` 是 `MultiResSpecDiscriminator`，具体实现做多分辨率谱判别。代码为了兼容训练容器还保留了 `wd` 模块，但当前不把它加入活跃优化器，也不使用 WavLM 对抗分支。

核心配置：内容隐藏维度 512、style 维度各 128；PLBERT/ALBERT 配置是 12 层、12 heads、隐藏维度 768。时长头 `max_dur=50`，每个 token 输出 50 个 logits。

基座五组参数共 548 个张量均严格匹配。新增训练辅助模块不属于原始 82M 推理权重，因此训练时参数和显存开销会明显大于最终推理模型。

### 为什么要重新训练两个 style encoder

官方推理权重提供了消费 style 向量的网络，当前基座中没有可直接拿来训练/导出的音频 style encoder。项目需要补上这两个编码器，把训练音频映射到解码器和韵律预测器使用的向量空间。

预检发现，普通随机输出头生成的向量接近零，会让预训练 ISTFT 解码器产生极大幅值。对同一批输入使用原生 voice style 后，输出峰值恢复到约 0.5–0.7。因此当前做法是：

- 使用 `zf_xiaobei.pt` 第 100 项的原生向量作为初始化中心。
- 声学输出头 bias 设置为前 128 维，随机 weight 缩小为原来的 0.01。
- 训练仍更新整个编码器；保留小的非零 weight，使共享特征层能收到梯度。
- Stage 2 将 Stage 1 的编码器参数原位拷贝给韵律编码器，再把输出头改为围绕后 128 维原生韵律向量初始化。

这个原生 voice 只提供稳定的起点，目标音色仍由 MajesticVoice 训练音频监督学习。该初始化解决了预检中的数值问题，最终音色相似度仍需评价。

## 4. Stage 1：先学会对齐、音色编码和声学重建

预算：4 epochs。更新 `text_aligner`、`text_encoder`、`style_encoder`、`decoder`，同时训练 `mpd/msd`。BERT、时长/韵律预测器和韵律编码器暂不更新，JDC 始终冻结。

```text
音素 → 文本编码；真实音频 → ASR对齐 → 按帧展开文本
真实音频 → 声学style、冻结JDC的F0、能量 → Decoder → 重建波形
更新对齐器、文本编码器、声学编码器、Decoder和判别器
```

一次 forward 的具体步骤：

1. 对齐器读取整句 Mel 和 token，输出文字 logits 与软对齐矩阵。
2. 去掉额外对齐位置，并屏蔽 padding；在 `no_grad` 下求最大单调路径。该路径提供稳定的 token—帧对应关系。
3. `text_encoder` 编码 token。训练时约 50% 概率使用软对齐展开、50% 使用单调路径展开；验证固定用单调路径。
4. 按 batch 内最短可用长度和上限 400 Mel 帧确定截取长度，随机取对应文本特征、Mel 和真实波形片段。
5. 冻结 JDC 提取真实 F0；从 Mel 的反归一化结果计算 log-norm 能量。
6. `style_encoder` 从该 Mel 片段提取声学 style。
7. `decoder` 使用内容特征、真实 F0、真实能量和声学 style，重建真实波形。
8. 分别更新判别器和生成器，并检查损失、梯度是否有限。

Stage 1 的波形路径使用真实音频提取的 F0/能量，因此这个阶段重建得好，并不自动意味着只输入文字就能达到同样效果。Stage 2 负责训练从文字预测这些条件的路径。

## 5. Stage 2：训练时长、F0、能量和韵律，再联合微调

Stage 1 完成后，supervisor 自动加载 `stage1_final.pth`，初始化韵律编码器并创建新的 Stage 2 优化器。正常长训要求 Stage 1 checkpoint 标记为已完成；有界预检才允许用中间 checkpoint。

| Stage 2 区间 | 更新模块 | 波形判别器 |
| --- | --- | --- |
| 前 1,000 步 | `bert`、`bert_encoder`、`predictor`、`predictor_encoder` | 暂不更新，GAN 项为 0 |
| 完成前 1,000 步后 | 上述模块 + `style_encoder`、`decoder`、可学习 `voicepack` | 更新 `mpd/msd`，启用 GAN 项 |

对齐器、文本编码器和 JDC 在 Stage 2 冻结。冻结 decoder 的前 1,000 步仍通过 decoder 把波形损失的梯度传回韵律预测路径；“冻结参数”不等于切断对输入的梯度。

```text
音素 → BERT及韵律预测器 → 时长、F0、能量
真实MAS对齐＋文本编码＋预测F0/能量＋声学style → Decoder → 波形
前1000步训练韵律；之后联合训练声学编码器、Decoder、固定voice和判别器
```

关键细节：

- 时长目标由单调对齐矩阵沿时间轴求和得到。
- 每个 token 的 50 维时长 logits 经 sigmoid 后求和，得到连续预测时长，用它做 L1 监督。
- 同时构造“第 k 帧是否仍属于该 token”的二值目标，使用 BCEWithLogits 训练 50 维 logits。
- **训练时使用音频对齐路径展开内容和韵律特征**，不会用取整后的预测时长来构造这条重建路径。
- F0 和能量由预测器输出，用真实音频提取值监督，并送入 decoder。
- 整句时长条件使用韵律编码器处理各条有效 Mel；随机声学片段的 F0/能量路径使用该片段提取的韵律 style。

这样既能通过连续损失训练时长头，也能通过波形损失训练韵律路径。推理时的 `round`、整数重复展开与这里的训练路径不同，不能简单地把推理函数去掉 `no_grad` 就替代训练器。

### 5.1 从 Stage 2 完成 1,000 步后直接训练 voicepack

配置见 `configs/stage2.json`。`voicepack.vector` 是一个独立的 `[1,256]` 参数，前 128 维是声学 style，后 128 维是韵律 style。它在 optimizer/DDP 创建之前注册，前 1,000 步冻结且不参与前向。完成第 1,000 步后、执行第 1,001 次更新前，用此时两个编码器对固定 96 条训练参考的均值原位初始化，广播到四卡，然后参与优化。

每卡 batch=16，随机选 8 条使用可学习 voice，其余 8 条使用参考编码器。一个样本的分支选择在整句时长、片段 F0/能量、decoder 三处保持一致。两路使用相同的真实对齐、预测 F0/能量和训练损失。这样固定部署向量直接收到梯度，同时参考编码器继续学习。验证默认保留参考路径；异步诊断另行检查固定 voice 路径。

```text
Stage 2 前 1000 步：
  真实 Mel → 参考韵律编码器 → predictor → 预测 F0/能量
  真实 Mel → 冻结声学编码器 ────────────→ 冻结 decoder → 波形损失

Stage 2 第 1001 次更新起：
  每卡随机一半：参考编码器 → [声学 style | 韵律 style]
  每卡另一半：可学习 voice → [前 128 维   | 后 128 维  ]
                           ↓             ↓
                         decoder ← predictor
                           ↓
                        波形及监督损失 → 更新模型、参考编码器和可学习 voice

部署保留：5 组 Kokoro 主干参数 + 学到的 256 维 voice
训练辅助：两个参考编码器、对齐器、JDC、判别器、WavLM，不随部署加载
```

voice 的学习率为 `1e-4`，使用 Stage 2 的学习率日程，weight decay 为 0。checkpoint 保存向量、初始化标志、初始向量和优化器状态；恢复后不重复初始化。TensorBoard 增加 `train/voicepack_delta_norm` 与 `train/voicepack_grad_norm`。

这是缩小固定 voice 与训练条件差异的实验设计，是否改善音色和可懂度要由 Stage 2 的固定 voice 评价确认。

## 6. 实际优化的损失函数

实现：[`training/train.py`](../training/train.py)、[`training/network.py`](../training/network.py)、[`vendor/styletts2/losses.py`](../vendor/styletts2/losses.py)。下面的系数与运行配置一致。

### 6.1 Stage 1 生成器损失

```text
L_G1 = 5 × L_mel
     + 1 × L_wavlm
     + 1 × L_GAN
     + 1 × L_s2s
     + 1 × L_mono
```

| 损失 | 当前计算方式 | 作用 |
| --- | --- | --- |
| `L_mel` | 三个分辨率的归一化 log-Mel 差异，按目标特征 L1 范数归一化后取平均 | 约束重建语音的频谱结构 |
| `L_wavlm` | 对真实/生成波形重采样至 16k，累加冻结 WavLM 各层 hidden state 的平均绝对误差 | 约束语音感知表示 |
| `L_GAN` | MPD + 多分辨率谱判别的生成器项、特征匹配项、相对损失项 | 约束波形和谱细节 |
| `L_s2s` | 对齐器输出与有效 token 的交叉熵，各条样本再平均 | 训练音频到文字的辅助对齐能力 |
| `L_mono` | 有效位置上软对齐与单调路径的平均绝对差 × **10** | 推动软对齐接近单调对应关系 |

注意 `L_mono` 自身已含 10 倍系数，表面上的外部权重为 1。

`L_mel` 沿用社区实现，FFT / hop / window 三组分别为：

```text
(1024, 120, 600)
(2048, 240, 1200)
(512,   50, 240)
```

使用 24k 的 MelSpectrogram，默认 128 个 Mel 通道，对 log-Mel 做 `(log(Mel + 1e-5) + 4) / 4`，再计算相对 L1 差异。它并不是简单的原始 waveform L1，也不是直接把三个线性 STFT 幅度相减。

判别器由真实片段和 `generated.detach()` 单独计算损失、反向和更新。随后冻结判别器参数，通过其生成器损失更新生成器。Feature matching 内部包含 2 倍缩放，TPRLS 相对项使用 `tau=0.04`。

BF16 下判别器输出相等可能令 TPRLS 的筛选集合为空。当前把空集合的贡献定义为零，并验证了该情况的 forward/backward 都是有限值。

### 6.2 Stage 2 生成器损失

```text
L_G2 = 5 × L_mel
     + 1 × L_wavlm
     + I_joint × L_GAN
     + 1 × L_duration
     + 20 × L_duration_ce
     + 1 × L_f0
     + 1 × L_energy
```

其中 `I_joint` 在该阶段前 1,000 步为 0，之后为 1。

```text
L_duration    = L1(sum(sigmoid(duration_logits)), aligned_duration)
L_duration_ce = BCEWithLogits(duration_logits, frame_occupancy_target)
L_f0          = SmoothL1(predicted_f0, JDC_f0) / 10
L_energy      = SmoothL1(predicted_energy, target_log_norm)
```

时长 L1 排除两端边界 token，BCE 使用有效 token。F0 损失内部除以 10。Stage 2 不再把 `s2s` 和 `mono` 加入训练损失，因为对齐器已冻结。

整个方案当前没有额外的显式说话人分类损失，也没有直接把 WavLM/CAMP 的评价相似度反向优化进网络。音色通过真实波形重建、style 编码和生成损失学习。

### 6.3 优化器、学习率和梯度

使用 AdamW：`betas=(0.0, 0.99)`、`eps=1e-9`、`weight_decay=1e-4`，独立 voice 参数的 weight decay 为 0。

| 参数组 | 基础学习率 |
| --- | ---: |
| BERT | `1e-5` |
| text encoder / bert encoder / predictor / decoder | `5e-5` |
| style encoder / predictor encoder / text aligner | `1e-4` |
| MPD / MSD 判别器 | `1e-4` |
| 可学习 voicepack（完成 Stage 2 前 1000 步后） | `1e-4` |

实际只更新当前阶段激活的参数组。每个阶段独立进行 500 步线性 warmup：第一个更新的倍率为 `1/500`。前 80% 预算后开始线性衰减，最低倍率为基础学习率的 0.1。Stage 2 新建优化器，学习率日程也重新开始。

生成器和判别器分别做最大范数 5 的梯度裁剪。出现非有限损失或梯度会抛错停止；不会静默跳过后假装训练成功。阶段第 1、第 100 步及 voice 首次激活的更新检查各活跃模块梯度是否非零、有限，并保存报告。

## 7. 四卡如何协同，显存为什么这样配置

四个进程分别绑定 GPU 0–3，各持有同一套生成器和判别器参数，读取全局 batch 的不同部分。DDP 同步梯度；生成器的 DDP 包装覆盖 predictor、`F0Ntrain` 和 decoder 的完整路径，避免关键训练调用绕过 DDP。

所有后续可能解冻的参数都先注册到 DDP，再按阶段切换 `requires_grad`。生成器使用 `find_unused_parameters=True` 处理阶段冻结，关闭自动 buffer 广播，并在 checkpoint 中保存各 rank 的 buffer 状态。

混合精度的实际边界：

- 外层前向主要使用 BF16，参数不是整网 `.half()`。
- 文本编码器使用 FP32：预检在 packed cuDNN LSTM 的 BF16 backward 中定位到了 NaN。
- 谱重建损失接收 FP32 波形。
- 冻结 WavLM 参数不更新，但生成波形分支仍有梯度，可以把感知损失传回生成器。
- 开启 TF32 与 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

容量测试结果：

| 测试 | 结果 |
| --- | --- |
| 每卡 24 条 | 触及 80GB 显存上限并 OOM，未用于正式配置 |
| 每卡 16 条，Stage 1 | 连续反向与更新通过，常规显存约 60GB |
| 每卡 16 条，Stage 2 联合训练 | 三步四卡预检通过，所有活跃模块梯度正常 |
| 四卡各 16 条最长 20 秒样本，Stage 2 | 前向、反向、优化与保存通过，显存约 68GB |

正式启动早期的 20 秒采样中，四卡平均 GPU 利用率约 75%–76%，峰值 98%–100%；每卡显存峰值约 60GB。短时计算阶段的采样曾达到 86%–99%。GPU 利用率会受变长数据、CPU 前端/读取、对齐路径计算、通信、保存和评价影响，不能把瞬时峰值解释成全程恒定 100%。

GPU 3 还运行异步评价，保留的显存用于 ASR 与评分模型。这次选择 batch 16 是依据实测容量和并行评价需求确定的。

## 8. 文本推理 Pipeline

最终推理读取五组 Kokoro 参数和一个 256 维 voice 向量，不需要训练用的真实音频、ASR 对齐器、JDC、判别器或 WavLM 感知网络。

```text
文本 → G2P → 音素IDs → BERT/bert_encoder＋voice后128维 → 时长及韵律
预测时长展开内容和韵律表示 → 预测F0/能量
展开内容＋F0/能量＋voice前128维 → Decoder/ISTFT → 24k波形
```

推理时长的公式：

```text
d_i = max(1, round(sum_k sigmoid(logit_i,k) / speed))
```

然后根据每个 token 的整数帧数构建展开矩阵。该矩阵同时用于内容表示和韵律表示，使两条路径落在相同时间轴上。`F0Ntrain` 这个函数名虽然带有 train，推理也用它预测 F0/能量。

Decoder 使用风格调制的残差结构和谐波源，最终通过 ISTFT 直接产生波形。导出的模型包含这个波形生成部分，部署时不额外调用 LITs 的 Vocos。

当前评价调用 `forward_with_tokens`，输入在前端已严格校验。没有使用会静默过滤未知字符的通用字符串入口。单次最多 510 个有效音素；当前评价不会静默截断超长文本。网页已支持按句末标点和换行分块后直接拼接，详见[Demo说明](../demo/README.md)；未额外插入静音或做接缝处理。

## 9. voicepack 如何导出

实现：[`training/evaluate.py`](../training/evaluate.py) 的 `export_and_synthesize`。

### 9.1 固定参考

从训练集冻结选择 96 条参考：中文 32、英文 32、混合 32，时长范围 2–12 秒，记录在：

```text
data/prepared/voicepack_references.jsonl
```

每条参考沿用训练相同的 24k 波形、首尾 padding、Mel 滤波器和归一化，分别通过对应编码器。每条音频得到一个 128 维向量后做等权平均，因此不是按音频时长加权。

```text
s_acoustic = mean(style_encoder(reference_i))
s_prosody  = mean(predictor_encoder(reference_i))
voice      = concat(s_acoustic, s_prosody)   # 256 维
```

上式用于初始化，以及每次导出的 `encoder_mean.pt` 对照文件。完成 Stage 2 前 1000 步后，部署使用 checkpoint 中直接优化的 `voicepack.vector`，不再以编码器均值覆盖。三种语言共享这个单音色向量；当前没有分别导出中文、英文、混合三个 voicepack，也没有拟合随文本长度变化的 style。

为了兼容常见 voicepack 的索引形状，把同一向量复制为 `[510, 1, 256]`。510 个位置内容相同，不代表训练了 510 种不同的风格。

### 9.2 两阶段导出含义不同

| checkpoint 阶段 | 声学 style | 韵律 style | 文件 |
| --- | --- | --- | --- |
| Stage 1 | 已训练声学编码器的参考均值 | 原生 `zf_xiaobei` 后 128 维 | `stage1_diagnostic.pt` |
| Stage 2，voice 尚未初始化 | 声学编码器参考均值 | 韵律编码器参考均值 | `majestic.pt` |
| Stage 2，voice 已初始化并参与优化 | 学到的 voice 前 128 维 | 学到的 voice 后 128 维 | `majestic.pt` |

Stage 1 的 predictor_encoder 尚未训练，不能因为其输出范数看起来正常就把它当作可用编码器。因此 Stage 1 导出明确标记为诊断组合。Stage 2 才是完整的目标声学/韵律组合。

导出的 `kokoro.pth` 仅包含：`bert`、`bert_encoder`、`predictor`、`text_encoder`、`decoder`。导出后再用推理结构严格加载检查。`export.json` 记录阶段、步数、conditioning 类型、参考清单哈希和 checkpoint 哈希。

Stage 2 联合阶段有一半训练样本直接使用部署 voice，以缩小条件分布差异。`encoder_mean.pt` 另存为对照，`export.json` 的 `conditioning` 明确标记实际部署使用均值还是可学习向量。

## 10. 验证集与 LITs 固定文本评价

当前保留同步验证和固定文本自由合成评价。

### 10.0 配对诊断与试听报告

专项配对重建/对齐/多条件诊断代码已于2026-09-21移除。后续训练保留常规验证和固定450条自由合成评估；仓库已有诊断汇总仅作历史记录；对应代码与本机旧诊断产物已清理，不再运行或展示在TensorBoard。详见[长文分析与新实验](long_text_duration.md)。

### 10.1 400 条验证集：有真实音频的重建检查

每 1,000 个阶段 step 和阶段结束，四卡分摊全部 400 条验证样本，batch=1、确定性裁剪，不更新参数。

按 `zh/en/mixed` 分组统计：

- Stage 1：谱重建、`s2s`、`mono`。
- Stage 2：谱重建、duration、duration BCE、F0、energy。

验证前后保存/恢复训练 RNG，避免评价消耗随机数后改变下一批训练的随机过程。

### 10.2 450 条固定文本：实际自由合成评价

沿用 LITs 的固定文本和参考协议，仅选 MajesticVoice 三组：

| 分组 | 全量条数 | 相似度参考 |
| --- | ---: | --- |
| `majestic_zh` | 200 | `reference/ts004_cn_000001.wav` |
| `majestic_en` | 200 | `candidates/wavs_24k/72003.wav` |
| `majestic_mixed` | 50 | `reference/ts004_cn_000001.wav` |

复制保留的原始 `eval_protocol.json` 仍包含 LJSpeech 信息及原始 650 条描述，用来保留 LITs 协议来源；**实际合成清单 `eval_manifest.jsonl` 已筛选为上述 450 条**。当前模型不会训练或评价 LJSpeech 音色。

```text
checkpoint → 导出模型及voice → 固定文本合成
波形 → ASR及音质/相似度评分 → JSON、TensorBoard和试听音频
```

具体执行逻辑：

1. 训练进程保存 checkpoint 后写 `eval_queue/*.json`。
2. 独立 worker 在 GPU 3 顺序运行 `synthesize → asr → metrics → summarize`，同时四卡训练继续。
3. 合成使用本项目 Kokoro 导出器。
4. ASR、metrics、summarize 复用 `/119010446/LITs/training/common/evaluate_checkpoint.py` 中的实现。
5. ASR 使用本地 Qwen3-ASR-1.7B、空提示 `context=''`、自动语言识别。
6. 报告 CER/WER 的逐条平均和按参考字符/词数加权的 micro 值，同时保存 WavLM/CAMP 相似度和 DNSMOS。
7. 中文主要看 CER，英文主要看 WER；混合文本结合 CER、英文词错误及试听。对纯中文计算出的英文词 WER 不应当作主要质量依据。
8. 每组的样例音频、数值指标写入 TensorBoard。

每个阶段第 100 步先评每组 8 条，共 24 条；每 1,000 步及阶段结束做全量 450 条。首个 Stage 1 全量评价会落在 global step 1,000；Stage 2 的间隔以自己的 `stage_step` 计算，TensorBoard 横轴仍使用累计 `global_step`。

评价异常会写 `failed.json` 和 `evaluation_status.json`，不会伪装成成功；当前失败作业不会自动重试。修复原因后删除对应失败标记，可让 worker 重新执行。模型合成时长要求在 0–120 秒内，检查 NaN/Inf，最终 WAV 按 PCM16 保存并裁到 `[-1,1]`；因此还要结合训练记录的原始生成峰值识别幅值问题，不能只看写盘后 WAV 的范围。

两阶段末尾各450条全量评价已完成，失败数均为0；指标、条件差异及诊断见[评估结果](evaluation_results.md)。

## 11. 保存、阶段切换与恢复

入口：[`scripts/run_training.py`](../scripts/run_training.py)。首次启动会冻结：

```text
runs/majestic_v1_20260920/
├── config.json                 # 本次实际配置
├── environment.txt             # pip freeze
├── source/                     # training、vendor、scripts 的代码快照
├── stage1.log / stage2.log
├── status.json                 # 最近一次训练日志状态
├── supervisor_status.json      # 父进程及阶段状态
├── checkpoints/
├── eval_queue/
├── eval/
└── tensorboard/
```

checkpoint 包含：

- 所有模型模块的权重，包括训练辅助模块；
- 生成器与判别器优化器状态；
- 当前阶段、累计步数、阶段步数；
- 下一个 epoch / batch 的位置；
- 每个 rank 的 Python、NumPy、Torch、CUDA RNG；
- 每个 rank 的模块 buffers；
- 训练清单 SHA256、完整配置、world size、最近验证结果。

保存使用临时文件再原子替换，避免把未完成文件当成可恢复 checkpoint。恢复会拒绝数据哈希、配置、阶段或卡数不一致的情况。DataLoader 使用独立 generator，减少重建迭代器对模型 RNG 的干扰。

断点预检比较了连续训练的第 10 步和从第 9 步恢复后的第 10 步：损失一致，全部模型状态在 `rtol=1e-5, atol=1e-5` 内一致，最大绝对差约 `5.46e-6`。CUDA 反向归约存在浮点非确定性，因此这里验证的是完整状态恢复和数值一致性，不承诺逐 bit 相同。

supervisor 在 Stage 1 正常结束后自动启动 Stage 2；任何阶段异常退出，记录失败并停止，不无限循环重启。两个阶段完成后写 `training_complete.json`。

## 12. 如何查看当前训练

项目目录：

```bash
cd /119010446/Kokoro-TTS-ZH
```

训练进度、父进程状态和评价状态：

```bash
cat runs/majestic_v1_20260920/status.json
cat runs/majestic_v1_20260920/supervisor_status.json
cat runs/majestic_v1_20260920/evaluation_status.json
tail -n 10 runs/majestic_v1_20260920/stage1.log
nvidia-smi
```

TensorBoard：

```text
http://服务器地址:32003/
```

已记录的主要标签：

| 标签 | 解释 |
| --- | --- |
| `train/loss` | 当前阶段总生成器损失 |
| `train/mel`, `train/wavlm`, `train/gan` | 主要声学损失 |
| `train/s2s`, `train/mono` | Stage 1 对齐监督 |
| `train/duration`, `train/duration_ce`, `train/f0`, `train/energy` | Stage 2 预测监督 |
| `train/discriminator` | 判别器损失 |
| `train/grad_norm` | 裁剪前生成器梯度范数 |
| `train/generated_peak`, `train/target_peak` | 波形峰值诊断；日志对 rank 数值取平均 |
| `lr/*` | 各参数组实际学习率 |
| `performance/samples_per_second` | 全局样本吞吐 |
| `performance/gpu_peak_gib` | rank 0 的累计峰值 allocated 显存，非四卡总和 |
| `val/zh/*`, `val/en/*`, `val/mixed/*` | 按语言拆分的重建验证指标 |
| 评价 run 下的 `majestic_*/*` | CER/WER、相似度和音质评分 |
| 评价 run 下的 `audio/*` | 固定文本样例 |

`batch/*` 只记录 rank 0 当前 batch 的语言计数，不是全局四卡语言比例。训练损失日志对各 rank 的标量取平均；验证指标则汇总全部验证样本。

只在训练进程已经退出时，用下列命令恢复：

```bash
.venv/bin/python scripts/run_training.py \
  --run-dir "$PWD/runs/majestic_v1_20260920"
```

同目录有文件锁，防止重复训练实例。重启评价和 TensorBoard 的命令可在 [README](../README.md) 找到；已有服务运行时不要重复占用相同端口。

## 13. 本项目相对社区代码的改动

社区来源固定为 `semidark/kikiri-tts` 的 `a12d0410e89841e6f3c09958ae5c072f90ae1d49`，训练子模块为 `b1956da84bf4a6ccc88f2440078024f1c4bfec7d`，推理子模块为 `b96fef95e6a746495f92443fac7c688f90fc57fc`。当前使用的是从这些实现整理出来的模块和本项目训练驱动，不能称为完全未经改动的社区训练复现。

主要改动及原因：

1. **严格加载**：只移除真实 `module.` 前缀并严格匹配，避免 `strict=False` 把参数全部漏加载后仍报告成功。
2. **优化器绑定**：初始化发生在创建优化器之前，编码器采用原位参数拷贝，避免替换模块后优化器仍指向旧参数。
3. **训练/导出特征一致**：统一 Mel、归一化和 padding，避免 style 编码输入分布不一致。
4. **可用的 style 初始化**：以原生向量为中心，修复随机 style 引起的巨大波形幅值。
5. **明确的阶段语义**：不靠编码器输出范数猜测是否训练过；Stage 1 导出明确使用原生韵律作为诊断条件。
6. **推理参数兼容**：AdaIN 的 InstanceNorm 使用 `affine=False`，与 v1.0 参数和训练实现一致。
7. **四卡与数值稳定性**：完整 DDP 边界、阶段冻结、FP32 文本 LSTM、非有限值检查、相对损失空集合处理。
8. **可恢复和可观察**：逐 rank 状态、数据哈希、固定源码快照、TensorBoard、LITs 异步评价。
9. **训练范围**：保留波形 GAN 与冻结 WavLM 感知损失，当前不启用 diffusion 或 SLM OOD 对抗分支。

`upstream/` 已删除。运行所需源代码在 `vendor/`，ASR/JDC 权重在 `models/auxiliary/`，原始许可证、提交号和审计用最小源码快照在 `provenance/`。删除 upstream 后调研探针重新运行通过，当前训练不依赖已删除目录。

## 14. 已验证与仍需观察的部分

已经验证：严格加载、前端词表和划分审计、四卡前向反向、活跃模块梯度、保存恢复、400 条验证、最长样本显存、两个阶段的严格导出与合成、LITs 完整评分链路、TensorBoard 服务。

训练已结束，以下问题仍需专项评估或人工试听：

- 中文声调、数字、专名和长句是否稳定；
- 英文口音、停顿和中英切换是否自然；
- 相似度指标改善是否对应人工感知改善；
- 固定可学习 voice 与参考 style 的评价差异，以及前者是否比 `encoder_mean.pt` 改善；
- 波形 GAN 是否稳定，以及谱损失、ASR 错误率、DNSMOS 是否出现相互矛盾的变化；
- 通用基座适配单音色后，其他原生 voice 的表现是否退化；当前没有保留多音色能力的专项训练目标。

固定文本评价已完成；独立保留测试集最终评分、人工听测和voice消融尚未完成，训练结束不等于完成交付验收。

## 15. 代码与审计入口

| 内容 | 路径 |
| --- | --- |
| 本轮固定配置（本机） | `runs/majestic_v1_20260920/config.json` |
| 本轮Stage 1源码快照（本机） | `runs/majestic_v1_20260920/source/training/train.py` |
| 项目训练驱动 | [`training/train.py`](../training/train.py) |
| 两阶段 forward | [`training/network.py`](../training/network.py) |
| 模型装配、特征和保存 | [`training/common.py`](../training/common.py) |
| 数据与 sampler | [`training/data.py`](../training/data.py) |
| 前端 | [`training/frontend.py`](../training/frontend.py) |
| 导出与评价适配 | [`training/evaluate.py`](../training/evaluate.py) |
| 两阶段 supervisor | [`scripts/run_training.py`](../scripts/run_training.py) |
| 异步评价 worker | [`scripts/watch_evaluation.py`](../scripts/watch_evaluation.py) |
| 数据审计 | [`reports/data_audit.json`](../reports/data_audit.json) |
| 前端审计 | [`reports/frontend_audit.json`](../reports/frontend_audit.json) |
| 预检记录 | [`reports/training_preflight.md`](../reports/training_preflight.md) |
| 恢复一致性检查 | [`reports/resume_check.json`](../reports/resume_check.json) |
| 源码及许可证来源 | [`provenance/sources.json`](../provenance/sources.json) |
