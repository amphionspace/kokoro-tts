# 自建7.48M大气女声学生：蒸馏方案

2026-09-22。用户授权完成方案与预检后直接训练。目标为中文、英文、中英混读共用的单音色、24kHz学生，教师为本项目20轮长样本加权82M最终模型。学生从随机初始化开始，不加载发布的英文学生权重；该权重仅用于核对结构和严格加载兼容性。

## 参考路线与适配边界

参考[oddadmix/Kokoro-7M-Distill](https://huggingface.co/oddadmix/Kokoro-7M-Distill)、[TRAINING.md](https://huggingface.co/oddadmix/Kokoro-7M-Distill/blob/9357717e6499717b769ab65de61797aa7c501c86/TRAINING.md)及其`training/train_student.py`，固定revision为`9357717e6499717b769ab65de61797aa7c501c86`。

采用其核心做法：教师生成音频，同时导出生成该音频的音素时长；学生在教师时长展开的条件下重建教师音频，再在自由合成路径评价。这次不做真实录音微调，因此不训练CTC对齐器，也不把教师预测时长配到原始VoxCPM2录音上。上游CTC章节用于真实录音微调，是另一条流程。

参考发布物有英文/阿拉伯开发路径和缺失的`_env`等依赖，不能原样启动。项目独立实现数据缓存、四卡DDP、完整断点、验证与LITs评估。沿用可配置的缩窄Kokoro模块，保留许可证与来源，避免宽松加载静默漏参数。参考实现的`disable_complex=True`还包含STFT固定buffer；已按其结构严格加载核对全部参数。

## 模型

参数总数为**7,477,702**，不含训练判别器、WavLM和固定voice。ALBERT隐藏192、12层共享参数、3个attention heads、中间768；文本/韵律隐藏160；decoder隐藏256、输出128、上采样初始通道128。风格维度仍为128+128，178个token槽位和通用v1.0完整音素表不变，已逐项确认与教师词表相同。模型配置为`configs/student_7m.json`。

使用教师的同一份固定`majestic.pt`作为学生条件并随学生导出，不换用af_heart或af_msa。学生独立随机初始化；时长头的bias按**训练缓存**的平均每token帧数初始化，避免50个sigmoid初始求和约25帧造成无意义的超长输出。其余参数均按新网络初始化，不使用英文模型权重。

## 教师数据

教师路径：`runs/majestic_s2_20ep_long_resume6k_20260921/eval/stage2_final`。对已有train 53,723条、val 400条分别生成整句音频，speed=1、不分句、不截断；608条保留测试集不用于训练。保留原划分和音素，val不混进train。

缓存包括PCM16波形、含BOS/EOS的token与整数时长、生成该波形的F0/能量。逐条校验`音频采样数 = sum(duration) × 600`、F0/能量帧数为时长帧数的两倍，拒绝非有限值、过长输出和异常幅值；记录削波比例。教师权重、voice、来源清单和协议均保存哈希。缓存由四个GPU worker生成，全部结束并检查索引完整、无重复、划分及文本一致后，才形成正式训练清单。

生成目标采用教师自己的推理结果；学生最多学习到该教师的行为，不能假定蒸馏会自动修复其长文本时长偏差。原始合成库音频仍保留，但不是本轮重建目标。

## 优化与预算

正式参数见`configs/distill_7m.json`。按用户确认采用四卡DDP、每卡batch16、全局64，20,000次更新，约128万抽样位置。每卡32/48的长样本预检也已通过，但正式训练使用16，不追求更大batch。最终值写入运行冻结配置。AdamW，betas=(0.8,0.99)、weight decay=0.01、生成器/判别器基础学习率2e-4；500步warmup与cosine衰减至0.1倍，梯度裁剪5。

教师整数时长用于展开学生特征；学生的时长预测接受L1监督。声学窗口约3秒，从合法起点均匀随机截取，整个decoder输出参与评分；不剔除两端、不额外抽样首尾，短句按可用长度缩短窗口。文本与韵律仍读取完整句子。循环层FP32，decoder/GAN/WavLM采用BF16；预检拦截非有限梯度。

生成器目标为：`5×多分辨率STFT + 5×log-Mel L1 + 5×duration L1 + 100×silence + WavLM + F0 + energy + GAN`。STFT采用线性幅度的谱收敛及log幅度L1，三组FFT为1024/2048/512。静音损失只在教师10ms RMS低于1e-3的窗口约束学生超过教师噪声底的RMS。F0为smooth-L1/10、energy为L1，是本项目加入的教师韵律直接监督；上游主脚本没有这两个显式项。GAN包含MPD与多分辨率谱判别、特征匹配和相对项，分别记录；前500步仅训练判别器，之后加入生成器GAN。WavLM冻结，用作感知特征匹配，不启用WavLM对抗分支。

## 预检、评价和产物保留

启动前检查：精确参数量与词表、三语言教师缓存的时长/音频一致性、随机学生前后向、四卡GAN与五模块梯度、断点恢复、导出/自由合成、ASR/音色评分链路，以及长样本batch显存。

每1000步保存checkpoint并对400条val做确定性声学/时长验证；第100步每种语言8条自由合成冒烟，后续每1000步和结束执行450条固定文本的原LITs ASR、WavLM/CAMP及DNSMOS。保存全部周期checkpoint，按三语言等权的中文CER/英文WER/混读CER平均值选择候选，平局比较平均CAMP；冒烟结果不参与候选选择。仍保留其他质量指标与音频供人工判断，不把最后一步当成必然最优。

固定450条用于训练期间观察，不能替代独立测试集验收。最终还需比较82M教师/7M学生的长文、重复漏读、静音底噪和CPU推理速度。训练模型、优化器、逐rank RNG/buffer、采样位置、配置及数据哈希全部进checkpoint；运行源码固定快照。已有baseline、教师、网页和本轮产物不自动清理。

## 正式运行（2026-09-22）

运行目录：`runs/majestic_student7m_boundary_20260922`。已启动四卡，每卡16、全局64，预算20,000步。教师缓存已完整生成：训练53,723条、100.6463小时，验证400条、0.7671小时，均无削波记录。训练使用原始随机初始化，不使用预检模型继续训练。

- TensorBoard：统一使用`http://服务器IP:32003`，新训练为`student7m_boundary_train`和`student7m_boundary_eval`。各模型loss定义和训练步数起点不同，不应仅按同一横坐标的总loss比较质量。
- 训练日志：`runs/majestic_student7m_boundary_20260922/train.log`。
- 当前状态：同目录`status.json`、`supervisor_status.json`、`evaluation_status.json`。
- checkpoint：`checkpoints/`；周期导出与评分：`eval/`；候选选择：`best.json`。
- 固定代码与参数：`source/`、`config.json`、`source_manifest.json`；教师缓存审计：`teacher_cache_audit.json`。

运行或恢复入口（已有实例时不要重复启动）：

```bash
.venv/bin/python -m distillation.run \
  --run-dir "$PWD/runs/majestic_student7m_boundary_20260922" \
  --cache "$PWD/data/distill7m_teacher_long20" \
  --config "$PWD/configs/distill_7m.json"
```

预检结果保存在[汇总](../reports/distill7m_preflight.json)和[恢复检查](../reports/distill7m_resume_check.json)。断点恢复的采样位置、CPU/CUDA RNG均一致，下一步前向loss一致到日志精度；CUDA反向不是逐位确定，模型参数差值RMS约1.24e-6、最大约3.03e-4，未宣称逐位复现。三步随机模型的评分只验证评估链路，CER/WER仍为100%，不能作音质成绩。

预检和benchmark的临时运行目录已清理，JSON证据保存在`reports/distill7m_preflight_details/`；正式训练及所有baseline/教师产物保留。[清理记录](../reports/cleanup_20260922.json)。上游源码冗余副本所在的`provenance/`已移除，必要来源和许可证位于`vendor/`，下载清单位于`configs/model_assets.json`。

## 2026-09-22 首尾监督修正并从零重训

旧运行 `majestic_student7m_20260922` 已按要求删除，仅保留仓库中的清理及指标记录。新运行 `majestic_student7m_boundary_20260922` 从随机初始化开始，不恢复旧学生、优化器或步数。教师缓存与首尾0 token保持不变。

声学训练与82M微调对齐为整个约3秒窗口参与重建、静音、WavLM和GAN损失，`context_frames=0`，不再剔除两端各0.5秒。按用户要求，不额外抽样句首句尾；从合法起点均匀随机截取窗口（包含最后一个合法起点），短句用整句。时长/F0/能量仍保留教师监督，不照搬82M排除边界的时长L1。其余四卡每卡16、20,000步及学习率不变。周期声学验证仍为确定性中间窗口，完整自由合成评价保留；声学loss定义已改变，不宜与旧曲线直接比较。

2026-09-22：按用户要求统一使用 TensorBoard **32003**，包含 baseline、82M教师和重训7M的训练/评估曲线；独立32005已关闭。重训曲线名为 `student7m_boundary_train` / `student7m_boundary_eval`。重新运行项目根目录的 `distillation.run` 时省略 `--tensorboard-port`，默认使用已有综合页面。
