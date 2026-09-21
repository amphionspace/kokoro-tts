# 社区 Kokoro 训练流程审查

> 历史调研记录：下文的源码检出状态、旧路径与待验证事项描述的是调研时情况。两阶段训练现已完成；实际实现见[训练技术参考](../docs/training_reference.md)，最终结果见[评估汇总](../docs/evaluation_results.md)。

审查日期：2026-09-20。主项目 `semidark/kikiri-tts@a12d0410e89841e6f3c09958ae5c072f90ae1d49`；训练子模块 `semidark/StyleTTS2@b1956da84bf4a6ccc88f2440078024f1c4bfec7d`。结论仅针对上述快照，不能泛化为全部社区项目或作者实际发布模型的历史训练过程。

## 判断

两阶段思路合理：先补齐音频编码器与对齐器并训练声学重建，再训练可从文本预测的时长、F0、能量，最终导出 Kokoro 所需模块。但现有实现不能直接作为我们 100h 中英混合数据的生产训练器。已发现确定性的实现缺陷，另有训练与推理条件不一致的问题需要对照实验。

本次未修改上游训练代码、未启动训练。进行了四个 CPU 小实验；代码 `scripts/review_community_probes.py`，结果 `reports/community_review_probes.json`。实验验证特定机制，不代表完整训练复现，也不证明已发布模型一定受这些问题影响。

## 1. Mel 预处理不一致：已确认，应优先修复

- `StyleTTS2/meldataset.py:33` 未指定 `sample_rate`，实际 torchaudio 默认 16000。
- `scripts/extract_voicepack.py:306` 和 `StyleTTS2/kokoro_tb_utils.py:78` 显式使用 24000。
- 两者都输入 24k 原波形，没有在这个比较中重采样。sample_rate 参数影响 Mel 滤波器，不是无关标签。
- 同一条本项目录音，不加任何额外 padding 的对照：两者均为 `[80,269]`，但归一化 Mel 平均绝对差约 **0.33936**，滤波器不同。
- 训练还在每端加 5000 个静音采样，提取不加；这是另一个需要统一或证明合理的输入分布差异。

应只有一个共享特征提取入口，并核对 JDC、ASR 等辅助预训练模型期待的特征。不能仅凭实际波形是 24k 就不加审计地改变所有旧模型的输入约定。配置文件里的 fmax 也不能代替检查实际调用。

参考：https://docs.pytorch.org/audio/main/generated/torchaudio.transforms.MelSpectrogram.html

## 2. checkpoint 加载在分布式/恢复路径中可能完全失效：已确认机制

- `StyleTTS2/models.py:864` 直接执行 `load_state_dict(params[key], strict=False)`，既不处理 `module.` 前缀，也不检查匹配结果；打印 loaded 在加载前发生。
- Stage 1 在 `accelerator.prepare(model)` 之后才加载权重（`train_first.py:160,183`）；DDP 包装和普通权重可能错配。
- Stage 2 先加载再 DataParallel，修复了一个方向，但保存直接取包裹模块的 state_dict（`train_second.py:909`），恢复时未去前缀，反方向仍然有问题。
- 对原样抽取的 load_checkpoint 函数做小模型测试：输出 **decoder loaded**，正常返回，但全部参数保持原值。

单卡 Stage 1 → 首次 Stage 2 不一定触发这个问题；不能说每条训练路径都会加载失败。正确处理是统一保存未包裹模型、兼容读取前缀，并对五个基座模块要求参数完整匹配；训练辅助模块的预期缺失另列。

## 3. 恢复分支的优化器参数引用错误：已确认机制、触发有条件

- `train_second.py:219` 创建 optimizer。
- 当 `load_pretrained=True` 时，`train_second.py:256` 随后用 deepcopy 替换 predictor_encoder。
- 优化器仍引用旧 Parameter。小实验中，新模块有梯度，但其参数与优化器参数对象交集为 **0**；step 后新参数未变化。
- 若用于 resume，还会覆盖已训练的 predictor_encoder；`iters=0` 的无条件赋值也丢失恢复步数。
- 推荐的 `second_stage_load_pretrained=false` 首次 Stage 2 分支在 optimizer 前 deepcopy（187 行），不触发同一个对象替换缺陷。应准确区分首次衔接和恢复训练。

应拆分初始化、Stage 1→2、resume 三种路径。只在初始化阶段建立/复制模块，之后再建 optimizer；resume 不替换已训练模块，验证 optimizer 引用覆盖、模型/Adam 状态与连续训练的一步一致性。

## 4. “编码器是否训练过”的判据不成立：已复现

`extract_voicepack.py:255` 通过随机输入下输出 norm 是否大于 1000 判断 predictor_encoder 未训练；小实验中，全新随机编码器 norm 为 **0.36086**，会通过这项判定。有限且不爆炸不代表经过训练，更不代表坐标与预训练 predictor 兼容。

该探测发生在 `.eval()` 之前，随机输入还会更新 spectral_norm 的估计 buffer。应从 checkpoint 显式记录训练阶段与模块更新次数，缺失必要模块则拒绝导出；数值检查只用于发现异常。

## 5. 逐句 style → 平均 voicepack：方法层面的风险，尚非效果定论

Stage 2 使用目标录音提取逐句/逐片段条件；duration 在完整句子 style 下训练，F0N 还使用片段 style。导出时 `extract_voicepack.py:369` 把若干句的 acoustic/prosodic style 分别平均，并复制到全部 510 个长度位置。

平均向量是一个可运行的近似，但没有证明等价于训练条件，也不是有内容/长度条件的韵律模型。尤其我们有中文、英文、混读以及不同语言参考，可能存在不同分布。不能从此直接断言平均必然造成口音或必须启用 diffusion。

应把部署条件纳入训练与验证：先以固定参考集提取的 voicepack 跑完整自由合成，比较逐句条件重建与真正文本生成的差距；必要时训练共享可学习 style 或混合使用固定/逐句 style。最终用同一 checkpoint 导出和评测，而非只看 teacher-forced Mel。

## 6. 跨阶段混用 encoder 属于补救手段，不能当通用原则

`extract_voicepack.py:239` 推荐 Stage 1 acoustic encoder + Stage 2 prosodic encoder；但 Stage 2 达 joint_epoch 后也更新 decoder 和 style_encoder（`train_second.py:630`）。换回 Stage 1 encoder 可能引入条件分布变化，需实测；不能根据该建议就认定 Stage 2 必然损坏编码器。

“不开 GAN 就导致 spectral_norm 漂移/崩溃”的文档解释也不足以成为一般规律。谱归一化 buffer 在 train 模式的更新本来就是预期机制，是否异常要检查 checkpoint、权重/优化器更新、特征一致性和 train/eval 模式。先排除上述确定性问题，再做对抗损失消融，不能直接把 `joint_epoch=3, lambda_slm=1` 当所有数据的稳定性保证。

## 7. 对我们数据还需适配的点

- `kokoro_symbols.TextCleaner` 静默丢弃未知字符。中文声调、英文重音及混读转换必须全量覆盖审计，不能只检查模型维数能否加载。
- `max_len` 默认 200，对应主要波形损失最多约 2.5 秒的 crop；batch 内最短句还会缩短它。duration 仍按整句监督，不能误称整个训练只看 2.5 秒。需要比较较长 crop 对语调/语言切换的影响，不能直接沿用固定 epoch 配方。
- `kokoro_tb_utils.extract_voicepack` 从 root_path 递归搜所有 WAV，会绕过 manifest；若根目录同时含 train/val/test，评测音色条件可能使用保留集。必须以固定的 train 参考清单采样。
- 目录脚本每次随机取音频进行 TensorBoard 音色提取，可能同时改变权重和参考条件；比较 checkpoint 时固定参考样本、随机种子和测试文本。
- 示例训练列表文档写 speaker_name，但 loader 对第三列执行 int(speaker_id)；我们的数据导出需明确转换为整数 speaker ID。
- 对齐器出错 `except: continue`、过短片段整 batch 跳过等行为必须按语言记录，否则有效数据比例可能偏离预定 50/25/25 小时。
- MultiOptimizer 仅保存 Adam 状态，未保存 scheduler/RNG；当前脚本也未见显式 scheduler.step 调用。不能以文档里的 scheduler 名称推断学习率实际在变化。应记录实际 LR，并实现完整恢复契约。

## 建议的落地顺序

1. 修复并验证 checkpoint/optimizer/预处理三个基础环节，固定前端、划分和参考音频。
2. 保留两阶段框架，但 Stage 1 先让缺失辅助组件稳定，低学习率调整已有基座，监测预训练知识与风格分布漂移。
3. Stage 2 验证时长、F0、能量各模块确实更新；逐步引入波形/对抗损失，明确冻结模块的状态与参数更新。
4. 尽早做真实 Kokoro 导出合成：固定 style、预测 duration/F0/energy，三类文本分别评测与试听。恢复前后同一 batch/seed 的一步比较必须通过。
5. 通过小样本过拟合与导出验证后再使用全量 100h。上述修复优先于增加 epoch、扩增数据或引入 diffusion。
