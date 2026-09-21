# Kokoro 通用 v1.0 大气女声适配：社区路线调研

> 历史调研记录：下文的源码检出状态、旧路径与待验证事项描述的是调研时情况。两阶段训练现已完成；实际实现见[训练技术参考](../docs/training_reference.md)，最终结果见[评估汇总](../docs/evaluation_results.md)。

日期：2026-09-20。用户要求：使用官方通用基座；复用 LITs 的中文、英文、中英混合大气女声数据。以下区分作者报告与本地验证，不将社区样例效果当成本项目效果。

| 项目 | 实际路线 | 对本项目的意义 |
|---|---|---|
| [semidark/kikiri-tts](https://github.com/semidark/kikiri-tts) | 补齐 StyleTTS2 训练组件，以 Kokoro 权重初始化，Stage 1/2 后导出 checkpoint 与 voicepack | 首选参考；作者发布德语基座及两个单音色模型，提供推理演示。本地已检出源码与子模块，尚未复现训练 |
| [tterrasson/kokoro-french](https://github.com/tterrasson/kokoro-french) | 基于 kikiri 的法语流程，包含数据处理、两阶段训练、导出和音色提取 | 参考跨语言适配与导出；作者特别提示 Stage 2 的 style encoder 漂移问题，提取时可加载 Stage 1 音色编码器，需在本项目验证 |
| [gushilabs/train-kokoro-encoder-styletts2](https://github.com/gushilabs/train-kokoro-encoder-styletts2) | 实验性补训未发布的 encoders，生成新 voicepack | 可作音色编码方向的参考，不能视为已验证的完整中英混合声学微调方案 |
| [sammy4321/Kokoro-Indic-Fine-Tuning](https://github.com/sammy4321/Kokoro-Indic-Fine-Tuning/blob/main/docs/JOURNEY.md) | Kannada 两阶段实验，作者报告加入 diffusion 训练与推理改善自然度 | 吸取韵律、裁剪长度和带宽检查经验；启用扩散改变运行路径，不直接作为标准 Kokoro 导出方案 |
| [jonirajala/kokoro_training](https://github.com/jonirajala/kokoro_training) | README 明确其模型是约 22M encoder-decoder transformer，外接 HiFi-GAN | 不能因名称相似就作为官方 82M 权重微调训练器 |

## 首选路线及代码核查

- 固定 `kikiri-tts` commit 和两个子模块 commit，见 `community_source_audit.json`。
- 本地比较了官方 v1.0 config 中全部 114 个显式 vocab 项与社区 `kokoro_symbols.py`，ID 不匹配数为 0。网络层有 178 个 embedding 位置，不等于 config 内有 178 个实际符号。这仅证明映射一致，尚未证明 Misaki 对全部中英文本的输出覆盖率。
- 检查 `train_second.py`：存在先 load checkpoint 再包 DataParallel 的逻辑；使用 `joint_epoch` 启动判别器；`slmadv.py` 有禁用 diffusion 时的分支。这些是实际源码检查，不是只根据仓库标题判断。
- 上游 config 同时有顶层配置和已标注为 dead config 的 `training:` 块，适配时仅以读取代码为准。
- 额外发现固定版本的 `models.py:load_checkpoint` 仍直接 `strict=False` 加载未统一去除 module 前缀；`train_second.py` 在创建 optimizer 后有 deepcopy 替换 predictor_encoder 的路径，且该路径没有针对 resume 的保护。这些是继续适配前需要修复并测试的问题，不能直接把原脚本视为已就绪。
- 源码中 `strict=False` 不能替代加载审计。应针对 bert、bert_encoder、predictor、text_encoder、decoder 全部统计 loaded/missing/unexpected/shape mismatch。
- 保留 Kokoro 的声学参数、词表和 iSTFTNet；LITs 的 Mel 参数、token IDs、duration target 不直接迁移。

## 数据与评估决策

三类训练数据使用同一 MajesticVoice 身份，按 LITs 冻结清单自然混合；约 50/25/25 小时是原配额，实际统计以 data_audit.json 为准。不额外加入 LJSpeech。

英文在 LITs 中使用单独批准过的英语参考，中文和混合使用中文参考。两者都要保留并分别评测，不把英文相似度与中文参考下的相似度直接混算。必要时在训练中提取逐句 style；是否共享一个最终 voicepack 需要通过跨语言试听决定。

先完成中英及混合 G2P 全量审计、小样本训练和标准 Kokoro 导出合成闭环，再运行全量训练。test 仅用于最终评价。当前没有新模型效果或训练收敛结论。
