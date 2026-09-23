# MajesticVoice 长文本 50 小时合成

目标为质检通过的中文25小时、英文12.5小时、中英混合12.5小时。每条至少三个有实质内容的完整句子，实际音频15–45秒。使用自然转写整段文本，不拼接、不改写、不补静音、不截断。句数由保守的标点/缩写/数字规则检查，不是语法解析。

本目录从 LITs 的 `data_generation/majestic_200h` 和 `majestic_voice` 迁入当前 Kokoro 项目，原文件路径和校验值见 `migration_sources.json`。合成和质检不再导入 LITs 脚本。模型环境仍复用 `tts-assets` 与 `UltraEval-Audio`；实际数据与参考音频仍在 DATA_TTS。

- 数据目录：`/ai_sds_wuzz/DATA_TTS/MajesticVoice_long50h_20260922`，当前项目入口 `data/majestic_long50h`。
- 原有100小时数据不修改。排除历史文本/音素及完整句子重复，合成前继续用FTS和编辑距离/包含关系去重。既有heldout来源组排除；这不构成语义等价去重保证。
- 使用当前 Kokoro Misaki 前端，最多510个有效音素，加首尾0后不超过512。VoxCPM2使用实际tokenizer（含中文字拆分）、参考文本和参考音频长度计算prefill，并预留1000生成token，总预算不超过服务4096。
- 保留之前批准的中文参考和英文reference-4，复制到新数据目录以避免历史产物清理影响。中文和混读用中文参考。
- VoxCPM2单次整段合成；双候选每轮、中文/英文最多4个候选，混读最多8个。只选一个完全通过质检的候选计时。
- ASR使用空上下文，保持原CER/WER、音色相似度、DNSMOS、削波和静音门槛。ASR输出预算增至1024，质检batch4，实际音频不足15秒直接淘汰。
- 等待 `majestic_student7m_boundary_20260922/training_complete.json` 且训练进程退出，再启动GPU0–3的VoxCPM2服务（8240–8243），每服务8个请求槽；ASR用GPU2，音色用GPU3，DNSMOS用GPU0。
- 达标并排空候选后做最终审计，关闭本任务服务，不启动任何模型训练。

在项目根目录使用 `.venv/bin/python data_generation/majestic_long/scan_text.py`，随后 `prepare_pool.py`。文本预检通过后用 `launch_when_ready.py` 启动等待器。所有入口默认新数据根，可通过 `MAJESTIC_LONG_ROOT` 指定；配置在数据根 `config.json`，本目录保留初始配方。

进度：`launch_status.json`、`collection_status.json`、`pipeline_progress.json`。日志：`logs/`。最终审计：`reports/final_audit.json`。候选和唯一通过的音频分别在 `candidates/` 与 `accepted/`，48k原始音频及24k训练音频均保存；只按通过质检的实际时长累计，文本估算时长不计入目标。
