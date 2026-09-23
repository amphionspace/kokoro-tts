# 100h + 长文本数据：82M 两阶段重训

用户指定将原有 100 小时与新增长文本合成数据合并，重新训练 82M Kokoro 的 Stage 1 和 Stage 2。

## 数据

原数据取 `data/prepared/train.jsonl`；新增数据取 `data/majestic_long50h/accepted/train.jsonl`。新增数据须先完成合成队列及最终审计。`scripts/prepare_combined150h.py` 随后检查新增文本、音素、音频与已有数据的重叠，检查来源组与验证/测试集的隔离，验证全部音频的单声道 24k 头信息与实际时长，并核对音素和官方词表 ID。

输出独立目录 `data/prepared_combined150h_20260923`，旧数据不修改，音频复用原路径。验证集、测试集、450条固定评价文本和96条训练音色参考逐字节复制。`preparation_report.json` 保存统计和输入/输出 SHA256；正式训练启动时检查输出哈希。

最终长文本审计与合并检查均已通过。训练集共 **63,226条、150.7982小时**（原53,723条99.9755小时 + 新9,503条50.8227小时），中文75.4697小时、英文37.7438小时、混读37.5847小时。验证400条、测试608条；全部64,234条音频通过头信息与时长检查。统计副本见 `reports/combined150h_data_audit.json`。

## 配置

- 官方通用 Kokoro-82M v1.0 基座，重新初始化 Stage 1。
- Stage 1：4轮，`configs/train_combined150h.json`。
- Stage 2：从本次完整 Stage 1 的最终 checkpoint 初始化，20轮，`configs/stage2_combined150h.json`。
- 四卡 DDP，每卡16条、全局64条；BF16，沿用既有学习率和损失。
- 每轮988步，Stage 1共3,952步，Stage 2共19,760步。
- Stage 2 沿用20轮实验的长样本采样方案：第13–17轮逐步提高最长20%样本权重，上限3倍。
- Stage 2 第1000步开始联合解码器与可学习音色训练。
- 独立运行目录 `runs/majestic_combined150h_20260923`；训练与评估源码在启动时冻结。
- TensorBoard 32003：`combined150h_stage1`、`combined150h_stage2`、`combined150h_eval`。

## 启动顺序

```bash
.venv/bin/python scripts/prepare_combined150h.py
.venv/bin/python scripts/check_combined150h.py
.venv/bin/python -u scripts/run_training.py \
  --run-dir "$PWD/runs/majestic_combined150h_20260923" \
  --config "$PWD/configs/train_combined150h.json" \
  --stage2-config "$PWD/configs/stage2_combined150h.json"
```

预检使用时长最长64条与音素数最多64条，在完整每卡batch16下执行两个阶段各2步。Stage 2 预检将联合训练和音色初始化提前到第0步，覆盖对应反向传播；这些临时权重不进入正式训练。结果保存 `reports/combined150h_preflight.json`。正式运行状态见运行目录的 `supervisor_status.json`、`status.json`，启动记录见 `launch_receipt.json`。

2026-09-23预检已通过：覆盖最长31.84秒、最多510音素，两个阶段损失与梯度有限，Stage 2音色参数取得非零梯度。正式训练已启动，Stage 1完成后监督进程自动启动Stage 2；固定文本评估进程也已启动。TensorBoard 32003已验证可访问。
