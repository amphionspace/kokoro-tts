# 环境、模型资产与来源

本仓库包含训练实现、必要的架构配置及轻量评估结果，不包含大权重、数据或本机虚拟环境。配置以本次实际运行版本为准；这里没有宣称从零安装环境已在另一台机器完整复现。

## 已提交的小文件

| 文件 | 用途与来源 |
|---|---|
| `configs/project.json` | 通用Kokoro v1.0的仓库、revision、数据路径 |
| `configs/train.json`、`configs/stage2.json` | 两阶段训练超参数及Stage 2可学习voice配置 |
| `models/Kokoro-82M/config.json` | 基座网络结构、音素词表；原始文件SHA256列于`configs/model_assets.json` |
| `models/wavlm-base-plus/config.json` | 冻结WavLM感知网络的结构配置 |
| `vendor/styletts2/Utils/ASR/config.yml` | ASRCNN对齐器结构配置；加载其中`model_params` |
| `vendor/styletts2/Utils/PLBERT/config.yml` | 上游兼容配置；本项目实际使用Kokoro自带BERT参数，不需另下载PLBERT权重 |
| `vendor/`、`configs/model_assets.json` | 必要源码、许可证、上游提交号与原始审计记录 |
| `requirements.txt` | 实测训练/前端依赖版本；不等于包含所有间接依赖的完整lockfile |

## 模型与辅助权重

完整文件列表、固定下载URL、文件大小、SHA256见[`model_assets.json`](../configs/model_assets.json)。配置小文件随Git提交，其余按清单下载：

```bash
python scripts/download_assets.py
python scripts/download_assets.py --verify-only
```

下载器先验证已有文件；校验不匹配会报错，不覆盖已有权重。下载使用临时文件，大小和SHA256通过后才放到正式路径。`download_base.py`仍保留，但它只下载Kokoro配置与主权重，不覆盖完整训练依赖。

| 本机资产 | 来源 | 固定版本 |
|---|---|---|
| `models/Kokoro-82M/kokoro-v1_0.pth`、`config.json`、`zf_xiaobei.pt` | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M/tree/f3ff3571791e39611d31c381e3a41a3af07b4987)；voice来自`voices/zf_xiaobei.pt` | `f3ff3571791e39611d31c381e3a41a3af07b4987` |
| `models/wavlm-base-plus/pytorch_model.bin`、`config.json` | [microsoft/wavlm-base-plus](https://huggingface.co/microsoft/wavlm-base-plus/tree/4c66d4806a428f2e922ccfa1a962776e232d487b) | `4c66d4806a428f2e922ccfa1a962776e232d487b` |
| `models/auxiliary/asr.pth` | [StyleTTS2 ASR/epoch_00080.pth](https://github.com/semidark/StyleTTS2/blob/b1956da84bf4a6ccc88f2440078024f1c4bfec7d/Utils/ASR/epoch_00080.pth) | `b1956da84bf4a6ccc88f2440078024f1c4bfec7d` |
| `models/auxiliary/jdc.t7` | [StyleTTS2 JDC/bst.t7](https://github.com/semidark/StyleTTS2/blob/b1956da84bf4a6ccc88f2440078024f1c4bfec7d/Utils/JDC/bst.t7) | 同上 |

ASR/JDC是训练辅助网络，不是最终部署模型。`zf_xiaobei.pt`用于稳定初始化及Stage 1的原生韵律诊断，不是训练目标女声的最终voice。

架构来自[semidark/kikiri-tts](https://github.com/semidark/kikiri-tts/tree/a12d0410e89841e6f3c09958ae5c072f90ae1d49)，其中StyleTTS2子模块固定如上，Kokoro子模块为[semidark/kokoro](https://github.com/semidark/kokoro/tree/b96fef95e6a746495f92443fac7c688f90fc57fc)。项目有自己的训练驱动和兼容修复，改动及许可证见[`vendor/sources.json`](../vendor/sources.json)、`vendor/*/LICENSE`和`vendor/kikiri_LICENSE`、`vendor/kikiri_NOTICE`。

## Python环境

本次使用Python 3.10.18、PyTorch/torchaudio 2.7.1+cu128、四张A800。现有`.venv`继承本机LITs环境，不能直接复制到其他机器。新环境可按以下命令准备，再检查平台依赖：

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
.venv/bin/python scripts/download_assets.py
```

Misaki使用`en.G2P(trf=False)`、美式英语、EspeakFallback以及`zh.ZHG2P(version=None)`。中文沿用通用v1.0兼容前端，混读的英文段单独路由；不使用v1.1-zh词表。英文模型`en_core_web_sm`实测为3.8.0。环境版本来自当前已跑通训练环境；本次仓库整理只验证本机资产哈希和导入，没有重建整个训练环境。

## 数据与外部评价依赖

训练原始清单来自本机LITs的MajesticVoice数据，约100小时、中文/英文/混读。数据没有可在此仓库自动下载的公开发行地址。准备新机器时需要提供自己的清单和WAV，并调整路径。`prepare_data.py --source PATH`接受本流程原始清单格式；随后运行`prepare_frontend.py`重新编码，不能复用LITs音素ID。

`training/evaluate.py`与诊断脚本调用本机`/119010446/LITs/training/common/evaluate_checkpoint.py`及LITs的数据质检模块；`scripts/watch_evaluation.py`使用独立ASR和metrics Python环境。它们尚未全部打包到本仓库，克隆仓库后不能把本机绝对路径直接当作可移植安装。

复现评分还需准备Qwen3-ASR-1.7B、WavLM/CAMPPlus、DNSMOS及LITs评价实现、450条固定文本和语言参考。现有评估汇总已随Git提交，可直接查看；重新评分需要先配置这些外部依赖。冻结模型和数据来源的审计文件位于`vendor/sources.json`和`reports/`；历史报告保留当时路径，不表示其他机器也具有该路径。

运行时的训练/评价源码快照与大体积checkpoint留在本机`runs/majestic_v1_20260920`。仓库中的源码用于后续维护；准确重现本轮数值应同时保留运行快照、数据哈希及`docs/evaluation/`中的协议信息。
