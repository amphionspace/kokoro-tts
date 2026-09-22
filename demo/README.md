# 大气女声网页 Demo

网页支持中文、英文及中英混读、自动语言识别、0.7–1.3倍语速、浏览器试听和WAV下载。使用指定Stage 2最终导出模型及直接优化的voice，不依赖LITs评价进程。界面为本地静态HTML/CSS/JavaScript，不加载第三方CDN。

## 当前本机实例（2026-09-22）

地址：`http://服务器IP:32002`。当前加载**20轮长样本加权模型**，默认开启“每句一块”；可关闭开关对照整段生成。页面与实际两句中文合成已验证返回HTTP 200，输出为24kHz单声道WAV、2个音频块。

| 模型 | 导出目录（相对项目根目录） | 用途 |
|---|---|---|
| 旧baseline，Stage 2十轮 | 已删除 | 历史指标保留在文档，不再可加载 |
| 延长至二十轮并提高长样本占比 | `runs/majestic_s2_20ep_long_resume6k_20260921/eval/stage2_final` | 当前网页加载，也是程序默认路径 |
| 完整重跑，Stage 1四轮＋Stage 2十轮 | `runs/majestic_baseline_full_rerun_20260921/eval/stage2_final` | 已完成，当前网页未加载 |

网页不提供模型下拉切换。切换模型需要停止原实例后以不同的`--export-dir`重新启动；不要在同一端口重复启动。`/healthz`中的model字段是通用名称，具体加载路径查本机`runs/web_demo/server.json`。这份文件记录当前后台实例；自行重启后应同步更新记录。

## 启动相同模型

从项目根目录执行。需要安装依赖时，先按[依赖准备](../docs/dependencies.md)配置训练/前端环境：

```bash
.venv/bin/python -m pip install -r demo/requirements.txt
CUDA_VISIBLE_DEVICES=0 .venv/bin/python demo/server.py \
  --host 0.0.0.0 --port 32002 --device cuda:0 \
  --export-dir "$PWD/runs/majestic_s2_20ep_long_resume6k_20260921/eval/stage2_final"
```

上面是前台命令。需要脱离终端后台运行时，改用以下命令（仅在该端口没有已有实例时）：

```bash
mkdir -p runs/web_demo
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python -u demo/server.py \
  --host 0.0.0.0 --port 32002 --device cuda:0 \
  --export-dir "$PWD/runs/majestic_s2_20ep_long_resume6k_20260921/eval/stage2_final" \
  >> runs/web_demo/server.log 2>&1 < /dev/null &
echo $! > runs/web_demo/server.pid
```

检查与日志：

```bash
curl --fail http://127.0.0.1:32002/healthz
tail -n 50 runs/web_demo/server.log
```

访问`http://服务器IP:32002`。默认使用可见的第一张GPU，CUDA不可用时回退CPU；可通过`--device cpu`或`--device cuda:0`显式指定。GPU选择也可用`CUDA_VISIBLE_DEVICES`控制。模型只加载一次，服务单进程运行，不应设置多个Uvicorn workers。

默认模型目录为`runs/majestic_s2_20ep_long_resume6k_20260921/eval/stage2_final`，必须包含：

- `kokoro.pth`：已训练的五组Kokoro推理权重。
- `majestic.pt`：导出的`[510,1,256]`固定大气女声voicepack。

另需仓库中的`models/Kokoro-82M/config.json`。模型权重不上传Git；在其他机器部署需要复制这两个导出文件，不能用官方未微调权重替代后仍声称是本轮大气女声。自定义导出位置：

```bash
.venv/bin/python demo/server.py --host 0.0.0.0 --port 32002 \
  --export-dir /path/to/export --device cuda:0
```

本次本机后台实例的PID与日志记录在`runs/web_demo/server.json`和`runs/web_demo/server.log`。它们属于本机运行状态，不入Git；启动命令本身是前台进程，如需开机自启应交给部署环境的服务管理器。

## 接口与行为

- `GET /healthz`：模型加载完成后返回`status=ready`。
- `POST /api/synthesize`：JSON含`text`、`language`（`auto/zh/en/mixed`）、`speed`和`sentence_chunking`（默认`true`），返回24kHz、单声道、PCM16 WAV。
- 返回头`X-Audio-Duration`为音频秒数，`X-Synthesis-Seconds`为服务器处理秒数，`X-Language`为所选语言。
- 默认开启“每句一块”：按中英文句末标点及换行拆分（中英文逗号均不切），每句最多510个有效音素，不设整篇字符数或总音频时长上限（实际受内存和请求连接限制）。先验证所有句子，再顺序合成并直接拼接，不额外添加静音或裁剪边界。
- 关闭开关或传`sentence_chunking=false`恢复整段模式：最多600字符、510个有效音素和120秒输出。切换不会截断文本。
- 超长单句、不支持字符或无效设置返回422，指出出错段号；不会丢弃内容。常见英文称谓、缩写、小数做了保护，但自动句界无法保证适用于所有缩写，必要时手动换行。
- 响应头`X-Chunk-Count`与`X-Sentence-Chunking`标注实际分块数量和模式。结果区也会显示模式，便于比较。
- 单次只执行一个合成请求；忙时返回429，防止并发推理挤占GPU。结果仅经HTTP返回，服务不主动保存用户输入或生成音频；标准访问日志记录请求路径和状态码。

```bash
curl http://127.0.0.1:32002/api/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好，欢迎来到语音工作室。","language":"auto","speed":1.0}' \
  --output /tmp/kokoro-demo.wav
```

本机验证覆盖中文、英文、混读实际合成，24k/单声道/PCM16输出，以及空文本、文本或音素超限、非法语速和语言设置。网页代码不会下载模型、修改训练产物或写入训练数据。

## 长文本试听方法与结果

旧baseline与当前模型的固定15组长文本比较见[对比报告](../docs/long_text_comparison_20260922.md)。一次性试听目录、音频包及旧baseline导出已按要求删除，JSON与文档结果保留。完整重跑baseline仍有checkpoint和最终导出，可以用于新的对比，但不是历史报告中的同一份旧权重。

测试模型本身的长上下文行为时，保持相同文本、speed=1，关闭“每句一块”。测试实际逐句朗读体验时再开启分句。两种模式应分别比较，不能用分句效果证明模型的长上下文时长偏差已消失。

小样本结论：中文/混读首句缩短略有减轻，音色相似度评分增加，但存在变差样本，尚未解决长文加速问题；英文首句对照只有一组。详细数字与局限以对比报告为准。
