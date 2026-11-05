# aicbili — 让 AI 真正「看」B站视频

> bili_summary 的下一代：从**只听声音**，升级到**连画面一起看懂**，并且更快。
> 全程不下载完整视频，只拉真正需要的那几 MB。

```bash
aicbili BV1xx411c7mD          # 一个链接 → 一份带画面证据的学习笔记
```

---

## 一、它解决什么问题

上一代 `bili_summary` 的链路是：`yt-dlp 下载整段音频 → faster-whisper 转写 → DeepSeek 总结`。
三个硬伤：

| 问题 | 后果 |
| --- | --- |
| 只处理音频 | 屏幕上写的代码、PPT 要点、演示操作、图表数据**全部丢失**；纯操作类视频几乎等于没看 |
| 必须下完整段音频 | 首次等待时间长，磁盘要留空间 |
| 转写是串行瓶颈 | 16 分钟视频 CPU 上要几分钟，而且这段时间什么都干不了 |

`aicbili` 换了一条链路：

```
                        ┌── 语音腿 ──────────────────────────────┐
解析（1s）──► 拿流地址 ─┤  B站字幕? ──有──► 1 秒拿到全文（跳过 ASR）│
                        │       └──无──► 抽音频(并发分窗) → ASR    │──► 对齐 ──► 总结
                        └── 画面腿 ──────────────────────────────┤
                           B站快照(≈1MB) → 场景变化检测 → 只抽变化处的高清帧
                                                    → 拼图 → 视觉模型
```

**两条腿完全并行**：画面腿通常十几秒跑完，语音腿结束时立刻对齐总结。

---

## 二、怎么做到「不下载视频也能看懂画面」

### 关键发现：B站 CDN 强制校验 Referer

实测（`curl -H "Range: bytes=0-63"`）：

| 资源 | 无 Referer | 带 `Referer: https://www.bilibili.com` |
| --- | --- | --- |
| 视频流 `*.bilivideo.com` | ❌ **403** | ✅ 206 |
| 音频流 `*.mcdn.bilivideo.cn` | ✅ 206 | ✅ 206 |
| 快照雪碧图 `bimp.hdslb.com` | ✅ 200 | ✅ 200 |

这解释了两件事：

1. **为什么不能简单把 B站链接丢给云端模型**——模型侧拉流会 403。
   （只有约 1/4 的视频能拿到匿名可访问的 `mcdn` 边缘节点地址，见下文 `--mode native`。）
2. **为什么必须自己抽帧**——本地 ffmpeg 可以注入 Referer，云端模型不行。

所以默认方案是：**本机只做「精准取样」，字节量压到最低**。

### 三重提速，都是实测出来的

**① 用 B站自己的「视频快照」做场景检测，而不是顺序读整条流**

B站给每条视频预生成了雪碧图：一张 10×10、每格 480×270，整片近 200 帧只要 **~1MB**。

| 做法 | 16 分钟视频实测 |
| --- | --- |
| `ffmpeg -vf select='gt(scene,0.3)'` 顺序读整条流 | **72.6 s**（还要读完整个文件） |
| 下载快照 → 算帧间差异 → 只取变化点 | **2.5 s 下载 + 0.8 s 计算** |

**② 高清帧用输入侧 seek 抽，只付目标位置附近的字节**

`ffmpeg -ss <t> -i <url> -frames:v 1` 走 HTTP Range，单帧 20~40KB。

| 做法 | 实测 |
| --- | --- |
| 8 帧并发 seek | **4.8 s**，共 130 KB |
| 24 帧并发 seek（16 分钟视频） | **11.7 s**，共 769 KB |

**③ 音频分窗并发拉取 + 优先使用 B站字幕**

单条 HTTP 连接到 B站 CDN 常常只有几十 KB/s，4~6 个时间窗并发基本能线性提速；
而配了 `SESSDATA` 后能用 B站 AI 字幕，**整段转写直接跳过**。

---

## 三、快速开始

**不需要 `pip install`。** 本项目除 `Pillow`（拼图用）外零第三方依赖，HTTP 全用标准库，
而主流发行版的系统 Python 基本都自带 Pillow。所以：

```bash
git clone <this-repo> && cd aicbili
python3 -c "import PIL; print('Pillow 已就绪')"   # 先确认一下

cp .env.example .env      # 填 API Key 与 SESSDATA
make doctor               # 环境自检
make run U='BV1xx411c7mD' # 分析一个视频
```

想在任何目录下敲 `aicbili`，链接一下即可（零安装、零网络）：

```bash
make link
aicbili --version
```

> 如果你的系统 Python 没有 Pillow：
> `sudo pacman -S python-pillow`（Arch）/ `sudo apt install python3-pil`（Debian/Ubuntu）/
> `sudo dnf install python3-pillow`（Fedora），或见下面的 venv 方案。

<details>
<summary>为什么不建议 <code>pip install -e .</code>（PEP 668）</summary>

Arch / Fedora / Debian 12+ 的系统 Python 受 PEP 668 保护，直接 pip 会报
`error: externally-managed-environment`。而且很多发行版已不再默认安装 `setuptools`，
连 `pip install -e . --no-build-isolation` 也会因缺构建后端失败。

所以本项目**刻意不依赖安装**：`bin/aicbili` 只是个 shell 启动器，
把项目根目录加进 `PYTHONPATH` 后调用 `python3 -m aicbili`。

如果你确实想要一个「正经」安装：

```bash
# 方案 A：pipx（独立环境，不污染系统）
pipx install --editable .

# 方案 B：虚拟环境（需要能访问 PyPI 装 Pillow）
python3 -m venv --system-site-packages .venv   # --system-site-packages 复用系统 Pillow，免下载
.venv/bin/pip install -e . --no-deps           # 有 setuptools 时才可行
# 或者干脆不做 editable 安装，直接：
AICBILI_PYTHON=$PWD/.venv/bin/python ./bin/aicbili --version
```
</details>

需要 **ffmpeg**（抽帧/抽音频用）：

```bash
sudo pacman -S ffmpeg       # Arch
sudo apt install ffmpeg     # Debian/Ubuntu
brew install ffmpeg         # macOS
winget install Gyan.FFmpeg  # Windows
```

### 最小配置（只要两样东西）

```ini
# ① 一把多模态模型的 Key —— DeepSeek 最省事，一把 key 同时看图 + 写总结
AICBILI_PROVIDER=deepseek
AICBILI_VLM_BASE_URL=https://api.deepseek.com/v1
AICBILI_VLM_MODEL=deepseek-flash
AICBILI_VLM_API_KEY=sk-你的key

# ② B站 SESSDATA —— 免费，且能跳过整个语音转写
AICBILI_SESSDATA=你浏览器里的 SESSDATA
```

`--provider` 预设：`deepseek` / `dashscope` / `ark` / `zhipu` / `siliconflow` / `openai` / `ollama`。
填完先跑 `aicbili doctor` —— 它会真的发一张图给模型，确认「这个 Key 到底能不能看图」。

### 成本到底多少

画面理解按图片计费，各家都会把图片压到固定像素预算（以 DeepSeek 为例）：
**单图上限 1024 tokens**，小于 ~544×544 的会被放大，更大的会被压到约 **1300×1300 像素**。

所以「几帧拼一张图」直接决定成本：

| 送法 | 图片数 | 图片 tokens | 每格实际分辨率 | 说明 |
| --- | --- | --- | --- | --- |
| 24 帧各发一张 | 24 | ~24 000 | 852×480 | 最贵，4 倍浪费 |
| 24 帧 → 6 张（4 帧/张） | 6 | ~6 100 | 640×387 | 稳 |
| **24 帧 → 4 张（6 帧/张，默认）** | 4 | **~4 100** | **640×387** | **甜点：格数变多但每格不变，纯省** |
| 24 帧 → 3 张（9 帧/张） | 3 | ~3 100 | 557×336 | 更省，但被降采样，小字开始糊 |

实测拼图尺寸（640px 宽单元格、16:9 源）：

```
 4 帧/张 → 1280×774  = 0.99MP ✅ 不降采样   每格 ≈ 640×387
 6 帧/张 → 1920×774  = 1.49MP ✅ 不降采样   每格 ≈ 640×387   ← 默认
 9 帧/张 → 1920×1161 = 2.23MP ⚠ 降采样×0.87 每格 ≈ 557×336
```

关键点：**6 帧/张在像素预算内，但每格分辨率与 4 帧/张完全相同** ——
所以它是纯赚的（同样的画面清晰度，少 1/3 的图片和 token）。

一个视频通常**几分钱到一毛钱**。真正的省钱大头其实是 **`SESSDATA`**：
有了 B站 AI 字幕，整个 ASR 环节直接消失（既不花钱也不花时间）。

---

## 四、四种模式

| 模式 | 做什么 | 下载量 | 适用 |
| --- | --- | --- | --- |
| `auto`（默认） | 字幕/ASR + 快照场景检测 + 抽帧 → 视觉模型 → 对齐总结 | ≈1~2 MB | 绝大多数视频 |
| `frames` | 同 auto，但强制抽帧 | ≈1~2 MB | 想统一行为时 |
| `native` | 把远端视频地址直接交给模型，模型自己看画面+听声音 | **0**（模型侧拉） | 见下方限制 |
| `audio` | 只转写不抽帧（等价于旧版能力） | ≈几 MB 音频 | 纯访谈/播客 |

### 完全本地 / 零 API 成本的可能性

| 环节 | 免费的本地方案 | 本机可行性（8 核 CPU、无独显、约 1.4GB 可用内存） |
| --- | --- | --- |
| 文稿 | **B站 AI 字幕**（配 SESSDATA） | ✅ 首选，免费且 <1s |
| 文稿 | **whisper.cpp**（`pacman -S whisper-cpp`） | ✅ 可行，base 模型约 2~5× 实时 |
| 画面 | 本地 VLM（llama.cpp `llama-mtmd-cli` + Qwen2.5-VL-3B） | ⚠️ 内存吃紧，CPU 推理约 5~15 分钟/视频 |
| 总结 | 本地文本模型 | ⚠️ 同上 |

结论：**文稿可以完全本地化**（尤其是有字幕时），但**画面理解不建议本地跑** ——
一台没有独显、可用内存 1.4GB 的机器跑 2B 视觉模型，会比云端慢一两个数量级，
而且 `deepseek-flash` 这类模型的图片计价通常一个视频只要几分钱。

### `--mode native` 到底是什么，有什么限制

这是「AI 直接读视频」最彻底的做法：请求体里放一个真正的 `video_url`，
字节由模型服务商去拉，本机**一个字节都不下**。

```json
{"type": "video_url", "video_url": {"url": "https://.../xxx.m4s?..."}}
```

限制很现实（实测 8 个视频，只有 2 个成功）：

- B站主 CDN 校验 Referer → 403，固定失败；
- 少数 `mcdn.bilivideo.cn` 边缘节点不校验，可以直接用；
- 想稳定使用，需要一个**会补 Referer 的反向代理**（Cloudflare Worker 之类，十几行），
  然后用 `--relay https://你的代理/` 指过去。

```
# 代理伪代码（CF Worker）
const up = await fetch(new URL(decodeURIComponent(pathname.slice(1))), {
  headers: { Referer: "https://www.bilibili.com", "User-Agent": "Mozilla/5.0" },
});
return new Response(up.body, { headers: { "Accept-Ranges": "bytes",
                                          "Content-Type": "video/mp4" } });
```

**建议**：日常用 `auto`（稳定、可解释、有逐帧画面记录）；`native` 当地址能匿名访问时再开，
它会自动降级回抽帧，不会失败。

---

## 五、输出长什么样

### 两种体裁

| 体裁 | 做法 | 结果 | 适合 |
| --- | --- | --- | --- |
| `--style article`（**默认**） | **等价重写**：先分章，再逐章把该段转录全部改写成书面正文 | 保留 90%+ 信息，可当教材通读 | 想「只读文章就等于上完课」 |
| `--style summary` | 压缩成要点 | 只保留 30%~50%，长视频更低 | 只想快速知道讲了啥 |

> 实测对比（18 个视频合计）：摘要式只保留 **35%** 的内容，最长的那份 40905 字被压成 3797 字（9%）。
> 论文式平均保留 **200%+**（口语碎片书面化后自然变长），且不添加原文没有的内容。

### `.md` — 论文式学习文章

```markdown
## 摘要                     ← 300~500 字 + 关键词
## 引言                     ← 400~800 字：背景、核心问题、全文结构
## 一、<章节标题>            ← 按主题划分（非等长），每章是一段完整书面正文
### 1.1 <小节>              ← 原文有示例/步骤时可再分小节
## 二、<章节标题>
...
## 结论                     ← 收拢被论证成立的结论 + 讲者的个人判断 + 可操作做法
## 术语与概念                ← 8~20 条，取自原文语境
## 附：章节时间索引          ← 每章对应视频的起止时间，方便回看
```

写作要求（写在 prompt 里的硬约束）：

- **不得丢失任何信息**——论点、推理、例子、类比、数字、公式、代码、命令、注意事项全保留
- 去掉口语冗余（嗯/啊/就是说/重复/寒暄），转书面学术语体
- 例子要写完整：不只说「举了个例子」，把例子内容和它要说明的道理都写出来
- 画面上的代码、命令、报错、界面文字**必须写进正文**
- 篇幅 0.8~1.5 倍原文，**不许注水，也不许省略**
- **不得添加原文没有的内容**

### `.txt` — 原始材料（md 里不重复）

```
【值得回看的片段】 / 【画面记录（逐帧）】 / 【多模态时间轴】 / 【完整文稿】
```

### `.srt` — 标准字幕

配 VLC / PotPlayer 直接对照原视频看。

### 想要更保真：`--verify`

逐章把成稿和原始转录对照，让模型列出**真正遗漏**的信息点并补写回去：

```bash
aicbili <url> --verify
```

调用次数约翻倍。重要资料建议开，日常可以不开。

### 旧报告迁移

改版前的报告可以用一条命令重排（**不调用任何 AI，不花钱**）：

```bash
aicbili reformat              # 处理 ./output 下所有 .md
aicbili reformat 某个目录/
```

## 六、常用命令

```bash
aicbili BV1xx411c7mD                    # 单个
aicbili urls.txt --jobs 3               # 批量（每行一个链接/BV号）
aicbili <url> --mode audio              # 只要转写
aicbili <url> --mode native             # 模型直读（能匿名访问时）
aicbili <url> --frames 40 --frame-width 1600   # 更细的画面采样
aicbili <url> --style summary           # 退回摘要式（短，但会丢信息）
aicbili <url> --verify                  # 逐章自检遗漏并补写（更保真）
aicbili <url> --no-ai                   # 只抽帧+转写，不调 AI（排错/离线）
aicbili <url> --dry-run                 # 只解析元信息，秒级验证
aicbili <url> --json                    # 机器可读结果
aicbili doctor                          # 环境自检（ffmpeg/API/字幕/快照）
aicbili config                          # 看当前生效配置
aicbili cache                           # 看缓存占用
aicbili reformat                        # 把旧格式报告重排成 md+txt（不调 AI）
```

### 调优速查

| 想要 | 调整 |
| --- | --- |
| 更快 | 配 `AICBILI_SESSDATA`（走字幕跳过 ASR）；降 `MAX_FRAMES`；降 `FRAME_WIDTH` |
| 更准 | 升 `MAX_FRAMES`；`SCENE_THRESHOLD` 调低（0.08）更敏感 |
| 更省 token | `SHEET_GRID=6`（默认，已是最优）；再省才用 9；或降 `FRAME_WIDTH` |
| 纯 PPT 课程 | `SCENE_THRESHOLD=0.08`，`SHEET_GRID=6` |
| 实拍 vlog | `SCENE_THRESHOLD=0.2`，`MAX_FRAMES=16` |

---

## 七、性能实测参考

本机（无 GPU、中国大陆网络，B站 CDN 速率波动很大）：

| 场景 | 耗时 | 下载量 | 说明 |
| --- | --- | --- | --- |
| 3.5 分钟 MV，`auto` | **10.9 ~ 15.5 s** | 1.3 MB 音频 + 130 KB 帧 | 8 帧 / 2 张拼图 / 2 次视觉调用 |
| 3.5 分钟 MV，`native` | **6.1 s** | **0** | 1 次调用，命中 mcdn 匿名地址 |
| 16 分钟课程，画面腿 | **11.7 s** | 985 KB 快照 + 769 KB 帧 | 24 帧；语音腿并行不受影响 |
| 16 分钟课程，音频腿 | **8.7 s**（好节点） | 1.25 MB | 4 窗口并发，约 24~81× 实时；坏节点会自动换 |
| （对照组）顺序读完整条流 | 72.6 s | 8.3 MB | 已弃用的做法 |

> 结论：`auto` 模式的总耗时 ≈ **max(语音腿, 画面腿)**，而不是两者相加；
> 而在能拿到 B站字幕的视频上，语音腿几乎归零。
>
> ⚠️ 波动主要来自 B站 CDN：同一台机器上，音频腿可能 3 秒也可能 60 秒。
> 所以内置了 CDN 测速、分窗并发、卡死自动换节点；画面腿只拉几百 KB，波动小得多。
> 同一条 3.5 分钟 MV 连续跑，`auto` 实测落在 **10.9 s ~ 28.9 s** 之间，属正常范围。

---

## 八、目录结构

```
aicbili/
├── bili.py       B站 API：解析/playurl(DASH)/快照/字幕/WBI 签名
├── media.py      ffmpeg 远端操作：抽帧、分窗抽音频、CDN 测速、匿名可达性探测
├── frames.py     雪碧图切分、场景变化检测、关键帧挑选、拼图
├── asr.py        三种转写后端（B站字幕 / 云端 / 本地 faster-whisper）
├── vision.py     视觉模型调用（拼图 image_url / 原生 video_url）+ 结构化解析
├── summarize.py  多模态时间轴对齐 + map-reduce 总结
├── pipeline.py   编排（两条腿并行、降级、缓存、断点复用）
├── config.py     .env / 环境变量 / 覆盖项
├── http.py       标准库 HTTP（零依赖）
└── cli.py        命令行
tests/
├── test_core.py  45 个单元测试（标准库 unittest，无需 pytest）
└── mock_vlm.py   假的 OpenAI 兼容服务，用来免 Key 端到端验证
```

跑测试：

```bash
python3 -m unittest tests.test_core -v
```

---

## 九、排错

| 现象 | 原因与处理 |
| --- | --- |
| `externally-managed-environment` | PEP 668。别用 pip，直接 `make link` 或 `python3 -m aicbili` |
| `aicbili: command not found` | 没链接。在项目目录 `make link`，或直接用 `python3 -m aicbili` |
| `403 Forbidden` 抽帧失败 | ffmpeg 没带上 Referer。用本项目（已内置注入），别手写 curl |
| `412` 风控 | 请求太密。配 `SESSDATA`，或降低 `--concurrency` |
| 只能看 480P | 没登录。配 `AICBILI_SESSDATA` |
| 「该视频没有快照」 | 该视频没有预生成雪碧图，会自动退化为均匀采样，正常 |
| 画面记录全是空的 | 视觉模型没配好，`aicbili doctor` 看 `AICBILI_VLM_*` |
| 视觉模型报 404/model not found | 模型名写错；会尝试 `AICBILI_VLM_FALLBACK_MODEL` |
| 转写很慢 | 用 `AICBILI_SESSDATA` 走字幕；或降 `AICBILI_WHISPER_MODEL`；或换云端 ASR |
| 本地转写装不上 | Python 3.13+ 可能没有 `ctranslate2` 轮子，用云端 ASR 或 3.11/3.12 |
| 音频拉取很慢 | 调大 `AICBILI_AUDIO_WINDOWS`（会做 CDN 测速挑最快节点） |

---

## 十、从 bili_summary 迁移

输入方式和旧版完全一致（BV 号 / 完整链接 / 分享文案 / 含链接的 txt），参数是超集：

| 旧版 `bili_summary` | 新版 `aicbili` |
| --- | --- |
| `python bili_summary.py <url>` | `aicbili <url>` |
| `--no-summary` | `--mode audio`（只转写）或 `--no-ai`（连视觉也关掉） |
| `--with-timestamps` | 自动产出 `.srt`（有逐句时间戳时） |
| `--skip-existing` / 断点续传 | 默认开启（`.aicbili_cache/<video>/meta.json` 记录状态，`--force` 重做） |
| `BILI_PARALLEL=N` | `--jobs N`（多视频）+ `--concurrency N`（单视频内并发） |
| `WHISPER_SIZE` | `AICBILI_WHISPER_MODEL` |
| `HF_ENDPOINT` 镜像 | 照旧可用（faster-whisper 会读这个环境变量） |
| 交互式粘贴链接 | 支持（不带参数且是终端时） |
| `--setup` 向导 / 注册全局命令 | **未移植**（直接改 `.env` 即可） |

旧版源码保留在 `legacy/`。

---

## 十一、License

MIT
