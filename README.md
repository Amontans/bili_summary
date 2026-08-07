# bili_summary — B站视频一键转文字稿（可选 AI 总结）

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)

输入 B站链接或 BV 号，自动完成 **下载音频 → Whisper 中文转写**，可选 **DeepSeek AI 总结**。
输出为**一个文件夹里的一堆纯文本文件**（每个视频一个 `.txt`），可直接交给 pi agent 等工具做后续处理。

支持批量处理、交互式输入、失败隔离（单个失败不影响其余）、首次运行自动配置环境。

---

## ✨ 特性

- ✅ **三种输入方式**：命令行参数 / 文件列表（`-f`）/ 交互模式（运行后逐行输入，空行开始）
- ✅ **一键配置**：`python bili_summary.py --setup` 像安装向导一样完成环境检查、API Key、镜像、模型预下载
- ✅ **自动清理**：交互生成的 `links.txt`、运行产生的 `__pycache__` 退出时自动删除（`--keep-links` 可保留）
- ✅ **扁平纯文本输出**：`output/<视频标题>.txt`，文件名自动取视频标题，方便喂给 pi 等 AI 工具
- ✅ **AI 总结可选**：`--no-summary` 只转写、不需要 API Key
- ✅ **零配置自举**：首次运行自动创建 `.venv` 并安装依赖，**下载即用**
- ✅ **可 pip 安装**：`pip install .` 或 `pip install git+...` 后获得 `bili-summary` 命令
- ✅ **跨平台**：Linux / macOS / Windows
- ✅ **交互式设置中心**：`--config` 像 pi agent 一样菜单式查看/修改全部配置（即时写入 `.env`，下次运行生效）；`--show-config` 只读查看当前生效值与来源
- ✅ **GPU 自动加速**：检测到 NVIDIA 显卡自动启用 CUDA+float16，无需手动配置；想关掉在 `.env` 设 `WHISPER_DEVICE=cpu` 即可
- ✅ **并行默认自动**：按 CPU 核数自动开启（上限 4 路，GPU 默认单路）；`-p 1` 或 `BILI_PARALLEL=1` 可手动关闭
- ✅ **并行流水线**：下载预取与转写重叠，不再“下完一个才下下一个”；`-p N` 多路并行转写（CPU 线程自动分摊）
- ✅ **断点续传**：`--skip-existing` 已转写过的视频直接跳过（只请求一次元数据，不重复下载），批量中断后重跑不浪费
- ✅ **字幕输出**：`--with-timestamps` 额外生成带时间戳的 `.srt` 字幕（与纯文本 `.txt` 并存）
- ✅ **模型自由定制**：`WHISPER_SIZE` 任选 tiny/base/small/medium/large-v3 **或本地模型目录路径**；`HF_HOME` 自定义模型缓存位置（`--setup` 向导可一键设置）
- ✅ 中文优先识别，失败自动切自动语种（兼容外语/音乐视频）
- ✅ huggingface.co 不可达时自动切换 hf-mirror.com 镜像

---

## 🚀 安装（任选一种）

### 方式 A：直接下载运行（推荐，零依赖）

```bash
git clone https://github.com/Amontans/bili_summary.git
cd bili_summary
python bili_summary.py --setup     # 一键配置：API Key、镜像地址、模型预下载
python bili_summary.py BV1GJ411x7h7 --no-summary
```

首次运行也会自动创建 `.venv`、安装依赖、下载 Whisper 模型（约 460MB）。
只需系统装有 Python 3.8+（ffmpeg 可选，缺省时自动用内置静态版）。

### 方式 B：pip 安装

```bash
pip install git+https://github.com/Amontans/bili_summary.git
# 之后直接使用命令（不再需要 python + 文件名）：
bili-summary BV1GJ411x7h7 --no-summary
```

### 方式 B2：本地注册全局命令（不污染系统，推荐；仅 Linux/macOS）

> **Windows 用户请跳过本节**：直接 `python bili_summary.py ...` 运行即可；或用 `pip install .` 安装后，命令位于 `.venv\Scripts\bili-summary.exe`（Windows 的 `pip install .` 与 Linux 用法完全一致）。

克隆后在项目目录执行两步，即可直接敲 `bili-summary`：

```bash
# 1. 以 editable 方式装进项目自己的 .venv（不复制代码、不装到系统）
python -m pip install -e . --no-deps
# 2. 在用户 bin 目录建软链接（~/.local/bin 需在 PATH 中）
ln -sf "$PWD/.venv/bin/bili-summary" ~/.local/bin/bili-summary

# 之后直接使用：
bili-summary --version
bili-summary BV1GJ411x7h7
bili-summary -i
```

**原理**：`pyproject.toml` 的 `[project.scripts]` 让 pip 生成一个极薄的启动器（shebang 指向 venv python + `from bili_summary import main; main()`），放到 PATH 里的 bin 目录后，系统就把 `python bili_summary.py xxx` 包装成了 `bili-summary xxx`。

撤销：`rm ~/.local/bin/bili-summary && python -m pip uninstall bili-summary`（editable 安装不产生额外拷贝，删链接即恢复原状）。

### 方式 C：下载 ZIP

GitHub 页面 → Code → Download ZIP → 解压后按方式 A 运行。

### Windows 零基础快速开始（双击即可）

1. 安装 [Python 3.9+](https://www.python.org/downloads/)：安装时**务必勾选 “Add python.exe to PATH”**；完成后 cmd 里 `python --version` 有输出即成功
2. 安装 [VC++ 运行库 (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe)（faster-whisper 依赖，缺了会报“无法加载 ctranslate2”或缺 DLL）
3. 下载本项目：`git clone ...`，或 GitHub 页面 → Code → Download ZIP 后解压
4. **双击 `start.bat`**（已处理好中文乱码、自动检测 Python），首次运行自动创建 `.venv` 并安装依赖（几百 MB，请耐心等待）
5. 运行 `python bili_summary.py --setup` 一键配置（只转写可跳过 API Key）→ 之后输入链接即可

> 不用启动器也行：cmd 里 `cd` 进项目目录后，所有 `python bili_summary.py ...` 命令照常可用。

---

## 🎛 一键配置：python bili_summary.py --setup

像安装向导一样交互式完成配置（**默认只动本项目目录与用户缓存；注册全局命令时会先征得你同意**）：

```text
$ python bili_summary.py --setup

========================================================
bili_summary 一键配置向导（默认只动本项目目录；注册全局命令时会经你确认）
========================================================
✅ Python 3.11
✅ ffmpeg: /usr/bin/ffmpeg
✅ 依赖齐全 (yt-dlp / faster-whisper / openai / imageio-ffmpeg)

🧠 Whisper 转写模型（核心大文件，约数百 MB）
   [1] 在线下载：自动从 HuggingFace/镜像拉取（规格与存放位置下面两步选）
   [2] 使用本地已安装模型：填目录路径直接调用，不下载
   请选择 [1/2]，回车=在线下载: 1
   在线下载规格: tiny / base / small / medium / large-v3（越大越准、越慢、越吃内存）
   当前 [small]，回车保留，或输入新规格: 

📂 模型缓存位置（在线下载的模型存放目录）
   [1] 用户缓存（默认）: ~/.cache/huggingface —— 跨项目复用
   [2] 项目内: models/ —— 随项目走，拷项目即带走模型
   [3] 自定义目录
   选择 [1/2/3]，回车保留当前: 1

🔑 DeepSeek API Key（仅“转写+AI总结”模式需要；只转写可回车跳过）
   请输入 API Key（回车跳过）: sk-xxxx
✅ DEEPSEEK_API_KEY 已写入 /path/bili_summary/.env
✅ huggingface.co 不可达，已写入 HF_ENDPOINT=https://hf-mirror.com
✅ 配置文件: /path/bili_summary/.env（权限 600，仅本用户可读）

⬇️ 预下载 Whisper small 模型（约 460MB；不下载则首次转写时自动下载）
   现在下载吗？[y/N] y
✅ Whisper small 模型就绪（存放于 ~/.cache/huggingface）

⌨️ 注册全局命令 bili-summary（之后直接敲 bili-summary，不再需要 python bili_summary.py）
   现在注册吗？[y/N] y
✅ 完成！新开终端直接输入 bili-summary 即可

✅ 配置完成！
========================================================
```

**它会做什么 / 不会做什么**：

| ✅ 只做（安全） | ❌ 绝不做（除非你明确确认） |
|---|---|
| 项目目录内创建 `.venv`（依赖自动安装） | 不修改 `~/.bashrc` / 系统环境变量 |
| 项目目录内写 `.env` 配置文件（权限 600） | 不往项目外写任何文件 |
| 模型下载到用户缓存 `~/.cache/huggingface`（可用 `HF_HOME` 自定义目录，可复用、可删） | 不强制安装任何系统软件 |
| 自定义 Whisper 模型规格（tiny~large-v3 或本地路径）与模型缓存目录 | 不需要你输入 sudo |
| 环境检查（`--setup --check` 只读体检） | 不需要你输入 sudo |
| 注册 `bili-summary` 全局命令（向导末尾可选）：Linux/macOS 写 `~/.local/bin` 软链接，Windows 写用户 PATH + 项目目录内 `bili-summary.bat`——**会先征得你同意**，且不碰系统级设置 | — |

> 说明：venv 与依赖在首次运行脚本时也会自动完成，`--setup` 额外处理 API Key、镜像地址、模型预下载三项。

---

## 📥 快速开始：交互模式

**运行后默认进入交互模式**，逐行输入链接，全部输完后**直接回车（空行）**开始处理：

```text
$ python bili_summary.py
[bili_summary] 📥 交互模式：请逐行输入 B站链接或BV号，全部输完后直接回车（空行）开始处理
   （随时按 Ctrl+C 可取消）
> BV1GJ411x7h7
> https://www.bilibili.com/video/BV1xx411c7mD?p=2
> [回车]
[bili_summary] 💾 已保存 2 个链接到 /path/links.txt
[bili_summary] 共 2 个视频待处理，模式: 转写 + AI 总结，输出目录: /path/output
...
```

- 输入的内容会写入 `links.txt`（含生成时间注释），之后可直接 `-f links.txt` 复用
- 显式使用 `-i` 也可强制进入交互模式（例如与 `-f` 混用时）
- 没有输入直接回车 → 打印帮助并退出

---

## ⚙️ 用法

```text
用法: python bili_summary.py [inputs...] [-i] [-f FILE] [-o OUTDIR] [-p N] [--no-summary] [--skip-existing] [--with-timestamps] [--keep-links] [--dry-run] [--setup] [--check]

  inputs            B站链接或BV号，可多个（无任何输入时自动进入交互模式）
  -i, --interactive 强制进入交互模式：逐行输入链接，空行开始处理
  -f, --file FILE   从文件读取链接列表（每行一个，# 开头为注释）
  -o, --outdir DIR  输出目录（默认 ./output）
  -p, --parallel N  并行路数（默认自动：CPU 按核数上限4，GPU 单路；N>1 多路同时转写；手动关闭用 -p 1，配置键 BILI_PARALLEL）
  --no-summary      只转写，不调用 DeepSeek 总结（无需 API Key）
  --skip-existing   断点续传：已有同名转写文件则直接跳过（只请求一次元数据，不下载）
  --with-timestamps 额外生成带时间戳的 .srt 字幕（与纯文本 .txt 并存）
  --keep-links      交互模式结束后保留 links.txt（默认自动删除）
  --dry-run         只预览：显示归一化 URL、抓取视频标题与输出文件名（不下载/不转写/不调AI）
  --setup           一键配置向导（API Key / 模型镜像 / 模型预下载）
  --config          交互式设置中心：菜单式查看/修改所有配置（即时写入 .env）
  --show-config     只读显示当前生效的配置（含来源: 环境变量/.env/默认）
  --check           只读环境检查（配合 --setup 使用）
  -V, --version     显示版本
```

### 预览（--dry-run）：顺带抓标题，提前看输出文件名

不下载音频、不转写、不调 AI，但会请求一次元数据拿到**视频标题**，直接预览将来的文件名：

```text
$ python bili_summary.py --dry-run BV1GJ411x7h7
链接解析预览（文件名取视频标题；抓不到标题时回退视频ID）
  https://www.bilibili.com/video/BV1GJ411x7h7
     标题: 【官方 MV】Never Gonna Give You Up - Rick Astley
     文件: 【官方 MV】Never Gonna Give You Up - Rick Astley.txt
```

视频已失效/无法抓取标题时给出警告并回退为视频 ID 命名。

### 支持的输入形式

| 形式 | 示例 | 处理方式 |
|---|---|---|
| 完整链接 | `https://www.bilibili.com/video/BV1GJ411x7h7` | 提取视频ID，重建为干净链接 |
| 带参数链接 | `https://www.bilibili.com/video/BV1xx411c7mD?p=2&spm_id_from=333.999` | 自动提取视频ID；仅保留 `?p=` 分P参数，跟踪参数（spm_id_from 等）自动丢弃 |
| 稍后再看/分享链接 | `https://www.bilibili.com/list/watchlater/?bvid=BV1tPMy6oEtG&oid=...` | 从 `bvid=` 查询参数提取视频ID（否则会被 yt-dlp 误判为列表） |
| 移动端链接 | `https://m.bilibili.com/video/BV1GJ411x7h7` | 提取视频ID，统一转为 www 链接 |
| b23 短链接 | `https://b23.tv/xxxxxx` | 无内嵌ID，原样传给 yt-dlp 自动解析跳转 |
| BV 号 | `BV1GJ411x7h7` | 自动补全为完整链接 |
| av 号（旧格式） | `av170001` | 自动补全为完整链接 |

> 含 `& ?` 等特殊字符的 URL 传参时**必须加引号**。所有输入按视频 ID 去重保序。

### 输出结构：一个文件夹 + 一堆文本文件

```text
output/                                ← -o 指定（默认 ./output）
├── 视频标题.txt                       ← ① 文字稿（文件名=视频标题，--no-summary 模式只有这类文件）
├── 视频标题.srt                       ← ①' 带时间戳字幕（仅 --with-timestamps）
├── 视频标题.summary.txt               ← ② AI 总结（仅开启总结时）
├── 另一个视频标题.txt
├── summaries.md                       ← ③ 全量汇总（仅开启总结时）
└── report.md                          ← ④ 处理报告（成功/跳过/失败，始终生成）
```

**文件名 = 视频标题**：自动去非法字符、压缩空白、截断 80 字；同批重名自动加 `_2`/`_3`；抓不到标题时回退为视频 ID。重复运行同一视频会**覆盖**同名文件，不同视频互不干扰。

### 配合 pi agent 使用（推荐工作流）

```bash
# 1. 批量转写（不需要 API Key）
python bili_summary.py -f links.txt --no-summary -o transcripts

# 2. 把 transcripts/ 下的 txt 交给 pi 处理
#    例如在 pi 中: 请阅读 transcripts/【官方 MV】Never Gonna Give You Up - Rick Astley.txt，提取要点并总结
```

### 自动清理（目录不乱）

运行结束后自动删除：
- 交互模式生成的 `links.txt`（需保留时加 `--keep-links`）
- 运行时产生的 `__pycache__`

输出目录 `output/` 里的转写/总结文件是产物，按你的 `-o` 指定存放，不会被误删。

### 交互式设置中心（--config，推荐日常改设置用）

像 pi agent 一样边看边改，所有修改即时写入 `.env`，下次运行生效：

```text
$ python bili_summary.py --config

⚙️  bili_summary 设置中心（修改即时写入 .env，下次运行生效）
    优先级: 命令行参数 > 真实环境变量 > .env > 默认

当前配置：
   1. DeepSeek API Key    sk-ca***180c  [环境变量]
   2. 总结主模型           deepseek-chat
   3. 总结备用模型          deepseek-reasoner
   4. 转写模型规格          small
   5. 推理设备             自动(cuda)          ← GPU 自动加速
   6. 量化类型             float16
   7. 并行转写路数          auto（→1路）          ← 并行默认自动
   8. 生成 .srt 字幕       关
   9. 断点续传             关
  10. 只转写不调AI          关
  11. 模型镜像地址          https://hf-mirror.com  [.env]
  12. 模型缓存位置          ~/.cache/huggingface
  13. 默认输出目录          ./output
  ----------------------------------------------
  14. 💾 完成（退出）

  输入编号修改，回车重显，q 退出: 4
```

- 每个条目标注来源（`[环境变量]` / `[.env]` / 默认），改了当场生效并落盘
- 想看当前配置而不改动：`python bili_summary.py --show-config`
- `--setup` 是首次初始化向导（含模型预下载），`--config` 是日常改设置的菜单，两者互补

### 环境变量与 .env 配置

配置优先级：**命令行参数 > 真实环境变量 > 项目目录 `.env` 文件 > 代码默认值**。

> 大多数运行参数（并行、断点续传、字幕、只转写）都可以不敲命令行，直接在 `.env` 里改：`BILI_PARALLEL` / `BILI_SKIP_EXISTING` / `BILI_WITH_TIMESTAMPS` / `BILI_NO_SUMMARY`。命令行显式给出时优先于文件。
>
> 硬件加速默认全自动：有 NVIDIA GPU 自动用 `cuda+float16`，否则 `cpu+int8`；`.env` 设 `WHISPER_DEVICE=cpu` 即关闭 GPU。
`.env` 由 `python bili_summary.py --setup` 生成，也可手写（每行 `KEY=VALUE`，`#` 注释，权限建议 600）。

| 变量 | 说明 | 默认 |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API Key，**仅总结模式需要** | - |
| `DEEPSEEK_BASE_URL` | DeepSeek API 地址 | `https://api.deepseek.com/v1` |
| `DEEPSEEK_MODEL` | 总结主模型 | `deepseek-chat` |
| `DEEPSEEK_MODEL_FALLBACK` | 主模型失败时的备用模型 | `deepseek-reasoner` |
| `WHISPER_SIZE` | Whisper 模型规格：`tiny`/`base`/`small`/`medium`/`large-v3`，或本地模型目录绝对路径（跳过下载直接加载） | `small` |
| `WHISPER_DEVICE` | 推理设备（留空=自动探测：有 NVIDIA GPU 用 cuda，否则 cpu） | 自动 |
| `WHISPER_COMPUTE_TYPE` | 量化类型（GPU 自动用 float16，CPU 用 int8） | 自动 |
| `HF_ENDPOINT` | HuggingFace 镜像地址（模型下载） | 自动探测，必要时 `https://hf-mirror.com` |
| `HF_HOME` | 模型缓存目录（自定义“模型安装位置”；Windows/Linux 均自动展开 `~`） | 默认 `~/.cache/huggingface` |
| `BILI_OUTPUT_DIR` | 默认输出目录 | `./output` |
| `BILI_VENV_DIR` | 虚拟环境目录（须为真实环境变量） | `脚本目录/.venv` |
| `BILI_PARALLEL` | 并行路数：`auto`=按核数自动（上限4，GPU单路），`1`=关闭并行，`N`=固定N路 | `auto` |
| `BILI_SKIP_EXISTING` | `1`=断点续传，跳过已转写视频（等价 `--skip-existing`） | `0` |
| `BILI_WITH_TIMESTAMPS` | `1`=额外生成 `.srt` 字幕（等价 `--with-timestamps`） | `0` |
| `BILI_NO_SUMMARY` | `1`=只转写不调 AI（等价 `--no-summary`） | `0` |

---

## ❓ 常见问题

**Q: 提示"未找到 ffmpeg"？**
安装 ffmpeg 即可：Ubuntu `sudo apt install ffmpeg`；macOS `brew install ffmpeg`；Windows 从 ffmpeg.org 下载并加入 PATH。没有 ffmpeg 时脚本会用 imageio-ffmpeg 内置的静态版兜底。

**Q: 首次运行提示 venv 创建失败？**
Debian/Ubuntu 可能需要先装 `python3-venv`（`sudo apt install python3-venv`）。

**Q: 模型下载很慢或失败？**
脚本会自动切换 hf-mirror.com 镜像；也可手动 `export HF_ENDPOINT=https://hf-mirror.com`。模型只需下载一次，缓存于 `~/.cache/huggingface`。

**Q: 模型（几百 MB 的大文件）放哪里？怎么用已下载好的模型？**
运行 `python bili_summary.py --setup`：在“Whisper 转写模型”一步选 `[1] 在线下载`（随后在“模型缓存位置”选默认 `~/.cache/huggingface` / 项目内 `models/` / 自定义目录），或选 `[2] 使用本地已安装模型` 直接填目录路径（如 `D:\models\whisper-large-v3`），脚本会直接加载、不再下载。也可手动改 `.env`：`HF_HOME` 控制缓存目录，`WHISPER_SIZE` 填本地路径即跳过下载。小文件（venv 与依赖）始终安装在项目内 `.venv`，无需选择。

**Q: 转写结果为空或过短？**
纯音乐、无人声视频无法转写属正常；中文识别失败会自动用自动语种重试一遍。

**Q: 一个视频失败会中断其他视频吗？**
不会。失败项会写入终端报告（总结模式下也写入 `summaries.md` 的"处理失败"节），其余视频照常处理。

**Q: 用 `--no-summary` 还需要 API Key 吗？**
不需要，转写全程本地完成，不调用任何云 API。

**Q: 转写很慢，怎么并行加速？**
默认已自动并行：按 CPU 核数开启（上限 4 路），下载也会提前预取与转写重叠。要调整用 `-p N`（或 `.env` 的 `BILI_PARALLEL`）：CPU 用户建议 `-p 2`~`-p 3`（每路模型约占 500MB~1GB 内存）；GPU 用户保持自动（单路）。

**Q: 有 NVIDIA 显卡，会自动用 GPU 吗？**
会。启动时自动检测，检测到即用 `cuda+float16`，无需配置。想关闭：`.env` 设 `WHISPER_DEVICE=cpu`。

**Q: 不想每次敲 `--skip-existing` / `--with-timestamps`？**
直接改项目目录 `.env`：`BILI_SKIP_EXISTING=1`、`BILI_WITH_TIMESTAMPS=1`，之后每次运行自动生效；命令行再给出时以命令行优先。

**Q: 批量处理中断了，能续跑吗？**
加 `--skip-existing` 重跑：已生成的转写文件会被跳过，只处理缺失的，且只请求一次元数据、不重复下载。

**Q: 想要带时间戳的字幕？**
加 `--with-timestamps`，每个视频会额外生成同名 `.srt` 字幕（纯文本 `.txt` 不受影响，仍可直接喂给 AI 工具）。

**Q: Windows 上怎么运行？**
直接 `python bili_summary.py ...`（或 `py bili_summary.py ...`）即可，用法与 Linux 完全一致；首次运行会自动创建 `.venv` 并安装依赖。若已 `pip install .`，直接敲 `bili-summary` 命令（`.venv\Scripts\bili-summary.exe`）。

**Q: Windows 上报错“无法加载 ctranslate2”或缺少 DLL？**
faster-whisper 依赖的 ctranslate2 需要 Microsoft Visual C++ 运行库，安装 [VC++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) 后重试即可。

**Q: 运行后目录里多了 links.txt / __pycache__？**
新版会自动删除：交互模式生成的 `links.txt` 和 `__pycache__` 退出时自动清理；需要保留 links.txt 加 `--keep-links`。

---

## 📁 项目结构

```text
bili_summary/
├── bili_summary.py        # 主程序（含 --setup 一键配置，依赖清单在代码内）
├── pyproject.toml         # pip 打包/安装配置
├── README.md
├── LICENSE                # MIT
├── .gitignore
└── .env.example           # 配置示例（复制为 .env 使用）
```

## 🛠 技术栈

| 环节 | 技术/参数 |
|---|---|
| 下载 | yt-dlp（bestaudio，noplaylist） |
| 转码 | ffmpeg → mp3（失败降级用原格式） |
| 转写 | faster-whisper `small` 模型，CPU/int8；中文+VAD 优先，失败自动切自动语种 |
| 分段 | 超过 30000 字按 20000 字切分（句末断开），仅供总结模式使用 |
| 总结（可选） | DeepSeek `deepseek-chat`（失败回退 `deepseek-reasoner`），temperature 0.3，500 字内，长文 map-reduce 分层合并 |

## 📄 许可

[MIT License](LICENSE)。仅供学习研究，请遵守 B站 用户协议与相关法律法规。
