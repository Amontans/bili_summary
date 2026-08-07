#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bili_summary.py — B站视频一键转文字稿 + AI 总结（支持多链接、可迁移）

流程: 链接/BV号 → yt-dlp 下载音频 → faster-whisper 中文转写 → DeepSeek 智能总结 → 清理临时文件

用法:
  python bili_summary.py                                    # 交互模式：逐行输入链接，空行开始处理
  python bili_summary.py <链接或BV号> [<链接或BV号> ...]    # 一次处理多个视频
  python bili_summary.py -f links.txt                       # 从文件读取链接（每行一个，# 开头为注释）
  python bili_summary.py -o output <链接...>                # 指定输出目录（默认 ./output）
  python bili_summary.py --no-summary <链接...>             # 只转写，不调用 AI 总结（无需 API Key）
  python bili_summary.py --setup                            # 一键配置向导（API Key/镜像/模型预下载）
  python bili_summary.py --dry-run -f links.txt             # 只预览链接解析结果，不实际处理

安装方式（GitHub）:
  git clone 后直接运行（自动创建 .venv 并安装依赖）；或 pip install . 安装为 bili-summary 命令
  配置: python bili_summary.py --setup 一键向导（API Key / 镜像地址 / 模型预下载，只动本项目与用户缓存）

配置（项目目录 .env 文件，优先级低于真实环境变量）:
  DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL / DEEPSEEK_MODEL_FALLBACK
  WHISPER_SIZE / WHISPER_DEVICE / WHISPER_COMPUTE_TYPE / HF_ENDPOINT / BILI_OUTPUT_DIR

自动清理:
  交互模式生成的 links.txt 与运行时产生的 __pycache__ 会在退出时自动删除
  （保留 links.txt 用 --keep-links）

输出（一个文件夹内的一堆纯文本文件）:
  output/<视频标题>.txt             每段视频的完整文字稿（文件名=视频标题，去非法字符/截断80字）
  output/<视频标题>.summary.txt     每段视频的AI总结（仅开启总结时）
  output/summaries.md              全部视频的汇总（仅开启总结时）

可迁移说明:
  - 依赖自动安装到脚本所在目录的 .venv（环境变量 BILI_VENV_DIR 可覆盖路径）
  - 拷贝整个文件夹到新机器，首次运行会自动重建虚拟环境并安装依赖
  - 需要: python3（ffmpeg 可选，缺省自动用内置静态版；DEEPSEEK_API_KEY 仅“转写+总结”模式需要）
"""
import argparse
import importlib.util
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR   = os.environ.get("BILI_VENV_DIR", os.path.join(SCRIPT_DIR, ".venv"))
LEGACY_VENV = os.path.expanduser("~/bili-venv")      # 旧版脚本遗留的虚拟环境，优先复用
REQUIREMENTS = ["yt-dlp", "faster-whisper", "openai", "imageio-ffmpeg"]   # pip 包名（唯一依赖清单）
IMPORT_NAMES = {p: p.replace("-", "_") for p in REQUIREMENTS}                 # 对应导入名

__version__ = "2.4.1"


def log(msg):
    print(f"[bili_summary] {msg}", flush=True)


def load_env_file(path):
    """极简 .env 加载（无第三方依赖）：KEY=VALUE，支持 # 注释与引号；不覆盖已存在的环境变量"""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


# ============ 自动引导: 确保依赖可用（可迁移核心） ============
def _venv_ok(python_bin):
    """判断某个解释器是否已装齐全部依赖（find_spec 需用导入名，如 yt_dlp）"""
    names = [IMPORT_NAMES[p] for p in REQUIREMENTS]
    code = ("import importlib.util,sys;"
            "sys.exit(0 if all(importlib.util.find_spec(m) for m in %r) else 1)" % names)
    try:
        return subprocess.run([python_bin, "-c", code], capture_output=True, timeout=30).returncode == 0
    except Exception:
        return False


def _venv_python(venv_dir):
    """跨平台定位虚拟环境的解释器（Unix: bin/python；Windows: Scripts/python.exe）"""
    for name in ("bin/python", "Scripts/python.exe"):
        p = os.path.join(venv_dir, name)
        if os.path.exists(p):
            return p
    return None


def _is_installed_package():
    """通过 pip 安装到 site-packages 时，依赖由 pip 管理，跳过自举"""
    return "site-packages" in SCRIPT_DIR


def ensure_deps():
    """若当前解释器缺依赖，则复用/创建脚本目录下的 .venv，并用其解释器重新执行本脚本"""
    if _is_installed_package():
        return
    missing = [p for p in REQUIREMENTS if importlib.util.find_spec(IMPORT_NAMES[p]) is None]
    if not missing:
        return
    # 1) 优先复用现成的虚拟环境（新位置 -> 旧版遗留位置），避免重复安装
    for vd in (VENV_DIR, LEGACY_VENV):
        vpy = _venv_python(vd)
        if vpy and _venv_ok(vpy):
            log(f"使用虚拟环境解释器: {vpy}")
            os.execv(vpy, [vpy] + sys.argv)
    # 2) 都没有则现场创建
    log(f"首次运行：创建虚拟环境 {VENV_DIR} 并安装依赖 {', '.join(REQUIREMENTS)} ...")
    subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
    vpy = _venv_python(VENV_DIR)
    subprocess.run([vpy, "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)
    log("正在安装依赖（首次需下载数百 MB，请耐心等待；失败会自动重试）...")
    try:
        subprocess.run([vpy, "-m", "pip", "install", "-q"] + REQUIREMENTS, check=True)
    except subprocess.CalledProcessError:
        log("依赖安装失败，正在重试一次...")
        subprocess.run([vpy, "-m", "pip", "install", "-q"] + REQUIREMENTS, check=True)
    log("依赖安装完成 ✅，自动重启脚本...")
    os.execv(vpy, [vpy] + sys.argv)


ensure_deps()

load_env_file(os.path.join(SCRIPT_DIR, ".env"))   # 加载项目目录下的 .env（真实环境变量优先）

# ============ 配置 ============
API_KEY   = os.environ.get("DEEPSEEK_API_KEY", "sk-xxxxxxxxxxxxxxxx")
BASE_URL  = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MODEL     = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")          # 主模型；失败时自动回退
MODEL_FB  = os.environ.get("DEEPSEEK_MODEL_FALLBACK", "deepseek-reasoner")
WHISPER_SIZE = os.environ.get("WHISPER_SIZE", "small")                 # small 模型
DEVICE    = os.environ.get("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")          # CPU 量化计算
CHUNK_LEN = 20000                      # 分段长度(字)
SPLIT_THRESHOLD = 30000                # 超过此长度才分段
TRANS_TIMEOUT = 7200                   # 转写超时(秒)，仅兜底防卡死
MAX_SUMMARY_TOKENS = 800

SYS_PROMPT = (
    "你是一个视频内容总结助手。请阅读下面提供的视频文字稿，"
    "提取视频的核心内容，用中文输出一份结构清晰、重点突出的总结。"
    "总结应包含：① 视频主题概述；② 主要观点与关键信息；③ 重点细节或结论。"
    "请使用小标题和要点列表，总结控制在500字以内。"
)

# ============ 工具 ============
def find_ffmpeg():
    """优先用 PATH 中的 ffmpeg，否则退回 imageio_ffmpeg 内置静态版"""
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError("未找到 ffmpeg，请先安装（apt install ffmpeg 或 pip install imageio-ffmpeg）")

def extract_bvid(text):
    m = re.search(r"BV[0-9A-Za-z]{10}", text)
    return m.group(0) if m else None

def extract_avid(text):
    m = re.search(r"av\d+", text)
    return m.group(0) if m else None

def with_retry(fn, name, retries=1, delay=3):
    """通用重试：失败给出明确提示并重试一次"""
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            log(f"⚠ {name} 失败（第 {attempt+1} 次）：{e}")
            if attempt < retries:
                log(f"→ {delay} 秒后重试...")
                time.sleep(delay)
            else:
                raise RuntimeError(f"{name} 多次尝试后仍失败：{e}") from e

def run_with_timeout(fn, timeout, name):
    """用 SIGALRM 兜底超时（仅防死锁，正常耗时远小于 timeout）；Windows 无 SIGALRM 则直接执行"""
    if not hasattr(signal, "SIGALRM"):
        return fn()
    def _alarm(signum, frame):
        raise TimeoutError(f"{name} 超时（>{timeout}s），疑似卡死")
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

# ============ 1. 下载音频 ============
def download_audio(url, outdir):
    import yt_dlp
    ffmpeg = find_ffmpeg()
    log(f"🎬 开始下载音频: {url}")
    # 不用 yt-dlp 的 FFmpegExtractAudio 后处理（需要 ffprobe），改为下载后手动转换
    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(outdir, "audio.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
    }
    def _dl():
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return info.get("title", "untitled")
    title = with_retry(_dl, "音频下载")
    cands = [os.path.join(outdir, f) for f in os.listdir(outdir) if f.startswith("audio.")]
    if not cands:
        raise RuntimeError("下载完成但未找到音频文件")
    audio = cands[0]
    # 用 ffmpeg 转成统一 mp3（失败则保留原格式，faster-whisper 可直接解码 m4a/aac）
    mp3 = os.path.join(outdir, "audio.mp3")
    cmd = [ffmpeg, "-y", "-i", audio, "-vn", "-acodec", "libmp3lame", "-q:a", "0", mp3]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=900)
        audio = mp3
    except Exception as e:
        log(f"⚠ 转 mp3 失败（{e}），直接使用原音频格式: {os.path.basename(audio)}")
    size = os.path.getsize(audio) / 1024 / 1024
    log(f"✅ 音频就绪: {os.path.basename(audio)} ({size:.1f} MB)，标题: {title}")
    return audio, title

# ============ 2. Whisper 转写 ============
def ensure_model_endpoint():
    """huggingface.co 不可达时自动切换到 hf-mirror.com 镜像；镜像不支持 Xet，需禁用"""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # 走经典 HTTP 下载，兼容镜像
    if os.environ.get("HF_ENDPOINT"):
        return
    import urllib.request
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=5)
    except Exception:
        log("↪ huggingface.co 不可达，自动切换模型镜像 hf-mirror.com")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def load_model():
    """加载 Whisper 模型（多视频共用同一实例，避免重复下载/加载）"""
    from faster_whisper import WhisperModel
    ensure_model_endpoint()
    log(f"🧠 加载 Whisper {WHISPER_SIZE} 模型（{DEVICE}/{COMPUTE_TYPE}），首次使用需下载模型...")
    return with_retry(lambda: WhisperModel(WHISPER_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE), "模型加载")

def transcribe(audio_path, model, save_path):
    log("🎧 开始中文语音识别（小模型 CPU int8，请耐心等待）...")

    def _run(pass_no):
        """第1遍: 中文+VAD；第2遍: 自动语种+不过滤（兼容音乐/外语视频）"""
        if pass_no == 1:
            segs, info = model.transcribe(
                audio_path, language="zh", vad_filter=True, beam_size=5,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
        else:
            segs, info = model.transcribe(
                audio_path, language=None, vad_filter=False, beam_size=5,
                condition_on_previous_text=False,
            )
        return "".join(s.text for s in segs).strip(), info.duration

    def _run_zh():
        return run_with_timeout(lambda: _run(1), TRANS_TIMEOUT, "语音识别")

    def _run_auto():
        return run_with_timeout(lambda: _run(2), TRANS_TIMEOUT, "语音识别(自动语种)")

    text, duration = with_retry(_run_zh, "语音识别(中文)", retries=1)
    min_chars = max(20, int(duration * 0.3))  # 按音频时长估算最低字数
    if len(text) < min_chars:
        log(f"⚠ 中文识别结果过短（{len(text)}字 / 音频{duration:.0f}s），尝试自动语种识别...")
        text, _ = with_retry(_run_auto, "语音识别(自动语种)", retries=1)
    text = text.strip()
    if len(text) < 10:
        raise RuntimeError("识别结果过短，音频可能为纯音乐或语音不清")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)
    log(f"✅ 转写完成，共 {len(text)} 字，已保存到 {save_path}")
    return text

# ============ 3. 分段 ============
def split_transcript(text):
    """超过 SPLIT_THRESHOLD 字按 CHUNK_LEN 分段（尽量在句末断开）"""
    if len(text) <= SPLIT_THRESHOLD:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_LEN, len(text))
        if end < len(text):
            cut = text.rfind("。", start, end)          # 优先句号
            if cut == -1:
                cut = max(text.rfind(c, start, end) for c in "！？\n")  # 其次任意中文标点/换行
            if cut != -1 and cut > start + CHUNK_LEN // 2:
                end = cut + 1
        chunks.append(text[start:end])
        start = end
    log(f"📑 文稿 {len(text)} 字，已自动分为 {len(chunks)} 段（每段约 {CHUNK_LEN} 字）")
    return chunks

# ============ 4. DeepSeek 总结 ============
def _chat(client, model, user_text):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": user_text},
        ],
        temperature=0.3,
        max_tokens=MAX_SUMMARY_TOKENS,
    )
    return resp.choices[0].message.content.strip()

def summarize_chunk(text, stage=""):
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120)
    def _call():
        try:
            return _chat(client, MODEL, text)
        except Exception as e:
            # 模型名不可用时回退到备用模型
            if "model" in str(e).lower() or "not found" in str(e).lower():
                log(f"↪ 模型 {MODEL} 不可用，回退 {MODEL_FB}")
                return _chat(client, MODEL_FB, text)
            raise
    return with_retry(_call, f"DeepSeek 总结{stage}", retries=1)

def summarize_all(transcript):
    chunks = split_transcript(transcript)
    if len(chunks) == 1:
        log("🤖 调用 DeepSeek 进行总结...")
        return summarize_chunk(chunks[0])
    log(f"🤖 分段总结 {len(chunks)} 段（每段先独立总结）...")
    partials = []
    for i, c in enumerate(chunks, 1):
        log(f"  ↳ 总结第 {i}/{len(chunks)} 段...")
        partials.append(summarize_chunk(c, stage=f"[第{i}段]"))
    merged = "\n\n".join(f"第{i}段总结：\n{p}" for i, p in enumerate(partials, 1))
    log("🧩 合并各段总结，生成最终总结...")
    prompt = ("以下是一个长视频分段的若干总结，请合并为一份完整的总结，"
              "去掉重复内容，保持结构清晰、重点突出，控制在500字以内：\n\n" + merged)
    return summarize_chunk(prompt, stage="[合并]")

# ============ 5. 输入/输出 ============
def normalize_url(text):
    """把 链接 / BV号 / av号 统一成完整 URL。
    兼容各类链接: 完整链接、带 ?bvid= 查询参数的分享/稍后再看链接、
    移动端 m.bilibili.com、分P链接(?p=) 等；b23 短链（无内嵌ID）原样透传给 yt-dlp。"""
    t = text.strip()
    # 1) 非 URL 输入（BV号/av号）：补全为视频页
    if not re.match(r"^https?://", t, re.I):
        for m in (extract_bvid(t), extract_avid(t)):
            if m:
                return f"https://www.bilibili.com/video/{m}"
        return t
    # 2) URL 输入：优先从链接/查询参数中提取视频ID，重建干净链接（避免被误判为列表/合集）
    vid = extract_bvid(t) or extract_avid(t)
    if vid:
        p = re.search(r"[?&]p=(\d+)", t)   # 仅保留分P参数（如 ?p=2），其余跟踪参数自动丢弃
        return f"https://www.bilibili.com/video/{vid}" + (f"?p={p.group(1)}" if p else "")
    # 3) 无视频ID的链接（如 b23.tv 短链）：原样透传给 yt-dlp 自动解析
    return t

def vid_key(text):
    """生成稳定的视频ID: BV号 / av号 / 截断的链接（用于目录与回退文件名）"""
    for m in (extract_bvid(text), extract_avid(text)):
        if m:
            return m
    return re.sub(r"[^\w\-]+", "_", text)[:60]


_WIN_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}   # Windows 保留设备名


def sanitize_title(title):
    """视频标题 → 安全文件名：去非法字符、压缩空白、去首尾点/空格、规避 Windows 保留名、截断 80 字；空则返回 None"""
    t = re.sub(r"[\\/:*?\"<>|\x00-\x1f]", "", title)   # Windows 非法字符 + 控制字符
    t = re.sub(r"\s+", " ", t).strip()
    t = t.strip(". ")
    if not t:
        return None
    t = t[:80]
    # Windows 下 CON/PRN/AUX/NUL/COM1-9/LPT1-9（含 CON.txt 等带扩展名形式）无法创建，加前缀规避
    if t.split(".")[0].upper() in _WIN_RESERVED:
        t = "_" + t
    return t


def make_output_name(title, key, used):
    """生成输出文件名：优先视频标题；同批重名自动加 _2/_3；标题不可用时回退视频ID"""
    base = sanitize_title(title) or key
    name, i = base, 2
    while name.lower() in used:
        name = f"{base}_{i}"
        i += 1
    used.add(name.lower())
    return name


class _NullLogger:
    """静默 yt-dlp 日志（仅用于 fetch_title 元数据探测，避免 ERROR 噪音）"""
    def debug(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass


def fetch_title(url):
    """仅抓取视频标题（不下载任何内容），失败返回 None"""
    import yt_dlp
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True,
                               "logger": _NullLogger()}) as ydl:
            return ydl.extract_info(url, download=False).get("title")
    except Exception:
        return None

def dedupe(items):
    """按视频 ID 去重保序"""
    seen, out = set(), []
    for it in items:
        key = vid_key(it).lower()
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out


def collect_inputs(args):
    """合并命令行参数与文件列表，去重保序"""
    items = list(args.inputs)
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    items.append(s)
    return dedupe(items)


def interactive_collect():
    """交互模式：逐行收集链接，空行（直接回车）结束输入"""
    print("📥 交互模式：请逐行输入 B站链接或BV号，全部输完后直接回车（空行）开始处理")
    print("   （随时按 Ctrl+C 可取消）")
    links = []
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        links.append(line)
    return links


def save_links_file(items, path="links.txt"):
    """把交互收集的链接写入 links.txt（含注释头，可被 -f 直接复用），返回绝对路径"""
    import datetime
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# 由 bili_summary 交互模式生成（{datetime.datetime.now():%Y-%m-%d %H:%M}）\n")
        for it in items:
            f.write(it + "\n")
    abs_path = os.path.abspath(path)
    log(f"💾 已保存 {len(items)} 个链接到 {abs_path}（退出后自动删除，保留请加 --keep-links）")
    return abs_path


def cleanup_artifacts():
    """退出时清理运行时产生的临时文件：交互生成的 links.txt（由调用方决定）与 __pycache__"""
    if not _is_installed_package():
        pc = os.path.join(SCRIPT_DIR, "__pycache__")
        if os.path.isdir(pc):
            shutil.rmtree(pc, ignore_errors=True)

def write_combined(outdir, results):
    """把所有视频总结合并成 summaries.md"""
    ok = [r for r in results if "error" not in r]
    bad = [r for r in results if "error" in r]
    lines = ["# B站视频总结汇总", ""]
    for i, r in enumerate(ok, 1):
        title = r.get("title") or r["url"]
        lines += [
            f"## {i}. {title}",
            "",
            f"- 链接: {r['url']}",
            f"- BV号: {r.get('bvid') or '-'}",
            f"- 文字稿: `{r['transcript_path']}`",
            "",
            r["summary"],
            "",
        ]
    if bad:
        lines += ["## 处理失败", ""]
        for r in bad:
            lines += [f"- {r['url']}: {r['error']}", ""]
    path = os.path.join(outdir, "summaries.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"📄 全量汇总已写入: {path}")

# ============ 6. 单视频处理 ============
def process_one(url, outdir, model_holder, summarize=True, used_names=None):
    """下载→转写→(可选)总结 一个视频，返回结果字典；失败抛异常由调用方捕获"""
    bvid = extract_bvid(url)
    log(f"🎯 目标: {url}（BV号: {bvid or '无'}）")
    tmpdir = tempfile.mkdtemp(prefix="bili_audio_")
    try:
        audio, title = download_audio(url, tmpdir)
        if model_holder["model"] is None:
            model_holder["model"] = load_model()   # Whisper 模型只加载一次，多视频复用
        key = vid_key(url)
        used = used_names if used_names is not None else set()
        fname = make_output_name(title, key, used)   # 文件名=视频标题
        os.makedirs(outdir, exist_ok=True)
        tpath = os.path.join(outdir, f"{fname}.txt")   # 扁平输出：每视频一个纯文本文件
        transcript = transcribe(audio, model_holder["model"], tpath)
        res = {"url": url, "bvid": bvid, "title": title,
               "key": key, "name": fname,
               "transcript_path": tpath, "transcript": transcript}
        if summarize:
            summary = summarize_all(transcript)
            spath = os.path.join(outdir, f"{fname}.summary.txt")
            with open(spath, "w", encoding="utf-8") as f:
                f.write(summary)
            res.update({"summary": summary, "summary_path": spath})
        return res
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        log("🧹 临时音频文件已清理")

# ============ 7. 一键配置向导（--setup，跨平台） ============
def set_env_value(path, key, value):
    """写入/更新 .env 中的键值（保留无关行）"""
    lines = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    found = False
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            if line.strip().startswith(key + "="):
                f.write(f"{key}={value}\n")
                found = True
            else:
                f.write(line)
        if not found:
            f.write(f"{key}={value}\n")


def _hf_reachable():
    import urllib.request
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=5)
        return True
    except Exception:
        return False


def setup_wizard(check_only=False):
    """一键配置向导：模型规格/缓存位置/API Key/镜像/预下载。只动项目目录与用户缓存，不碰系统"""
    global WHISPER_SIZE
    print("\n" + "=" * 56)
    print("bili_summary 一键配置向导（只动本项目目录与用户缓存，不碰系统）")
    print("=" * 56)

    # 1) 环境检查
    print(f"✅ Python {sys.version.split()[0]}")
    if shutil.which("ffmpeg"):
        print(f"✅ ffmpeg: {shutil.which('ffmpeg')}")
    else:
        print("⚠️ 未检测到系统 ffmpeg —— 无需安装，脚本会自动用 imageio-ffmpeg 内置静态版")
    missing = [p for p in REQUIREMENTS if importlib.util.find_spec(IMPORT_NAMES[p]) is None]
    if missing:
        print(f"❌ 依赖缺失: {', '.join(missing)}（请先正常运行一次脚本，会自动安装）")
        sys.exit(1)
    print("✅ 依赖齐全 (yt-dlp / faster-whisper / openai / imageio-ffmpeg)")
    if check_only:
        print("\n检查完毕（未做任何修改）")
        return

    env_file = os.path.join(SCRIPT_DIR, ".env")

    # 2) Whisper 模型规格（可自定义：内置规格或本地模型目录路径）
    print("\n🧠 Whisper 模型规格（音频转文字的核心模型）")
    print("   内置可选: tiny / base / small / medium / large-v3（越大越准、越慢、越吃内存）")
    print("   也可输入本地模型目录的绝对路径（faster-whisper 直接加载该路径，跳过下载）")
    cur_size = os.environ.get("WHISPER_SIZE", "small")
    try:
        ans = input(f"   当前 [{cur_size}]，直接回车保留，或输入新规格/路径: ").strip()
    except (EOFError, KeyboardInterrupt):
        ans = ""
    if ans:
        WHISPER_SIZE = ans
        os.environ["WHISPER_SIZE"] = ans
        set_env_value(env_file, "WHISPER_SIZE", ans)
        print(f"✅ WHISPER_SIZE = {ans}（已写入 {env_file}）")
    else:
        print(f"✅ 保留 WHISPER_SIZE = {cur_size}")

    # 3) 模型缓存位置（默认 ~/.cache/huggingface，可自由指定目录）
    print("\n📂 Whisper 模型缓存位置（模型下载后存放的目录）")
    cur_hf = os.environ.get("HF_HOME")
    hint = f"当前 [{cur_hf}]" if cur_hf else "当前 [默认 ~/.cache/huggingface]"
    try:
        ans = input(f"   {hint}，直接回车保留，或输入新目录（如 D:\\models 或 ~/models）: ").strip()
    except (EOFError, KeyboardInterrupt):
        ans = ""
    if ans:
        set_env_value(env_file, "HF_HOME", ans)
        os.environ["HF_HOME"] = ans
        print(f"✅ HF_HOME = {ans}（已写入 {env_file}，模型将缓存到该目录）")
    else:
        print("✅ 使用默认缓存位置（~/.cache/huggingface）")

    # 4) DeepSeek API Key（仅总结模式需要）
    print("\n🔑 DeepSeek API Key（仅“转写+AI总结”模式需要；只转写可回车跳过）")
    try:
        key = input("   请输入 API Key（回车跳过）: ").strip()
    except (EOFError, KeyboardInterrupt):
        key = ""
    existing = os.environ.get("DEEPSEEK_API_KEY", "")
    if key:
        set_env_value(env_file, "DEEPSEEK_API_KEY", key)
        print(f"✅ DEEPSEEK_API_KEY 已写入 {env_file}")
    elif existing and not existing.startswith("sk-xxx"):
        print("✅ 保留已有的 DEEPSEEK_API_KEY")
    else:
        print("⚠️ 未配置 API Key —— 仅可使用 --no-summary 转写模式")

    # 5) 模型下载镜像地址
    if os.environ.get("HF_ENDPOINT"):
        print(f"✅ 模型镜像已配置: {os.environ['HF_ENDPOINT']}")
    elif _hf_reachable():
        print("✅ huggingface.co 可达，无需镜像")
    else:
        set_env_value(env_file, "HF_ENDPOINT", "https://hf-mirror.com")
        print("✅ huggingface.co 不可达，已写入 HF_ENDPOINT=https://hf-mirror.com")

    if os.path.exists(env_file):
        os.chmod(env_file, 0o600)
        print(f"✅ 配置文件: {env_file}（权限 600，仅本用户可读）")

    # 6) Whisper 模型预下载（按所选规格，仅首次）
    print(f"\n⬇️ 预下载 Whisper {WHISPER_SIZE} 模型（约数百 MB；不下载则首次转写时自动下载）")
    print("   若 WHISPER_SIZE 填的是本地路径，此步会直接验证该路径可用性")
    try:
        ans = input("   现在下载吗？[y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans = ""
    if ans in ("y", "yes"):
        ensure_model_endpoint()
        print("   下载中，请耐心等待...")
        try:
            load_model()
            print(f"✅ Whisper {WHISPER_SIZE} 模型就绪（缓存于 {os.environ.get('HF_HOME', '~/.cache/huggingface')}）")
        except Exception as e:
            print(f"⚠️ 模型预下载失败: {e}（可重跑 --setup 或首次转写时自动下载）")
    else:
        print("⚠️ 跳过模型预下载")

    print("\n" + "=" * 56)
    print("✅ 配置完成！使用示例:")
    print("   python bili_summary.py -i                                 # 交互模式")
    print("   python bili_summary.py BV1GJ411x7h7 --no-summary -o out    # 只转写")
    print("=" * 56)


# ============ 主流程 ============
class _HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """帮助格式：保留原始换行 + 显示参数默认值"""
    pass


def parse_args(argv):
    ap = argparse.ArgumentParser(
        prog="bili_summary.py",
        description=(
            "B站视频一键转文字稿 + AI 总结（支持多链接、可迁移）\n"
            "\n"
            "流程: 链接/BV号 → yt-dlp 下载音频 → faster-whisper 中文转写 → (可选) DeepSeek AI 总结\n"
            "输出: 一个文件夹内的一堆纯文本文件（默认 ./output/<视频标题>.txt）\n"
            "\n"
            "特性:\n"
            "  · 三种输入方式: 命令行参数 / 文件列表(-f) / 交互模式(-i，无参数自动进入)\n"
            "  · 文件名=视频标题（自动去非法字符/压缩空白/截断80字，同批重名加 _2）\n"
            "  · --no-summary 只转写，无需 API Key\n"
            "  · --setup 一键配置（API Key / 模型镜像 / 模型预下载）\n"
            "  · 自动清理: 交互生成的 links.txt 与运行产生的 __pycache__ 退出时自动删除\n"
            "  · 首次运行自动创建 .venv 并安装依赖，拷贝即用"
        ),
        epilog=(
            "示例:\n"
            "  python bili_summary.py --setup                        # 一键配置（API Key/镜像/模型预下载）\n"
            "  python bili_summary.py -i                             # 交互模式：逐行输入链接，空行开始处理\n"
            "  python bili_summary.py BV1GJ411x7h7                   # 单个视频（默认: 转写 + AI 总结）\n"
            "  python bili_summary.py BV1GJ411x7h7 BV1xx411c7mD      # 多个视频\n"
            "  python bili_summary.py -f links.txt                   # 从文件读取链接\n"
            "  python bili_summary.py --no-summary -f links.txt -o transcripts   # 只转写\n"
            "\n"
            "输出结构:\n"
            "  output/<视频标题>.txt            完整文字稿（文件名=视频标题，去非法字符/截断80字）\n"
            "  output/<视频标题>.summary.txt    每段视频的 AI 总结（仅开启总结时）\n"
            "  output/summaries.md              全部视频汇总（仅开启总结时）\n"
            "\n"
            "配置（优先级: 环境变量 > 项目目录 .env > 默认值）:\n"
            "  DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL / DEEPSEEK_MODEL_FALLBACK\n"
            "  WHISPER_SIZE / WHISPER_DEVICE / WHISPER_COMPUTE_TYPE / HF_ENDPOINT / BILI_OUTPUT_DIR\n"
            "\n"
            "注意:\n"
            "  · 含 & ? 的 URL 传参时请加引号\n"
            "  · 未配置 API Key 时需加 --no-summary\n"
            "  · 交互模式生成的 links.txt 与 __pycache__ 退出时自动删除（--keep-links 保留）"
        ),
        formatter_class=_HelpFormatter,
    )
    ap.add_argument("inputs", nargs="*", help="B站链接或BV号，可多个（无任何输入时自动进入交互模式）")
    ap.add_argument("-i", "--interactive", action="store_true",
                    help="交互模式：逐行输入链接，直接回车（空行）开始处理")
    ap.add_argument("-f", "--file", help="从文件读取链接列表（每行一个，# 开头为注释）")
    ap.add_argument("-o", "--outdir", default=os.environ.get("BILI_OUTPUT_DIR", "output"),
                    help="输出目录（可用环境变量 BILI_OUTPUT_DIR 覆盖）")
    ap.add_argument("--no-summary", action="store_true",
                    help="只转写，不调用 DeepSeek 总结（无需 API Key；输出为纯文本文件）")
    ap.add_argument("--keep-links", action="store_true",
                    help="交互模式结束后保留 links.txt（默认自动删除）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只预览：显示归一化 URL、抓取视频标题与输出文件名（不下载/不转写/不调AI）")
    ap.add_argument("--setup", action="store_true",
                    help="一键配置向导：API Key / 模型镜像 / 模型预下载（--setup --check 只读体检）")
    ap.add_argument("--check", action="store_true", help="只读环境检查（配合 --setup 使用）")
    ap.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    return ap.parse_args(argv)

def report(outdir, results):
    ok = [r for r in results if "error" not in r]
    bad = [r for r in results if "error" in r]
    summary_mode = any("summary" in r for r in ok)
    print("\n" + "#" * 56)
    print(f"📊 汇总报告: 成功 {len(ok)} / 共 {len(results)}")
    print("#" * 56)
    for i, r in enumerate(ok, 1):
        title = r.get("title") or r["url"]
        print(f"  ✅ [{i}] {title}")
        print(f"      ├ 文字稿: {r['transcript_path']}")
        if "summary" in r:
            print(f"      └ 总结:   {r['summary_path']}")
    for r in bad:
        print(f"  ❌ {r['url']}: {r['error']}")
    if summary_mode:
        print(f"  📄 全量汇总: {os.path.join(outdir, 'summaries.md')}")
    print(f"  📁 输出目录: {outdir}")

def main():
    args = parse_args(sys.argv[1:])
    if args.setup:
        setup_wizard(check_only=args.check)
        sys.exit(0)
    links = collect_inputs(args)
    links_file = None   # 交互模式生成的 links.txt（退出时自动删除）
    # 交互模式：显式 -i，或没有任何输入且终端可交互时自动进入
    if args.interactive or (not args.inputs and not args.file and not args.dry_run and sys.stdin.isatty()):
        more = interactive_collect()
        if more:
            links_file = save_links_file(more)
            links = dedupe(links + more)
    try:
        if args.dry_run:
            if not links:
                parse_args(["--help"])
                sys.exit(1)
            print("链接解析预览（文件名取视频标题；抓不到标题时回退视频ID）")
            used = set()
            for raw in links:
                url = normalize_url(raw)
                title = fetch_title(url)
                if title:
                    fname = make_output_name(title, vid_key(url), used)
                    print(f"  {url}")
                    print(f"     标题: {title}")
                    print(f"     文件: {fname}.txt")
                else:
                    key = vid_key(url)
                    used.add(key.lower())
                    print(f"  {url}")
                    print(f"     ⚠️ 标题获取失败（视频可能已失效），文件名回退为: {key}.txt")
            sys.exit(0)
        if not links:
            parse_args(["--help"])
            sys.exit(1)
        if not args.no_summary and (not API_KEY or API_KEY.startswith("sk-xxx")):
            print("❌ 未配置 DeepSeek API Key，请设置环境变量 DEEPSEEK_API_KEY 后重试（或加 --no-summary 只转写）。")
            sys.exit(1)
        outdir = os.path.abspath(args.outdir)
        os.makedirs(outdir, exist_ok=True)
        mode = "转写 + AI 总结" if not args.no_summary else "仅转写（不调用 AI）"
        log(f"共 {len(links)} 个视频待处理，模式: {mode}，输出目录: {outdir}")

        model_holder = {"model": None}   # Whisper 模型全局复用
        used_names = set()               # 输出文件名去重（标题相同则加 _2/_3）
        results = []
        for i, raw in enumerate(links, 1):
            url = normalize_url(raw)
            log(f"\n===== [{i}/{len(links)}] 开始处理 {url} =====")
            try:
                res = process_one(url, outdir, model_holder, summarize=not args.no_summary, used_names=used_names)
                results.append(res)
                print("\n" + "=" * 56)
                print(f"📝 {'AI 总结' if 'summary' in res else '转写结果'} {i}/{len(links)}")
                print("=" * 56)
                if "summary" in res:
                    print(res["summary"])
                else:
                    print(f"文字稿已保存: {res['transcript_path']}（共 {len(res['transcript'])} 字）")
                print("=" * 56)
            except Exception as e:
                log(f"❌ 处理失败: {e}")
                results.append({"url": url, "error": str(e)})

        if not args.no_summary:
            write_combined(outdir, results)
        report(outdir, results)
    finally:
        # 自动清理：links.txt 与 __pycache__
        if links_file and not args.keep_links:
            try:
                os.remove(links_file)
                log(f"🧹 已自动删除临时文件 {links_file}（如需保留请加 --keep-links）")
            except OSError:
                pass
        cleanup_artifacts()

if __name__ == "__main__":
    main()
