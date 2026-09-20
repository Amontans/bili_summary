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
  python bili_summary.py --setup                            # 一键配置向导（含注册 bili-summary 全局命令）
  python bili_summary.py --dry-run -f links.txt             # 只预览链接解析结果，不实际处理
  python bili_summary.py -p 2 -f links.txt                  # 并行：下载预取与转写重叠，2 路同时转写
  python bili_summary.py --skip-existing -f links.txt       # 断点续传：已转写过的视频直接跳过
  python bili_summary.py --with-timestamps BV1xx...         # 额外生成带时间戳的 .srt 字幕
  python bili_summary.py --config                            # 交互式设置中心（菜单式改配置，写入 .env）
  python bili_summary.py --show-config                       # 只读显示当前生效配置
  python bili_summary.py --background -f links.txt           # 后台独立会话运行（关终端/断 SSH 也不中断）
  python bili_summary.py --force -f links.txt                # 忽略断点续传状态，全部重做

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
  output/<视频标题>.srt             带时间戳字幕（仅 --with-timestamps）
  output/<视频标题>.summary.txt     每段视频的AI总结（仅开启总结时）
  output/summaries.md              全部视频的汇总（仅开启总结时）
  output/report.md                 处理报告（成功/跳过/失败/未完成，始终生成）
  output/.bili_state.json          断点续传状态（已完成清单，重跑时秒级跳过）
  output/.bili_cache/<BV号>/       音频与分片进度缓存（续跑免下载；完成后自动清理）

中断保护（避免各类意外中断造成白跑）:
  · 内存自适应：按可用内存自动限制并行路数，低于阈值时暂停启动新任务，避免被系统 OOM 杀掉
  · 长音频分片：超过阈值的长音频（直播回放等）自动切成 N 分钟分片转写，内存占用恒定、逐片落盘
  · 断点续传：每个视频完成后立即写状态；中断后重跑同一条命令，已完成/已下载/已转写的分片自动跳过
  · 优雅停止：Ctrl+C / kill / 关终端（SIGINT/SIGTERM/SIGHUP）只停止接新任务，进度与缓存全部保留
  · 后台运行：--background 把任务放到独立会话，输出写 logs/*.log，关终端不影响

可迁移说明:
  - 依赖自动安装到脚本所在目录的 .venv（环境变量 BILI_VENV_DIR 可覆盖路径）
  - 拷贝整个文件夹到新机器，首次运行会自动重建虚拟环境并安装依赖
  - 需要: python3（ffmpeg 可选，缺省自动用内置静态版；DEEPSEEK_API_KEY 仅“转写+总结”模式需要）
"""
import argparse
import importlib.util
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR   = os.environ.get("BILI_VENV_DIR", os.path.join(SCRIPT_DIR, ".venv"))
LEGACY_VENV = os.path.expanduser("~/bili-venv")      # 旧版脚本遗留的虚拟环境，优先复用
REQUIREMENTS = ["yt-dlp", "faster-whisper", "openai", "imageio-ffmpeg"]   # pip 包名（唯一依赖清单）
IMPORT_NAMES = {p: p.replace("-", "_") for p in REQUIREMENTS}                 # 对应导入名

__version__ = "2.5.0"


_print_lock = threading.Lock()   # 多线程打印互斥，避免日志交错


def log(msg):
    with _print_lock:
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

_REAL_ENV = set(os.environ)                       # 进程自带的真实环境变量（不含 .env 注入，供配置来源判定）
load_env_file(os.path.join(SCRIPT_DIR, ".env"))   # 加载项目目录下的 .env（真实环境变量优先）

# ============ 配置 ============
API_KEY   = os.environ.get("DEEPSEEK_API_KEY", "sk-xxxxxxxxxxxxxxxx")
BASE_URL  = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MODEL     = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")          # 主模型；失败时自动回退
MODEL_FB  = os.environ.get("DEEPSEEK_MODEL_FALLBACK", "deepseek-reasoner")
WHISPER_SIZE = os.environ.get("WHISPER_SIZE", "small")                 # small 模型
DEVICE    = os.environ.get("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")          # CPU 量化计算

# ---- 自动硬件/并行探测与配置读取（可在 .env 或环境变量中手动覆盖）----
def cfg_flag(key):
    """读取 .env/环境变量中的开关（1/true/yes/on 视为开）"""
    return os.environ.get(key, "").strip().lower() in ("1", "true", "yes", "on")


# ============ 中断保护：内存探测 / 优雅停止 / 断点续传状态 ============
# 目标：任何意外中断（内存不足被 kill、Ctrl+C、关终端、断网、断电）都不丢已完成的工作，
#       重跑同一条命令即可自动续跑：不重复下载、不重复转写。
MEM_RESERVE_GB    = float(os.environ.get("BILI_MEM_RESERVE_GB", "1.0"))     # 给系统/其他程序预留
MEM_PER_WORKER_GB = float(os.environ.get("BILI_MEM_PER_WORKER_GB", "1.5"))  # 每路转写内存预算
LOW_MEM_MB        = int(os.environ.get("BILI_LOW_MEM_MB", "700"))           # 可用内存低于此值告警
CHUNK_SEC         = int(os.environ.get("BILI_CHUNK_SEC", "1800"))           # 长音频分片秒数（0=不分片）
CHUNK_THRESHOLD   = int(os.environ.get("BILI_CHUNK_THRESHOLD", "2400"))     # 超过此秒数才分片
MIN_DISK_MB       = int(os.environ.get("BILI_MIN_DISK_MB", "600"))          # 下载前磁盘剩余下限
KEEP_CACHE        = cfg_flag("BILI_KEEP_CACHE")                             # 成功后也保留音频缓存
STATE_NAME        = ".bili_state.json"       # 断点续传状态（哪些视频已完成）
CACHE_NAME        = ".bili_cache"            # 音频/分片缓存目录（续跑免下载）


class InterruptedByUser(Exception):
    """收到中断信号（Ctrl+C / kill / 关终端）时中止当前视频，进度保留"""


_stop_event = threading.Event()
_stop_reason = {"why": ""}
_signal_count = {"n": 0}


def stop_requested():
    """是否已请求停止（各工作线程/分片循环都会检查）"""
    return _stop_event.is_set()


def request_stop(reason="收到中断信号"):
    if not _stop_event.is_set():
        _stop_reason["why"] = reason
    _stop_event.set()


def install_signal_handlers():
    """SIGINT/SIGTERM/SIGHUP → 优雅停止（保存状态、保留缓存），再按一次立即退出"""
    def _handler(signum, frame):
        _signal_count["n"] += 1
        try:
            name = signal.Signals(signum).name
        except Exception:
            name = str(signum)
        if _signal_count["n"] >= 2:
            log(f"⛔ 再次收到 {name}，立即退出（已完成进度与缓存已落盘，重跑可续）")
            os._exit(130)
        request_stop(f"收到 {name}")
        log(f"🛑 收到 {name}：安全停止中 —— 不再接新任务，当前分片/视频收尾后退出；"
            f"进度已保存，再跑一次同样的命令即可续传（再按一次 Ctrl+C 立即退出）")
    for sg in ("SIGINT", "SIGTERM", "SIGHUP"):
        s = getattr(signal, sg, None)
        if s is None:
            continue
        try:
            signal.signal(s, _handler)
        except Exception:
            pass


def mem_available_bytes():
    """系统可用内存（字节）；探测不到返回 None"""
    try:                                            # Linux / WSL
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    if sys.platform == "win32":
        try:                                        # Windows
            import ctypes
            class _MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = _MemStatus()
            st.dwLength = ctypes.sizeof(_MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return int(st.ullAvailPhys)
        except Exception:
            return None
    try:                                            # macOS / 其他 Unix
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return None


def mem_report_line():
    """一行人类可读的内存信息（用于日志）"""
    avail = mem_available_bytes()
    if avail is None:
        return "内存未知"
    return f"可用内存 {avail / 1024 ** 3:.1f} GB"


def free_disk_bytes(path):
    try:
        return shutil.disk_usage(path).free
    except Exception:
        return None


def wait_for_memory(need_gb, tag="转写", max_wait_s=1800):
    """可用内存不足时等待（避免被系统 OOM 杀掉）；返回 False = 收到停止信号"""
    need = int(need_gb * 1024 ** 3)
    waited, t0, last_log = 0.0, time.time(), 0.0
    while True:
        if stop_requested():
            return False
        avail = mem_available_bytes()
        if avail is None or avail >= need:
            return True
        if time.time() - last_log > 30:
            last_log = time.time()
            log(f"⏳ 可用内存不足（{avail / 1024 ** 3:.2f} GB < {need_gb:.1f} GB），暂停启动新的{tag}，"
                f"等待其他任务释放内存...")
        if waited > max_wait_s:
            log(f"⚠️ 等待内存已超过 {max_wait_s // 60} 分钟，仍继续（若被系统杀掉，重跑会自动续传）")
            return True
        time.sleep(2)
        waited += 2


def start_memory_watchdog():
    """后台看门狗：内存告急时提前告警（比被 OOM 杀掉多一次自救机会）"""
    def _loop():
        while not stop_requested():
            avail = mem_available_bytes()
            if avail is not None and avail < LOW_MEM_MB * 1024 * 1024:
                log(f"⚠️ 可用内存仅剩 {avail / 1024 ** 2:.0f} MB，随时可能被系统 OOM 杀掉；"
                    f"建议 Ctrl+C 安全停止后改用 -p 1 续跑（进度不会丢）")
                time.sleep(30)
                continue
            time.sleep(20)
    threading.Thread(target=_loop, daemon=True, name="mem-watchdog").start()


def _parallel_cap_by_memory():
    """按可用内存推算最多能并行几路转写"""
    avail = mem_available_bytes()
    if avail is None:
        return 99
    budget = avail / 1024 ** 3 - MEM_RESERVE_GB
    return max(1, int(budget // MEM_PER_WORKER_GB))


# ---- 断点续传状态（output/.bili_state.json）----
_state_lock = threading.Lock()


def state_path(outdir):
    return os.path.join(outdir, STATE_NAME)


def load_state(outdir):
    """读取断点续传状态 {vid_key: {...}}；损坏时忽略并重新判断"""
    try:
        with open(state_path(outdir), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        log(f"⚠ 状态文件无法读取（{e}），忽略并从零判断")
        return {}


def save_state(outdir, state):
    """原子写入状态文件（临时文件 + rename，断电也不会写坏）；调用方需持有 _state_lock"""
    try:
        os.makedirs(outdir, exist_ok=True)
        items = {k: v for k, v in state.items() if not k.startswith("_")}
        if len(items) > 2000:                       # 状态文件不无限膨胀
            items = dict(sorted(items.items(),
                                key=lambda kv: kv[1].get("ts", 0) if isinstance(kv[1], dict) else 0)[-2000:])
        payload = dict(items, _updated=time.strftime("%Y-%m-%d %H:%M:%S"))
        tmp = state_path(outdir) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        os.replace(tmp, state_path(outdir))
    except Exception as e:
        log(f"⚠ 状态写入失败: {e}")


def update_state(outdir, state, key, **rec):
    """更新单个视频状态并立即落盘（进程被 kill 也只会丢正在跑的那一个）"""
    rec.setdefault("ts", int(time.time()))
    with _state_lock:                 # 写入与落盘都在锁内，避免多线程同时改/迭代出错
        state[key] = rec
        save_state(outdir, state)


def resume_ok(state, key, outdir=None):
    """已完成且文字稿仍在 → 可秒级跳过（不联网、不下载）"""
    rec = state.get(key) or {}
    if not isinstance(rec, dict) or rec.get("status") != "done":
        return False
    p = rec.get("transcript")
    return bool(p) and os.path.exists(p)


# ---- 音频/分片缓存（避免中断后重复下载与重复转写）----
def workdir_for(outdir, key):
    """该视频的缓存目录（音频 + 分片进度）"""
    return os.path.join(outdir, CACHE_NAME, key)


def prune_cache_dir(outdir):
    """清理已完成的空缓存目录（成功后会删掉 BV 子目录，这里顺手把空壳目录也删掉）"""
    root = os.path.join(outdir, CACHE_NAME)
    try:
        if not os.path.isdir(root):
            return
        for name in os.listdir(root):
            sub = os.path.join(root, name)
            if os.path.isdir(sub) and not os.listdir(sub):
                os.rmdir(sub)
        if not os.listdir(root):
            os.rmdir(root)
    except OSError:
        pass


def cleanup_stale_tempdirs(max_age_h=24):
    """清理历史遗留的 bili_audio_* 临时目录（旧版本崩溃/断电残留），避免占满磁盘"""
    tmp = tempfile.gettempdir()
    now, removed = time.time(), 0
    try:
        for name in os.listdir(tmp):
            if not name.startswith("bili_audio_"):
                continue
            p = os.path.join(tmp, name)
            try:
                if now - os.path.getmtime(p) > max_age_h * 3600:
                    shutil.rmtree(p, ignore_errors=True)
                    removed += 1
            except OSError:
                pass
    except Exception:
        pass
    if removed:
        log(f"🧹 已清理 {removed} 个历史遗留的临时音频目录")


def check_disk_space(path, need_mb=None):
    """磁盘剩余空间检查：不足时给出明确提示（避免下到一半磁盘满）"""
    need_mb = need_mb or MIN_DISK_MB
    free = free_disk_bytes(path)
    if free is None:
        return
    if free < need_mb * 1024 * 1024:
        raise RuntimeError(f"磁盘空间不足：剩余 {free / 1024 ** 2:.0f} MB < 需要的 {need_mb} MB"
                           f"（可用 BILI_MIN_DISK_MB 调整阈值；或清理 {path}）")


_device_explicit = False                 # 用户是否显式指定了设备（自动回退只作用于自动探测）


def _cuda_usable():
    """验证 CUDA 是否真的可用：设备计数 > 0 且 cuBLAS/cuDNN 运行库可加载。
    ctranslate2 的 get_cuda_device_count() 在缺少驱动/运行库时可能误报，
    缺 libcublas.so.12 会在转写时才崩溃，这里提前拦截。"""
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() <= 0:
            return False
    except Exception:
        return False
    import ctypes
    for lib in ("libcublas.so.12", "libcudart.so.12", "libcublasLt.so.12"):
        try:
            ctypes.CDLL(lib)
        except OSError:
            return False
    return True


def detect_best_device():
    """自动硬件加速：检测到可用 NVIDIA GPU → cuda+float16；否则 CPU+int8。
    手动关闭：.env/环境变量设 WHISPER_DEVICE=cpu（强制 CPU）；设 cuda 则强制 GPU；auto=自动探测"""
    global DEVICE, COMPUTE_TYPE, _device_explicit
    explicit = os.environ.get("WHISPER_DEVICE", "").strip().lower()
    if explicit and explicit != "auto":            # 用户已显式指定（cpu/cuda），尊重之
        _device_explicit = True
        DEVICE = explicit
        if os.environ.get("WHISPER_COMPUTE_TYPE"):
            COMPUTE_TYPE = os.environ["WHISPER_COMPUTE_TYPE"]
        return
    if _cuda_usable():
        DEVICE, COMPUTE_TYPE = "cuda", "float16"
        os.environ["WHISPER_DEVICE"] = DEVICE
        os.environ["WHISPER_COMPUTE_TYPE"] = COMPUTE_TYPE
        log("🖥️ 检测到 NVIDIA GPU，已自动启用 CUDA 加速（float16）；如需关闭：.env 设 WHISPER_DEVICE=cpu")
    else:
        log("💻 未检测到可用 GPU（或无 CUDA 运行库），使用 CPU（int8）")
        DEVICE, COMPUTE_TYPE = "cpu", "int8"
        os.environ["WHISPER_DEVICE"] = DEVICE
        os.environ["WHISPER_COMPUTE_TYPE"] = COMPUTE_TYPE


def resolve_parallel(cli_value):
    """并行路数：命令行显式 > 配置文件 BILI_PARALLEL > 自动（CPU 按核数与可用内存取小；GPU 默认单路）。
    手动关闭并行：-p 1 或 .env 设 BILI_PARALLEL=1"""
    def _auto():
        if DEVICE == "cuda":
            return 1
        cores = os.cpu_count() or 4
        n = min(4, max(1, (cores + 1) // 2))
        cap = _parallel_cap_by_memory()
        if cap < n:
            avail = mem_available_bytes()
            log(f"🧠 内存自适应：{mem_report_line()} → 并行路数由 {n} 下调为 {cap}"
                f"（每路预算 {MEM_PER_WORKER_GB:g} GB，预留 {MEM_RESERVE_GB:g} GB）")
            n = cap
        return max(1, n)

    if cli_value is not None and cli_value > 0:
        cap = _parallel_cap_by_memory()
        if cap < cli_value:
            log(f"⚠ 手动指定 {cli_value} 路，但{mem_report_line()}，建议最多 {cap} 路："
                f"内存不足时可能被系统杀掉（被杀也不会丢进度，重跑即可续传）")
        return cli_value
    cfg = os.environ.get("BILI_PARALLEL", "").strip().lower()
    if cfg and cfg != "auto":
        try:
            v = int(cfg)
            if v > 0:
                cap = _parallel_cap_by_memory()
                if cap < v:
                    log(f"⚠ BILI_PARALLEL={v} 高于内存建议值 {cap}，继续但存在被系统杀掉的风险")
                return v
        except ValueError:
            log(f"⚠ BILI_PARALLEL 值 {cfg} 无法识别，改用自动")
    return _auto()


# ============ 交互式设置中心（--config / --show-config） ============
# (key, 显示名, 类型, 说明, 枚举选项)
SETTING_ITEMS = [
    ("DEEPSEEK_API_KEY",      "DeepSeek API Key",   "secret",  "转写+AI总结模式需要；只转写可留空", None),
    ("DEEPSEEK_MODEL",        "总结主模型",         "string",  "如 deepseek-chat", None),
    ("DEEPSEEK_MODEL_FALLBACK", "总结备用模型",    "string",  "主模型失败时回退", None),
    ("WHISPER_SIZE",          "转写模型规格",       "enum",    "tiny/base/small/medium/large-v3 或本地路径",
     ["tiny", "base", "small", "medium", "large-v3"]),
    ("WHISPER_DEVICE",        "推理设备",           "enum",    "auto=自动探测GPU(推荐)；cpu=关闭GPU；cuda=强制GPU",
     ["auto", "cpu", "cuda"]),
    ("WHISPER_COMPUTE_TYPE",  "量化类型",           "enum",    "GPU 用 float16，CPU 用 int8", ["int8", "float16", "float32"]),
    ("BILI_PARALLEL",         "并行转写路数",       "parallel", "auto/N；1=关闭并行", None),
    ("BILI_WITH_TIMESTAMPS",  "生成 .srt 字幕",     "bool",    "每个视频额外输出带时间戳字幕", None),
    ("BILI_SKIP_EXISTING",    "断点续传",           "bool",    "已转写过的视频自动跳过", None),
    ("BILI_NO_SUMMARY",       "只转写不调AI",       "bool",    "无需 API Key", None),
    ("HF_ENDPOINT",           "模型镜像地址",       "string",  "留空=自动探测（大陆建议 hf-mirror.com）", None),
    ("HF_HOME",               "模型缓存位置",       "string",  "留空=~/.cache/huggingface", None),
    ("BILI_OUTPUT_DIR",       "默认输出目录",       "string",  "留空=./output", None),
    ("BILI_CHUNK_SEC",        "长音频分片秒数",     "int",     "0=关闭分片；长直播回放建议 1800（每片30分钟，内存恒定且可续传）", None),
    ("BILI_MEM_PER_WORKER_GB", "每路内存预算(GB)",  "string",  "自动并行按可用内存估算时使用，越大越保守", None),
    ("BILI_KEEP_CACHE",       "保留音频缓存",       "bool",    "1=转写完成后也保留 .bili_cache（便于重跑/排查，占磁盘）", None),
]


def _mask_key(v):
    """API Key 打码显示"""
    if not v or v.startswith("sk-xxx"):
        return "未配置"
    return v[:5] + "***" + v[-4:]


def _env_source(key):
    """配置来源: 真实环境变量 / .env 文件 / 默认值"""
    if key in _REAL_ENV:
        return "环境变量"
    env_file = os.path.join(SCRIPT_DIR, ".env")
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith(key + "="):
                    return ".env"
    return "默认"


def _display_value(item):
    """按类型显示当前生效值（含代码默认值）"""
    key, name, typ, hint, _ = item
    val = os.environ.get(key, "")
    if typ == "secret":
        return _mask_key(val or "")
    if typ == "bool":
        eff = val or "0"
        return "开 ✅" if eff.lower() in ("1", "true", "yes", "on") else "关"
    if key == "WHISPER_DEVICE":
        return val or f"自动({DEVICE})"
    if key == "BILI_PARALLEL":
        eff = val or "auto"
        return eff + (f"（→{resolve_parallel(None)}路）" if eff.lower() == "auto" else "")
    if key == "DEEPSEEK_MODEL":
        return val or "deepseek-chat"
    if key == "DEEPSEEK_MODEL_FALLBACK":
        return val or "deepseek-reasoner"
    if key == "WHISPER_SIZE":
        return val or "small"
    if key == "WHISPER_COMPUTE_TYPE":
        return val or "int8"
    if key == "BILI_OUTPUT_DIR":
        return val or "./output"
    if key == "HF_HOME":
        return val or "~/.cache/huggingface"
    if key == "HF_ENDPOINT":
        return val or "自动探测"
    return val or "（默认）"


def _edit_item(item):
    """修改单个配置项：即时写入 .env 并更新内存（含设备/并行联动）"""
    global DEVICE, COMPUTE_TYPE
    key, name, typ, hint, choices = item
    env_file = os.path.join(SCRIPT_DIR, ".env")
    cur = os.environ.get(key, "")
    print(f"\n▶ 修改「{name}」（{hint}）")
    print(f"   当前: {_display_value(item)}")
    if typ == "enum":
        for i, c in enumerate(choices, 1):
            mark = "  ← 当前" if c == cur else ""
            print(f"     {i}. {c}{mark}")
        ans = input("   选择编号，或输入自定义值，回车保留: ").strip()
        if not ans:
            return
        if ans.isdigit() and 1 <= int(ans) <= len(choices):
            val = choices[int(ans) - 1]
        else:
            val = ans
    elif typ == "bool":
        ans = input("   0=关 1=开（回车保留）: ").strip().lower()
        if not ans:
            return
        val = "1" if ans in ("1", "y", "yes", "on") else "0"
    elif typ == "parallel":
        ans = input("   auto 或数字（1=关闭并行，回车保留）: ").strip()
        if not ans:
            return
        val = ans
    else:   # secret / string
        ans = input("   输入新值（回车保留；输入 del 清空恢复默认）: ").strip()
        if not ans:
            return
        if ans.lower() == "del":
            remove_env_key(env_file, key)
            os.environ.pop(key, None)
            print(f"   ✅ {key} 已清空（恢复默认）")
            return
        val = ans
    set_env_value(env_file, key, val)
    os.environ[key] = val
    if key == "WHISPER_DEVICE":                       # 同步本会话的全局设备状态
        DEVICE = val
        COMPUTE_TYPE = "float16" if val == "cuda" else "int8"
    elif key == "WHISPER_COMPUTE_TYPE":
        COMPUTE_TYPE = val
    print(f"   ✅ {key} = {val} 已写入 {env_file}")


def settings_menu():
    """⚙️ 交互式设置中心：pi 式菜单，逐项查看/修改，即时写入 .env"""
    detect_best_device()   # 与运行态保持一致
    print("\n" + "=" * 56)
    print("⚙️  bili_summary 设置中心（修改即时写入 .env，下次运行生效）")
    print("    优先级: 命令行参数 > 真实环境变量 > .env > 默认")
    print("=" * 56)
    while True:
        print("\n当前配置：")
        for i, item in enumerate(SETTING_ITEMS, 1):
            key = item[0]
            src = _env_source(key)
            flag = "" if src == "默认" else f"  [{src}]"
            print(f"  {i:>2}. {item[1]:<18} {_display_value(item)}{flag}")
        print("  " + "-" * 46)
        print(f"  {len(SETTING_ITEMS) + 1:>2}. 💾 完成（退出）")
        try:
            ans = input("\n  输入编号修改，回车重显，q 退出: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if ans.lower() in ("q", "quit", "exit"):
            break
        if not ans:
            continue
        if ans.isdigit():
            n = int(ans)
            if 1 <= n <= len(SETTING_ITEMS):
                _edit_item(SETTING_ITEMS[n - 1])
                continue
            if n == len(SETTING_ITEMS) + 1:
                break
        print("  ⚠ 无法识别，请输入列表中的编号或 q")
    print("\n✅ 设置已保存到 .env（下次运行生效；命令行参数可临时覆盖）")


def show_config():
    """只读显示当前生效配置（--show-config）"""
    detect_best_device()
    print("\n" + "=" * 56)
    print("📋 bili_summary 当前生效配置（优先级: 环境变量 > .env > 默认）")
    print("=" * 56)
    for item in SETTING_ITEMS:
        key, name, typ, hint, _ = item
        print(f"  {name:<18} {_display_value(item):<20} [{_env_source(key)}]")
    print(f"  {'生效设备':<18} {DEVICE}/{COMPUTE_TYPE}")
    print(f"  {'并行路数':<18} {resolve_parallel(None)}")
    print(f"  {'内存':<18} {mem_report_line()}（每路预算 {MEM_PER_WORKER_GB:g} GB，预留 {MEM_RESERVE_GB:g} GB）")
    print(f"  {'长音频分片':<18} " + (f"每 {_fmt_len(CHUNK_SEC)} 一片，超过 {_fmt_len(CHUNK_THRESHOLD)} 才分片"
                                     if CHUNK_SEC > 0 else "已关闭"))
    print("=" * 56)
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
    """通用重试：失败给出明确提示并重试一次；已请求停止时不再重试"""
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            if stop_requested():                  # 安全停止中：不再重试/不死等
                raise InterruptedByUser(_stop_reason["why"] or "中断") from e
            log(f"⚠ {name} 失败（第 {attempt+1} 次）：{e}")
            if attempt < retries:
                log(f"→ {delay} 秒后重试...")
                time.sleep(delay)
            else:
                raise RuntimeError(f"{name} 多次尝试后仍失败：{e}") from e

def run_with_timeout(fn, timeout, name):
    """用 SIGALRM 兜底超时（仅防死锁，正常耗时远小于 timeout）；
    Windows 无 SIGALRM、或运行在工作线程时（信号仅主线程可用）直接执行"""
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
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
def _download_progress_hook(tag="", interval=15):
    """yt-dlp 下载进度 → 本工具日志（限频，避免并行时刷屏/后台日志爆大）"""
    last = {"t": 0.0}
    pre = f"[{tag}] " if tag else ""
    def _hook(d):
        if stop_requested():          # 安全停止：立即中断下载（.part 保留，下次续传）
            raise InterruptedByUser(_stop_reason["why"] or "中断")
        st = d.get("status")
        if st == "downloading":
            now = time.time()
            if now - last["t"] < interval:
                return
            last["t"] = now
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            pct = f"{done * 100 / total:.0f}%" if total else "?"
            speed = (d.get("speed") or 0) / 1024
            log(f"{pre}⬇️ 下载中 {pct}（{done / 1024 ** 2:.1f} MB，{speed:.0f} KB/s）")
        elif st == "finished":
            log(f"{pre}⬇️ 下载完成，转换音频格式中...")
    return _hook


def _terminate_proc(proc):
    """先 TERM 后 KILL，确保子进程（ffmpeg 等）被真正收掉"""
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    except Exception:
        pass


def run_subprocess_cancellable(cmd, timeout=1800, tag="ffmpeg"):
    """运行子进程；收到停止请求/超时立即终止（Ctrl+C 不再卡在耗时的音频转换/切分上）"""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    err = b""
    deadline = time.time() + timeout
    try:
        while True:
            try:
                _, err = proc.communicate(timeout=1)
                break
            except subprocess.TimeoutExpired:
                if stop_requested():
                    _terminate_proc(proc)
                    raise InterruptedByUser(_stop_reason["why"] or "中断")
                if time.time() > deadline:
                    _terminate_proc(proc)
                    raise RuntimeError(f"{tag} 超时（>{timeout}s）")
    finally:
        if proc.poll() is None:
            _terminate_proc(proc)
    if proc.returncode != 0:
        raise RuntimeError(f"{tag} 失败（退出码 {proc.returncode}）："
                           f"{err.decode('utf-8', 'ignore')[-300:]}")
    return proc.returncode


def download_audio(url, outdir):
    import yt_dlp
    ffmpeg = find_ffmpeg()
    log(f"🎬 开始下载音频: {url}")
    check_disk_space(outdir)
    # 不用 yt-dlp 的 FFmpegExtractAudio 后处理（需要 ffprobe），改为下载后手动转换
    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(outdir, "audio.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 5,
        "fragment_retries": 5,
        "continuedl": True,           # 断点续传：中断后 .part 文件保留，下次接着下
        "socket_timeout": 30,
        "noprogress": True,           # 进度改用 progress_hooks 限频输出
        "progress_hooks": [_download_progress_hook(vid_key(url))],
    }
    def _dl():
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return info.get("title", "untitled")
    title = with_retry(_dl, "音频下载")
    cands = [os.path.join(outdir, f) for f in os.listdir(outdir)
             if f.startswith("audio.") and not f.endswith((".part", ".ytdl"))]
    if not cands:
        raise RuntimeError("下载完成但未找到音频文件")
    audio = cands[0]
    # 用 ffmpeg 转成统一 mp3（失败则保留原格式，faster-whisper 可直接解码 m4a/aac）
    # 先写 .part 再改名：即使中途被杀/断电，也不会留下半个 mp3 被当成完整音频
    mp3 = os.path.join(outdir, "audio.mp3")
    mp3_tmp = mp3 + ".part"
    src_audio = audio
    if os.path.abspath(src_audio) != os.path.abspath(mp3):
        cmd = [ffmpeg, "-y", "-i", src_audio, "-vn", "-acodec", "libmp3lame", "-q:a", "0", mp3_tmp]
        try:
            run_subprocess_cancellable(cmd, timeout=1800, tag="转 mp3")
            os.replace(mp3_tmp, mp3)
            audio = mp3
            try:                          # 删掉源文件，避免缓存占双份磁盘
                os.remove(src_audio)
            except OSError:
                pass
        except InterruptedByUser:
            raise
        except Exception as e:
            log(f"⚠ 转 mp3 失败（{e}），直接使用原音频格式: {os.path.basename(src_audio)}")
            audio = src_audio
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

_model_local = threading.local()          # 每个线程独立的 Whisper 模型实例
_model_lock = threading.Lock()            # 模型实例创建互斥（避免并发重复下载/加载）


def load_model(cpu_threads=0):
    """加载 Whisper 模型（多线程并行时每个线程各持一个实例，共享同一份磁盘缓存）"""
    from faster_whisper import WhisperModel
    ensure_model_endpoint()
    kwargs = {"device": DEVICE, "compute_type": COMPUTE_TYPE}
    if cpu_threads:
        kwargs["cpu_threads"] = cpu_threads
    log(f"🧠 加载 Whisper {WHISPER_SIZE} 模型（{DEVICE}/{COMPUTE_TYPE}），首次使用会自动下载模型...")
    return with_retry(lambda: WhisperModel(WHISPER_SIZE, **kwargs), "模型加载")


def get_worker_model(cpu_threads=0):
    """线程内惰性加载模型（每个线程只加载一次，多视频复用）；
    并发加载前等内存（每个实例约 0.5~0.7 GB，不够时暂停，避免被系统 OOM 杀掉）"""
    m = getattr(_model_local, "model", None)
    if m is None:
        if not wait_for_memory(min(MEM_PER_WORKER_GB, 0.8), "模型加载"):
            raise InterruptedByUser(_stop_reason["why"] or "用户中断")
        with _model_lock:                       # 只串行化“创建”，各线程最终各持一个实例
            m = getattr(_model_local, "model", None)
            if m is None:
                m = load_model(cpu_threads)
                _model_local.model = m
    return m

def _fmt_srt_ts(t):
    """秒 → SRT 时间戳 HH:MM:SS,mmm"""
    ms = int(round(max(0, t) * 1000))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segs, path):
    """把带时间戳的分段写成 .srt 字幕（与纯文本 .txt 并存）"""
    lines = []
    for i, seg in enumerate(segs, 1):
        text = (seg.text or "").strip()
        if not text:
            continue
        lines += [str(i), f"{_fmt_srt_ts(seg.start)} --> {_fmt_srt_ts(seg.end)}", text, ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def append_srt(segs, offset, path, next_index=1):
    """把某个分片的分段按时间偏移追加到 .srt（分片续传时不会重写已写入的条目）"""
    lines, i = [], next_index
    for seg in segs:
        text = (seg.text or "").strip()
        if not text:
            continue
        lines += [str(i), f"{_fmt_srt_ts(seg.start + offset)} --> {_fmt_srt_ts(seg.end + offset)}", text, ""]
        i += 1
    if lines:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return i


def _fmt_len(sec):
    """秒 → 人类可读（1小时2分 / 30分 / 45秒）"""
    sec = int(round(sec))
    if sec >= 3600:
        return f"{sec // 3600}小时{(sec % 3600) // 60}分"
    if sec >= 60:
        return f"{sec // 60}分{sec % 60}秒" if sec % 60 else f"{sec // 60}分钟"
    return f"{sec}秒"


def _chunk_plan(duration, chunk_sec):
    """长音频分片计划 [(start, length), ...]；并把过短的尾巴片合并到前一片（避免末尾几秒单独成片）"""
    import math
    n = max(1, int(math.ceil(duration / chunk_sec)))
    bounds = [(i * chunk_sec, min(chunk_sec, duration - i * chunk_sec)) for i in range(n)]
    if len(bounds) >= 2 and bounds[-1][1] < min(60.0, chunk_sec * 0.25):
        s, l = bounds[-2]
        bounds[-2] = (s, l + bounds[-1][1])
        bounds.pop()
    return bounds


def audio_duration(path):
    """用 ffmpeg 读取音频时长（秒）；失败返回 None（则按不分片处理）"""
    try:
        ffmpeg = find_ffmpeg()
        p = subprocess.run([ffmpeg, "-hide_banner", "-i", path],
                           capture_output=True, text=True, timeout=120)
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", p.stderr or "")
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    except Exception as e:
        log(f"⚠ 读取音频时长失败（{e}），按不分片处理")
    return None


def cut_audio_chunk(src, start, length, dst):
    """用 ffmpeg 截取一个音频分片（16k 单声道 wav）：分片转写让内存占用恒定"""
    ffmpeg = find_ffmpeg()
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", src,
           "-vn", "-ac", "1", "-ar", "16000", dst]
    run_subprocess_cancellable(cmd, timeout=1800, tag="音频切分")
    return dst


def _json_load(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _json_store(path, data):
    """原子写入小 JSON（分片进度）"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


def _whisper_pass(m, audio_path, pass_no):
    """第1遍: 中文+VAD；第2遍: 自动语种+不过滤（兼容音乐/外语视频）"""
    if pass_no == 1:
        segs, info = m.transcribe(
            audio_path, language="zh", vad_filter=True, beam_size=5,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
    else:
        segs, info = m.transcribe(
            audio_path, language=None, vad_filter=False, beam_size=5,
            condition_on_previous_text=False,
        )
    segs = list(segs)   # 物化分段（含时间戳，供 .srt 使用）
    return "".join(s.text for s in segs).strip(), info.duration, segs


def _cuda_fallback(fn, model):
    """自动探测到 CUDA 但运行库缺失/损坏时（如缺 libcublas.so.12），降级到 CPU 重试一次。
    返回 (结果, 实际使用的模型)，供后续分片继续使用正确模型"""
    global DEVICE, COMPUTE_TYPE
    try:
        return fn(model), model
    except Exception as e:
        msg = str(e).lower()
        if (DEVICE == "cuda" and not _device_explicit
                and any(k in msg for k in ("cublas", "cudnn", "cudart", "cublaslt", "cuda"))):
            log(f"⚠️ 语音识别遇 CUDA 运行库故障（{e}），自动回退 CPU 重试...")
            DEVICE, COMPUTE_TYPE = "cpu", "int8"
            os.environ["WHISPER_DEVICE"] = DEVICE
            os.environ["WHISPER_COMPUTE_TYPE"] = COMPUTE_TYPE
            cpu_model = load_model(0)
            _model_local.model = cpu_model
            return fn(cpu_model), cpu_model
        raise


def _transcribe_once(audio_path, model):
    """单次（单片）识别：中文失败/过短时自动改用语种自动识别；返回 (text, segs, model)"""
    (text, duration, segs), model = _cuda_fallback(
        lambda m: with_retry(lambda: run_with_timeout(lambda: _whisper_pass(m, audio_path, 1),
                                                      TRANS_TIMEOUT, "语音识别"),
                             "语音识别(中文)", retries=1), model)
    min_chars = max(20, int(duration * 0.3))     # 按音频时长估算最低字数
    if len(text) < min_chars:
        log(f"⚠ 中文识别结果过短（{len(text)}字 / 音频{duration:.0f}s），尝试自动语种识别...")
        (text, duration, segs), model = _cuda_fallback(
            lambda m: with_retry(lambda: run_with_timeout(lambda: _whisper_pass(m, audio_path, 2),
                                                          TRANS_TIMEOUT, "语音识别(自动语种)"),
                                 "语音识别(自动语种)", retries=1), model)
    return text.strip(), segs, model


def _transcribe_chunked(audio_path, model, save_path, duration, with_timestamps, workdir):
    """长音频分片转写：逐片解码→识别→立即落盘（内存恒定）；中断后从下一片继续"""
    workdir = workdir or (os.path.dirname(os.path.abspath(save_path)) or ".")
    os.makedirs(workdir, exist_ok=True)
    base = os.path.basename(save_path)
    part = os.path.join(workdir, base + ".part.txt")          # 已完成分片的文字稿（追加）
    srt_part = os.path.join(workdir, base + ".part.srt")
    prog_path = os.path.join(workdir, base + ".progress.json")
    plan = _chunk_plan(duration, CHUNK_SEC)
    n_chunks = len(plan)
    log(f"🧩 长音频 {_fmt_len(duration)} → 切成 {n_chunks} 片（每片 {_fmt_len(CHUNK_SEC)}）转写："
        f"内存占用恒定，中断后可从已完成分片继续")

    prog = _json_load(prog_path) or {}
    if (prog.get("n_chunks") != n_chunks or prog.get("chunk_sec") != CHUNK_SEC
            or not os.path.exists(part)):
        prog = {"n_chunks": n_chunks, "chunk_sec": CHUNK_SEC, "done": 0, "pass": 1, "srt_index": 1}
        for p in (part, srt_part):
            if os.path.exists(p):
                os.remove(p)
    done = int(prog.get("done", 0))
    pass_no = int(prog.get("pass", 1))
    srt_index = int(prog.get("srt_index", 1))
    if done:
        log(f"📦 上次中断于第 {done}/{n_chunks} 片之后，从第 {done + 1} 片继续（免重复转写）")

    m = model
    min_chars_total = max(20, int(duration * 0.3))

    def _run_chunks(from_idx, pno):
        nonlocal m, done, srt_index
        for i in range(from_idx, n_chunks):
            if stop_requested():
                raise InterruptedByUser(_stop_reason["why"] or "用户中断")
            if not wait_for_memory(MEM_PER_WORKER_GB, "转写"):
                raise InterruptedByUser(_stop_reason["why"] or "用户中断")
            start = i * CHUNK_SEC
            length = min(CHUNK_SEC, duration - start)
            wav = os.path.join(workdir, f"chunk_{i:05d}.wav")
            log(f"  ↳ 分片 {i + 1}/{n_chunks}（{_fmt_len(start)}~{_fmt_len(start + length)}）解码 + 识别中...")
            t0 = time.time()
            cut_audio_chunk(audio_path, start, length, wav)
            try:
                text_i, segs_i, m = _transcribe_once(wav, m)
            finally:
                try:
                    os.remove(wav)
                except OSError:
                    pass
            with open(part, "a", encoding="utf-8") as f:
                f.write(text_i + "\n")
            if with_timestamps:
                srt_index = append_srt(segs_i, start, srt_part, srt_index)
            done = i + 1
            _json_store(prog_path, {"n_chunks": n_chunks, "chunk_sec": CHUNK_SEC,
                                    "done": done, "pass": pno, "srt_index": srt_index})
            log(f"  ↳ 分片 {i + 1}/{n_chunks} 完成（{len(text_i)} 字，用时 {time.time() - t0:.0f}s）；"
                f"进度已保存，中断后从这里继续")
        with open(part, "r", encoding="utf-8") as f:
            return f.read().strip()

    ok = False
    try:
        text = _run_chunks(done, pass_no if done else 1)
        if pass_no == 1 and len(text) < min_chars_total:
            log(f"⚠ 中文识别结果过短（{len(text)}字 / 音频{duration:.0f}s），改用自动语种重跑全部分片...")
            for p in (part, srt_part):
                if os.path.exists(p):
                    os.remove(p)
            done, srt_index = 0, 1
            _json_store(prog_path, {"n_chunks": n_chunks, "chunk_sec": CHUNK_SEC, "done": 0,
                                    "pass": 2, "srt_index": 1})
            text = _run_chunks(0, 2)
        ok = True
    finally:
        if ok:                                   # 中断/失败时保留 .part 与进度，供下次续传
            for p in (part, prog_path):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass
    if with_timestamps and os.path.exists(srt_part):
        os.replace(srt_part, os.path.splitext(save_path)[0] + ".srt")
    return text


def transcribe(audio_path, model, save_path, with_timestamps=False, workdir=None):
    log(f"🎧 开始语音识别（{DEVICE}/{COMPUTE_TYPE}，请耐心等待）...")
    duration = audio_duration(audio_path)
    segs = None
    if CHUNK_SEC > 0 and duration and duration > CHUNK_THRESHOLD:
        # 长音频（直播回放等）：分片转写，内存恒定、逐片落盘、可续跑
        text = _transcribe_chunked(audio_path, model, save_path, duration, with_timestamps, workdir)
    else:
        if not wait_for_memory(MEM_PER_WORKER_GB, "转写"):
            raise InterruptedByUser(_stop_reason["why"] or "用户中断")
        text, segs, model = _transcribe_once(audio_path, model)
    if len(text) < 10:
        raise RuntimeError("识别结果过短，音频可能为纯音乐或语音不清")
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)
    if with_timestamps:
        srt_path = os.path.splitext(save_path)[0] + ".srt"
        if segs is not None:
            write_srt(segs, srt_path)
        log(f"🎞 字幕已保存: {srt_path}")
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


def write_report(outdir, results, stopped=False):
    """始终生成 report.md：成功/跳过/失败/未完成清单（中断后也能反推还剩什么）"""
    ok = [r for r in results if "error" not in r and "skipped" not in r and "interrupted" not in r]
    skip = [r for r in results if r.get("skipped")]
    bad = [r for r in results if r.get("error")]
    pend = [r for r in results if r.get("interrupted")]
    lines = ["# bili_summary 处理报告", "",
             f"- 成功: {len(ok)}，跳过(已完成): {len(skip)}，失败: {len(bad)}，未完成: {len(pend)}",
             f"- 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"- 运行状态: {'已中断（可续跑）' if stopped or pend else '完成'}", ""]
    for i, r in enumerate(ok, 1):
        title = r.get("title") or r["url"]
        lines += [f"## {i}. {title}", "", f"- 链接: {r['url']}", f"- BV号: {r.get('bvid') or '-'}",
                  f"- 文字稿: `{r['transcript_path']}`"]
        if r.get("srt_path"):
            lines += [f"- 字幕: `{r['srt_path']}`"]
        if r.get("summary_path"):
            lines += [f"- 总结: `{r['summary_path']}`"]
        lines.append("")
    if skip:
        lines += ["## 已跳过（之前已完成）", ""]
        for r in skip:
            lines += [f"- {r.get('title') or r['url']}: `{r.get('transcript_path') or '-'}`", ""]
    if pend:
        lines += ["## 未完成（重跑同一条命令即可续传；已完成/已下载/已转写分片会自动跳过）", ""]
        for r in pend:
            lines += [f"- {r['url']}", ""]
    if bad:
        lines += ["## 处理失败", ""]
        for r in bad:
            lines += [f"- {r['url']}: {r['error']}", ""]
    lines += ["## 断点续传状态", "",
              f"- 状态文件: `{state_path(outdir)}`（删掉它并从 report 中移除链接即可全部重做）",
              f"- 音频缓存: `{os.path.join(outdir, CACHE_NAME)}/`（续跑免下载；成功后自动清理）", ""]
    path = os.path.join(outdir, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"📄 处理报告已写入: {path}")

# ============ 6. 单视频处理 ============
def _cached_audio(workdir):
    """缓存目录里已就绪的音频（跳过下载用）"""
    try:
        cands = [os.path.join(workdir, f) for f in os.listdir(workdir)
                 if f.startswith("audio.") and not f.endswith((".part", ".ytdl"))]
    except OSError:
        return None
    cands = [p for p in cands if os.path.getsize(p) > 32 * 1024]
    if not cands:
        return None
    if os.path.exists(os.path.join(workdir, "audio.mp3")):    # 优先用转好的 mp3
        return os.path.join(workdir, "audio.mp3")
    return max(cands, key=os.path.getsize)


def _cache_meta(workdir, **kw):
    """缓存元信息（标题/链接）：中断后重跑不必重新联网取标题"""
    p = os.path.join(workdir, "meta.json")
    if kw:
        data = _json_load(p) or {}
        data.update(kw)
        _json_store(p, data)
        return data
    return _json_load(p) or {}


def download_one(url, outdir, key=None):
    """下载音频到缓存目录，返回 (缓存目录, 音频路径, 标题)。
    已下载过（哪怕上次只下了一半）则直接复用/续传，失败时不删目录"""
    key = key or vid_key(url)
    tmpdir = workdir_for(outdir, key)
    os.makedirs(tmpdir, exist_ok=True)
    cached = _cached_audio(tmpdir)
    if cached:
        title = (_cache_meta(tmpdir).get("title")) or url
        log(f"📦 复用上次已下载的音频（免重新下载）: {os.path.basename(cached)}，标题: {title}")
        return tmpdir, cached, title
    log(f"🎯 目标: {url}（BV号: {extract_bvid(url) or '无'}）")
    try:
        audio, title = download_audio(url, tmpdir)
        _cache_meta(tmpdir, url=url, title=title, audio=os.path.basename(audio))
        return tmpdir, audio, title
    except Exception:
        # 保留 .part / 已下好的音频 + 分片进度，下次重跑可续传；目录为空才删
        try:
            if not os.listdir(tmpdir):
                os.rmdir(tmpdir)
        except OSError:
            pass
        raise


def process_one(url, outdir, tmpdir, audio, title, summarize=True, used_names=None,
                name_lock=None, with_timestamps=False, cpu_threads=0):
    """对已下载音频执行 转写→(可选)总结，返回结果字典；失败抛异常由调用方捕获"""
    bvid = extract_bvid(url)
    key = vid_key(url)
    if name_lock is not None:
        with name_lock:
            fname = make_output_name(title, key, used_names)   # 文件名=视频标题
    else:
        fname = make_output_name(title, key, used_names)
    model = get_worker_model(cpu_threads)          # 每个线程复用自己已加载的模型实例
    os.makedirs(outdir, exist_ok=True)
    tpath = os.path.join(outdir, f"{fname}.txt")   # 扁平输出：每视频一个纯文本文件
    _cache_meta(tmpdir, title=title, transcript=tpath)
    transcript = transcribe(audio, model, tpath, with_timestamps=with_timestamps,
                            workdir=tmpdir)        # 分片进度也放在缓存目录，续跑不丢
    res = {"url": url, "bvid": bvid, "title": title,
           "key": key, "name": fname,
           "transcript_path": tpath, "transcript": transcript}
    if with_timestamps:
        res["srt_path"] = os.path.splitext(tpath)[0] + ".srt"
    if summarize:
        summary = summarize_all(transcript)
        spath = os.path.join(outdir, f"{fname}.summary.txt")
        with open(spath, "w", encoding="utf-8") as f:
            f.write(summary)
        res.update({"summary": summary, "summary_path": spath})
    return res


def finish_one(url, outdir, tmpdir, audio, title, **kw):
    """包装 process_one：成功后清理缓存目录；失败/中断则保留，供重跑续传"""
    ok = False
    try:
        res = process_one(url, outdir, tmpdir, audio, title, **kw)
        ok = True
        return res
    finally:
        if ok and not KEEP_CACHE:
            shutil.rmtree(tmpdir, ignore_errors=True)
            log("🧹 临时音频与分片缓存已清理")
        elif ok:
            log(f"📦 已保留音频缓存（BILI_KEEP_CACHE=1）: {tmpdir}")
        else:
            log(f"📦 音频与分片进度已保留: {tmpdir}（重跑会自动续传，不再重新下载/转写）")

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


def remove_env_key(path, key):
    """从 .env 中删除某个键（保留其他行）"""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(ln for ln in lines if not ln.strip().startswith(key + "="))


# ============ 注册全局命令 bili-summary（--setup 向导内调用） ============
def register_cli_cmd():
    """把 bili-summary 注册为全局命令（之后直接敲 bili-summary 即可）：
    ① 在本项目 .venv 内 pip install -e . --no-deps 生成启动器（不复制代码、不装到系统）
    ② Linux/macOS: 软链接到 ~/.local/bin/bili-summary；Windows: 生成 bili-summary.bat 并加入用户 PATH
    ③ 会经用户确认后才修改用户环境（~/.local/bin 或用户 PATH）"""
    print("\n⌨️ 注册全局命令 bili-summary（之后直接敲 bili-summary，不再需要 python bili_summary.py）")
    try:
        ans = input("   现在注册吗？[y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans = ""
    if ans not in ("y", "yes"):
        print("   已跳过（以后可重跑 --setup 注册；或手动见 README 方式 B2）")
        return
    vpy = _venv_python(VENV_DIR)
    if not vpy:
        print("   ⚠ 未找到虚拟环境 .venv，请先正常运行一次脚本后再注册")
        return
    print("   ① 在 .venv 内安装启动器（pip install -e . --no-deps）...")
    try:
        subprocess.run([vpy, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"], check=True)
        print("   ✅ 启动器已生成")
    except Exception as e:
        print(f"   ❌ 启动器安装失败: {e}")
        print("      可手动执行: .venv/bin/pip install -e . --no-deps（Windows: .venv\\Scripts\\pip install -e . --no-deps）")
        return
    if sys.platform == "win32":
        _register_win32_cli()
    else:
        _register_unix_cli()


def _register_unix_cli():
    """Linux/macOS: 软链接 .venv/bin/bili-summary -> ~/.local/bin/"""
    bin_dir = os.path.expanduser("~/.local/bin")
    os.makedirs(bin_dir, exist_ok=True)
    src = os.path.join(VENV_DIR, "bin", "bili-summary")
    dst = os.path.join(bin_dir, "bili-summary")
    try:
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
        print(f"   ✅ 已创建软链接: {dst}")
        if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
            print(f"   ⚠ 当前 PATH 不含 {bin_dir}，先加入一次（以后无需再改）:")
            print(f"     echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> ~/.bashrc && source ~/.bashrc")
        print("   ✅ 完成！新开终端直接输入 bili-summary 即可")
    except Exception as e:
        print(f"   ❌ 创建软链接失败: {e}")
        print("      可手动: ln -sf <项目目录>/.venv/bin/bili-summary ~/.local/bin/bili-summary")


def _register_win32_cli():
    """Windows: 生成 bili-summary.bat 并把项目目录加入用户 PATH（不碰系统级 PATH）"""
    project = SCRIPT_DIR
    bat = os.path.join(project, "bili-summary.bat")
    try:
        with open(bat, "w", encoding="ascii") as f:
            f.write('@echo off\r\n"%%~dp0.venv\\Scripts\\bili-summary.exe" %%*\r\n')
        print(f"   ✅ 已生成 {bat}")
    except Exception as e:
        print(f"   ❌ 生成 bili-summary.bat 失败: {e}")
        return
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0,
                             winreg.KEY_READ | winreg.KEY_SET_VALUE)
        try:
            cur, _ = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            cur = ""
        parts = [p for p in cur.split(";") if p.strip()]
        if project in parts:
            print("   ✅ 项目目录已在用户 PATH 中")
        else:
            parts.append(project)
            winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, ";".join(parts))
            try:
                import ctypes
                ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x001A, 0, "Environment", 0, 1000, None)
            except Exception:
                pass
            print(f"   ✅ 已将 {project} 加入用户 PATH（新开 cmd 生效）")
        winreg.CloseKey(key)
        print("   ✅ 完成！新开 cmd 直接输入 bili-summary 即可")
    except Exception as e:
        print(f"   ⚠ 自动加入用户 PATH 失败: {e}")
        print("      请手动: 设置 → 系统 → 关于 → 高级系统设置 → 环境变量 → 用户变量 Path → 新建 → 粘贴:")
        print(f"      {project}")


def _hf_reachable():
    import urllib.request
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=5)
        return True
    except Exception:
        return False


def _is_local_model_path(s):
    """判断 WHISPER_SIZE 是否指向本地模型目录（而非 tiny/base/... 内置规格名）"""
    s = (s or "").strip()
    if not s:
        return False
    p = os.path.expanduser(s)
    if os.path.isdir(p) or os.path.exists(p):
        return True
    if s.startswith((".", "/", "\\", "~")) or re.match(r"^[A-Za-z]:[\\/]", s):
        return True
    return False


def setup_wizard(check_only=False):
    """一键配置向导：模型规格/缓存位置/API Key/镜像/预下载。只动项目目录与用户缓存，不碰系统"""
    global WHISPER_SIZE
    print("\n" + "=" * 56)
    print("bili_summary 一键配置向导（默认只动本项目目录；注册全局命令时会经你确认）")
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

    # 2) 硬件加速（GPU 自动 / 强制 CPU）
    print("\n🖥️ 硬件加速（GPU 默认自动开启）")
    print(f"   当前: 自动（{DEVICE}/{COMPUTE_TYPE}）")
    print("   [1] 自动（推荐）：有 NVIDIA GPU 用 CUDA+float16，否则 CPU+int8")
    print("   [2] 强制 CPU（关闭 GPU 加速）")
    try:
        ans = input("   请选择 [1/2]，回车保留: ").strip()
    except (EOFError, KeyboardInterrupt):
        ans = ""
    if ans == "2":
        set_env_value(env_file, "WHISPER_DEVICE", "cpu")
        set_env_value(env_file, "WHISPER_COMPUTE_TYPE", "int8")
        print("✅ 已关闭 GPU 加速（WHISPER_DEVICE=cpu 写入 .env）")
    elif ans == "1":
        remove_env_key(env_file, "WHISPER_DEVICE")
        remove_env_key(env_file, "WHISPER_COMPUTE_TYPE")
        print("✅ 已设为自动（下次运行重新探测 GPU）")

    # 3) 并行转写路数
    print("\n⚙️ 并行转写路数（默认自动开启）")
    print("   自动：CPU 按核数（上限4路），GPU 默认单路；输入 1 即关闭并行")
    cur_p = os.environ.get("BILI_PARALLEL", "auto")
    try:
        ans = input(f"   当前 [{cur_p}]，回车保留，或输入数字: ").strip()
    except (EOFError, KeyboardInterrupt):
        ans = ""
    if ans:
        set_env_value(env_file, "BILI_PARALLEL", ans)
        print(f"✅ BILI_PARALLEL = {ans}（已写入 {env_file}）")

    # 4) Whisper 模型来源（大文件，明确询问：在线下载 / 本地已安装）
    print("\n🧠 Whisper 转写模型（核心大文件，约数百 MB）")
    print("   [1] 在线下载：自动从 HuggingFace/镜像拉取（规格与存放位置下面两步选）")
    print("   [2] 使用本地已安装模型：填目录路径直接调用，不下载")
    try:
        src = input("   请选择 [1/2]，回车=在线下载: ").strip()
    except (EOFError, KeyboardInterrupt):
        src = ""
    if src == "2":
        path = input("   输入本地模型目录绝对路径（如 D:\\models\\whisper-large-v3）: ").strip()
        if path:
            WHISPER_SIZE = path
            os.environ["WHISPER_SIZE"] = path
            set_env_value(env_file, "WHISPER_SIZE", path)
            print(f"✅ 已设置 WHISPER_SIZE = {path}（转写时直接加载，不再下载）")
        else:
            print("⚠️ 未输入路径，回退为在线下载（下一步选规格）")
            src = ""
    if src != "2":
        print("   在线下载规格: tiny / base / small / medium / large-v3（越大越准、越慢、越吃内存）")
        cur_size = os.environ.get("WHISPER_SIZE", "small")
        try:
            ans = input(f"   当前 [{cur_size}]，回车保留，或输入新规格: ").strip()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        if ans:
            WHISPER_SIZE = ans
            os.environ["WHISPER_SIZE"] = ans
            set_env_value(env_file, "WHISPER_SIZE", ans)
            print(f"✅ WHISPER_SIZE = {ans}（已写入 {env_file}）")
        else:
            print(f"✅ 保留 WHISPER_SIZE = {cur_size}")

    # 5) 模型缓存位置（仅在线下载需要：默认 / 项目内 / 自定义）
    print("\n📂 模型缓存位置（在线下载的模型存放目录）")
    print("   [1] 用户缓存（默认）: ~/.cache/huggingface —— 跨项目复用")
    print("   [2] 项目内: models/ —— 随项目走，拷项目即带走模型")
    print("   [3] 自定义目录")
    cur_hf = os.environ.get("HF_HOME")
    if cur_hf:
        print(f"   当前: {cur_hf}")
    try:
        ans = input("   选择 [1/2/3]，回车保留当前: ").strip()
    except (EOFError, KeyboardInterrupt):
        ans = ""
    if ans == "1":
        remove_env_key(env_file, "HF_HOME")
        os.environ.pop("HF_HOME", None)
        print("✅ 使用默认缓存 ~/.cache/huggingface（跨项目复用）")
    elif ans == "2":
        val = os.path.join(SCRIPT_DIR, "models")
        set_env_value(env_file, "HF_HOME", val)
        os.environ["HF_HOME"] = val
        print(f"✅ 模型将缓存到项目内: {val}")
    elif ans == "3":
        val = input("   输入缓存目录绝对路径（如 D:\\models_cache）: ").strip()
        if val:
            set_env_value(env_file, "HF_HOME", val)
            os.environ["HF_HOME"] = val
            print(f"✅ HF_HOME = {val}（已写入 {env_file}）")
        else:
            print("⚠️ 未输入，保留当前缓存位置")
    else:
        print("   保留当前缓存位置")

    # 6) DeepSeek API Key（仅总结模式需要）
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

    # 7) 模型下载镜像地址
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

    # 8) 模型预下载（本地已安装模型则验证跳过；在线规格才询问下载）
    if _is_local_model_path(WHISPER_SIZE):
        p = os.path.expanduser(WHISPER_SIZE)
        if os.path.isdir(p):
            print(f"✅ 使用本地已安装模型: {p}（目录存在，转写时直接加载，无需下载）")
        else:
            print(f"⚠️ 本地模型路径不存在: {p}，请确认（转写时会报错；可重跑 --setup 改回在线下载）")
    else:
        print(f"\n⬇️ 预下载 Whisper {WHISPER_SIZE} 模型（约数百 MB；不下载则首次转写时自动下载）")
        try:
            ans = input("   现在下载吗？[y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        if ans in ("y", "yes"):
            ensure_model_endpoint()
            print("   下载中，请耐心等待...")
            try:
                load_model()
                print(f"✅ Whisper {WHISPER_SIZE} 模型就绪（存放于 {os.environ.get('HF_HOME', '~/.cache/huggingface')}）")
            except Exception as e:
                print(f"⚠️ 模型预下载失败: {e}（可重跑 --setup 或首次转写时自动下载）")
        else:
            print("⚠️ 跳过模型预下载（首次转写时会自动下载）")

    # 9) 注册全局命令 bili-summary（可选，经确认才改用户环境）
    register_cli_cmd()

    print("\n" + "=" * 56)
    print("✅ 配置完成！使用示例:")
    print("   bili-summary -i                                     # 交互模式（已注册全局命令时）")
    print("   python bili_summary.py -i                           # 未注册时也可这样运行")
    print("   bili-summary BV1GJ411x7h7 --no-summary -o out        # 只转写")
    print("=" * 56)


# ============ 主流程 ============
class _HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """帮助格式：保留原始换行 + 显示参数默认值（隐藏 None 默认）"""
    def _get_help_string(self, action):
        help_text = super()._get_help_string(action)
        if action.default is None:
            help_text = help_text.replace(" (default: %(default)s)", "")
        return help_text


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
            "  · 中断保护: 内存自适应并行 + 长音频分片 + 断点续传（重跑自动续）+ Ctrl+C/关终端优雅停止\n"
            "  · 自动清理: 交互生成的 links.txt 与运行产生的 __pycache__ 退出时自动删除\n"
            "  · 首次运行自动创建 .venv 并安装依赖，拷贝即用"
        ),
        epilog=(
            "示例:\n"
            "  python bili_summary.py --setup                        # 一键配置（API Key/镜像/模型预下载）\n"
            "  python bili_summary.py -i                             # 交互模式：逐行输入链接，空行开始处理\n"
            "  python bili_summary.py BV1GJ411x7h7                   # 单个视频（默认: 转写 + AI 总结）\n"
            "  python bili_summary.py -f links.txt                   # 从文件读取链接（最推荐：中断后重跑即可续）\n"
            "  python bili_summary.py --no-summary -f links.txt -o transcripts   # 只转写\n"
            "  python bili_summary.py --background -f links.txt      # 后台跑，关终端不中断\n"
            "\n"
            "中断保护（避免白跑）:\n"
            "  · 断点续传：每个视频完成后立即写 output/.bili_state.json；重跑同一条命令\n"
            "    已完成的秒级跳过、下到一半的音频继续下、长音频已转写的分片不重做\n"
            "  · 内存自适应：按可用内存自动限制并行路数，内存告急时暂停接新任务\n"
            "  · 长音频分片：超过阈值的长音频自动分片转写，内存恒定；直播回放不再被 OOM 杀死\n"
            "  · 优雅停止：Ctrl+C / kill / 关终端（SIGINT/SIGTERM/SIGHUP）只停止接新任务，\n"
            "    当前分片收尾后退出，进度/缓存全部保留（再按一次 Ctrl+C 立即退出）\n"
            "  · 全量重做：--force（或删掉 .bili_state.json）\n"
            "\n"
            "输出结构:\n"
            "  output/<视频标题>.txt            完整文字稿（文件名=视频标题，去非法字符/截断80字）\n"
            "  output/<视频标题>.summary.txt    每段视频的 AI 总结（仅开启总结时）\n"
            "  output/summaries.md              全部视频汇总（仅开启总结时）\n"
            "  output/report.md                 处理报告（含未完成清单，始终生成）\n"
            "  output/.bili_state.json          断点续传状态（已完成清单）\n"
            "  output/.bili_cache/<BV号>/       音频与分片缓存（续跑免下载，成功后自动删）\n"
            "\n"
            "配置（优先级: 环境变量 > 项目目录 .env > 默认值）:\n"
            "  DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL / DEEPSEEK_MODEL_FALLBACK\n"
            "  WHISPER_SIZE / WHISPER_DEVICE / WHISPER_COMPUTE_TYPE / HF_ENDPOINT / BILI_OUTPUT_DIR\n"
            "  BILI_PARALLEL / BILI_SKIP_EXISTING / BILI_WITH_TIMESTAMPS / BILI_NO_SUMMARY\n"
            "  BILI_CHUNK_SEC / BILI_CHUNK_THRESHOLD / BILI_MEM_PER_WORKER_GB / BILI_KEEP_CACHE\n"
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
    ap.add_argument("--no-summary", action="store_true", default=None,
                    help="只转写，不调用 DeepSeek 总结（无需 API Key；输出为纯文本文件；未指定时读配置 BILI_NO_SUMMARY）")
    ap.add_argument("--keep-links", action="store_true",
                    help="交互模式结束后保留 links.txt（默认自动删除）")
    ap.add_argument("-p", "--parallel", type=int, default=None, metavar="N",
                    help="并行处理线程数（默认自动：CPU 按核数上限4，GPU 单路；手动关闭用 -p 1，配置键 BILI_PARALLEL）")
    ap.add_argument("--skip-existing", action="store_true", default=None,
                    help="断点续传：输出目录已有同名转写文件时直接跳过（仅请求一次元数据，不下载；配置键 BILI_SKIP_EXISTING）")
    ap.add_argument("--with-timestamps", action="store_true", default=None,
                    help="额外生成带时间戳的 .srt 字幕文件（与纯文本 .txt 并存；配置键 BILI_WITH_TIMESTAMPS）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只预览：显示归一化 URL、抓取视频标题与输出文件名（不下载/不转写/不调AI）")
    ap.add_argument("--setup", action="store_true",
                    help="一键配置向导：硬件/并行/模型/API Key/镜像/预下载/注册 bili-summary 命令（--setup --check 只读体检）")
    ap.add_argument("--config", action="store_true",
                    help="交互式设置中心：菜单式查看/修改所有配置（即时写入 .env）")
    ap.add_argument("--show-config", action="store_true",
                    help="只读显示当前生效的配置（来源: 环境变量/.env/默认）")
    ap.add_argument("--check", action="store_true", help="只读环境检查（配合 --setup 使用）")
    ap.add_argument("-b", "--background", action="store_true",
                    help="后台运行：放到独立会话（关终端/断 SSH 不中断），日志写入 logs/*.log")
    ap.add_argument("--force", action="store_true",
                    help="忽略断点续传状态，全部重做（默认已完成/已下载/已转写分片自动跳过）")
    ap.add_argument("--chunk-sec", type=int, default=None, metavar="N",
                    help="长音频分片秒数（0=关闭；默认 1800=每片30分钟，仅长音频分片；配置键 BILI_CHUNK_SEC）")
    ap.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    return ap.parse_args(argv)

def results_from_state(links, state):
    """从状态文件重建结果列表（用于运行时意外异常/中断后仍能生成完整报告）"""
    out = []
    for raw in links:
        url = normalize_url(raw)
        key = vid_key(url)
        rec = state.get(key) or {}
        st = rec.get("status")
        if st == "done" and rec.get("transcript"):
            out.append({"url": url, "key": key, "title": rec.get("title") or url,
                        "skipped": True, "transcript_path": rec.get("transcript")})
        elif st == "failed":
            out.append({"url": url, "key": key, "title": rec.get("title") or url,
                        "error": rec.get("error") or "失败"})
        else:
            out.append({"url": url, "key": key, "interrupted": True})
    return out


def run_pipeline(links, outdir, args, state):
    """流水线并行：下载线程池预取音频 → N 个处理线程转写/总结。
    - 默认 N=1：转写仍顺序进行，但下载提前预取（下载与转写重叠，不再“下完一个才下下一个”）
    - N>1：多个视频同时转写（每线程独立模型实例，CPU 线程数自动分摊）
    - 断点续传：已完成的秒级跳过；中断时保留音频与分片进度，重跑可续
    - 失败隔离：单个视频下载/转写失败不影响其他视频"""
    import concurrent.futures as cf
    n = max(1, args.parallel or 1)
    cpu_threads = 0
    if n > 1 and DEVICE == "cpu":
        cores = os.cpu_count() or 4
        cpu_threads = max(1, cores // n)
        log(f"⚙️ 并行 {n} 路转写：每路 CPU {cpu_threads} 线程（共 {cores} 核）")
    elif n > 1 and DEVICE != "cpu":
        log(f"⚠️ GPU 设备并行 {n} 路会成倍占用显存，建议 --parallel 1")

    used_names = set()                 # 输出文件名去重（标题相同则加 _2/_3）
    name_lock = threading.Lock()
    results = {}
    q = queue.Queue(maxsize=n + 2)     # 限制在途下载数量，形成背压

    def _mark(key, **rec):
        """更新断点续传状态并立即落盘（被杀也只会丢正在跑的那一个）"""
        update_state(outdir, state, key, **rec)

    def dl_worker(idx, url):
        key = vid_key(url)
        try:
            if stop_requested():
                raise InterruptedByUser(_stop_reason["why"] or "中断")
            tmpdir, audio, title = download_one(url, outdir, key)
            q.put((idx, url, {"tmpdir": tmpdir, "audio": audio, "title": title}))
        except InterruptedByUser as e:
            q.put((idx, url, {"interrupted": str(e)}))
        except Exception as e:
            _mark(key, status="failed", url=url, error=str(e))
            q.put((idx, url, {"error": str(e)}))

    def tx_worker():
        while True:
            item = q.get()
            idx, url = -1, ""
            try:
                if item is None:       # 结束哨兵
                    break
                idx, url, payload = item
                key = vid_key(url)
                if "error" in payload:
                    results[idx] = {"url": url, "key": key, "error": payload["error"]}
                    log(f"❌ 下载失败 {url}: {payload['error']}")
                    continue
                if "interrupted" in payload or stop_requested():
                    results[idx] = {"url": url, "key": key, "interrupted": True}
                    _mark(key, status="interrupted", url=url)
                    log(f"⏸ 跳过（已请求停止）: {url}")
                    continue
                res = finish_one(url, outdir, payload["tmpdir"], payload["audio"], payload["title"],
                                 summarize=not args.no_summary, used_names=used_names,
                                 name_lock=name_lock, with_timestamps=args.with_timestamps,
                                 cpu_threads=cpu_threads)
                results[idx] = res
                _mark(key, status="done", title=payload["title"], url=url,
                      transcript=res["transcript_path"])
                with _print_lock:
                    print("\n" + "=" * 56)
                    print(f"📝 {'AI 总结' if 'summary' in res else '转写结果'} {idx + 1}/{len(links)}")
                    print("=" * 56)
                    if "summary" in res:
                        print(res["summary"])
                    else:
                        print(f"文字稿已保存: {res['transcript_path']}（共 {len(res['transcript'])} 字）")
                    print("=" * 56)
            except InterruptedByUser as e:
                key = vid_key(url)
                results[idx] = {"url": url, "key": key, "interrupted": True}
                _mark(key, status="interrupted", url=url)
                log(f"⏸ 中断（音频与分片进度已保留，重跑续传）: {url}　{e}")
            except Exception as e:
                key = vid_key(url)
                results[idx] = {"url": url, "key": key, "error": str(e)}
                _mark(key, status="failed", url=url, error=str(e))
                log(f"❌ 处理失败 {url}: {e}")
            finally:
                q.task_done()

    # 先启动处理线程（消费队列），再提交下载任务，避免队列满时下载线程阻塞成死锁
    workers = [threading.Thread(target=tx_worker, daemon=True) for _ in range(n)]
    for w in workers:
        w.start()

    n_dl = min(len(links), max(2, n + 1))          # 下载线程略多于处理线程，保证预取
    futs = []
    dl_pool = cf.ThreadPoolExecutor(max_workers=n_dl)
    try:
        for idx, raw in enumerate(links):
            url = normalize_url(raw)
            key = vid_key(url)
            if not args.force and resume_ok(state, key, outdir):    # 断点续传：秒级跳过（不联网）
                rec = state.get(key) or {}
                results[idx] = {"url": url, "key": key, "title": rec.get("title") or url,
                                "skipped": True, "transcript_path": rec.get("transcript")}
                log(f"⏭ 已完成，秒级跳过（重做加 --force）: {rec.get('title') or url}")
                continue
            if args.skip_existing:
                title = fetch_title(url)           # 仅元数据请求，不下载
                fname = sanitize_title(title) if title else None
                cand = os.path.join(outdir, (fname or key) + ".txt")
                if os.path.exists(cand):
                    results[idx] = {"url": url, "key": key, "title": title or url,
                                    "skipped": True, "transcript_path": cand}
                    _mark(key, status="done", title=title or url, url=url, transcript=cand)
                    log(f"⏭ 已存在，跳过: {cand}")
                    continue
            if stop_requested():                   # 已请求停止：不再提交新任务
                results[idx] = {"url": url, "key": key, "interrupted": True}
                continue
            log(f"\n===== [{idx + 1}/{len(links)}] 开始处理 {url} ===")
            futs.append(dl_pool.submit(dl_worker, idx, url))
        cf.wait(futs)
    finally:
        for _ in range(n):                         # 下载全部入队后发结束哨兵
            q.put(None)
        dl_pool.shutdown()
    for w in workers:
        w.join()
    return [results.get(i, {"url": links[i], "key": vid_key(links[i]), "interrupted": True})
            for i in range(len(links))]


def report(outdir, results, stopped=False):
    ok = [r for r in results if "error" not in r and "skipped" not in r and "interrupted" not in r]
    skip = [r for r in results if r.get("skipped")]
    bad = [r for r in results if r.get("error")]
    pend = [r for r in results if r.get("interrupted")]
    summary_mode = any("summary" in r for r in ok)
    print("\n" + "#" * 56)
    print(f"📊 汇总报告: 成功 {len(ok)}，跳过(已完成) {len(skip)}，失败 {len(bad)}，"
          f"未完成 {len(pend)} / 共 {len(results)}")
    print("#" * 56)
    for i, r in enumerate(ok, 1):
        title = r.get("title") or r["url"]
        print(f"  ✅ [{i}] {title}")
        print(f"      ├ 文字稿: {r['transcript_path']}")
        if "srt_path" in r:
            print(f"      ├ 字幕:   {r['srt_path']}")
        if "summary" in r:
            print(f"      └ 总结:   {r['summary_path']}")
    for r in skip:
        print(f"  ⏭ {r.get('title') or r['url']}: 已完成，跳过")
    for r in bad:
        print(f"  ❌ {r['url']}: {r['error']}")
    for r in pend:
        print(f"  ⏳ {r['url']}: 未完成（重跑会自动续传）")
    if summary_mode:
        print(f"  📄 全量汇总: {os.path.join(outdir, 'summaries.md')}")
    print(f"  📄 处理报告: {os.path.join(outdir, 'report.md')}")
    print(f"  🧾 断点状态: {state_path(outdir)}")
    print(f"  📁 输出目录: {outdir}")
    if pend or stopped:
        print("\n" + "-" * 56)
        print("⏳ 还有未完成的视频：直接重跑同一条命令即可续传")
        print("   （已完成/已下载/已转写分片都会自动跳过，加 --force 可全部重做）")
        print("-" * 56)

def spawn_background():
    """把当前任务重新放到后台独立会话运行（关终端/断 SSH/挂断都不会中断）。
    输出（含 yt-dlp 进度）统一写入 logs/bili-summary-<时间>.log"""
    logdir = os.path.join(SCRIPT_DIR, "logs")
    os.makedirs(logdir, exist_ok=True)
    logpath = os.path.join(logdir, f"bili-summary-{time.strftime('%Y%m%d-%H%M%S')}.log")
    argv = [sys.executable, os.path.abspath(__file__)]
    argv += [a for a in sys.argv[1:] if a not in ("--background", "-b")]
    env = dict(os.environ, BILI_BG_CHILD="1")
    try:
        logf = open(logpath, "ab")
    except OSError as e:
        log(f"❌ 无法创建日志文件 {logpath}: {e}")
        sys.exit(1)
    with logf:
        proc = subprocess.Popen(argv, stdout=logf, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True,
                                cwd=os.getcwd(), env=env)
    print(f"🚀 已在后台启动（PID {proc.pid}，独立会话：关终端/断 SSH 不中断）")
    print(f"📄 日志: {logpath}")
    print(f"👀 跟踪: tail -f '{logpath}'")
    print(f"⛔ 停止: kill {proc.pid}（优雅停止，进度会保留；再跑同样命令续传）")
    print(f"🧾 状态: {os.path.join(os.path.abspath(os.environ.get('BILI_OUTPUT_DIR', 'output')), STATE_NAME)}")


def main():
    global CHUNK_SEC
    args = parse_args(sys.argv[1:])
    # --background：自己转到后台再退出（子进程带 BILI_BG_CHILD 标记，不会递归）
    if args.background and not os.environ.get("BILI_BG_CHILD"):
        if not (args.inputs or args.file or args.interactive):
            print("❌ --background 需配合链接或 -f links.txt 使用（后台无法交互输入）")
            sys.exit(1)
        spawn_background()
        sys.exit(0)
    install_signal_handlers()     # Ctrl+C / kill / 关终端 → 优雅停止，进度不丢
    detect_best_device()          # GPU 自动加速（.env 设 WHISPER_DEVICE=cpu 可关闭）
    # 配置优先级：命令行显式 > .env/环境变量 > 自动默认（GPU/并行默认全自动开启）
    args.parallel = resolve_parallel(args.parallel)
    args.no_summary = args.no_summary if args.no_summary is not None else cfg_flag("BILI_NO_SUMMARY")
    args.skip_existing = args.skip_existing if args.skip_existing is not None else cfg_flag("BILI_SKIP_EXISTING")
    args.with_timestamps = args.with_timestamps if args.with_timestamps is not None else cfg_flag("BILI_WITH_TIMESTAMPS")
    if args.chunk_sec is not None:
        CHUNK_SEC = max(0, args.chunk_sec)
    if args.setup:
        setup_wizard(check_only=args.check)
        sys.exit(0)
    if args.show_config:
        show_config()
        sys.exit(0)
    if args.config:
        settings_menu()
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
        cleanup_stale_tempdirs()          # 清理旧版崩溃残留的临时目录
        start_memory_watchdog()           # 内存告急提前告警（避免直接被 OOM 杀）
        state = load_state(outdir)        # 断点续传状态
        mode = "转写 + AI 总结" if not args.no_summary else "仅转写（不调用 AI）"
        log(f"共 {len(links)} 个视频待处理，模式: {mode}，输出目录: {outdir}")
        log(f"🛡️ 中断保护: {mem_report_line()}；并行 {args.parallel} 路；"
            + (f"长音频按 {_fmt_len(CHUNK_SEC)} 分片转写" if CHUNK_SEC > 0 else "长音频不分片")
            + "；已完成/已下载/已转写分片自动跳过（--force 可全部重做）")
        if not args.force:
            done_before = sum(1 for raw in links
                              if resume_ok(state, vid_key(normalize_url(raw)), outdir))
            if done_before:
                log(f"⏭ 其中 {done_before} 个已完成，将秒级跳过")

        results = None
        stopped = False
        try:
            results = run_pipeline(links, outdir, args, state)   # 流水线：下载预取 + N 路并行转写
        except Exception as e:                                   # 意外异常也不丢报告与进度
            stopped = stop_requested()
            log(f"❌ 运行中断（{e}）；已完成进度已保存在状态文件中，重跑可续传")
        finally:
            stopped = stopped or stop_requested()
            results = results or results_from_state(links, state)
            if not args.no_summary:
                write_combined(outdir, results)
            prune_cache_dir(outdir)           # 顺手清掉已完成的空缓存目录
            write_report(outdir, results, stopped=stopped)
            report(outdir, results, stopped=stopped)
        if stopped:
            log("🛑 已安全停止（进度与音频/分片缓存均已保存）")
            log("▶ 续跑: 重新执行同一条命令即可自动续传（无需加任何参数）")
            sys.exit(130)
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
