"""远端媒体处理：直接对 B站 DASH 地址做 HTTP Range 抽取，不落地完整视频。

关键点
------
* B站视频 CDN（bilivideo.com）强制校验 `Referer`，缺了就是 403。
  这里统一注入 `Referer` + `User-Agent`，由本机的 ffmpeg 去拉，所以稳定可用。
* 抽帧用输入侧 `-ss`（seek），ffmpeg 只会 Range 请求目标位置附近的数据，
  单帧通常 20~40KB；16 分钟视频抽 8 帧实测约 5 秒。
* 音频按需转码成 16k 单声道 mp3，实测 ~97 倍速，比整段下载快一个量级。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from .bili import REFERER, Stream

STOP = threading.Event()

# 全局网络并发闸门：不管是几个视频、多少线程，同时打到 B站 CDN 的连接数封顶。
# 实测 30 条并发会把 CDN 打崩（821 KB/s → 23 KB/s），所以这是必须的。
_CONN_SEM = threading.BoundedSemaphore(8)


def configure_global_limit(n: int) -> None:
    """设置全局网络并发上限（在所有视频之间共享）。"""
    global _CONN_SEM
    _CONN_SEM = threading.BoundedSemaphore(max(1, int(n)))


def _conn_slot():
    return _CONN_SEM


class FFmpegError(RuntimeError):
    pass


# 正在跑的 ffmpeg 子进程 PID，中断时要一并杀掉，否则会留下孤儿进程
_LIVE_PROCS: set = set()
_LIVE_LOCK = threading.Lock()


def kill_children() -> None:
    """杀掉所有在跑的 ffmpeg 子进程（中断/退出时调用）。"""
    import signal as _sig

    with _LIVE_LOCK:
        pids = list(_LIVE_PROCS)
        _LIVE_PROCS.clear()
    for pid in pids:
        try:
            os.kill(pid, _sig.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def request_stop() -> None:
    STOP.set()
    kill_children()


def stop_requested() -> bool:
    return STOP.is_set()


def ffmpeg_path() -> str:
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg  # 可选依赖

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise FFmpegError(
            "未找到 ffmpeg。请安装：\n"
            "  Ubuntu/Debian: sudo apt install ffmpeg\n"
            "  macOS: brew install ffmpeg\n"
            "  Windows: winget install Gyan.FFmpeg\n"
            "或 pip install imageio-ffmpeg"
        )


def has_ffmpeg() -> bool:
    try:
        ffmpeg_path()
        return True
    except FFmpegError:
        return False


def _needs_referer(url: str) -> bool:
    host = urllib.parse.urlsplit(url).netloc.lower()
    return "bilibili" in host or "bilivideo" in host or "hdslb" in host


def http_header_arg(url: str, user_agent: str = "") -> Optional[str]:
    """构造 ffmpeg 的 -headers 参数值（B站需要 Referer，否则 403）。"""
    if not _needs_referer(url):
        return None
    ua = user_agent or (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
    return f"Referer: {REFERER}\r\nUser-Agent: {ua}\r\n"


def _run(cmd: Sequence[str], timeout: float = 180.0, tag: str = "ffmpeg",
         watch_path: Optional[str] = None, stall_secs: float = 0.0) -> None:
    """跑一个 ffmpeg 子进程。

    watch_path + stall_secs：盯着输出文件大小，长时间没增长就判定为「连接卡死」并杀掉。
    B站 CDN 偶尔会给出一个只握手不出数据的节点，没有这一步会一直干等。
    """
    if stop_requested():
        raise FFmpegError("已请求停止")
    proc = subprocess.Popen(
        list(cmd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    with _LIVE_LOCK:
        _LIVE_PROCS.add(proc.pid)
    t0 = time.time()
    last_size = -1
    last_progress = t0
    try:
      while True:
        if proc.poll() is not None:
            break
        if stop_requested():
            proc.kill()
            raise FFmpegError("已请求停止")
        now = time.time()
        if now - t0 > timeout:
            proc.kill()
            _, err = proc.communicate()
            raise FFmpegError(f"{tag} 超时({timeout:.0f}s): {err[-400:]}")
        if watch_path and stall_secs > 0:
            try:
                size = os.path.getsize(watch_path)
            except OSError:
                size = 0
            if size != last_size:
                last_size, last_progress = size, now
            elif now - last_progress > stall_secs:
                proc.kill()
                proc.communicate()
                raise FFmpegError(f"{tag} 卡住：{stall_secs:.0f}s 无数据增长（已丢弃该节点）")
        time.sleep(0.2)
    finally:
        with _LIVE_LOCK:
            _LIVE_PROCS.discard(proc.pid)
        if proc.poll() is None:
            proc.kill()
    _, err = proc.communicate()
    if proc.returncode != 0:
        tail = (err or "").strip().splitlines()[-6:]
        raise FFmpegError(f"{tag} 失败(rc={proc.returncode}): " + " | ".join(tail))


@dataclass
class MediaResult:
    path: str
    seconds: float
    bytes: int

    @property
    def kb(self) -> float:
        return self.bytes / 1024.0


# --------------------------------------------------------------------------
# 单帧抽取
# --------------------------------------------------------------------------
def grab_frame(stream: Stream, ts: float, out_path: str, *, width: int = 1280,
               quality: int = 4, user_agent: str = "", timeout: float = 90.0) -> MediaResult:
    """在指定时间点抽一帧（不下载整段视频）。"""
    ff = ffmpeg_path()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    last_err: Optional[Exception] = None
    for url in stream.all_urls:
        hdr = http_header_arg(url, user_agent)
        cmd: List[str] = [ff, "-y", "-nostdin", "-loglevel", "error"]
        if hdr:
            cmd += ["-headers", hdr]
        # 输入侧 seek：依赖 HTTP Range，速度最快
        cmd += ["-ss", f"{max(0.0, float(ts)):.3f}", "-i", url, "-frames:v", "1", "-an", "-sn"]
        # 等比缩放，最长边不超过 width，并且宽高都取偶数
        cmd += ["-vf", f"scale='min({width},iw)':-2:flags=lanczos", "-q:v", str(quality)]
        cmd += ["-f", "image2", out_path]
        try:
            with _conn_slot():
                _run(cmd, timeout=timeout, tag=f"抽帧@{ts:.0f}s")
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                return MediaResult(out_path, ts, os.path.getsize(out_path))
            last_err = FFmpegError("抽帧产出为空")
        except FFmpegError as e:
            last_err = e
            continue
    raise last_err or FFmpegError("抽帧失败")


# --------------------------------------------------------------------------
# 音频抽取
# --------------------------------------------------------------------------
def extract_audio(stream: Stream, out_path: str, *, start: Optional[float] = None,
                  duration: Optional[float] = None, bitrate: str = "48k",
                  sample_rate: int = 16000, mono: bool = True,
                  user_agent: str = "", timeout: float = 3600.0) -> MediaResult:
    """转码抽取音频（16k 单声道 mp3，体积小、转写够用）。"""
    ff = ffmpeg_path()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path + ".part"
    last_err: Optional[Exception] = None
    for url in stream.all_urls:
        hdr = http_header_arg(url, user_agent)
        cmd: List[str] = [ff, "-y", "-nostdin", "-loglevel", "error"]
        if hdr:
            cmd += ["-headers", hdr]
        if start and start > 0:
            cmd += ["-ss", f"{start:.3f}"]
        cmd += ["-i", url, "-vn", "-sn", "-dn"]
        if duration:
            cmd += ["-t", f"{duration:.3f}"]
        cmd += ["-ac", "1" if mono else "2", "-ar", str(sample_rate)]
        cmd += ["-c:a", "libmp3lame", "-b:a", bitrate]
        cmd += ["-f", "mp3", tmp]
        try:
            _run(cmd, timeout=timeout, tag="抽音频")
            if os.path.exists(tmp) and os.path.getsize(tmp) > 1024:
                os.replace(tmp, out_path)
                return MediaResult(out_path, duration or 0.0, os.path.getsize(out_path))
            last_err = FFmpegError("音频产出为空")
        except FFmpegError as e:
            last_err = e
            for p in (tmp,):
                try:
                    os.remove(p)
                except OSError:
                    pass
            continue
    raise last_err or FFmpegError("音频抽取失败")


def download_bytes(url: str, *, user_agent: str = "", timeout: float = 60.0,
                   retries: int = 3) -> bytes:
    """带 Referer 的简单下载（快照雪碧图 / 字幕），只用于小文件。"""
    from . import http as _http

    hdr = {"User-Agent": user_agent or "Mozilla/5.0", "Referer": REFERER}
    try:
        with _conn_slot():
            return _http.get_bytes(url, headers=hdr, timeout=timeout, retries=retries)
    except Exception as e:
        raise FFmpegError(f"下载失败 {url[:80]}: {e}") from e


# --------------------------------------------------------------------------
# CDN 测速：B站同一路流会给出 baseUrl + 多个 backupUrl，不同边缘节点速度差很多
# --------------------------------------------------------------------------
def rank_urls(stream: Stream, *, sample: int = 262144, timeout: float = 20.0,
              user_agent: str = "", log=print) -> List[str]:
    """并发测速，按速度从快到慢返回候选地址。"""
    import concurrent.futures as cf

    urls = stream.all_urls
    if len(urls) <= 1:
        return urls

    def speed(u: str) -> Tuple[str, float]:
        from . import http as _http

        headers = {}
        hdr = http_header_arg(u, user_agent)
        if hdr:
            for line in hdr.splitlines():
                if ": " in line:
                    k, v = line.split(": ", 1)
                    headers[k] = v
        t0 = time.time()
        try:
            body = _http.get_bytes(u, headers=headers, timeout=timeout, retries=1,
                                   byte_range=(0, max(1024, sample) - 1))
        except Exception:
            return u, 0.0
        dt = max(1e-3, time.time() - t0)
        return u, len(body) / dt / 1024.0

    with cf.ThreadPoolExecutor(max_workers=len(urls)) as ex:
        results = list(ex.map(speed, urls))
    results.sort(key=lambda r: r[1], reverse=True)
    if results and results[0][1] > 0:
        best_host = results[0][0].split("/")[2]
        log(f"  🚀 CDN 测速: 最快节点 {best_host} ({results[0][1]:.0f} KB/s)"
            f"，共 {len(urls)} 个候选")
    return [u for u, sp in results]


# --------------------------------------------------------------------------
# 并行分窗抽取音频：把长音频切成若干时间窗同时拉，突破单连接限速
# --------------------------------------------------------------------------
def extract_audio_parallel(stream: Stream, out_path: str, duration: float, *,
                           windows: int = 4, bitrate: str = "48k",
                           sample_rate: int = 16000, user_agent: str = "",
                           timeout: float = 1800.0, stall_secs: float = 45.0,
                           log=print) -> MediaResult:
    """把音频按时间窗并发抽取后拼接。

    单条 HTTP 连接在 B站 CDN 上往往只有几十 KB/s，
    4~6 个窗口并发基本能线性提速；每个窗口是独立 seek，互不影响。
    """
    import concurrent.futures as cf
    import tempfile

    ff = ffmpeg_path()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    duration = float(duration or 0)
    windows = max(1, min(int(windows), 12))
    if duration <= 0 or windows == 1:
        return extract_audio(stream, out_path, bitrate=bitrate, sample_rate=sample_rate,
                             user_agent=user_agent, timeout=timeout)

    urls = rank_urls(stream, user_agent=user_agent, log=log) or stream.all_urls
    win_len = duration / windows
    tmpdir = Path(tempfile.mkdtemp(prefix="aicbili_audio_", dir=str(out.parent)))

    def one(i: int) -> Optional[str]:
        if stop_requested():
            return None
        start = win_len * i
        length = min(win_len + 0.5, duration - start)   # 轻微重叠，避免接缝丢字
        dst = tmpdir / f"w{i:02d}.mp3"
        for url in urls:
            hdr = http_header_arg(url, user_agent)
            cmd = [ff, "-y", "-nostdin", "-loglevel", "error"]
            if hdr:
                cmd += ["-headers", hdr]
            cmd += ["-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", url,
                    "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(sample_rate),
                    "-c:a", "libmp3lame", "-b:a", bitrate, "-f", "mp3", str(dst)]
            try:
                with _conn_slot():
                    _run(cmd, timeout=timeout, tag=f"音频窗口{i}",
                         watch_path=str(dst), stall_secs=stall_secs)
                if dst.exists() and dst.stat().st_size > 1024:
                    return str(dst)
            except FFmpegError as e:
                log(f"    ⚠ 窗口 {i} 失败（{url.split('/')[2]}）: {e}")
                try:
                    dst.unlink()
                except OSError:
                    pass
        return None

    t0 = time.time()
    try:
        with cf.ThreadPoolExecutor(max_workers=windows) as ex:
            parts = [p for p in ex.map(one, range(windows)) if p]
        if not parts:
            raise FFmpegError("所有音频窗口都失败")
        if len(parts) == 1:
            os.replace(parts[0], out_path)
        else:
            listing = tmpdir / "list.txt"
            listing.write_text("\n".join(f"file '{p}'" for p in parts), encoding="utf-8")
            _run([ff, "-y", "-nostdin", "-loglevel", "error", "-f", "concat", "-safe", "0",
                  "-i", str(listing), "-c", "copy", "-f", "mp3", out_path + ".part"],
                 timeout=timeout, tag="音频拼接")
            os.replace(out_path + ".part", out_path)
        size = os.path.getsize(out_path)
        log(f"  🎧 音频 {size / 1024:.0f} KB：{len(parts)}/{windows} 个窗口并发，"
            f"用时 {time.time() - t0:.1f}s（{duration / max(1e-6, time.time() - t0):.0f}x 实时）")
        return MediaResult(out_path, duration, size)
    finally:
        import shutil as _sh

        _sh.rmtree(tmpdir, ignore_errors=True)


__all__ = [
    "FFmpegError", "MediaResult", "STOP", "configure_global_limit", "download_bytes",
    "extract_audio", "kill_children",
    "extract_audio_parallel", "ffmpeg_path", "grab_frame", "has_ffmpeg",
    "http_header_arg", "probe_anonymous", "rank_urls", "referer_free_url",
    "relay_url", "request_stop", "stop_requested",
]


# --------------------------------------------------------------------------
# 匿名可达性探测：判断这个视频地址能否直接交给云端模型去拉
# --------------------------------------------------------------------------
def probe_anonymous(url: str, timeout: float = 8.0) -> bool:
    """不带 Referer 请求 64 字节，看是否 200/206。"""
    from . import http as _http

    try:
        body = _http.get_bytes(url, headers={"User-Agent": "Mozilla/5.0"},
                               timeout=timeout, retries=1, byte_range=(0, 63))
        return len(body) > 0
    except Exception:
        return False


def referer_free_url(stream: Stream, *, probe: bool = True, log=print) -> Optional[str]:
    """在 baseUrl/backupUrl 里找一条匿名可访问的地址（B站只有部分 mcdn 节点放行）。"""
    if not probe:
        for u in stream.all_urls:
            host = urllib.parse.urlsplit(u).netloc.lower()
            if "mcdn" in host:
                return u
        return None
    for u in stream.all_urls:
        if probe_anonymous(u):
            log(f"🌐 找到匿名可访问的视频地址（{urllib.parse.urlsplit(u).netloc}），可交给模型原生读取")
            return u
    return None


def relay_url(relay: str, video_url: str) -> str:
    """把视频地址包进自建中转代理（该代理负责补 Referer）。"""
    return relay.rstrip("/") + "/" + urllib.parse.quote(video_url, safe="")
