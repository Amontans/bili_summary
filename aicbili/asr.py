"""语音转写：三种后端，自动挑最快能用的。

优先级（auto）
--------------
1. ``subtitle``  —— 直接用 B站字幕（人工 CC 或官方 AI 字幕）。有登录 Cookie 时
   通常 1 秒内拿到全文，**完全跳过 ASR**，这是最快的路径。
2. ``openai``    —— 任意 OpenAI 兼容的 /audio/transcriptions 端点（云端转写）。
3. ``local``     —— faster-whisper 本地转写（不需要联网 AI，但需要额外依赖与算力）。
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from .config import Config


@dataclass
class Segment:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Transcript:
    segments: List[Segment]
    source: str = ""          # subtitle / local-whisper / openai-asr
    language: str = ""
    elapsed: float = 0.0

    @property
    def text(self) -> str:
        return "".join(s.text.strip() + "\n" for s in self.segments).strip()

    @property
    def chars(self) -> int:
        return sum(len(s.text.strip()) for s in self.segments)

    def with_timestamps(self) -> str:
        return "\n".join(f"[{fmt_ts(s.start)}] {s.text.strip()}" for s in self.segments)

    def window(self, start: float, end: float) -> str:
        buf = [s.text.strip() for s in self.segments if s.end >= start and s.start <= end]
        return " ".join(t for t in buf if t)


def fmt_ts(sec: float) -> str:
    s = int(max(0.0, sec))
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:d}:{s % 60:02d}"


def fmt_srt_ts(sec: float) -> str:
    ms = int(round(max(0.0, sec) * 1000))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: Sequence[Segment], path: str) -> None:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{fmt_srt_ts(seg.start)} --> {fmt_srt_ts(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------
# 后端 1：B站字幕（最快）
# --------------------------------------------------------------------------
def from_cues(cues: Sequence[Dict[str, Any]], source: str = "subtitle") -> Transcript:
    segs: List[Segment] = []
    for c in cues:
        txt = str(c.get("text") or "").strip()
        if not txt:
            continue
        segs.append(Segment(start=float(c.get("start") or 0.0),
                            end=float(c.get("end") or 0.0), text=txt))
    return Transcript(segments=segs, source=source)


# --------------------------------------------------------------------------
# 后端 2：本地 faster-whisper
# --------------------------------------------------------------------------
def local_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def transcribe_local(audio_path: str, cfg: Config, log=print) -> Transcript:
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise RuntimeError(
            "未安装 faster-whisper。安装：\n"
            "  pip install 'aicbili[local-asr]'\n"
            f"或改用云端 ASR / B站字幕（原始错误：{e}）"
        )

    device = cfg.whisper_device
    compute = cfg.whisper_compute
    if not device:
        device, compute = _auto_device()
    log(f"🎧 本地转写：模型={cfg.whisper_model} 设备={device} 精度={compute}")
    t0 = time.time()
    model = WhisperModel(cfg.whisper_model, device=device, compute_type=compute)
    segments, info = model.transcribe(
        audio_path,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        beam_size=5,
        condition_on_previous_text=False,
        word_timestamps=False,
    )
    segs: List[Segment] = []
    for s in segments:
        txt = (s.text or "").strip()
        if txt:
            segs.append(Segment(start=float(s.start), end=float(s.end), text=txt))
    return Transcript(segments=segs, source="local-whisper",
                      language=getattr(info, "language", "") or "",
                      elapsed=time.time() - t0)


def _auto_device():
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


# --------------------------------------------------------------------------
# 后端 3：OpenAI 兼容的云端 ASR
# --------------------------------------------------------------------------
def transcribe_openai(audio_path: str, cfg: Config, log=print) -> Transcript:
    from . import http as _http

    p = cfg.vlm
    if not p.api_key:
        raise RuntimeError("云端 ASR 需要配置 AICBILI_VLM_API_KEY")
    model = os.environ.get("AICBILI_ASR_MODEL") or "whisper-1"
    url = p.base_url.rstrip("/") + "/audio/transcriptions"
    log(f"☁️ 云端转写：{model} @ {p.base_url}")
    t0 = time.time()
    with open(audio_path, "rb") as f:
        blob = f.read()
    payload = _http.post_multipart(
        url,
        {"model": model, "response_format": "verbose_json"},
        [("file", os.path.basename(audio_path), blob, "audio/mpeg")],
        headers={"Authorization": f"Bearer {p.api_key}"},
        timeout=900,
    )
    segs: List[Segment] = []
    for s in payload.get("segments") or []:
        txt = str(s.get("text") or "").strip()
        if txt:
            segs.append(Segment(start=float(s.get("start") or 0),
                                end=float(s.get("end") or 0), text=txt))
    if not segs and payload.get("text"):
        return Transcript(segments=[Segment(0, 0, str(payload["text"]).strip())],
                          source="openai-asr", elapsed=time.time() - t0)
    return Transcript(segments=segs, source="openai-asr",
                      language=str(payload.get("language") or ""),
                      elapsed=time.time() - t0)


# --------------------------------------------------------------------------
# 后端 3：whisper.cpp（本地、免费，Arch 官方仓库有包）
# --------------------------------------------------------------------------
_CPP_BIN_NAMES = ("whisper-cli", "whisper-cpp", "main")
_MODEL_HINTS = (
    "~/.cache/whisper-cpp", "~/whisper.cpp/models", "./models",
    "/usr/share/whisper-cpp", "/usr/local/share/whisper-cpp", "~/.local/share/whisper.cpp",
)


def whisper_cpp_bin() -> Optional[str]:
    """找 whisper.cpp 的可执行文件。"""
    import shutil as _sh

    for name in _CPP_BIN_NAMES:
        p = _sh.which(name)
        if p:
            return p
    return None


def find_whisper_cpp_model(explicit: str = "") -> Optional[str]:
    """定位 ggml 模型文件。找不到就返回 None（由调用方给出下载提示）。"""
    import glob as _glob

    if explicit:
        p = os.path.expanduser(explicit)
        if os.path.isfile(p):
            return p
    for hint in _MODEL_HINTS:
        d = os.path.expanduser(hint)
        if not os.path.isdir(d):
            continue
        for pat in ("ggml-*.bin", "*.bin", "for-tests-*.bin"):
            hits = sorted(_glob.glob(os.path.join(d, pat)),
                          key=lambda f: (0 if "base" in f else 1 if "small" in f else 2,
                                         len(f)))
            hits = [h for h in hits if "mmproj" not in h]   # 排除多模态投影文件
            if hits:
                return hits[0]
    return None


def _to_wav16k(audio_path: str, out_wav: str) -> None:
    """whisper.cpp 只吃 16kHz 单声道 WAV，先用 ffmpeg 转一下（本地，很快）。"""
    from .media import FFmpegError, _run, ffmpeg_path

    try:
        _run([ffmpeg_path(), "-y", "-nostdin", "-loglevel", "error", "-i", audio_path,
              "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out_wav],
             timeout=900, tag="转 wav")
    except FFmpegError as e:
        raise RuntimeError(f"转 WAV 失败: {e}") from e


def _parse_whisper_cpp_json(path: str) -> List[Segment]:
    """解析 whisper.cpp 的 -oj 输出。offsets 单位是毫秒。"""
    import json as _json

    try:
        with open(path, encoding="utf-8") as f:
            data = _json.load(f)
    except Exception:
        return []
    segs: List[Segment] = []
    for item in data.get("transcription") or []:
        txt = str(item.get("text") or "").strip()
        if not txt:
            continue
        off = item.get("offsets") or {}
        try:
            start = float(off.get("from") or 0) / 1000.0
            end = float(off.get("to") or 0) / 1000.0
        except (TypeError, ValueError):
            continue
        segs.append(Segment(start=start, end=end, text=txt))
    return segs


def transcribe_whisper_cpp(audio_path: str, cfg: Config, log=print) -> Transcript:
    import json as _json
    import subprocess
    import tempfile

    binary = cfg.whisper_cpp_bin or whisper_cpp_bin()
    if not binary:
        raise RuntimeError(
            "未找到 whisper.cpp。安装：\n"
            "  Arch:   sudo pacman -S whisper-cpp\n"
            "  其他:   编译 https://github.com/ggml-org/whisper.cpp 或用 AICBILI_WHISPER_CPP_BIN 指定路径"
        )
    model = find_whisper_cpp_model(cfg.whisper_cpp_model)
    if not model:
        raise RuntimeError(
            "找不到 whisper.cpp 的 ggml 模型。下载一个（base 约 148MB，small 约 488MB）：\n"
            "  mkdir -p ~/.cache/whisper-cpp && cd ~/.cache/whisper-cpp\n"
            "  curl -L -o ggml-base.bin \\\n"
            "    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin\n"
            "或用 AICBILI_WHISPER_CPP_MODEL 指定已有 .bin 路径"
        )
    t0 = time.time()
    log(f"🎙 whisper.cpp 本地转写：{os.path.basename(model)}（{os.path.basename(binary)}）")
    with tempfile.TemporaryDirectory(prefix="aicbili_wcpp_") as td:
        wav = os.path.join(td, "audio.wav")
        _to_wav16k(audio_path, wav)
        prefix = os.path.join(td, "out")
        cmd = [binary, "-m", model, "-f", wav, "-oj", "-of", prefix,
               "-t", str(max(2, (os.cpu_count() or 4) - 1))]
        lang = (cfg.whisper_language or "").strip()
        if lang:
            cmd += ["-l", lang]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  errors="replace", timeout=7200)
        except subprocess.TimeoutExpired:
            raise RuntimeError("whisper.cpp 超时")
        json_path = prefix + ".json"
        if not os.path.exists(json_path):
            tail = (proc.stderr or proc.stdout or "")[-500:]
            raise RuntimeError(f"whisper.cpp 没有产出 JSON（rc={proc.returncode}）: {tail}")
        segs = _parse_whisper_cpp_json(json_path)
        lang_got = ""
        try:
            lang_got = str(_json.load(open(json_path, encoding="utf-8"))
                           .get("result", {}).get("language") or "")
        except Exception:
            pass
    if not segs:
        raise RuntimeError("whisper.cpp 转写结果为空")
    return Transcript(segments=segs, source="whisper.cpp", language=lang_got,
                      elapsed=time.time() - t0)


def available_backends(cfg: Config) -> List[str]:
    """列出当前环境下**真的能用**的转写后端（不联网、不下载）。

    用途：在没有字幕的视频上，先判断「下了音频有没有意义」，
    避免白白下载几 MB 音频才发现根本转写不了。
    """
    out: List[str] = []
    base = (cfg.vlm.base_url or "").lower()
    if cfg.asr_backend == "openai":
        return ["openai"] if cfg.vlm.api_key else []
    if cfg.asr_backend in ("whisper_cpp", "whisper-cpp", "cpp"):
        return ["whisper_cpp"] if (whisper_cpp_bin()
                                   and find_whisper_cpp_model(cfg.whisper_cpp_model)) else []
    if cfg.asr_backend == "local":
        return ["local"] if local_available() else []
    # auto
    if cfg.vlm.api_key and "deepseek" not in base:
        out.append("openai")
    if whisper_cpp_bin() and find_whisper_cpp_model(cfg.whisper_cpp_model):
        out.append("whisper_cpp")
    if local_available():
        out.append("local")
    return out


def transcribe(audio_path: str, cfg: Config, log=print) -> Transcript:
    """按配置选择后端转写本地音频。"""
    backend = cfg.asr_backend
    errors: List[str] = []
    order: List[str]
    if backend == "subtitle":
        raise RuntimeError("ASR 后端被指定为 subtitle，但该视频没有可用字幕")
    if backend == "local":
        order = ["local"]
    elif backend == "openai":
        order = ["openai"]
    elif backend in ("whisper_cpp", "whisper-cpp", "cpp"):
        order = ["whisper_cpp"]
    else:
        # auto：云 ASR 最快 → whisper.cpp（本地零成本）→ faster-whisper
        # 但 DeepSeek 这类端点根本没有 /audio/transcriptions，别去白撞一次
        base = (cfg.vlm.base_url or "").lower()
        no_audio_api = "deepseek" in base
        order = ["whisper_cpp", "local"] if no_audio_api else ["openai", "whisper_cpp", "local"]
    for name in order:
        try:
            if name == "local":
                if not local_available():
                    raise RuntimeError("faster-whisper 未安装")
                return transcribe_local(audio_path, cfg, log=log)
            if name == "whisper_cpp":
                return transcribe_whisper_cpp(audio_path, cfg, log=log)
            return transcribe_openai(audio_path, cfg, log=log)
        except Exception as e:
            errors.append(f"{name}: {e}")
            log(f"⚠ {name} 转写不可用：{e}")
    raise RuntimeError("没有可用的转写后端：\n  " + "\n  ".join(errors))


__all__ = [
    "Segment", "Transcript", "available_backends", "find_whisper_cpp_model", "fmt_srt_ts", "fmt_ts",
    "from_cues", "local_available", "transcribe", "transcribe_local",
    "transcribe_openai", "transcribe_whisper_cpp", "whisper_cpp_bin", "write_srt",
]
