"""配置：真实环境变量 > .env 文件 > 默认值。

所有键都以 AICBILI_ 前缀，避免与旧项目/其他工具冲突。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional

DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

# 各家的 OpenAI 兼容端点，方便 --provider 快速切换
PROVIDER_PRESETS: Dict[str, Dict[str, str]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "dashscope": {  # 阿里云百炼（通义千问 VL）
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-vl-max-latest",
    },
    "ark": {  # 火山方舟（豆包）
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "doubao-1.5-vision-pro-32k",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4v-plus",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen2.5-VL-72B-Instruct",
    },
    "deepseek": {
        # deepseek-flash 已支持图片输入（OpenAI 兼容格式），看画面 + 写总结一把 key 搞定
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-flash",
    },
    "ollama": {
        # 本地推理（可选）：ollama serve 后即可，无需真实 API Key（随便填一个非空值）
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5vl:3b",
    },
}


def _parse_env_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return out
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        k, _, v = line.partition("=")
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        out[k.strip()] = v
    return out


def _candidate_env_files(explicit: Optional[str] = None):
    if explicit:
        yield Path(explicit).expanduser()
        return
    for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
        for name in (".env", ".env.local"):
            yield base / name


def load_env(explicit: Optional[str] = None) -> Dict[str, str]:
    """合并 .env 与环境变量（后者优先）。"""
    merged: Dict[str, str] = {}
    for p in _candidate_env_files(explicit):
        if p.is_file():
            merged.update(_parse_env_file(p))
    for k in list(merged) + [k for k in os.environ if k.startswith("AICBILI_")]:
        if k in os.environ:
            merged[k] = os.environ[k]
    return merged


def _int(v, default: int) -> int:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return default


def _float(v, default: float) -> float:
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return default


def _bool(v, default: bool = False) -> bool:
    if v is None or v == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass
class Provider:
    """一个 OpenAI 兼容的多模态/文本端点。"""

    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = ""
    fallback_model: str = ""
    timeout: float = 120.0

    @property
    def ready(self) -> bool:
        return bool(self.api_key and self.model)


@dataclass
class Config:
    # ── B站登录态（可选；解锁 1080P 与 B站 AI 字幕）──
    sessdata: str = ""
    cookie: str = ""
    user_agent: str = DEFAULT_UA

    # ── 多模态（看图/看视频）与纯文本总结 ──
    vlm: Provider = field(default_factory=Provider)
    text: Provider = field(default_factory=Provider)

    # ── 抽帧策略 ──
    max_frames: int = 24          # 送给视觉模型的画面数量上限
    frame_width: int = 1280       # 单帧最长边（像素）
    jpeg_quality: int = 4         # ffmpeg -q:v，2 最好 ~ 31 最差
    scene_threshold: float = 0.12 # 场景变化阈值（0~1，越低越敏感）
    sheet_grid: int = 6           # 每张拼图放几帧：6=3x2（兼顾清晰度与 token）
    use_storyboard: bool = True   # 用 B站快照雪碧图做场景检测（极省流量）

    # ── 音频/转写 ──
    asr_backend: str = "auto"     # auto|subtitle|openai|whisper_cpp|local
    whisper_model: str = "small"
    whisper_device: str = ""      # 留空=自动
    whisper_compute: str = ""
    whisper_cpp_model: str = ""   # ggml-*.bin 路径（留空自动找）
    whisper_cpp_bin: str = ""     # whisper-cli 路径（留空自动找）
    whisper_language: str = "zh"  # whisper.cpp 强制语言，速度更快更准
    audio_bitrate: str = "48k"
    audio_windows: int = 4        # 分窗并发抽取音频，突破单连接限速
    audio_stall: int = 45         # 单个窗口多少秒没数据增长就换节点

    # ── 运行时 ──
    concurrency: int = 6
    mode: str = "auto"            # auto|native|frames|audio
    quality: int = 80             # 期望清晰度 qn（1080P=80）
    output_dir: str = "./output"
    cache_dir: str = ""
    keep_cache: bool = False
    retries: int = 3
    # 推理模型（如 DeepSeek）默认会先输出一大段思维链，既慢又贵。
    # auto = 认识的推理端点(deepseek)自动关掉「看画面」时的思考、总结用 low；其他端点不发这个参数
    thinking: str = "auto"        # auto|off|low|high|max
    verbose: bool = False
    relay: str = ""               # 可选：注入 Referer 的反向代理，供「原生视频直读」模式使用

    # ---------------------------------------------------------------
    @classmethod
    def load(cls, env_file: Optional[str] = None, **overrides) -> "Config":
        env = load_env(env_file)
        g = lambda k, d=None: env.get("AICBILI_" + k, d)  # noqa: E731

        preset_name = (g("PROVIDER") or "").strip().lower()
        preset = PROVIDER_PRESETS.get(preset_name, {})

        vlm = Provider(
            api_key=g("VLM_API_KEY") or g("API_KEY") or "",
            base_url=g("VLM_BASE_URL") or preset.get("base_url") or "https://api.openai.com/v1",
            model=g("VLM_MODEL") or preset.get("model") or "",
            fallback_model=g("VLM_FALLBACK_MODEL") or "",
            timeout=_float(g("VLM_TIMEOUT"), 120.0),
        )
        # 文本总结默认复用多模态端点（同一把 key 通常通用），可用 TEXT_* 覆盖
        text = Provider(
            api_key=g("TEXT_API_KEY") or vlm.api_key,
            base_url=g("TEXT_BASE_URL") or vlm.base_url,
            model=g("TEXT_MODEL") or vlm.model,
            fallback_model=g("TEXT_FALLBACK_MODEL") or "",
            timeout=_float(g("TEXT_TIMEOUT"), 120.0),
        )

        cfg = cls(
            sessdata=g("SESSDATA") or "",
            cookie=g("COOKIE") or "",
            user_agent=g("UA") or DEFAULT_UA,
            vlm=vlm,
            text=text,
            max_frames=_int(g("MAX_FRAMES"), 24),
            frame_width=_int(g("FRAME_WIDTH"), 1280),
            jpeg_quality=_int(g("JPEG_QUALITY"), 4),
            scene_threshold=_float(g("SCENE_THRESHOLD"), 0.12),
            sheet_grid=_int(g("SHEET_GRID"), 4),
            use_storyboard=_bool(g("USE_STORYBOARD"), True),
            asr_backend=(g("ASR_BACKEND") or "auto").strip().lower(),
            whisper_model=g("WHISPER_MODEL") or "small",
            whisper_device=g("WHISPER_DEVICE") or "",
            whisper_compute=g("WHISPER_COMPUTE") or "",
            whisper_cpp_model=g("WHISPER_CPP_MODEL") or "",
            whisper_cpp_bin=g("WHISPER_CPP_BIN") or "",
            whisper_language=g("WHISPER_LANGUAGE") or "zh",
            audio_bitrate=g("AUDIO_BITRATE") or "48k",
            audio_windows=_int(g("AUDIO_WINDOWS"), 4),
            audio_stall=_int(g("AUDIO_STALL"), 45),
            concurrency=_int(g("CONCURRENCY"), 6),
            mode=(g("MODE") or "auto").strip().lower(),
            quality=_int(g("QUALITY"), 80),
            output_dir=g("OUTPUT_DIR") or "./output",
            cache_dir=g("CACHE_DIR") or "",
            keep_cache=_bool(g("KEEP_CACHE"), False),
            retries=_int(g("RETRIES"), 3),
            thinking=(g("THINKING") or "auto").strip().lower(),
            verbose=_bool(g("VERBOSE"), False),
            relay=(g("RELAY") or "").rstrip("/"),
        )

        for key, val in overrides.items():
            if val is None:
                continue
            if "." in key:  # 例如 vlm.model
                obj, _, attr = key.partition(".")
                setattr(getattr(cfg, obj), attr, val)
            elif hasattr(cfg, key):
                setattr(cfg, key, val)
        return cfg

    # ---------------------------------------------------------------
    @property
    def cookie_header(self) -> str:
        """拼出请求 B站 用的 Cookie 头。"""
        parts = [self.cookie] if self.cookie else []
        if self.sessdata and "SESSDATA" not in (self.cookie or ""):
            parts.append(f"SESSDATA={self.sessdata}")
        return "; ".join(p for p in parts if p)

    @property
    def logged_in(self) -> bool:
        return bool(self.sessdata or "SESSDATA" in (self.cookie or ""))

    def cache_root(self) -> Path:
        """缓存根目录。注意：这里**不**创建目录（查看配置时不该产生副作用），
        真正要写入的调用方自己 mkdir。"""
        if self.cache_dir:
            return Path(self.cache_dir).expanduser()
        return Path(self.output_dir).expanduser() / ".aicbili_cache"

    def masked(self) -> Dict[str, object]:
        def m(v: str) -> str:
            return (v[:4] + "…" + v[-4:]) if len(v) > 12 else ("***" if v else "")

        return {
            "登录态": "已配置" if self.logged_in else "未配置（480P + 仅官方字幕）",
            "多模态模型": f"{self.vlm.model} @ {self.vlm.base_url}" if self.vlm.model else "未配置",
            "多模态 Key": m(self.vlm.api_key),
            "文本模型": f"{self.text.model} @ {self.text.base_url}" if self.text.model else "未配置",
            "抽帧上限": self.max_frames,
            "帧宽": self.frame_width,
            "场景阈值": self.scene_threshold,
            "ASR": self.asr_backend,
            "并发": self.concurrency,
            "模式": self.mode,
            "期望清晰度": self.quality,
            "输出": self.output_dir,
            "缓存": str(self.cache_root()),
            "中转代理": self.relay or "未配置",
        }

    def asdict(self, reveal_secrets: bool = False) -> Dict[str, object]:
        """导出为 dict。**默认脱敏**，避免 -v/日志把 Key 打出来。"""
        d = asdict(self)
        d["vlm"] = asdict(self.vlm)
        d["text"] = asdict(self.text)
        if not reveal_secrets:
            def _mask(v: str) -> str:
                if not v:
                    return ""
                return v[:5] + "…" + v[-4:] if len(v) > 12 else "***"

            for k in ("sessdata", "cookie"):
                if d.get(k):
                    d[k] = _mask(str(d[k]))
            for prov in ("vlm", "text"):
                if d[prov].get("api_key"):
                    d[prov]["api_key"] = _mask(str(d[prov]["api_key"]))
        return d
