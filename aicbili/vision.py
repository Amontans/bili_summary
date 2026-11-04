"""视觉理解：把画面交给多模态模型，得到「画面里到底有什么」的结构化笔记。

两种用法
--------
* ``describe_sheet()``：一屏拼图（4 帧）一次请求 → 省请求数、省 token、更快。
  这是默认路径，任何支持 ``image_url`` 的视觉模型都能用。
* ``watch_video()``：直接把远端视频地址当 ``video_url`` 发给支持原生视频的模型
  （豆包/通义/智谱等）。字节完全由模型侧拉取，本地零下载 —— 但要求该地址
  允许跨域匿名访问（B站主 CDN 校验 Referer，只有部分 mcdn 边缘节点放行，
  或自建一个注 Referer 的中转代理）。
"""
from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import Config, Provider

SHEET_PROMPT = """你在帮用户「看」一个视频的画面。下面是同一段视频里按时间顺序截取的 {n} 帧，
每帧下方有编号与时间戳（格式 #序号 时:分:秒），拼成一张图。

请逐帧描述你**真正看到**的内容，只写从画面上能确认的信息，不要凭字幕/标题猜测。
重点关注：
1. 画面上的文字：PPT 标题与要点、代码、命令、公式、图表数值、UI 界面元素、报错信息。
   尽量原样抄写关键文字（尤其是代码与专有名词）。
2. 画面类型：slides(幻灯片) / code(写代码或终端) / demo(软件操作演示) / chart(图表数据)
   / talking(人物口播) / other。
3. 这一帧作为「理解视频」是否重要（比如出现新概念、新章节、实际操作步骤、结果展示）。

严格输出 JSON，不要额外解释：
{{"frames":[{{"i":1,"kind":"slides","text":"画面文字与内容的简要描述","important":true}}],
 "gist":"这 {n} 帧合起来在讲什么（一句话）"}}

视频标题：{title}
"""

NATIVE_PROMPT = """请完整「观看」这个视频（包含画面与声音），并输出：
1) 视频主题与一句话总结；
2) 按时间分章的要点（给时间范围），每章同时说明**画面上出现的东西**（幻灯片/代码/演示/图表）；
3) 视频里出现的所有关键术语、代码片段、公式、数字；
4) 讲者明确强调的结论或注意事项。
只依据视频本身内容，不要编造。

视频标题：{title}
"""


# --------------------------------------------------------------------------
# 推理模型的思维链控制
# --------------------------------------------------------------------------
def thinking_fields(cfg, task: str = "vision") -> Dict[str, Any]:
    """按配置生成控制「思考模式」的请求字段。

    DeepSeek 这类模型默认 thinking=enabled / effort=high，会先把 token 全花在
    reasoning_content 上，导致 content 为空、看起来像「模型不回答」。
    看画面属于感知任务，关掉思考又快又省钱。
    """
    mode = (getattr(cfg, "thinking", "auto") or "auto").lower()
    is_reasoner = "deepseek" in (cfg.vlm.base_url or "").lower()
    if mode == "auto":
        if not is_reasoner:
            return {}
        mode = "off" if task == "vision" else "low"
    if mode in ("off", "disabled", "none"):
        return {"thinking": {"type": "disabled"}}
    if mode in ("low", "high", "max"):
        return {"reasoning_effort": mode}
    return {}


_STRIP_ON_400 = ("thinking", "reasoning_effort")


class VisionError(RuntimeError):
    pass


@dataclass
class FrameNote:
    ts: float
    kind: str = "other"
    text: str = ""
    important: bool = False

    def line(self) -> str:
        from .asr import fmt_ts

        flag = "★" if self.important else "·"
        return f"{flag} [{fmt_ts(self.ts)}] ({self.kind}) {self.text}"


@dataclass
class VisualNote:
    items: List[FrameNote] = field(default_factory=list)
    gist: str = ""
    source: str = ""
    elapsed: float = 0.0

    def as_markdown(self) -> str:
        if not self.items:
            return self.gist or ""
        lines = [f"- {i.line()}" for i in self.items]
        head = f"**画面概要**：{self.gist}\n\n" if self.gist else ""
        return head + "\n".join(lines)

    def merge(self, other: "VisualNote") -> "VisualNote":
        """合并另一批画面笔记（新的优先，gist 去重）。"""
        items = {round(i.ts, 1): i for i in self.items}
        for i in other.items:
            items[round(i.ts, 1)] = i
        gists: List[str] = []
        for g in (self.gist, other.gist):
            g = (g or "").strip()
            if g and g not in gists:
                gists.append(g)
        return VisualNote(
            items=sorted(items.values(), key=lambda x: x.ts),
            gist=" ".join(gists),
            source=self.source or other.source,
            elapsed=self.elapsed + other.elapsed,
        )


def _extract_json(text: str) -> Optional[dict]:
    """从模型输出里尽量挖出 JSON（容忍 ```json 包裹与前后废话）。"""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    for candidate in (t, t[t.find("{"): t.rfind("}") + 1] if "{" in t else ""):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def _message_text(data: Any) -> str:
    """从 chat/completions 响应里取出正文（兼容 content 为分段列表的情况）。"""
    try:
        choice = data["choices"][0]
        msg = choice["message"]
    except (KeyError, IndexError, TypeError):
        raise VisionError(f"响应格式异常: {str(data)[:300]}")
    # 推理模型常见故障：token 预算被 reasoning_content 吃光，content 为空
    finish = str(choice.get("finish_reason") or "")
    if not (msg.get("content") or "").strip() and finish == "length":
        usage = data.get("usage") or {}
        detail = usage.get("completion_tokens_details") or {}
        r = detail.get("reasoning_tokens", usage.get("completion_tokens", "?"))
        raise VisionError(
            f"输出被思考过程吃光了（finish_reason=length，reasoning_tokens={r}）。"
            "把 AICBILI_THINKING 设为 off，或调大 max_tokens。")
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                out.append(str(part.get("text") or ""))
            else:
                out.append(str(part))
        return "".join(out)
    if content is None and msg.get("reasoning_content"):
        return str(msg["reasoning_content"])
    return str(content or "")


class VisionModel:
    def __init__(self, cfg: Config, log=print):
        self.cfg = cfg
        self.log = log
        self.prov: Provider = cfg.vlm

    # ------------------------------------------------ 基础
    @property
    def ready(self) -> bool:
        return self.prov.ready

    def _chat(self, content: List[Dict[str, Any]], *, model: Optional[str] = None,
              max_tokens: int = 2000, temperature: float = 0.2) -> str:
        from . import http as _http

        url = self.prov.base_url.rstrip("/") + "/chat/completions"
        hdr = {"Authorization": f"Bearer {self.prov.api_key}",
               "Content-Type": "application/json"}
        # 候选模型按序尝试；换模型不能消耗重试次数（否则 retries=1 时永远走不到备选）
        models = [model or self.prov.model]
        if self.prov.fallback_model and self.prov.fallback_model != models[0]:
            models.append(self.prov.fallback_model)

        extra = thinking_fields(self.cfg, task="vision")
        last: Optional[Exception] = None
        for i, mdl in enumerate(models):
            payload = {
                "model": mdl,
                "messages": [{"role": "user", "content": content}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                **extra,
            }
            for attempt in range(max(1, self.cfg.retries)):
                try:
                    data = _http.post_json(url, payload, headers=hdr,
                                           timeout=self.prov.timeout, retries=1)
                    return _message_text(data).strip()
                except Exception as e:
                    last = e
                    # 某些端点不认识 thinking/reasoning_effort → 去掉再试
                    if any(k in payload for k in _STRIP_ON_400) and \
                            any(w in str(e).lower() for w in ("unrecognized", "unknown",
                                                              "unsupported", "invalid_request",
                                                              "extra", "400")):
                        for k in _STRIP_ON_400:
                            payload.pop(k, None)
                        self.log("↪ 端点不支持 thinking 参数，已去掉重试")
                        continue
                    msg = str(e).lower()
                    looks_like_model_error = ("model" in msg or "404" in msg
                                              or "-2013" in msg or "not found" in msg)
                    if looks_like_model_error and i + 1 < len(models):
                        self.log(f"↪ 模型 {mdl} 不可用，改用 {models[i + 1]}")
                        break                      # 直接换下一个模型，不占重试次数
                    if attempt + 1 < self.cfg.retries:
                        time.sleep(1.2 * (attempt + 1))
        raise VisionError(f"视觉模型调用失败: {last}")

    # ------------------------------------------------ 看拼图
    def describe_sheet(self, sheet_jpeg: bytes, timestamps: Sequence[float],
                       title: str = "", *, extra_hint: str = "") -> VisualNote:
        t0 = time.time()
        b64 = base64.b64encode(sheet_jpeg).decode()
        prompt = SHEET_PROMPT.format(n=len(timestamps), title=title or "(未提供)")
        if extra_hint:
            prompt += f"\n补充背景：{extra_hint}\n"
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
            },
        ]
        raw = self._chat(content, max_tokens=1800)
        note = self._parse_sheet(raw, timestamps)
        note.source = "sheet"
        note.elapsed = time.time() - t0
        return note

    def _parse_sheet(self, raw: str, timestamps: Sequence[float]) -> VisualNote:
        data = _extract_json(raw)
        note = VisualNote()
        if not isinstance(data, dict):
            # 模型没按格式返回 → 退化成整段文本，仍然有用
            note.gist = raw.strip()[:4000]
            note.items = [FrameNote(ts=float(ts), kind="other", text="(见画面概要)")
                          for ts in timestamps]
            return note
        note.gist = str(data.get("gist") or "").strip()
        frames = data.get("frames") or []
        idx_map = list(range(1, len(timestamps) + 1))
        for f in frames if isinstance(frames, list) else []:
            if not isinstance(f, dict):
                continue
            try:
                i = int(f.get("i") or 0)
            except (TypeError, ValueError):
                i = 0
            ts = timestamps[i - 1] if 0 < i <= len(timestamps) else None
            if ts is None and idx_map:
                ts = timestamps[min(len(timestamps) - 1, max(0, i - 1))]
            if ts is None:
                continue
            note.items.append(FrameNote(
                ts=float(ts),
                kind=str(f.get("kind") or "other")[:16],
                text=str(f.get("text") or "").strip()[:1200],
                important=bool(f.get("important")),
            ))
        if not note.items and note.gist:
            note.items = [FrameNote(ts=float(ts), kind="other", text="(见画面概要)")
                          for ts in timestamps]
        note.items.sort(key=lambda x: x.ts)
        return note

    # ------------------------------------------------ 原生看视频
    def watch_video(self, video_url: str, title: str = "", *, max_tokens: int = 3000) -> str:
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": NATIVE_PROMPT.format(title=title or "(未提供)")},
            {"type": "video_url", "video_url": {"url": video_url}},
        ]
        return self._chat(content, max_tokens=max_tokens, temperature=0.2)


# --------------------------------------------------------------------------
# 能力自检：这个「模型 + Key」到底能不能看图？
# --------------------------------------------------------------------------
def probe_multimodal(cfg: Config, timeout: float = 90.0) -> Tuple[bool, str]:
    """发一张带文字的小图，问模型图里写了什么。

    文本模型（比如只支持文字的 deepseek-chat）会在这一步暴露出来，
    比等到跑完整条流水线才 400 报错要好得多。
    """
    import io

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (200, 80), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([4, 4, 195, 75], outline=(0, 0, 0), width=2)
    d.text((16, 26), "OK 42", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    probe_extra = thinking_fields(cfg, task="vision")
    payload = {
        "model": cfg.vlm.model,
        **probe_extra,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "这张图里印着什么字？只回答图中文字本身，不要解释。"},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        # 推理模型的思考也要占 token，32 根本不够 → 给 512 并关掉思考
        "max_tokens": 512 if probe_extra else 64,
        "temperature": 0,
    }
    url = cfg.vlm.base_url.rstrip("/") + "/chat/completions"
    hdr = {"Authorization": f"Bearer {cfg.vlm.api_key}"}
    from . import http as _http

    try:
        data = _http.post_json(url, payload, headers=hdr, timeout=timeout, retries=1)
        reply = _message_text(data).strip()
        ok = "42" in reply or "OK" in reply.upper()
        return ok, reply[:120]
    except Exception as e:
        return False, str(e)[:300]


__all__ = ["FrameNote", "VisualNote", "VisionError", "VisionModel", "NATIVE_PROMPT",
           "SHEET_PROMPT", "probe_multimodal"]
