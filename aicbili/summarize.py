"""把「画面笔记」与「语音文稿」按时间轴对齐，再交给文本模型总结。

对齐是关键：只喂音频会漏掉屏幕上写的东西，只喂画面会漏掉讲者说的话。
所以这里先拼成一条**多模态时间轴**（同一时间点：画面看到什么 + 讲者说了什么），
再分段总结、合并，避免长视频超出上下文。
"""
from __future__ import annotations

import time
from typing import List, Optional

from .asr import Transcript, fmt_ts
from .config import Config, Provider
from .vision import FrameNote, VisualNote

SYS_PROMPT = (
    "你是短视频/课程内容分析专家。你会同时收到视频的『画面记录』和『语音文稿』，"
    "两者按时间对齐。请只依据给定材料作答，不要编造视频里没有的内容。"
    "输出使用简体中文 Markdown，条理清晰、信息密度高，保留具体术语、数字、代码与时间点。"
)

FINAL_PROMPT = """下面是同一个视频的「画面记录」与「语音文稿」按时间对齐后的材料。

请输出一份可直接阅读的学习笔记，严格按以下结构，**并严格控制在给定的条数与总字数内**
（材料可能很长，但笔记是给人快速抓重点的，不要穷举所有时间点）：

## 一句话总结
（不超过 60 字：讲什么 + 对谁有用）

## 内容大纲
（**最多 8 章**，每章 `[起止时间] 章节标题`，下面 1~3 条要点；点明这段画面在做什么）

## 核心知识点
（**最多 12 条**，每条标注时间点；代码/公式/命令原样给出，其余用概括）

## 画面证据
（**最多 8 条**屏幕上出现的关键内容：幻灯片标题、代码片段、图表数值、界面操作）

## 结论与行动项
（**最多 5 条**，讲者强调的结论、注意事项、可立刻上手的做法）

## 值得回看的片段
（**3~5 条**，时间点 + 一句话理由）

全篇正文控制在 **2000~3000 字**。宁可少而准，不要多而全。

---
{material}
"""

CHUNK_PROMPT = """这是一个长视频的第 {i}/{n} 段材料（含画面记录与语音文稿）。请提取这一段的要点：

- 按时间顺序列出 3~8 条要点，每条标注时间点
- 明确记录画面上出现的代码/公式/图表等关键内容
- 不要写总结性套话，只保留信息

材料：
{material}
"""


# --------------------------------------------------------------------------
# 时间轴对齐
# --------------------------------------------------------------------------
def align_timeline(transcript: Optional[Transcript], visual: Optional[VisualNote],
                   *, window: float = 25.0, max_visual: int = 200) -> str:
    """生成多模态时间轴文本：画面笔记为主干，附上该时刻附近的语音。

    transcript 可以为 None（原生直读模式没有单独转写，画面笔记里已含语音信息）。
    """
    if transcript is None or not transcript.segments:
        if visual is None:
            return ""
        # as_markdown() 里已经带了「画面概要」，这里不要再加一遍
        return visual.as_markdown()
    lines: List[str] = []
    items: List[FrameNote] = list(visual.items) if visual else []
    items.sort(key=lambda x: x.ts)
    items = items[:max_visual]

    if not items:
        return "（无画面记录，仅语音文稿）\n" + transcript.with_timestamps()

    used: List[int] = []
    for it in items:
        if it.text and "(见画面概要)" not in it.text:
            lines.append(f"[{fmt_ts(it.ts)}] 【画面·{it.kind}】{it.text}")
            # 附带这一时刻前后的语音，帮助模型理解画面在讲什么
            speech = transcript.window(max(0.0, it.ts - window), it.ts + window)
            if speech:
                lines.append(f"　　　【语音】{speech[:800]}")
        else:
            used.append(1)
    if visual and visual.gist:
        lines.insert(0, f"【整片画面印象】{visual.gist}\n")

    # 语音里没有被任何画面覆盖到的部分（比如纯口播段落）也补进来
    covered = [(it.ts - window, it.ts + window) for it in items]
    tail: List[str] = []
    for seg in transcript.segments:
        mid = (seg.start + seg.end) / 2.0
        if not any(a <= mid <= b for a, b in covered):
            tail.append(f"[{fmt_ts(seg.start)}] 【语音】{seg.text.strip()}")
    if tail:
        lines.append("")
        lines.append("【未被画面覆盖的语音段落】")
        lines.extend(tail[:400])
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 文本模型调用
# --------------------------------------------------------------------------
def _chat(prov: Provider, system: str, user: str, *, cfg: Config,
          max_tokens: int = 8000, log=print) -> str:
    from .http import post_json
    from .vision import TruncatedOutput, _message_text, thinking_fields

    url = prov.base_url.rstrip("/") + "/chat/completions"
    hdr = {"Authorization": f"Bearer {prov.api_key}"}
    models = [prov.model] + ([prov.fallback_model] if prov.fallback_model else [])
    last: Optional[Exception] = None
    for model in models:
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0.3,
            "max_tokens": max_tokens,
            **thinking_fields(cfg, task="summary"),
        }
        truncations = 0
        for attempt in range(3):
            # 生成长文本要花时间：timeout 必须随 max_tokens 放大，
            # 否则大预算请求 100% 会在半路超时，白等一轮
            budget = int(payload["max_tokens"])
            timeout = max(prov.timeout, 90.0 + budget / 25.0)
            try:
                data = post_json(url, payload, headers=hdr,
                                 timeout=timeout, retries=1)
                return _message_text(data).strip()
            except TruncatedOutput as e:
                last = e
                if truncations < 2 and payload["max_tokens"] < 16384:
                    truncations += 1
                    payload["max_tokens"] = min(16384, payload["max_tokens"] * 2)
                    log(f"↪ 报告被截断，max_tokens → {payload['max_tokens']} 重试")
                    continue
                raise
            except Exception as e:
                last = e
                msg = str(e).lower()
                if "model" in msg or "404" in msg:
                    log(f"↪ 文本模型 {model} 不可用，尝试备选")
                    break
                if attempt + 1 < cfg.retries:
                    time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"总结模型调用失败: {last}")


def _split_material(material: str, limit: int = 18000) -> List[str]:
    if len(material) <= limit:
        return [material]
    chunks, buf = [], []
    size = 0
    for line in material.splitlines(keepends=True):
        if size + len(line) > limit and buf:
            chunks.append("".join(buf))
            buf, size = [], 0
        buf.append(line)
        size += len(line)
    if buf:
        chunks.append("".join(buf))
    return chunks


def summarize_material(material: str, cfg: Config, log=print) -> str:
    """map-reduce：长材料先分段提炼，再合并成最终笔记。"""
    prov = cfg.text if cfg.text.ready else cfg.vlm
    if not prov.ready:
        raise RuntimeError("未配置总结模型（AICBILI_VLM_API_KEY / AICBILI_TEXT_MODEL）")
    chunks = _split_material(material)
    if len(chunks) == 1:
        log(f"🤖 生成笔记（材料 {len(material)} 字）...")
        return _chat(prov, SYS_PROMPT, FINAL_PROMPT.format(material=material),
                     cfg=cfg, max_tokens=8000, log=log)

    log(f"📑 材料 {len(material)} 字，分 {len(chunks)} 段提炼后合并")
    parts: List[str] = []
    for i, ch in enumerate(chunks, 1):
        log(f"  ↳ 第 {i}/{len(chunks)} 段...")
        parts.append(_chat(prov, SYS_PROMPT,
                           CHUNK_PROMPT.format(i=i, n=len(chunks), material=ch),
                           cfg=cfg, max_tokens=5000, log=log))
    merged = "\n\n".join(f"### 第{i}段\n{p}" for i, p in enumerate(parts, 1))
    log("🧩 合并分段要点，生成最终笔记...")
    return _chat(prov, SYS_PROMPT, FINAL_PROMPT.format(material=merged),
                 cfg=cfg, max_tokens=10000, log=log)


__all__ = ["align_timeline", "summarize_material", "SYS_PROMPT", "FINAL_PROMPT"]
