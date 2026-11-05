"""论文式学习文章：把口语转录改写成可直接当教材读的完整正文。

和「摘要」的根本区别
--------------------
摘要是压缩，**必然丢信息**（实测长视频只保留 9%~35%）。
这里走的是「等价重写」：先把全片规划成章节，再**逐章把该段转录全部改写成书面正文**，
最后补摘要/引言/结论/术语表。信息量基本守恒，只是把口语变成文章。

流水线
------
    plan_chapters()  全片转录 → 章节划分（按主题，不按等长）
    write_section()  每章：该段完整转录 + 该段画面记录 → 书面正文（并行）
    make_abstract()  章节标题+brief → 摘要 + 关键词
    make_intro()     同上 → 引言
    make_back()      成稿 → 结论 + 术语表
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .asr import Transcript, fmt_ts
from .config import Config
from .vision import VisualNote, _extract_json

SYS = (
    "你是学术编辑与教材作者，擅长把讲座、访谈、教程的口语内容改写成严谨、完整、"
    "可独立阅读的中文书面文章。你的第一原则是**忠实且不遗漏**，第二原则是**书面化与结构化**。"
)

PLAN_PROMPT = """下面是一个视频的完整转录（带 [时:分:秒] 标记），以及画面记录的概要。

请把它规划成 {min_c}~{max_c} 个章节：

1. 按**主题**划分，绝不按等长时间机械切分；每章应是一个相对完整的知识点或论证单元。
2. 必须**覆盖全片、首尾相接**：第 1 章从 0 秒开始，最后一章结束于 {duration} 秒，中间不留空档。
3. 标题用学术化名词短语（如「梯度下降的收敛条件」），不要用「第一部分」「开场」这类空标题。
4. `brief` 用 1~2 句说明：这一章讲了什么、用了什么例子或论证。
5. 时间用整数秒。

严格输出 JSON，不要任何解释：
{{"chapters":[{{"title":"…","start":0,"end":312,"brief":"…"}}]}}

视频时长：{duration} 秒
视频标题：{title}

--- 完整转录 ---
{transcript}
"""

SECTION_PROMPT = """你是学术编辑，正在把一段视频讲座的口语转录改写成**教科书级的书面正文**。

## 铁律
**不得丢失任何信息。** 原文中的每一个论点、推理步骤、举例、类比、数字、公式、代码、
命令、注意事项、讲者的判断与保留意见，都必须出现在改写结果中。
宁可写长、宁可展开，也不许省略或用"等等""略"带过。

## 改写要求
1. 去掉纯口语成分：嗯/啊/这个/就是说/然后呢、口吃、重复描述、寒暄、与主题无关的闲聊。
2. 转成书面学术语体：完整句子、主语明确、逻辑连接词（首先/其次/因此/然而/具体而言）。
   把"我们来看一下""大家注意""对不对"这类课堂话术改成直接陈述。
3. 保留原论证结构：前提 → 推理 → 结论；若是操作演示，按步骤依次写清。
4. 例子必须写完整：不仅说"举了个例子"，要把例子的具体内容和它要说明的道理都写出来。
5. 代码、命令、公式、专有名词、数字**原样保留**（用行内代码或代码块）。
6. **禁止元叙述**：不要出现"本段介绍了""讲者提到""接下来我们讲""综上所述本节"。
   直接写内容本身。
7. 内容较长时可用 `###` 分小节，但不要写本章主标题，也不要写"小结/总结"。
8. 只输出 Markdown 正文，不要前言、后记、解释。
9. **篇幅**：改写后字数控制在原文的 **0.8~1.5 倍**。原文是口语碎片（如字幕逐句换行），
   书面化后自然会变长，但不许为了凑长度重复啰嗦，也不许为求简洁而省略。
10. **不得添加原文没有的内容**：不要补充你自己的背景知识、数据、建议或评价；
    原文没说的，就不要写。

## 本章定位
第 {i}/{n} 章：{title}
应覆盖：{brief}
（对应视频 {t0} – {t1}）

## 原文转录
{transcript}

## 同一时段的画面内容
用于还原"这个""这里"指代的东西。**画面中出现的代码、命令、文件名、报错信息、
界面文字、数字与图表数值，只要对理解有作用，就必须写进正文**（例如演示代码的完整写法、
运行结果、报错原文）。纯装饰性的画面（水印、纯人物镜头）可略。
"""

ABSTRACT_PROMPT = """下面是一篇由视频讲座整理成的长文的章节清单。请为它写：

1. **摘要**：300~500 字，说明：讨论的问题、覆盖的主要内容与论证、得出的结论或可行做法。
   要求信息密度高、可直接替代阅读前的预期管理。
2. **关键词**：4~6 个，用「、」分隔。

输出格式（严格）：
## 摘要
<正文>

**关键词**：词1、词2、词3

章节清单：
{outline}
"""

INTRO_PROMPT = """下面是一篇由视频讲座整理成的长文的章节清单与摘要。

请写一节 **引言**（400~800 字），要求：
- 交代背景：这个话题为什么值得讨论、通常的困惑或误区是什么；
- 说明本文要解决的核心问题；
- 概述全文的组织结构（"本文首先…随后…最后…"）；
- 用连贯段落写，不要用列表，不要写"本文由视频整理而成"这类元信息。

只输出引言正文（不要"## 引言"标题）。

视频标题：{title}
摘要：{abstract}

章节清单：
{outline}
"""

BACK_PROMPT = """下面是一篇长文的章节清单和正文节选。请写两部分：

## 结论
（400~700 字）把全文的主要结论收拢：哪些是被论证成立的、哪些是讲者的个人判断、
对读者最可操作的做法是什么。用连贯段落，不要罗列。

## 术语与概念
（8~20 条）从文中提取关键术语、专有名词、公式或命令，每条一行：
`- **术语**：一句话解释（取自原文语境，不要编造）`

章节清单：
{outline}

正文（每章给出首尾节选；"……（中略）……"处是被省略的段落，不代表原文没有内容，
**不要**据此写"本文未展开""节选未说明"这类话）：
{body}
"""


VERIFY_PROMPT = """下面是一段视频的【原始转录】和据其改写的【文章正文】。

请逐条对照，找出**正文遗漏**的信息点，只关注实质内容：
具体数字、代码/命令/文件名、报错原文、专有名词、举例的具体内容、明确的结论或注意事项。

判定标准：
- 只列**真正没被表达**的；正文用了别的说法但意思已覆盖的，**不算遗漏**。
- 不要挑剔措辞、顺序、详略。不要提改进建议。
- 每条一行：「- 遗漏点：… ｜ 原文：…（引用一小句）」
- 如果确实没有遗漏，只输出四个字：无遗漏

【原始转录】
{origin}

【文章正文】
{body}
"""

PATCH_PROMPT = """下面是一段【文章正文】，以及审校发现的【遗漏点】。

请把这些遗漏点的信息**补写进正文的恰当位置**，要求：
- 保持原文的行文风格与逻辑顺序，读起来是一篇连贯文章，不要变成补丁或列表堆叠。
- 只补信息，不要改写无关段落，不要新增原文没有的内容。
- 输出**完整正文**（Markdown），不要只输出改动部分，不要加任何说明。

【遗漏点】
{gaps}

【正文】
{body}
"""


class ArticleError(RuntimeError):
    pass


@dataclass
class Chapter:
    title: str
    start: float
    end: float
    brief: str = ""
    body: str = ""

    @property
    def span(self) -> str:
        return f"{fmt_ts(self.start)}–{fmt_ts(self.end)}"


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------
def _chat(cfg: Config, prompt: str, *, max_tokens: int, log=print,
          temperature: float = 0.35) -> str:
    from .summarize import _chat as _sum_chat

    prov = cfg.text if cfg.text.ready else cfg.vlm
    if not prov.ready:
        raise ArticleError("未配置总结模型（AICBILI_VLM_API_KEY / AICBILI_TEXT_MODEL）")
    return _sum_chat(prov, SYS, prompt, cfg=cfg, max_tokens=max_tokens, log=log)


def _ts_transcript(transcript: Transcript, max_chars: int = 60000) -> str:
    """带时间戳的转录；过长时按比例抽样（每句仍保留时间点，便于规划章节）。"""
    lines = [f"[{fmt_ts(s.start)}] {s.text.strip()}" for s in transcript.segments if s.text.strip()]
    if not lines:
        return ""
    total = sum(len(x) + 1 for x in lines)
    if total <= max_chars:
        return "\n".join(lines)
    # 抽样：保留头尾，其余等间隔取
    keep = max(40, int(len(lines) * max_chars / total))
    if keep >= len(lines):
        return "\n".join(lines)
    step = len(lines) / keep
    picked = sorted({int(i * step) for i in range(keep)} | {0, len(lines) - 1})
    return "\n".join(lines[i] for i in picked if 0 <= i < len(lines))


def _visual_for(visual: Optional[VisualNote], start: float, end: float,
                limit: int = 40) -> str:
    if not visual or not visual.items:
        return "（无画面记录）"
    buf = []
    for it in visual.items:
        if start - 5 <= it.ts <= end + 5 and it.text:
            buf.append(f"[{fmt_ts(it.ts)}] {it.text.strip()}")
        if len(buf) >= limit:
            break
    return "\n".join(buf) if buf else "（该时段无画面记录）"


def _outline(chapters: Sequence[Chapter]) -> str:
    return "\n".join(f"{i}. 「{c.title}」（{c.span}）：{c.brief}"
                     for i, c in enumerate(chapters, 1))


# --------------------------------------------------------------------------
# 阶段 1：章节规划
# --------------------------------------------------------------------------
def plan_chapters(transcript: Transcript, visual: Optional[VisualNote], cfg: Config,
                  *, title: str = "", log=print) -> List[Chapter]:
    duration = transcript.segments[-1].end if transcript.segments else 0.0
    dur = int(duration) or 1
    # 章节数与内容量挂钩：约每 3~5 分钟一章，限制在 4~14
    n_min = max(4, min(14, int(dur / 600) + 3))
    n_max = max(n_min + 2, min(16, int(dur / 300) + 3))

    prompt = PLAN_PROMPT.format(min_c=n_min, max_c=n_max, duration=dur,
                                title=title or "(未提供)",
                                transcript=_ts_transcript(transcript))
    raw = _chat(cfg, prompt, max_tokens=2000, log=log, temperature=0.2)
    data = _extract_json(raw)
    chapters: List[Chapter] = []
    for c in (data or {}).get("chapters") or []:
        try:
            chapters.append(Chapter(title=str(c.get("title") or "").strip() or "未命名章节",
                                    start=float(c.get("start") or 0),
                                    end=float(c.get("end") or 0),
                                    brief=str(c.get("brief") or "").strip()))
        except (TypeError, ValueError):
            continue
    if not chapters:
        raise ArticleError(f"章节规划失败，模型输出: {raw[:200]}")

    # 兜底修正：首尾相接、去掉空章、限制重叠
    chapters.sort(key=lambda c: c.start)
    chapters[0].start = 0.0
    for i, c in enumerate(chapters):
        if c.end <= c.start:
            c.end = min(duration, c.start + max(60.0, duration / len(chapters)))
        if i + 1 < len(chapters):
            nxt = chapters[i + 1]
            if c.end > nxt.start:                 # 有重叠就切到中点
                c.end = max(c.start + 1, (c.start + nxt.start) / 2)
            if nxt.start < c.end:                 # 仍然逆序则顺延
                nxt.start = c.end
    chapters[-1].end = max(chapters[-1].end, duration)
    log(f"  🗂 规划 {len(chapters)} 章：" +
        " ｜ ".join(c.title[:12] for c in chapters[:6]) + ("…" if len(chapters) > 6 else ""))
    return chapters


# --------------------------------------------------------------------------
# 阶段 2：逐章改写
# --------------------------------------------------------------------------
def write_section(ch: Chapter, i: int, n: int, transcript: Transcript,
                  visual: Optional[VisualNote], cfg: Config, log=print) -> str:
    text = transcript.window(ch.start, ch.end)
    if not text.strip():
        return ""
    prompt = SECTION_PROMPT.format(
        i=i, n=n, title=ch.title, brief=ch.brief or "（自由发挥，覆盖全部内容）",
        t0=fmt_ts(ch.start), t1=fmt_ts(ch.end),
        transcript=text, visual=_visual_for(visual, ch.start, ch.end))
    body = _chat(cfg, prompt, max_tokens=8000, log=log)
    return body.strip()


def verify_section(ch: Chapter, transcript: Transcript, cfg: Config, log=print) -> bool:
    """对照原文检查一章是否漏信息；有遗漏就补写。返回是否补过。"""
    origin = transcript.window(ch.start, ch.end).strip()
    if not origin or not ch.body:
        return False
    gaps = _chat(cfg, VERIFY_PROMPT.format(origin=origin, body=ch.body),
                 max_tokens=1500, log=log, temperature=0.1)
    g = gaps.strip()
    if not g or "无遗漏" in g[:20]:
        return False
    log(f"  🔎 「{ch.title[:16]}」发现遗漏：{len(g.splitlines())} 条，正在补写...")
    try:
        fixed = _chat(cfg, PATCH_PROMPT.format(gaps=g, body=ch.body),
                      max_tokens=8000, log=log, temperature=0.25).strip()
        if len(fixed) > len(ch.body) * 0.6:      # 防呆：补写结果不能明显变短
            ch.body = fixed
            return True
    except Exception as e:
        log(f"  ⚠ 补写失败（保留原正文）: {e}")
    return False


def _write_all(chapters: List[Chapter], transcript: Transcript,
               visual: Optional[VisualNote], cfg: Config, log=print) -> None:
    """并行改写各章；单章失败不影响其他章（失败的那章退回原文摘录）。"""
    n = len(chapters)
    workers = max(1, min(cfg.concurrency, 4))

    def one(idx: int) -> Tuple[int, str, Optional[str]]:
        ch = chapters[idx]
        try:
            return idx, write_section(ch, idx + 1, n, transcript, visual, cfg, log), None
        except Exception as e:  # noqa: BLE001
            return idx, "", str(e)

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, i): i for i in range(n)}
        for fut in cf.as_completed(futs):
            idx, body, err = fut.result()
            ch = chapters[idx]
            if body:
                ch.body = body
                log(f"  ✍️ {idx + 1}/{n} 「{ch.title[:20]}」 {len(body)} 字")
            else:
                # 降级：这一章直接用原文，保证信息不丢
                ch.body = ("> （本章改写失败，以下为该段原始转录）\n\n"
                           + transcript.window(ch.start, ch.end).strip())
                log(f"  ⚠ {idx + 1}/{n} 「{ch.title[:20]}」改写失败，已回退为原文转录: {err}")


# --------------------------------------------------------------------------
# 阶段 3：摘要 / 引言 / 结论
# --------------------------------------------------------------------------
def make_front(chapters: List[Chapter], title: str, cfg: Config, log=print) -> Tuple[str, str]:
    outline = _outline(chapters)
    abstract_raw = _chat(cfg, ABSTRACT_PROMPT.format(outline=outline),
                         max_tokens=2000, log=log, temperature=0.3)
    abstract = abstract_raw.strip()
    if abstract.startswith("## 摘要"):
        abstract = abstract[len("## 摘要"):].lstrip()
    intro = _chat(cfg, INTRO_PROMPT.format(title=title or "(未提供)", abstract=abstract[:1200],
                                           outline=outline),
                  max_tokens=3000, log=log, temperature=0.35).strip()
    return abstract.strip(), intro.strip()


def make_back(chapters: List[Chapter], cfg: Config, log=print) -> str:
    """生成结论 + 术语表。

    注意：这里必须给到**每章的首尾**。早期版本只取前 600 字，结果章末的关键结论
    （比如「字符串和整数不能相加」）没进上下文，模型就在结论里写下
    「本节选未展开」这种错误的自我否定。
    """
    chunks = []
    for c in chapters:
        if not c.body:
            continue
        b = c.body.strip()
        if len(b) > 1400:
            b = b[:800] + "\n……（中略）……\n" + b[-600:]   # 首 + 尾，宁可重叠
        chunks.append(f"【{c.title}】\n{b}")
    body = "\n\n".join(chunks)[:30000]
    return _chat(cfg, BACK_PROMPT.format(outline=_outline(chapters), body=body),
                 max_tokens=5000, log=log, temperature=0.3).strip()


# --------------------------------------------------------------------------
# 组装
# --------------------------------------------------------------------------
def _strip_h2(text: str, name: str) -> str:
    for h in (f"## {name}", f"# {name}", name):
        if text.lstrip().startswith(h):
            return text.lstrip()[len(h):].lstrip()
    return text


def build_article(title: str, transcript: Transcript, visual: Optional[VisualNote],
                  cfg: Config, log=print) -> str:
    """生成论文式正文（不含一级标题，供 pipeline 统一加头部）。"""
    if not transcript or not transcript.segments:
        raise ArticleError("没有文稿，无法生成论文式文章")

    chapters = plan_chapters(transcript, visual, cfg, title=title, log=log)

    log(f"  ✍️ 逐章改写（{len(chapters)} 章，{min(cfg.concurrency, 4)} 路并行）...")
    _write_all(chapters, transcript, visual, cfg, log)

    if getattr(cfg, "verify", False):
        log(f"  🔎 逐章自检遗漏（{len(chapters)} 章）...")
        fixed = 0
        for i, ch in enumerate(chapters, 1):
            try:
                if verify_section(ch, transcript, cfg, log):
                    fixed += 1
            except Exception as e:
                log(f"  ⚠ 第{i}章自检失败: {e}")
        log(f"  🔎 自检完成：补写 {fixed} 章")

    log("  📝 生成摘要与引言...")
    try:
        abstract, intro = make_front(chapters, title, cfg, log)
    except Exception as e:
        log(f"  ⚠ 摘要/引言生成失败（不影响正文）: {e}")
        abstract, intro = "", ""

    log("  📝 生成结论与术语表...")
    try:
        back = make_back(chapters, cfg, log)
    except Exception as e:
        log(f"  ⚠ 结论生成失败: {e}")
        back = ""

    parts: List[str] = []
    if abstract:
        parts += ["## 摘要", "", abstract, ""]
    if intro:
        parts += ["## 引言", "", intro, ""]

    numerals = "一二三四五六七八九十"
    for i, ch in enumerate(chapters, 1):
        mark = numerals[i - 1] if i <= len(numerals) else str(i)
        parts += [f"## {mark}、{ch.title}", "", ch.body.strip(), ""]

    if back:
        parts += [back, ""]

    # 章节时间索引（附录，便于回看原视频）
    parts += ["---", "", "## 附：章节时间索引", ""]
    for i, ch in enumerate(chapters, 1):
        parts.append(f"- **{ch.title}** ｜ {ch.span}")
    return "\n".join(parts).strip()


__all__ = ["ArticleError", "Chapter", "build_article", "plan_chapters", "write_section"]
