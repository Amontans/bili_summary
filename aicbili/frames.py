"""画面理解：用 B站快照做「场景变化检测」，再按需抽高清帧。

这里的核心取舍
--------------
* 顺序读完整条视频流做 `select='gt(scene,...)'` 很慢（实测 16 分钟视频要 72 秒），
  而且长视频会读完整个文件 —— 与「不下载视频」的目标相悖。
* B站给每条视频预生成了「视频快照」雪碧图：整片近 200 帧、总共约 1MB、
  2~4 秒就能下完。用它算帧间差异，就能以极低成本找到「画面真正变了」的时刻。
* 于是：**雪碧图决定看哪里，高清 seek 决定看什么**。
  16 分钟视频实测 8 帧并行 seek 只要 4.7 秒。
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from .bili import Storyboard

# 缩小到这个尺寸再比较差异：足够敏感，又不吃 CPU
_CMP_W, _CMP_H = 64, 36


@dataclass
class Candidate:
    """快照里的一帧：时间点 + 缩略图。"""

    ts: float
    image: Image.Image
    diff: float = 0.0


# --------------------------------------------------------------------------
# 雪碧图解析
# --------------------------------------------------------------------------
def split_sprite(data: bytes, cols: int, rows: int, cell_w: int = 0,
                 cell_h: int = 0) -> List[Image.Image]:
    """把一张 10x10 的雪碧图切成小图（按行优先）。"""
    img = Image.open(io.BytesIO(data))
    img.load()
    if img.mode != "RGB":
        img = img.convert("RGB")
    cols = max(1, cols)
    rows = max(1, rows)
    w, h = img.size
    cw = cell_w or w // cols
    ch = cell_h or h // rows
    cw = min(cw, w)
    ch = min(ch, h)
    out: List[Image.Image] = []
    for r in range(rows):
        for c in range(cols):
            box = (c * cw, r * ch, (c + 1) * cw, (r + 1) * ch)
            if box[2] > w or box[3] > h:
                continue
            cell = img.crop(box)
            out.append(cell)
    return out


def storyboard_candidates(sb: Storyboard, sheet_bytes: Sequence[bytes],
                          limit: int = 400) -> List[Candidate]:
    """把雪碧图列表还原成带时间戳的候选帧序列。"""
    per_sheet = sb.cols * sb.rows
    cands: List[Candidate] = []
    for si, data in enumerate(sheet_bytes):
        try:
            cells = split_sprite(data, sb.cols, sb.rows, sb.cell_w, sb.cell_h)
        except Exception:
            continue
        for ci, cell in enumerate(cells):
            fi = si * per_sheet + ci
            if fi >= len(sb.timestamps):
                break
            ts = float(sb.timestamps[fi])
            # 有些视频开头会有重复时间戳，跳过避免浪费视觉预算
            if cands and abs(ts - cands[-1].ts) < 0.5:
                continue
            cands.append(Candidate(ts=ts, image=cell))
            if len(cands) >= limit:
                return cands
    return cands


# --------------------------------------------------------------------------
# 变化检测
# --------------------------------------------------------------------------
def _signature(img: Image.Image) -> Tuple[List[int], List[float]]:
    """灰度缩略 + 亮度直方图，作为一帧的「指纹」。"""
    g = img.convert("L").resize((_CMP_W, _CMP_H), Image.BILINEAR)
    px = list(g.tobytes())   # 不用 getdata()，Pillow 14 起已废弃
    hist = g.histogram()
    total = float(sum(hist)) or 1.0
    return px, [h / total for h in hist]


def _diff(sig_a, sig_b) -> float:
    """0~1：像素差异(权重高) + 直方图差异(补充对整体位移的敏感度)。"""
    pa, ha = sig_a
    pb, hb = sig_b
    n = min(len(pa), len(pb)) or 1
    pix = sum(abs(pa[i] - pb[i]) for i in range(n)) / (255.0 * n)
    hdiff = 0.5 * sum(abs(ha[i] - hb[i]) for i in range(min(len(ha), len(hb))))
    return 0.65 * pix + 0.35 * hdiff


def score_changes(cands: Sequence[Candidate]) -> None:
    """就地写入每一帧与前一帧的差异值。"""
    if not cands:
        return
    cands[0].diff = 1.0
    prev = _signature(cands[0].image)
    for c in cands[1:]:
        cur = _signature(c.image)
        c.diff = _diff(prev, cur)
        prev = cur


def pick_keyframes(cands: Sequence[Candidate], max_frames: int = 24,
                   threshold: float = 0.12, min_gap: Optional[float] = None
                   ) -> List[Candidate]:
    """挑出「画面真正变了」的关键帧。

    规则：
      1. 首帧永远要（交代开场画面）。
      2. 差异超过阈值、且与上一个入选项间隔够远的帧，视为换场/翻页。
      3. 数量超过上限时，保留差异最大的那些，再按时间排序。
      4. 场景太静的片子（幻灯片、口播），变化点太少 → 均匀补帧。
    """
    if not cands:
        return []
    max_frames = max(1, max_frames)
    if min_gap is None:
        gap = 0.0
        if len(cands) > 1:
            gap = cands[1].ts - cands[0].ts
        min_gap = max(3.0, gap * 1.5)

    scored = list(cands)
    score_changes(scored)

    chosen: List[Candidate] = [scored[0]]
    for c in scored[1:]:
        if c.diff >= threshold and (c.ts - chosen[-1].ts) >= min_gap:
            chosen.append(c)

    if len(chosen) > max_frames:
        head = chosen[0]
        rest = sorted(chosen[1:], key=lambda c: c.diff, reverse=True)[: max_frames - 1]
        chosen = [head] + sorted(rest, key=lambda c: c.ts)

    # 太少 → 均匀补帧，保证长视频也有足够采样密度
    if len(chosen) < max_frames and len(scored) >= max_frames:
        need = max_frames - len(chosen)
        span_start, span_end = scored[0].ts, scored[-1].ts
        have = [c.ts for c in chosen]
        pool = [c for c in scored if c.ts not in have]
        if pool and span_end > span_start:
            slot = (span_end - span_start) / float(need + 1)
            for i in range(1, need + 1):
                target = span_start + slot * i
                if not pool:
                    break
                best = min(pool, key=lambda c: abs(c.ts - target))
                chosen.append(best)
                pool.remove(best)
        chosen = sorted(chosen, key=lambda c: c.ts)
    return chosen[:max_frames]


# --------------------------------------------------------------------------
# 拼图（少发请求、少花 token）
# --------------------------------------------------------------------------
def _font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "NotoSansCJK-Regular.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _ts_label(sec: float) -> str:
    s = int(sec)
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:d}:{s % 60:02d}"


def contact_sheet(shots: Sequence[Tuple[float, Image.Image]], *, cells: int = 4,
                  cell_w: int = 640, label: bool = True) -> Image.Image:
    """把若干帧拼成一张带时间戳标注的图，便于单次请求喂给视觉模型。"""
    shots = [s for s in shots if s[1] is not None]
    if not shots:
        raise ValueError("没有可拼接的画面")
    cells = max(1, cells)
    # 单元格数与列数的约定：1/2 单行，4=2x2，6=3x2，9=3x3，其余按 sqrt 推算
    layout = {1: (1, 1), 2: (2, 1), 3: (2, 2), 4: (2, 2), 6: (3, 2), 9: (3, 3)}
    cols, rows = layout.get(cells, (2, int(math.ceil(cells / 2)) or 1))
    rows = max(rows, int(math.ceil(len(shots) / cols)) or 1)

    label_h = 26 if label else 0
    # 统一单元格宽高，避免拼图出现错位
    ar = []
    for _, im in shots:
        ar.append(im.height / max(1, im.width))
    avg_ar = sum(ar) / len(ar) if ar else 0.5625
    cw = max(160, int(cell_w))
    ch = max(90, int(round(cw * avg_ar)))
    W = cols * cw
    H = rows * (ch + label_h)
    sheet = Image.new("RGB", (W, H), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    font = _font(18)
    for i, (ts, im) in enumerate(shots[: cols * rows]):
        r, c = divmod(i, cols)
        y0 = r * (ch + label_h)
        x0 = c * cw
        im2 = im.convert("RGB").resize((cw, ch), Image.LANCZOS)
        sheet.paste(im2, (x0, y0))
        if label_h:
            tag = f"#{i + 1}  {_ts_label(ts)}"
            draw.rectangle([x0, y0 + ch, x0 + cw, y0 + ch + label_h], fill=(0, 0, 0))
            draw.text((x0 + 6, y0 + ch + 3), tag, fill=(255, 220, 80), font=font)
    return sheet


def jpeg_bytes(img: Image.Image, quality: int = 82, max_side: Optional[int] = None) -> bytes:
    if max_side:
        w, h = img.size
        longest = max(w, h)
        if longest > max_side:
            scale = max_side / float(longest)
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def build_sheets(shots: Sequence[Tuple[float, Image.Image]], *, cells: int = 4,
                 cell_w: int = 640) -> List[Tuple[Image.Image, List[float]]]:
    """把镜头按 cells 一组切分并拼图，返回 [(拼图, [时间戳...])]。"""
    out: List[Tuple[Image.Image, List[float]]] = []
    cells = max(1, cells)
    for i in range(0, len(shots), cells):
        group = shots[i: i + cells]
        if not group:
            continue
        out.append((contact_sheet(group, cells=cells, cell_w=cell_w),
                    [ts for ts, _ in group]))
    return out


__all__ = [
    "Candidate", "build_sheets", "contact_sheet", "jpeg_bytes", "pick_keyframes",
    "score_changes", "split_sprite", "storyboard_candidates",
]
