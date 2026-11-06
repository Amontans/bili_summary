"""B站 API 封装：解析视频、取 DASH 流地址、快照雪碧图、字幕。

这里刻意不走 yt-dlp：直接拿 playurl 的 DASH 地址，交给 ffmpeg 远端抽帧/抽音频，
可以按 HTTP Range 只取需要的字节，是「不下载完整视频」的前提。
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import http as _http

API = "https://api.bilibili.com"
REFERER = "https://www.bilibili.com"

_MIXIN_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
    33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
    61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
    36, 20, 34, 44, 52,
]

_BV_RE = re.compile(r"BV[0-9A-Za-z]{10}")
_AV_RE = re.compile(r"av(\d+)", re.I)


class BiliError(RuntimeError):
    """B站接口返回了非 0 的 code。"""

    def __init__(self, code: int, message: str, where: str = ""):
        self.code = code
        self.message = message
        self.where = where
        super().__init__(f"{where}: code={code} {message}" if where else f"code={code} {message}")


# --------------------------------------------------------------------------
# 数据模型
# --------------------------------------------------------------------------
@dataclass
class Page:
    cid: int
    page: int
    part: str
    duration: int


@dataclass
class Stream:
    """一条 DASH 流（视频或音频）。"""

    url: str
    backups: List[str] = field(default_factory=list)
    id: int = 0
    codec: str = ""
    width: int = 0
    height: int = 0
    bandwidth: int = 0
    mime: str = ""

    @property
    def all_urls(self) -> List[str]:
        return [self.url] + list(self.backups)

    def host(self) -> str:
        try:
            return urllib.parse.urlsplit(self.url).netloc
        except ValueError:
            return ""


@dataclass
class Dash:
    duration: int
    videos: List[Stream]
    audios: List[Stream]

    def best_video(self, max_height: int = 2160) -> Optional[Stream]:
        """挑分辨率最高的 H.264 流（兼容性最好；无 H.264 时退其它编码）。"""
        cands = [v for v in self.videos if v.height <= max_height] or self.videos
        if not cands:
            return None
        h264 = [v for v in cands if v.codec.startswith("avc1")]
        pool = h264 or cands
        return max(pool, key=lambda v: (v.height, v.bandwidth))

    def best_audio(self) -> Optional[Stream]:
        if not self.audios:
            return None
        # 只要普通 AAC 音轨：杜比/FLAC 解析更麻烦且没必要
        return max(self.audios, key=lambda a: a.bandwidth)


@dataclass
class Storyboard:
    """B站预生成的视频快照（雪碧图）：整片几十~几百帧，通常只有 1MB 左右。"""

    sheets: List[str]
    cols: int
    rows: int
    cell_w: int
    cell_h: int
    timestamps: List[int]
    sampled_total: int = 0        # 场景检测实际用了多少帧（可能被均匀降采样）

    @property
    def frame_count(self) -> int:
        return len(self.timestamps)

    @property
    def interval(self) -> float:
        """相邻帧的时间间隔（取正差的中位数，容忍开头重复时间戳）。"""
        gaps = [b - a for a, b in zip(self.timestamps, self.timestamps[1:]) if b - a > 0]
        if not gaps:
            return 0.0
        gaps.sort()
        return float(gaps[len(gaps) // 2])


@dataclass
class SubtitleTrack:
    lan: str
    lan_doc: str
    url: str
    is_ai: bool = False


@dataclass
class Video:
    bvid: str
    aid: int
    cid: int
    page: int
    title: str
    part: str
    duration: int
    owner: str = ""
    desc: str = ""
    pages: List[Page] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.bvid}_p{self.page}" if self.page > 1 else self.bvid

    @property
    def page_url(self) -> str:
        u = f"{REFERER}/video/{self.bvid}"
        if self.page > 1:
            u += f"?p={self.page}"
        return u


# --------------------------------------------------------------------------
# 客户端
# --------------------------------------------------------------------------
class BiliClient:
    def __init__(self, cookie: str = "", user_agent: str = "", timeout: float = 20.0,
                 retries: int = 3):
        self.timeout = timeout
        self.retries = max(1, retries)
        self._wbi_keys: Optional[Tuple[str, str]] = None
        self._wbi_ts = 0.0
        headers = {
            "User-Agent": user_agent or (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
            "Referer": REFERER,
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Origin": REFERER,
        }
        if cookie:
            headers["Cookie"] = cookie
        self._headers = headers

    def close(self) -> None:      # 纯标准库实现，无需关闭连接池
        return None

    def __enter__(self) -> "BiliClient":
        return self

    def __exit__(self, *a) -> None:
        self.close()

    # ---------------------------- 基础请求 ----------------------------
    def _get_json(self, url: str, params: Optional[dict] = None, *,
                  where: str = "", allow_codes: Tuple[int, ...] = ()) -> Dict[str, Any]:
        last: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                r = _http.request("GET", url, headers=self._headers, params=params,
                                  timeout=self.timeout, retries=1)
                if r.status == 412:
                    raise BiliError(-412, "被风控限流（412），请稍后再试或配置 Cookie", where)
                data = r.json()
                code = data.get("code", 0)
                if code != 0 and code not in allow_codes:
                    raise BiliError(code, data.get("message") or data.get("msg") or "", where)
                return data
            except BiliError:
                raise
            except Exception as e:  # 网络抖动重试
                last = e
                if attempt + 1 < self.retries:
                    time.sleep(0.8 * (attempt + 1))
        raise BiliError(-1, f"请求失败: {last}", where)

    # ---------------------------- WBI 签名 ----------------------------
    def _get_wbi_keys(self) -> Tuple[str, str]:
        if self._wbi_keys and time.time() - self._wbi_ts < 3600:
            return self._wbi_keys
        data = self._get_json(f"{API}/x/web-interface/nav", where="nav")
        wbi = (data.get("data") or {}).get("wbi_img") or {}
        img = str(wbi.get("img_url") or "")
        sub = str(wbi.get("sub_url") or "")
        img_key = img.rsplit("/", 1)[-1].split(".")[0]
        sub_key = sub.rsplit("/", 1)[-1].split(".")[0]
        if not img_key or not sub_key:
            raise BiliError(-1, "无法获取 WBI 密钥", "nav")
        self._wbi_keys = (img_key, sub_key)
        self._wbi_ts = time.time()
        return self._wbi_keys

    def sign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """按 B站 WBI 规则给参数加 wts / w_rid。"""
        img_key, sub_key = self._get_wbi_keys()
        raw = img_key + sub_key
        mixin = "".join(raw[i] for i in _MIXIN_TAB)[:32]
        p = dict(params)
        p["wts"] = int(time.time())
        items = []
        for k in sorted(p):
            v = str(p[k])
            v = re.sub(r"[!'()*]", "", v)
            items.append((k, v))
        query = urllib.parse.urlencode(items)
        w_rid = hashlib.md5((query + mixin).encode()).hexdigest()
        p["w_rid"] = w_rid
        return p

    # ---------------------------- 解析输入 ----------------------------
    @staticmethod
    def parse_target(text: str) -> Dict[str, Optional[Any]]:
        """从链接/BV号/av号里提取 bvid / aid / 分P。"""
        t = (text or "").strip()
        page = 1
        m = re.search(r"[?&]p=(\d+)", t)
        if m:
            page = max(1, int(m.group(1)))
        bv = _BV_RE.search(t)
        if bv:
            return {"bvid": bv.group(0), "aid": None, "page": page}
        av = _AV_RE.search(t)
        if av:
            return {"bvid": None, "aid": int(av.group(1)), "page": page}
        return {"bvid": None, "aid": None, "page": page}

    def expand_short_link(self, text: str) -> str:
        """b23.tv 短链 → 真实链接。"""
        t = (text or "").strip()
        if not re.search(r"https?://b23\.tv/", t, re.I):
            return t
        try:
            r = _http.request("GET", t, headers=self._headers, timeout=self.timeout, retries=1)
            return r.url
        except Exception:
            return t

    # ---------------------------- 元信息 ----------------------------
    def pages(self, bvid: str) -> List[Page]:
        d = self._get_json(f"{API}/x/player/pagelist", {"bvid": bvid}, where="pagelist")
        out = []
        for it in d.get("data") or []:
            out.append(Page(cid=int(it["cid"]), page=int(it.get("page") or 1),
                            part=str(it.get("part") or ""), duration=int(it.get("duration") or 0)))
        return out

    def view(self, bvid: str) -> Dict[str, Any]:
        d = self._get_json(f"{API}/x/web-interface/view", {"bvid": bvid}, where="view")
        return d.get("data") or {}

    def resolve(self, text: str, page: Optional[int] = None) -> Video:
        """把任意输入（完整链接/短链/BV/av/分享文案）解析成 Video。"""
        if not text or not text.strip():
            raise BiliError(-1, "空输入", "resolve")
        expanded = self.expand_short_link(text)
        tgt = self.parse_target(expanded)
        if tgt["bvid"] is None and tgt["aid"] is None:
            raise BiliError(-1, f"无法从输入中识别 BV/av 号: {text[:80]}", "resolve")
        if tgt["bvid"] is None:
            d = self._get_json(f"{API}/x/web-interface/view", {"aid": tgt["aid"]}, where="view")
            data = d.get("data") or {}
            bvid = data.get("bvid")
            if not bvid:
                raise BiliError(-1, "aid 无法转换为 bvid", "resolve")
        else:
            bvid = tgt["bvid"]
            data = self.view(bvid)

        want_page = int(page or tgt["page"] or 1)
        pages = self.pages(bvid)
        if not pages:
            raise BiliError(-1, "该视频没有可播放分P", "resolve")
        want_page = min(max(1, want_page), len(pages))
        cur = pages[want_page - 1]

        return Video(
            bvid=bvid,
            aid=int(data.get("aid") or 0),
            cid=cur.cid,
            page=want_page,
            title=str(data.get("title") or bvid),
            part=cur.part,
            duration=cur.duration or int(data.get("duration") or 0),
            owner=str((data.get("owner") or {}).get("name") or ""),
            desc=str(data.get("desc") or ""),
            pages=pages,
        )

    # ---------------------------- 播放流 ----------------------------
    def dash(self, bvid: str, cid: int, quality: int = 80) -> Dash:
        params = {
            "bvid": bvid,
            "cid": cid,
            "fnval": 4048,      # DASH + 4K + HDR + 杜比 + 8K
            "fnver": 0,
            "fourk": 1,
            "qn": quality,
        }
        d = self._get_json(f"{API}/x/player/playurl", params, where="playurl")
        data = d.get("data") or {}
        dash = data.get("dash")
        if not dash:
            raise BiliError(-1, "该视频没有 DASH 流（可能是付费/地区限制/番剧）", "playurl")

        videos: List[Stream] = []
        for v in dash.get("video") or []:
            videos.append(Stream(
                url=str(v.get("baseUrl") or v.get("base_url") or ""),
                backups=[str(u) for u in (v.get("backupUrl") or v.get("backup_url") or [])],
                id=int(v.get("id") or 0),
                codec=str(v.get("codecs") or ""),
                width=int(v.get("width") or 0),
                height=int(v.get("height") or 0),
                bandwidth=int(v.get("bandwidth") or 0),
                mime=str(v.get("mimeType") or v.get("mime_type") or ""),
            ))
        audios: List[Stream] = []
        audio_items = list(dash.get("audio") or [])
        for d in (dash.get("dolby") or {}).get("audio") or []:
            audio_items.append(d)          # 杜比音轨同样是 AAC 封装，可以直接用
        for a in audio_items:
            audios.append(Stream(
                url=str(a.get("baseUrl") or a.get("base_url") or ""),
                backups=[str(u) for u in (a.get("backupUrl") or a.get("backup_url") or [])],
                id=int(a.get("id") or 0),
                codec=str(a.get("codecs") or ""),
                bandwidth=int(a.get("bandwidth") or 0),
                mime=str(a.get("mimeType") or a.get("mime_type") or ""),
            ))
        flac = (dash.get("flac") or {}).get("audio")
        if flac:
            audios.append(Stream(
                url=str(flac.get("baseUrl") or ""),
                backups=[str(u) for u in (flac.get("backupUrl") or [])],
                id=int(flac.get("id") or 0), codec=str(flac.get("codecs") or ""),
                bandwidth=int(flac.get("bandwidth") or 0),
            ))
        return Dash(duration=int(dash.get("duration") or 0), videos=videos, audios=audios)

    # ---------------------------- 快照雪碧图 ----------------------------
    def storyboard(self, bvid: str, cid: int, attempts: int = 3) -> Optional[Storyboard]:
        """取快照雪碧图。这个接口偶尔返回空的 index（风控/抖动），所以要多试几次。"""
        data: Dict[str, Any] = {}
        for i in range(max(1, attempts)):
            try:
                d = self._get_json(f"{API}/x/player/videoshot",
                                   {"bvid": bvid, "cid": cid, "index": 1}, where="videoshot")
            except BiliError:
                return None
            data = d.get("data") or {}
            if data.get("indexs") or data.get("index"):
                break
            time.sleep(0.7 * (i + 1))
        sheets = []
        for u in data.get("image") or []:
            u = str(u)
            sheets.append("https:" + u if u.startswith("//") else u)
        ts = data.get("indexs") or data.get("index") or []
        try:
            ts = [int(x) for x in ts]
        except (TypeError, ValueError):
            ts = []
        if not sheets or not ts:
            return None
        return Storyboard(
            sheets=sheets,
            cols=int(data.get("img_x_len") or 10),
            rows=int(data.get("img_y_len") or 10),
            cell_w=int(data.get("img_x_size") or 0),
            cell_h=int(data.get("img_y_size") or 0),
            timestamps=ts,
        )

    # ---------------------------- 字幕 ----------------------------
    def subtitles(self, bvid: str, cid: int) -> List[SubtitleTrack]:
        """取字幕轨列表（需要登录 Cookie 才能看到 B站 AI 字幕）。"""
        try:
            params = self.sign({"bvid": bvid, "cid": cid})
            d = self._get_json(f"{API}/x/player/wbi/v2", params, where="player/v2")
        except BiliError:
            try:  # 回退到非 wbi 接口（无登录也能用）
                d = self._get_json(f"{API}/x/player/v2",
                                   {"bvid": bvid, "cid": cid}, where="player/v2")
            except BiliError:
                return []
        sub = ((d.get("data") or {}).get("subtitle") or {})
        out: List[SubtitleTrack] = []
        for it in sub.get("subtitles") or []:
            url = str(it.get("subtitle_url") or "")
            if url.startswith("//"):
                url = "https:" + url
            if not url:
                continue
            lan = str(it.get("lan") or "")
            out.append(SubtitleTrack(
                lan=lan,
                lan_doc=str(it.get("lan_doc") or lan),
                url=url,
                is_ai=bool(it.get("ai_status")) or lan.startswith("ai-"),
            ))
        return out

    def fetch_subtitle(self, track: SubtitleTrack) -> List[Dict[str, Any]]:
        """下载并解析字幕 JSON → [{start, end, text}, ...]。"""
        try:
            data = _http.get_json(track.url, headers=self._headers,
                                 timeout=15, retries=2)
        except Exception:
            return []
        cues = []
        for b in data.get("body") or []:
            txt = str(b.get("content") or "").strip()
            if not txt:
                continue
            try:
                cues.append({
                    "start": float(b.get("from") or 0),
                    "end": float(b.get("to") or 0),
                    "text": txt,
                })
            except (TypeError, ValueError):
                continue
        return cues


def pick_subtitle(tracks: List[SubtitleTrack]) -> Optional[SubtitleTrack]:
    """优先中文人工字幕 → 中文 AI 字幕 → 任意中文 → 第一条。"""

    def score(t: SubtitleTrack) -> Tuple[int, int]:
        zh = 1 if (t.lan.startswith("zh") or "中文" in t.lan_doc) else 0
        manual = 0 if t.is_ai else 1
        return (zh, manual)

    if not tracks:
        return None
    return max(tracks, key=score)
