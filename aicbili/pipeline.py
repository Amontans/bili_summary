"""编排：一次分析 = 元信息 → (字幕 | 音频转写) ∥ 画面理解 → 对齐 → 总结。

并行设计（这是「更快」的核心）
------------------------------
时间线上真正串行的只有「解析 + 取字幕列表」约 1 秒。之后分成两条腿同时跑：

    ┌─ 语音腿：抽音频(97x 实时) → 转写(慢)
    └─ 画面腿：下快照(≈1MB) → 场景检测 → 并行抽高清帧 → 视觉模型(快)

画面腿通常十几秒就跑完，等语音腿一结束立刻对齐总结。对长视频来说，
整体耗时基本等于「转写耗时」，而画面理解是白送的。
如果能用上 B站字幕，转写耗时也几乎归零。
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import threading
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from . import asr, frames, media, summarize, vision
from .bili import BiliClient, BiliError, Stream, SubtitleTrack, Video, pick_subtitle
from .config import Config

MAX_FILENAME = 90


# --------------------------------------------------------------------------
@dataclass
class Result:
    video: Optional[Video] = None
    transcript: Optional[asr.Transcript] = None
    visual: Optional[vision.VisualNote] = None
    summary: str = ""
    mode: str = ""
    files: Dict[str, str] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    skipped: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def _safe_name(title: str, key: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|\n\r\t]", " ", title or key).strip()
    name = re.sub(r"\s+", " ", name)
    return (name[:MAX_FILENAME] or key) + f" [{key}]"


_REVIEW_HEADINGS = ("## 值得回看的片段", "# 值得回看的片段", "值得回看的片段")


def _strip_heading(text: str) -> str:
    """去掉正文开头那一行标题（模型会写成 '## 值得回看的片段'）。

    只剥离开头的 # 标题行，正文里的 markdown 列表/加粗原样保留。
    """
    lines = text.strip().splitlines()
    while lines and lines[0].strip().startswith("#"):
        lines.pop(0)
    return "\n".join(lines).strip()


def _split_review(summary: str) -> Tuple[str, str]:
    """把总结切成「笔记本体」与「值得回看的片段」两部分。"""
    if not summary:
        return "", ""
    for h in _REVIEW_HEADINGS:
        i = summary.find(h)
        if i != -1:
            return summary[:i].rstrip(), summary[i:].strip()
    return summary.strip(), ""


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class Analyzer:
    def __init__(self, cfg: Config, log=print):
        self.cfg = cfg
        self.log = log
        self.client = BiliClient(cookie=cfg.cookie_header, user_agent=cfg.user_agent,
                                 retries=cfg.retries)
        self.vlm = vision.VisionModel(cfg, log=log)

    def close(self) -> None:
        self.client.close()

    # ------------------------------------------------------------------
    def analyze(self, url: str, *, force: bool = False, no_ai: bool = False,
                dry_run: bool = False) -> Result:
        t_start = time.monotonic()
        res = Result()
        try:
            video = self.client.resolve(url)
        except BiliError as e:
            res.error = f"解析失败: {e}"
            return res
        res.video = video
        assert video is not None
        key = video.key
        workdir = self.cfg.cache_root() / key
        workdir.mkdir(parents=True, exist_ok=True)
        outdir = Path(self.cfg.output_dir).expanduser()
        outdir.mkdir(parents=True, exist_ok=True)
        stem = _safe_name(video.title, key)
        md_path = outdir / f"{stem}.md"

        meta = _load_json(workdir / "meta.json")
        if not force and md_path.exists() and meta.get("status") == "done":
            res.skipped = True
            res.files = dict(meta.get("files") or {})
            res.timings = dict(meta.get("timings") or {})
            res.mode = str(meta.get("mode") or "")
            res.summary = (meta.get("summary") or "")[:400]
            self.log(f"⏭ 已完成，跳过（--force 可重做）: {video.title[:50]}")
            return res

        self.log(f"▶ {video.title}")
        self.log(f"  {video.bvid} P{video.page} · {video.duration // 60}分{video.duration % 60}秒"
                 + (f" · UP: {video.owner}" if video.owner else ""))

        # ---------------- 1. 播放流 ----------------
        try:
            dash = self.client.dash(video.bvid, video.cid, quality=self.cfg.quality)
        except BiliError as e:
            res.error = f"取播放流失败: {e}"
            return res
        vstream = dash.best_video()
        astream = dash.best_audio()
        if vstream is None:
            res.error = "没有可用的视频流"
            return res
        duration = float(dash.duration or video.duration or 0)
        self.log(f"  DASH: {vstream.width}x{vstream.height} {vstream.codec.split('.')[0]}"
                 f" | 音频 {astream.bandwidth // 1000 if astream else 0}kbps"
                 f" | 时长 {int(duration)}s")

        mode = self.cfg.mode
        want_vision = mode in ("auto", "frames", "native") and not dry_run

        # ---------------- 2. 字幕探测（最便宜，通常 <300ms） ----------------
        t0 = time.monotonic()
        track: Optional[SubtitleTrack] = None
        try:
            track = pick_subtitle(self.client.subtitles(video.bvid, video.cid))
        except Exception as e:
            self.log(f"  ⚠ 取字幕列表失败（忽略）: {e}")
        res.timings["subtitles_probe"] = time.monotonic() - t0
        if track:
            self.log(f"  📜 发现字幕轨: {track.lan_doc}{' (AI)' if track.is_ai else ''}")

        # ---------------- 3a. 原生直读：让模型自己看远端视频 ----------------
        # 成功的话既不用抽帧也不用转写，耗时只剩一次模型调用。
        native_note: Optional[vision.VisualNote] = None
        native_wanted = mode == "native" or (bool(self.cfg.relay) and mode == "auto")
        if native_wanted and want_vision and not no_ai:
            if not self.vlm.ready:
                self.log("  ⚠ 未配置视觉模型，无法原生直读，改用抽帧模式")
            else:
                native_note = self._try_native(video, vstream)

        if native_note is not None:
            res.visual = native_note
            # 模型已经把声音也听了，这里只顺手拿一下免费的字幕（有就用，没有就算）
            transcript = self._subtitle_only(video, track)
            res.transcript = transcript
            if transcript:
                self.log(f"  ✅ 附带字幕文稿 {transcript.chars} 字（{transcript.source}）")
        else:
            # ---------------- 3b. 语音腿 + 画面腿 并行 ----------------
            pool = cf.ThreadPoolExecutor(max_workers=max(2, self.cfg.concurrency))
            try:
                f_transcript = pool.submit(self._speech_leg, video, track, astream,
                                           workdir, dry_run)
                # --no-ai 时仍然抽帧（便宜且有诊断价值），只是不调用视觉模型
                f_visual = (pool.submit(self._frames_leg, video, vstream, duration, workdir,
                                        describe=not no_ai) if want_vision else None)

                try:
                    transcript = f_transcript.result()
                except Exception as e:
                    self.log(f"  ⚠ 转写失败（继续用画面结果出报告）: {e}")
                    transcript = None
                res.transcript = transcript
                if transcript:
                    self.log(f"  ✅ 文稿 {transcript.chars} 字 / {len(transcript.segments)} 段"
                             f"（{transcript.source}，{transcript.elapsed:.1f}s）")

                visual = None
                if f_visual is not None:
                    try:
                        visual = f_visual.result()
                    except Exception as e:
                        self.log(f"  ⚠ 画面理解失败（降级为纯文稿）: {e}")
                res.visual = visual
            finally:
                pool.shutdown(wait=True)

        if res.visual:
            self.log(f"  🖼 画面笔记 {len(res.visual.items)} 条"
                     f"（{res.visual.source}，{res.visual.elapsed:.1f}s）")

        # ---------------- 4. 对齐 + 总结 ----------------
        transcript = res.transcript
        visual = res.visual
        files: Dict[str, str] = {}
        if transcript and transcript.chars > 0 and transcript.segments \
                and transcript.segments[-1].end > 0:
            srt_path = outdir / f"{stem}.srt"
            asr.write_srt(transcript.segments, str(srt_path))
            files["srt"] = str(srt_path)

        timeline = ""
        if transcript or visual:
            timeline = summarize.align_timeline(transcript, visual)
            if timeline:
                (workdir / "timeline.md").write_text(timeline, encoding="utf-8")

        # 原生直读时模型已经给出完整结构化结果，不必再花一次调用重写
        native_only = bool(visual and visual.source == "native" and not transcript)
        if native_only:
            res.summary = visual.items[0].text if visual.items else ""
            self.log("  ⏩ 原生直读已产出结构化结果，跳过二次总结")
        elif (not dry_run and not no_ai and not media.stop_requested()
              and (transcript or visual) and timeline):
            t0 = time.monotonic()
            try:
                res.summary = summarize.summarize_material(timeline, self.cfg, log=self.log)
                res.timings["summary"] = time.monotonic() - t0
            except Exception as e:
                res.error = f"总结失败: {e}"
                self.log(f"  ⚠ {e}")
        elif dry_run:
            res.summary = "（dry-run：仅解析，未抽帧/转写/总结）"
        elif no_ai:
            res.summary = "（--no-ai：已跳过 AI 总结，下方为转写与画面记录）"

        # ---------------- 5. 落盘 ----------------
        if res.transcript:
            res.timings["transcript"] = res.transcript.elapsed
        if res.visual:
            res.timings["visual"] = res.visual.elapsed
        res.timings["total"] = time.monotonic() - t_start
        res.files = files
        res.mode = mode
        # 拆分总结：md 只留「笔记本体」，「值得回看的片段」挪到 txt，
        # 这样 md / txt / srt 三者内容不重复。
        note_body, review = _split_review(res.summary)
        md = self._render_markdown(video, res, note_body)
        md_path.write_text(md, encoding="utf-8")
        files["report"] = str(md_path)

        txt = self._render_text(video, res, review, timeline)
        txt_path = outdir / f"{stem}.txt"
        txt_path.write_text(txt, encoding="utf-8")
        files["transcript"] = str(txt_path)

        # dry-run 不算完成，否则下次会被误判为「已完成」而跳过
        status = "dry-run" if dry_run else ("done" if res.ok else "partial")
        _dump_json(workdir / "meta.json", {
            "status": status,
            "bvid": video.bvid, "cid": video.cid, "page": video.page,
            "title": video.title, "part": video.part,
            "mode": mode,
            "transcript_source": transcript.source if transcript else "",
            "visual_frames": len(visual.items) if visual else 0,
            "timings": res.timings,
            "files": files,
            "summary": res.summary[:400],
            "error": res.error,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        if res.ok and not self.cfg.keep_cache:
            self._prune(workdir, keep=("meta.json", "timeline.md"))
        return res

    # ------------------------------------------------------------------
    def _speech_leg(self, video: Video, track: Optional[SubtitleTrack],
                    astream: Optional[Stream], workdir: Path, dry_run: bool
                    ) -> Optional[asr.Transcript]:
        # 1) 有字幕就用字幕：零转写耗时
        if track is not None and self.cfg.asr_backend in ("auto", "subtitle"):
            t0 = time.monotonic()
            try:
                cues = self.client.fetch_subtitle(track)
                if cues:
                    tr = asr.from_cues(cues, source=f"subtitle:{track.lan}")
                    tr.elapsed = time.monotonic() - t0
                    self.log(f"  📜 使用 B站字幕（跳过 ASR）: {len(cues)} 条")
                    return tr
                self.log("  ⚠ 字幕为空，改用音频转写")
            except Exception as e:
                self.log(f"  ⚠ 字幕下载失败（改用音频）: {e}")

        if astream is None:
            self.log("  ⚠ 没有音频流，无法转写")
            return None

        # 抽音频前先确认「有没有能用的转写后端」——没有就别白下几 MB
        backends = asr.available_backends(self.cfg)
        if not backends:
            self.log("  ⏭ 没有可用的转写后端，跳过音频抽取（不浪费流量）")
            self.log("     想要文稿：配 SESSDATA 用 B站字幕，或 sudo pacman -S whisper-cpp")
            return None

        # 2) 抽音频 → 转写
        audio_path = workdir / "audio.mp3"
        t0 = time.monotonic()
        if dry_run and audio_path.exists():
            self.log(f"  ♻ 复用缓存音频 {audio_path.name}")
        elif not dry_run:
            self.log("  🎧 抽取音频（远端，不落地视频）...")
            windows = max(1, min(self.cfg.audio_windows, int(self.cfg.concurrency)))
            media.extract_audio_parallel(
                astream, str(audio_path), duration=float(video.duration or 0),
                windows=windows, bitrate=self.cfg.audio_bitrate,
                stall_secs=float(self.cfg.audio_stall),
                user_agent=self.cfg.user_agent, log=self.log)
            self.log(f"  ✅ 音频就绪，总计 {time.monotonic() - t0:.1f}s")
        if dry_run:
            self.log("  ⏭ dry-run：跳过转写")
            return None
        tr = asr.transcribe(str(audio_path), self.cfg, log=self.log)
        if not self.cfg.keep_cache:
            try:
                audio_path.unlink()
            except OSError:
                pass
        return tr

    # ------------------------------------------------------------------
    def _try_native(self, video: Video, vstream: Stream) -> Optional[vision.VisualNote]:
        """把远端视频地址直接交给模型，让它在服务端「看」整段视频。

        前提是地址能匿名访问：B站主 CDN 校验 Referer 会 403，
        只有部分 mcdn 边缘节点放行（约 1/4 视频）；否则需要用 --relay 自建中转。
        """
        url: Optional[str] = None
        if self.cfg.relay:
            url = media.relay_url(self.cfg.relay, vstream.url)
            self.log(f"  🌐 通过中转代理直读视频: {self.cfg.relay}")
        else:
            url = media.referer_free_url(vstream, log=self.log)
        if not url:
            self.log("  ⚠ 该视频地址不允许匿名访问（B站 CDN 校验 Referer），改用抽帧模式")
            return None
        t0 = time.monotonic()
        try:
            text = self.vlm.watch_video(url, title=video.title)
        except Exception as e:
            self.log(f"  ⚠ 原生直读失败（{e}），改用抽帧模式")
            return None
        note = vision.VisualNote(
            items=[vision.FrameNote(ts=0.0, kind="native", text=text, important=True)],
            gist="（模型原生观看整段视频，含画面与声音）",
            source="native", elapsed=time.monotonic() - t0)
        return note

    # ------------------------------------------------------------------
    def _subtitle_only(self, video: Video, track: Optional[SubtitleTrack]
                       ) -> Optional[asr.Transcript]:
        """只取字幕，不碰音频（原生直读模式下用来补充精确文稿）。"""
        if track is None:
            return None
        if self.cfg.asr_backend not in ("auto", "subtitle"):
            return None
        t0 = time.monotonic()
        try:
            cues = self.client.fetch_subtitle(track)
        except Exception:
            return None
        if not cues:
            return None
        tr = asr.from_cues(cues, source=f"subtitle:{track.lan}")
        tr.elapsed = time.monotonic() - t0
        return tr

    def _frames_leg(self, video: Video, vstream: Stream, duration: float,
                    workdir: Path, describe: bool = True) -> Optional[vision.VisualNote]:
        cfg = self.cfg
        leg_t0 = time.monotonic()
        frames_dir = workdir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # ---- A. 快照雪碧图 → 场景变化检测 ----
        keyframes: List[frames.Candidate] = []
        if cfg.use_storyboard:
            t0 = time.monotonic()
            sb = self.client.storyboard(video.bvid, video.cid)
            if sb and sb.frame_count >= 8:
                try:
                    blobs = [media.download_bytes(u, user_agent=cfg.user_agent,
                                                  timeout=90) for u in sb.sheets]
                    cands = frames.storyboard_candidates(sb, blobs)
                    if cands:
                        keyframes = frames.pick_keyframes(
                            cands, max_frames=cfg.max_frames,
                            threshold=cfg.scene_threshold)
                        self.log(f"  🎞 快照 {sb.frame_count} 帧/{len(blobs)} 张 "
                                 f"({sum(len(b) for b in blobs) // 1024} KB) → 选 {len(keyframes)} 个关键时刻"
                                 f"（{time.monotonic() - t0:.1f}s）")
                except Exception as e:
                    self.log(f"  ⚠ 快照处理失败，改用均匀采样: {e}")
            else:
                self.log("  ⚠ 该视频没有快照，改用均匀采样")

        if not keyframes:  # 兜底：均匀采样时间点
            n = max(2, min(cfg.max_frames, 12))
            step = max(1.0, duration / (n + 1))
            keyframes = [frames.Candidate(ts=step * (i + 1)) for i in range(n)]
            self.log(f"  🎞 均匀采样 {len(keyframes)} 个时间点")

        timestamps = [c.ts for c in keyframes]

        # ---- B. 并行抽高清帧（只取需要的字节） ----
        # 先给视频流选最快的 CDN 节点，帧抽取得更快
        order = media.rank_urls(vstream, sample=131072, timeout=12.0,
                                user_agent=cfg.user_agent, log=self.log)
        if order:
            vstream = Stream(url=order[0], backups=order[1:], id=vstream.id,
                             codec=vstream.codec, width=vstream.width,
                             height=vstream.height, bandwidth=vstream.bandwidth)
        t0 = time.monotonic()
        shots: List[Tuple[float, Image.Image]] = []

        # 抽帧状态：支持「签名过期后重新取流」+「连续失败熔断」
        st = {
            "stream": vstream,
            "lock": threading.Lock(),
            "consecutive": 0,      # 连续失败次数
            "fail": 0,
            "refreshed": 0,
            "abort": False,
            "logged": 0,           # 已详细打印的错误条数（避免刷屏）
        }

        def _refresh_stream() -> bool:
            """签名过期/DNS 变化后，重新取一次 playurl（至多 2 次）。"""
            with st["lock"]:
                if st["refreshed"] >= 2 or st["abort"]:
                    return False
                st["refreshed"] += 1
                n = st["refreshed"]
            self.log(f"  🔄 流地址可能已过期，重新获取（第 {n} 次）...")
            try:
                d = self.client.dash(video.bvid, video.cid, quality=cfg.quality)
                fresh = d.best_video()
                if fresh is None:
                    return False
                order = media.rank_urls(fresh, sample=65536, timeout=10.0,
                                        user_agent=cfg.user_agent, log=lambda m: None)
                if order:
                    fresh = Stream(url=order[0], backups=order[1:], id=fresh.id,
                                   codec=fresh.codec, width=fresh.width,
                                   height=fresh.height, bandwidth=fresh.bandwidth)
                with st["lock"]:
                    st["stream"] = fresh
                    st["consecutive"] = 0
                return True
            except Exception as e:
                self.log(f"  ⚠ 重新取流失败: {media.scrub(str(e), 120)}")
                return False

        def _grab(ts: float) -> Optional[Tuple[float, Image.Image]]:
            out = frames_dir / f"f{int(ts):05d}.jpg"
            if st["abort"] or media.stop_requested():
                return None
            for attempt in (0, 1):
                try:
                    with st["lock"]:
                        cur = st["stream"]
                    if not out.exists() or out.stat().st_size == 0:
                        media.grab_frame(cur, ts, str(out), width=cfg.frame_width,
                                         quality=cfg.jpeg_quality,
                                         user_agent=cfg.user_agent)
                    img = Image.open(out).convert("RGB")
                    with st["lock"]:
                        st["consecutive"] = 0
                    return ts, img
                except Exception as e:
                    if media.stop_requested():
                        return None
                    with st["lock"]:
                        st["fail"] += 1
                        st["consecutive"] += 1
                        consec = st["consecutive"]
                        show = st["logged"] < 3
                        if show:
                            st["logged"] += 1
                    if show:
                        self.log(f"  ⚠ 抽帧 @{int(ts)}s 失败: {media.scrub(str(e), 160)}")
                    # 签名过期 / DNS 故障 → 换新地址再试一次
                    if attempt == 0 and media.looks_like_expired(e) and _refresh_stream():
                        continue
                    # 连续失败太多 → 熔断，别再白跑剩下几十帧
                    if consec >= 6:
                        with st["lock"]:
                            already = st["abort"]
                            st["abort"] = True
                        if not already:
                            self.log(f"  ⛔ 连续 {consec} 次抽帧失败（网络/DNS/签名过期），"
                                     "放弃本视频的画面腿。可稍后重跑（有缓存会自动跳过已完成部分）")
                    return None
            return None

        ex = cf.ThreadPoolExecutor(max_workers=max(1, cfg.concurrency))
        futs = {}
        try:
            futs = {ex.submit(_grab, ts): ts for ts in timestamps}
            for fut in cf.as_completed(futs):
                if media.stop_requested():
                    break
                try:
                    item = fut.result()
                except Exception:
                    item = None
                if item:
                    shots.append(item)
        finally:
            # 中断时把还没跑的任务取消掉，避免继续提交/刷屏
            for f in list(futs):
                f.cancel()
            ex.shutdown(wait=True, cancel_futures=True)
        shots.sort(key=lambda x: x[0])
        if st["fail"]:
            self.log(f"  ℹ 抽帧失败 {st['fail']} 次 / 重新取流 {st['refreshed']} 次")
        self.log(f"  📸 抽 {len(shots)}/{len(timestamps)} 帧，用时 {time.monotonic() - t0:.1f}s"
                 f"（共 {sum((frames_dir / f'f{int(t):05d}.jpg').stat().st_size for t, _ in shots) // 1024} KB）")
        if not shots:
            return None

        if not describe:
            note = vision.VisualNote(source="frames-only")
            note.items = [vision.FrameNote(ts=ts, kind="frame", important=False)
                          for ts, _ in shots]
            note.elapsed = time.monotonic() - leg_t0
            self.log(f"  ℹ 已保存 {len(shots)} 帧到 {frames_dir}（未调用视觉模型）")
            return note

        # ---- C. 拼图 → 视觉模型（并行） ----
        cells = max(1, min(cfg.sheet_grid, 9))
        sheets = frames.build_sheets(shots, cells=cells,
                                     cell_w=max(320, min(720, cfg.frame_width // 2)))
        # 注意：这里**不能**预先塞占位条目 —— merge() 以时间戳为键，
        # 占位条目会把随后拿到的真实描述顶掉。
        note = vision.VisualNote(source="sheet")

        def _describe(pack):
            img, ts_list = pack
            payload = frames.jpeg_bytes(img, quality=85)
            return self.vlm.describe_sheet(payload, ts_list, title=video.title)

        t0 = time.monotonic()
        ex2 = cf.ThreadPoolExecutor(max_workers=max(1, min(4, cfg.concurrency)))
        try:
            futs = [ex2.submit(_describe, s) for s in sheets]
            for f in cf.as_completed(futs):
                if media.stop_requested():
                    break
                try:
                    note = note.merge(f.result())
                except Exception as e:
                    self.log(f"  ⚠ 视觉请求失败: {e}")
        finally:
            for f in futs:
                f.cancel()
            ex2.shutdown(wait=True, cancel_futures=True)
        note.elapsed = time.monotonic() - leg_t0    # 含快照、抽帧、视觉调用全过程
        self.log(f"  👁 视觉理解完成：{len(note.items)} 条笔记"
                 f"（{len(sheets)} 张拼图，画面腿共 {note.elapsed:.1f}s）")
        return note

    # ------------------------------------------------------------------
    def _render_text(self, video: Video, res: Result, review: str, timeline: str) -> str:
        """纯文本产物：值得回看的片段 + 全部原始材料（md 里不再重复这些）。"""
        bar = "━" * 56
        head = [
            video.title,
            f"{video.page_url}",
            f"UP 主：{video.owner or '未知'} ｜ 时长：{video.duration // 60} 分 "
            f"{video.duration % 60} 秒",
            f"文稿：{res.transcript.source if res.transcript else '无'}"
            f"（{res.transcript.chars if res.transcript else 0} 字） ｜ "
            f"画面：{res.visual.source + f' · {len(res.visual.items)} 条' if res.visual else '未启用'}",
            "",
        ]
        parts: List[str] = [chr(10).join(head)]

        if review:
            parts += [bar, "【值得回看的片段】", bar, _strip_heading(review), ""]
        if res.visual and res.visual.items:
            parts += [bar, "【画面记录（逐帧）】", bar, res.visual.as_markdown(), ""]
        if timeline:
            parts += [bar, "【多模态时间轴（画面 + 语音对齐）】", bar, timeline, ""]
        if res.transcript and res.transcript.chars:
            parts += [bar, "【完整文稿】", bar, res.transcript.text, ""]
        return chr(10).join(parts) + chr(10)

    # ------------------------------------------------------------------
    def _render_markdown(self, video: Video, res: Result, note_body: str) -> str:
        t = res.timings
        head = [
            f"# {video.title}" + (f" · {video.part}" if video.page > 1 and video.part else ""),
            "",
            f"- **链接**：{video.page_url}",
            f"- **UP 主**：{video.owner or '未知'}",
            f"- **时长**：{video.duration // 60} 分 {video.duration % 60} 秒",
            "- **文稿来源**：" + (
                f"{res.transcript.source}（{res.transcript.chars} 字）"
                if res.transcript and res.transcript.chars else "无（仅有画面）"),
            f"- **画面来源**：{(res.visual.source + f' · {len(res.visual.items)} 条') if res.visual else '未启用'}",
            f"- **耗时**：总 {t.get('total', 0):.1f}s"
            + (f" · 转写 {res.transcript.elapsed:.1f}s" if res.transcript else "")
            + (f" · 画面 {res.visual.elapsed:.1f}s" if res.visual else "")
            + (f" · 总结 {t.get('summary', 0):.1f}s" if t.get("summary") else ""),
            "",
        ]
        if res.error:
            head += [f"> ⚠ {res.error}", ""]
        body = note_body or "（未生成总结）"
        parts = ["\n".join(head), body]
        # 画面记录 / 时间轴 / 完整文稿 / 值得回看 都在同名 .txt 里，md 保持精简
        return "\n".join(parts) + "\n"

    # ------------------------------------------------------------------
    def _prune(self, workdir: Path, keep: Sequence[str] = ()) -> None:
        for child in list(workdir.iterdir()):
            if child.name in keep:
                continue
            try:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink()
            except OSError:
                pass


__all__ = ["Analyzer", "Result"]


# --------------------------------------------------------------------------
# 旧格式迁移：把「全塞在 md 里」的旧报告拆成新的 md（笔记）+ txt（原始材料）
# --------------------------------------------------------------------------
_MARK_FRAMES = "## 附：画面记录（逐帧）"
_MARK_TIMELINE = "## 附：多模态时间轴"
_MARK_TRANS = "## 附：完整文稿"


def reformat_report(md_path: Path) -> Tuple[bool, str]:
    """把一份旧格式 .md 重排成新格式（不调用任何 AI）。

    返回 (是否改动, 说明)。
    旧格式：md 里同时有笔记 + 画面记录 + 时间轴 + 完整文稿（+ 值得回看）
    新格式：md 只留笔记；txt 收走 值得回看 + 三个附录
    """
    text = md_path.read_text(encoding="utf-8")
    if _MARK_FRAMES not in text:          # 已经是新格式（或没有附录）
        return False, "已是新格式"

    # 1) 拆出头部（标题 + 元信息）
    lines = text.splitlines()
    head_end = next((i for i, l in enumerate(lines) if l.startswith("## ")), len(lines))
    header = "\n".join(lines[:head_end]).rstrip()

    body = "\n".join(lines[head_end:])

    # 2) 拆出三个附录
    def cut(src: str, start_mark: str, end_mark: Optional[str]) -> Tuple[str, str]:
        a = src.find(start_mark)
        if a == -1:
            return src, ""
        b = src.find(end_mark, a + len(start_mark)) if end_mark else -1
        if b == -1:
            return src[:a], src[a:]
        return src[:a] + src[b:], src[a:b]

    body, raw_trans = cut(body, _MARK_TRANS, None)
    body, raw_time = cut(body, _MARK_TIMELINE, _MARK_TRANS)
    body, raw_frames = cut(body, _MARK_FRAMES, _MARK_TIMELINE)

    # 3) 笔记里拆出「值得回看的片段」
    note, review = _split_review(body.strip())
    note = note.replace("\n---\n", "\n").strip()

    # 4) 收拾附录文本
    def clean(chunk: str, mark: str) -> str:
        if not chunk:
            return ""
        chunk = chunk.replace(mark, "").strip()
        # 时间轴原本包在 ``` 里
        if chunk.startswith("```"):
            chunk = chunk.split("\n", 1)[-1]
        if chunk.endswith("```"):
            chunk = chunk.rsplit("```", 1)[0]
        return chunk.strip().strip("-").strip()

    frames_txt = clean(raw_frames, _MARK_FRAMES)
    time_txt = clean(raw_time, _MARK_TIMELINE)
    trans_txt = clean(raw_trans, _MARK_TRANS)

    # 5) 写回 md（只留笔记）
    md_path.write_text(f"{header}\n\n{note}\n", encoding="utf-8")

    # 6) 写同名 .txt
    bar = "━" * 56
    parts = [header, ""]
    if review:
        parts += [bar, "【值得回看的片段】", bar, _strip_heading(review), ""]
    if frames_txt:
        parts += [bar, "【画面记录（逐帧）】", bar, frames_txt, ""]
    if time_txt:
        parts += [bar, "【多模态时间轴（画面 + 语音对齐）】", bar, time_txt, ""]
    if trans_txt:
        parts += [bar, "【完整文稿】", bar, trans_txt, ""]
    txt_path = md_path.with_suffix(".txt")
    txt_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return True, f"md {len(text.encode())//1024}KB → {md_path.stat().st_size//1024}KB, " \
                 f"txt {txt_path.stat().st_size//1024}KB"
