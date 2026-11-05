"""命令行入口。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from . import __version__, media
from .config import PROVIDER_PRESETS, Config

BANNER = r"""
   ___  _____ ____  ____  _     ___
  / _ \|_   _| __ )| __ )(_)   |_ _|
 | |_| | | | |  _ \|  _ \| |    | |
 |  _  | | | | |_) | |_) | |    | |
 |_| |_| |_| |____/|____/|_|___|___|
  让 AI 真正「看」B 站视频 · v{v}
"""


def log(msg: str) -> None:
    print(msg, flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aicbili",
        description="让 AI 真正「看」B站视频：远端抽帧 + 音频转写 + 多模态总结，全程不下载完整视频。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  aicbili BV1xx411c7mD
  aicbili https://www.bilibili.com/video/BV1xx411c7mD?p=3 --frames 30
  aicbili urls.txt --jobs 3          # 批量（每行一个链接）
  aicbili <url> --no-ai              # 只抽帧+转写，不调 AI
  aicbili <url> --mode native        # 让模型直接读远端视频（需匿名可访问或配 --relay）
  aicbili doctor                     # 环境自检
""",
    )
    p.add_argument("inputs", nargs="*", help="视频链接 / BV 号 / av 号 / 含链接的文本文件")
    p.add_argument("--version", action="version", version=f"aicbili {__version__}")

    g = p.add_argument_group("分析方式")
    g.add_argument("--mode", choices=["auto", "frames", "native", "audio"], default=None,
                   help="auto=抽帧+转写(默认) frames=只抽帧 native=模型原生直读视频 audio=只转写")
    g.add_argument("--frames", type=int, default=None, dest="max_frames",
                   help="最多抽取多少画面（默认 24）")
    g.add_argument("--frame-width", type=int, default=None, help="单帧最长边像素（默认 1280）")
    g.add_argument("--scene-threshold", type=float, default=None,
                   help="场景变化阈值 0~1，越低越敏感（默认 0.12）")
    g.add_argument("--quality", type=int, default=None, help="期望清晰度 qn：80=1080P(需登录) 32=480P")
    g.add_argument("--no-storyboard", action="store_true", help="不用快照，直接均匀采样抽帧")

    g = p.add_argument_group("模型")
    g.add_argument("--provider", choices=sorted(PROVIDER_PRESETS), default=None,
                   help="预设服务商（openai/dashscope/ark/zhipu/siliconflow/deepseek）")
    g.add_argument("--model", default=None, help="视觉模型名")
    g.add_argument("--text-model", default=None, help="纯文本总结模型名（默认与视觉模型相同）")
    g.add_argument("--base-url", default=None, help="OpenAI 兼容端点")
    g.add_argument("--api-key", default=None, help="API Key（优先用环境变量）")
    g.add_argument("--relay", default=None,
                   help="自建中转代理地址：把视频地址包一层补 Referer，供 native 模式直读")

    g = p.add_argument_group("B站")
    g.add_argument("--sessdata", default=None, help="B站 Cookie 里的 SESSDATA：解锁 1080P 与官方 AI 字幕（强烈建议）")

    g = p.add_argument_group("运行")
    g.add_argument("--jobs", type=int, default=1, help="同时处理几个视频（默认 1）")
    g.add_argument("--concurrency", type=int, default=None, help="单个视频内的并发数（默认 6）")
    g.add_argument("--out", default=None, help="输出目录（默认 ./output）")
    g.add_argument("--cache-dir", default=None, help="缓存目录")
    g.add_argument("--keep-cache", action="store_true", help="保留中间文件（音频/帧）")
    g.add_argument("--force", action="store_true", help="忽略已有结果，强制重做")
    g.add_argument("--no-ai", action="store_true",
                   help="跳过视觉模型与总结（仍抽帧+转写，便于排错/离线使用）")
    g.add_argument("--dry-run", action="store_true",
                   help="只解析视频信息，不抽帧不转写不总结（秒级验证链接与配置）")
    g.add_argument("--json", action="store_true", help="以 JSON 输出结果摘要")
    g.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("cmd", nargs="?", default="analyze",
                   choices=["analyze", "doctor", "config", "cache", "models"])
    return p


def cfg_from_args(a: argparse.Namespace) -> Config:
    overrides = {}
    if a.mode:
        overrides["mode"] = a.mode
    if a.max_frames:
        overrides["max_frames"] = a.max_frames
    if a.frame_width:
        overrides["frame_width"] = a.frame_width
    if a.scene_threshold is not None:
        overrides["scene_threshold"] = a.scene_threshold
    if a.quality:
        overrides["quality"] = a.quality
    if a.concurrency:
        overrides["concurrency"] = a.concurrency
    if a.out:
        overrides["output_dir"] = a.out
    if a.cache_dir:
        overrides["cache_dir"] = a.cache_dir
    if a.sessdata:
        overrides["sessdata"] = a.sessdata
    if a.relay:
        overrides["relay"] = a.relay
    if a.keep_cache:
        overrides["keep_cache"] = True
    if a.verbose:
        overrides["verbose"] = True
    if a.provider:
        preset = PROVIDER_PRESETS[a.provider]
        overrides.setdefault("vlm.base_url", preset["base_url"])
        if not a.model:
            overrides["vlm.model"] = preset["model"]
    if a.base_url:
        overrides["vlm.base_url"] = a.base_url
    if a.model:
        overrides["vlm.model"] = a.model
    if a.text_model:
        overrides["text.model"] = a.text_model
    if a.api_key:
        overrides["vlm.api_key"] = a.api_key

    cfg = Config.load(**overrides)
    if a.provider and not a.text_model:
        cfg.text.model = cfg.vlm.model
        cfg.text.base_url = cfg.vlm.base_url
    if a.no_storyboard:
        cfg.use_storyboard = False
    return cfg


def collect_inputs(inputs: List[str]) -> List[str]:
    """展开 .txt 文件里的链接、去掉分享文案里的多余字符、去重。"""
    import re

    urls: List[str] = []
    for raw in inputs:
        p = Path(raw).expanduser()
        if p.is_file() and p.suffix.lower() in (".txt", ".list", ".md"):
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.extend(_extract_links(line))
        else:
            urls.extend(_extract_links(raw))
    seen, out = set(), []
    for u in urls:
        k = u.strip()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _extract_links(text: str) -> List[str]:
    import re

    t = text.strip()
    if not t:
        return []
    found = re.findall(r"https?://[^\s\u4e00-\u9fff]+", t)
    if found:
        return [u.rstrip("，,。.、)）】]") for u in found]
    if re.search(r"BV[0-9A-Za-z]{10}", t) or re.search(r"\bav\d+", t, re.I):
        return [t]
    return []


# --------------------------------------------------------------------------
def cmd_analyze(cfg: Config, urls: List[str], a: argparse.Namespace) -> int:
    from .pipeline import Analyzer

    if not urls:
        # 和旧版一样：没给参数时支持交互式粘贴（一行一个，空行结束）
        if sys.stdin.isatty():
            log("粘贴 B站链接 / BV 号（每行一个，空行结束）：")
            buf: List[str] = []
            while True:
                try:
                    line = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not line:
                    break
                buf.append(line)
            urls = collect_inputs(buf)
        if not urls:
            log("❗ 请提供视频链接 / BV 号，或用 -h 看用法")
            return 2
    log(BANNER.format(v=__version__) if not a.json else "")
    if not a.json:
        for k, v in cfg.masked().items():
            log(f"  {k:<10} {v}")
        log("")

    no_ai = bool(a.no_ai)
    dry = bool(a.dry_run)
    # 批量时把「每视频并发」按 jobs 分摊，并给所有视频设一个全局连接上限，
    # 否则 N 个视频 × 每视频 (抽帧线程 + 音频窗口) 会把 B站 CDN 打崩。
    per_video = cfg.concurrency
    if a.jobs > 1 and cfg.concurrency > 0:
        per_video = max(2, cfg.concurrency // a.jobs)
        cfg.concurrency = per_video
        if not a.json:
            log(f"  ⚙ 并发分摊：{a.jobs} 个视频 × 每视频 {per_video} 并发"
                f"（全局上限 {max(per_video, a.jobs * per_video)} 条连接）")
    media.configure_global_limit(max(2, per_video * min(a.jobs, 2)))

    # 本地转写也吃 CPU：批量时按 jobs 分摊线程，避免 N 个 whisper 抢 8 个核
    cores = os.cpu_count() or 4
    if a.jobs > 1 and not cfg.whisper_threads:
        cfg.whisper_threads = max(2, (cores - 1) // a.jobs)
        if not a.json:
            log(f"  ⚙ 本地转写线程：每视频 {cfg.whisper_threads}（共 {cores} 核 / {a.jobs} 路）")

    # 预检：一个转写后端都没有的话，没字幕的视频将只有画面
    if not a.json and not dry:
        from . import asr as _asr
        if not _asr.available_backends(cfg) and not cfg.logged_in:
            log("  ⚠ 未检测到任何转写后端：没有 AI 字幕的视频将只出画面、无文稿")

    analyzer = Analyzer(cfg, log=(lambda m: None) if a.json else log)
    results = []
    t0 = time.time()

    def one(url: str):
        try:
            return analyzer.analyze(url, force=a.force, no_ai=no_ai, dry_run=dry)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            from .pipeline import Result
            r = Result()
            r.error = str(e)
            return r

    import concurrent.futures as cf

    ex = cf.ThreadPoolExecutor(max_workers=max(1, a.jobs))
    try:
        if a.jobs <= 1:
            for u in urls:
                if media.stop_requested():
                    break
                results.append(one(u))
        else:
            futs = [ex.submit(one, u) for u in urls]
            for f in cf.as_completed(futs):
                if media.stop_requested():
                    break
                try:
                    results.append(f.result())
                except Exception as e:
                    from .pipeline import Result
                    r = Result(); r.error = str(e); results.append(r)
    except KeyboardInterrupt:
        media.request_stop()
        media.kill_children()
        log("\n⏹ 已中断，正在停止（几秒内退出）...")
        for f in locals().get("futs", []):
            f.cancel()
        ex.shutdown(wait=False, cancel_futures=True)
        time.sleep(min(3.0, 0.1))
        sys.stdout.flush()
        os._exit(130)          # 直接退出：避免 ThreadPool 在解释器关闭时刷 traceback
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
        analyzer.close()

    total = time.time() - t0
    if a.json:
        payload = []
        for r in results:
            payload.append({
                "title": r.video.title if r.video else "",
                "url": r.video.page_url if r.video else "",
                "mode": r.mode,
                "error": r.error,
                "skipped": r.skipped,
                "transcript_source": r.transcript.source if r.transcript else "",
                "transcript_chars": r.transcript.chars if r.transcript else 0,
                "visual_items": len(r.visual.items) if r.visual else 0,
                "timings": {k: round(v, 2) for k, v in r.timings.items()},
                "files": r.files,
            })
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if all(r.ok for r in results) else 1

    ok = [r for r in results if r.ok]
    bad = [r for r in results if not r.ok]
    log("")
    log("=" * 64)
    for r in results:
        if r.skipped:
            log(f"⏭ 跳过（已完成）: {r.files.get('report', '')}")
        elif r.ok:
            extra = ""
            if r.transcript:
                extra += f" 文稿 {r.transcript.chars} 字"
            if r.visual:
                extra += f" | 画面 {len(r.visual.items)} 条"
            log(f"✅ {r.video.title[:40]}{extra}")
            log(f"   → {r.files.get('report', '')}")
        else:
            log(f"❌ {r.video.title[:40] if r.video else '?'}: {r.error}")
    log("=" * 64)
    log(f"完成 {len(ok)}/{len(results)} 个，总耗时 {total:.1f}s"
        + (f"，平均 {total / len(ok):.1f}s/个" if ok else ""))
    return 0 if not bad else 1


# --------------------------------------------------------------------------
def cmd_doctor(cfg: Config) -> int:
    log(f"aicbili {__version__} 环境自检\n")
    problems = 0

    # ffmpeg
    try:
        ff = media.ffmpeg_path()
        import subprocess
        ver = subprocess.run([ff, "-version"], capture_output=True, text=True).stdout.splitlines()[0]
        log(f"✅ ffmpeg: {ff}\n   {ver[:80]}")
    except Exception as e:
        log(f"❌ ffmpeg 不可用: {e}")
        problems += 1

    # B站 API
    from .bili import BiliClient
    t0 = time.time()
    try:
        c = BiliClient(cookie=cfg.cookie_header, user_agent=cfg.user_agent)
        v = c.resolve("BV1wD4y1o7AS")
        log(f"✅ B站 API 可用（{(time.time() - t0) * 1000:.0f}ms）: {v.title[:40]}")
        subs = c.subtitles(v.bvid, v.cid)
        if cfg.logged_in:
            log(f"   字幕轨: {[t.lan_doc for t in subs] or '该视频无'}")
        else:
            log("   ⚠ 未配置 SESSDATA：最高 480P，且看不到 B站 AI 字幕（速度优势用不上）")
        d = c.dash(v.bvid, v.cid)
        bv = d.best_video()
        if bv:
            log(f"   最佳视频流: {bv.width}x{bv.height} {bv.codec.split('.')[0]}")
        sb = c.storyboard(v.bvid, v.cid)
        log(f"   快照: {sb.frame_count if sb else 0} 帧"
            + (f"（{sb.interval:.0f}s 一帧，{len(sb.sheets)} 张）" if sb else ""))
        c.close()
    except Exception as e:
        log(f"❌ B站 API 异常: {e}")
        problems += 1

    # 视觉模型：不仅看配置，还真的发一张图试试能不能认出来
    if cfg.vlm.ready:
        local = any(h in cfg.vlm.base_url for h in ("localhost", "127.0.0.1"))
        log(f"🔍 视觉模型: {cfg.vlm.model} @ {cfg.vlm.base_url}"
            + ("（本地）" if local else "") + "，正在试拍一张图...")
        from .vision import probe_multimodal

        ok, msg = probe_multimodal(cfg)
        if ok:
            log(f"✅ 多模态可用：模型正确读出了测试图里的文字（{msg[:40]}）")
        else:
            log(f"❌ 这个模型/Key 看不了图：{msg}")
            log("   常见原因：模型名是纯文本模型（如 deepseek-chat / gpt-3.5-turbo）、"
                "Key 无效、或余额不足")
            problems += 1
    else:
        log("❌ 视觉模型未配置：设 AICBILI_VLM_API_KEY 与 AICBILI_VLM_MODEL（画面理解需要）")
        problems += 1

    # 转写：按「最省事 → 最麻烦」列出可用性
    from . import asr

    speech_ok = False
    if cfg.logged_in:
        log("✅ 文稿首选通道: B站 AI 字幕（已配 SESSDATA，有字幕时零耗时）")
        speech_ok = True
    wcpp_bin = asr.whisper_cpp_bin()
    wcpp_model = asr.find_whisper_cpp_model(cfg.whisper_cpp_model) if wcpp_bin else None
    if wcpp_bin and wcpp_model:
        log(f"✅ 本地 whisper.cpp: {wcpp_bin} + {os.path.basename(wcpp_model)}")
        speech_ok = True
    elif wcpp_bin:
        log(f"⚠ whisper.cpp 已装（{wcpp_bin}）但缺 ggml 模型，下载一个即可：")
        log("   mkdir -p ~/.cache/whisper-cpp && cd ~/.cache/whisper-cpp && \\")
        log("   curl -L -O https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin")
    elif asr.local_available():
        log("✅ 本地转写: faster-whisper")
        speech_ok = True
    else:
        log("⚠ 无本地转写（faster-whisper 未装、whisper.cpp 未装）")
        log("   Arch 一条命令搞定：sudo pacman -S whisper-cpp，再下载 ggml-base.bin")
    if not speech_ok:
        log("   ⚠ 结果：本次只能出「画面」报告，没有文稿")

    # 快照可达性
    try:
        media.download_bytes(
            "https://bimp.hdslb.com/videoshotpvhdboss/28092009765_pcq9db-0001.jpg",
            user_agent=cfg.user_agent, timeout=20, retries=1)
        log("✅ 快照雪碧图可下载")
    except Exception as e:
        log(f"⚠ 快照下载失败（会退化为均匀采样）: {e}")

    log("")
    log("自检完成：" + ("一切就绪 🎉" if problems == 0 else f"{problems} 项需要处理"))
    return 0 if problems == 0 else 1


def cmd_config(cfg: Config) -> int:
    log("当前生效配置：\n")
    for k, v in cfg.masked().items():
        log(f"  {k:<12} {v}")
    envf = Path(".env")
    log("")
    log(f".env: {envf.resolve() if envf.exists() else '不存在（可从 .env.example 复制）'}")
    return 0


def cmd_cache(cfg: Config) -> int:
    root = cfg.cache_root()
    log(f"缓存目录: {root}")
    if not root.exists():
        log("（空）")
        return 0
    total = 0
    for p in sorted(root.iterdir()):
        if p.is_dir():
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            total += size
            log(f"  {p.name:<28} {size / 1024:8.0f} KB")
    log(f"合计 {total / 1024 / 1024:.2f} MB")
    return 0


def cmd_reformat(cfg: Config, inputs: List[str]) -> int:
    """把旧格式报告（全塞在 md 里）重排成 md + txt，不调用任何 AI。"""
    from .pipeline import reformat_report

    target = Path(inputs[0]).expanduser() if inputs else Path(cfg.output_dir).expanduser()
    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(target.glob("*.md"))
    else:
        log(f"❗ 找不到路径: {target}")
        return 2
    if not files:
        log(f"❗ {target} 里没有 .md 文件")
        return 2

    changed = skipped = failed = 0
    for f in files:
        try:
            did, msg = reformat_report(f)
        except Exception as e:
            failed += 1
            log(f"  ❌ {f.name[:40]}: {e}")
            continue
        if did:
            changed += 1
            log(f"  ✅ {f.name[:44]}  {msg}")
        else:
            skipped += 1
    log("")
    log(f"完成：重排 {changed} 份，已是新格式 {skipped} 份，失败 {failed} 份")
    return 0 if failed == 0 else 1


def cmd_models() -> int:
    log("内置服务商预设（--provider）：\n")
    for name, p in PROVIDER_PRESETS.items():
        log(f"  {name:<12} {p['model']:<34} {p['base_url']}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    try:
        return _main(argv)
    except BrokenPipeError:
        # 例如 `aicbili config | head`：安静退出，不要打 traceback
        try:
            sys.stdout.close()
        except Exception:
            pass
        return 0
    except KeyboardInterrupt:
        media.request_stop()
        print("\n⏹ 已中断", file=sys.stderr)
        return 130


def _main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # 允许 `aicbili doctor` 这类子命令出现在任意位置
    cmd = "analyze"
    for i, tok in enumerate(argv):
        if tok in ("doctor", "config", "cache", "models", "reformat") and not tok.startswith("-"):
            cmd = tok
            argv.pop(i)
            break
    a = build_parser().parse_args(argv)
    cfg = cfg_from_args(a)

    if cfg.verbose:
        log(f"[cfg] {cfg.asdict()}")

    if cmd == "doctor":
        return cmd_doctor(cfg)
    if cmd == "config":
        return cmd_config(cfg)
    if cmd == "cache":
        return cmd_cache(cfg)
    if cmd == "models":
        return cmd_models()
    if cmd == "reformat":
        return cmd_reformat(cfg, a.inputs)
    return cmd_analyze(cfg, collect_inputs(a.inputs), a)


if __name__ == "__main__":
    raise SystemExit(main())
