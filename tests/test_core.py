"""核心单元测试（只用标准库 unittest，无需 pytest）。"""
from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image

from aicbili import frames, summarize
from aicbili.asr import Segment, Transcript, from_cues, fmt_srt_ts, fmt_ts, write_srt
from aicbili.bili import BiliClient, Storyboard, SubtitleTrack, pick_subtitle
from aicbili.config import Config, load_env
from aicbili.media import http_header_arg, relay_url
from aicbili.vision import FrameNote, VisualNote, _extract_json


class TestParsing(unittest.TestCase):
    def test_bvid_from_url(self):
        t = BiliClient.parse_target("https://www.bilibili.com/video/BV1xx411c7mD?p=3")
        self.assertEqual(t["bvid"], "BV1xx411c7mD")
        self.assertEqual(t["page"], 3)

    def test_bvid_bare(self):
        self.assertEqual(BiliClient.parse_target("BV1GJ411x7h7")["bvid"], "BV1GJ411x7h7")

    def test_share_text(self):
        t = BiliClient.parse_target("【标题】 https://www.bilibili.com/video/BV1rpWjevEip?share_source=copy")
        self.assertEqual(t["bvid"], "BV1rpWjevEip")

    def test_av_number(self):
        t = BiliClient.parse_target("av12345")
        self.assertEqual(t["aid"], 12345)
        self.assertIsNone(t["bvid"])

    def test_page_default(self):
        self.assertEqual(BiliClient.parse_target("BV1GJ411x7h7")["page"], 1)

    def test_garbage(self):
        t = BiliClient.parse_target("随便一段文字")
        self.assertIsNone(t["bvid"])
        self.assertIsNone(t["aid"])


class TestSubtitle(unittest.TestCase):
    def test_prefer_manual_chinese(self):
        tracks = [
            SubtitleTrack("en-US", "English", "u1"),
            SubtitleTrack("ai-zh", "中文（自动生成）", "u2", is_ai=True),
            SubtitleTrack("zh-CN", "中文（中国）", "u3"),
        ]
        self.assertEqual(pick_subtitle(tracks).url, "u3")

    def test_ai_chinese_beats_english(self):
        tracks = [SubtitleTrack("en-US", "English", "u1"),
                  SubtitleTrack("ai-zh", "中文（自动生成）", "u2", is_ai=True)]
        self.assertEqual(pick_subtitle(tracks).url, "u2")

    def test_empty(self):
        self.assertIsNone(pick_subtitle([]))

    def test_from_cues(self):
        tr = from_cues([{"start": 0, "end": 2.5, "text": "你好"},
                        {"start": 2.5, "end": 5, "text": "  "},
                        {"start": 5, "end": 8, "text": "世界"}])
        self.assertEqual(len(tr.segments), 2)
        self.assertEqual(tr.text, "你好\n世界")
        self.assertEqual(tr.chars, 4)


class TestTimeFormat(unittest.TestCase):
    def test_fmt_ts(self):
        self.assertEqual(fmt_ts(65), "1:05")
        self.assertEqual(fmt_ts(3725), "1:02:05")

    def test_srt_ts(self):
        self.assertEqual(fmt_srt_ts(3661.5), "01:01:01,500")

    def test_write_srt(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.srt")
            write_srt([Segment(0, 1.5, "第一句"), Segment(1.5, 3, "第二句")], p)
            body = Path(p).read_text(encoding="utf-8")
            self.assertIn("00:00:00,000 --> 00:00:01,500", body)
            self.assertIn("第二句", body)


class TestStoryboard(unittest.TestCase):
    def test_interval_median(self):
        sb = Storyboard([], 10, 10, 480, 270, [0, 0, 5, 10, 15, 20])
        self.assertEqual(sb.interval, 5.0)

    def test_interval_single(self):
        self.assertEqual(Storyboard([], 10, 10, 0, 0, [7]).interval, 0.0)

    def _sprite(self, cols=5, rows=4, cw=64, ch=36, color=None):
        img = Image.new("RGB", (cols * cw, rows * ch), color or (0, 0, 0))
        for r in range(rows):
            for c in range(cols):
                v = (r * cols + c) * 8 % 255
                img.paste((v, v, v), (c * cw, r * ch, (c + 1) * cw, (r + 1) * ch))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()

    def test_split_sprite_count_and_order(self):
        cells = frames.split_sprite(self._sprite(5, 4), 5, 4)
        self.assertEqual(len(cells), 20)
        self.assertEqual(cells[0].size, (64, 36))
        # 行优先：第 6 张应该是第二行第一个，颜色更亮
        self.assertLess(sum(cells[0].convert("L").tobytes()),
                        sum(cells[5].convert("L").tobytes()))

    def test_storyboard_candidates_respects_timestamps(self):
        sb = Storyboard(["x"], 5, 4, 64, 36, list(range(0, 200, 10)))
        cands = frames.storyboard_candidates(sb, [self._sprite()])
        self.assertEqual(len(cands), 20)
        self.assertEqual(cands[0].ts, 0.0)
        self.assertEqual(cands[1].ts, 10.0)

    def test_storyboard_candidates_dedupes_same_ts(self):
        sb = Storyboard(["x"], 5, 4, 64, 36, [0, 0, 0, 10, 20, 30, 40, 50, 60, 70])
        cands = frames.storyboard_candidates(sb, [self._sprite()])
        self.assertEqual(len(cands), 8)   # 3 个 0 只留 1 个


class TestKeyframes(unittest.TestCase):
    def _cands(self, n=30, switch_at=15):
        out = []
        for i in range(n):
            color = (10, 10, 10) if i < switch_at else (240, 240, 240)
            out.append(frames.Candidate(ts=float(i * 5), image=Image.new("RGB", (64, 36), color)))
        return out

    def test_detects_hard_cut(self):
        cands = self._cands()
        kf = frames.pick_keyframes(cands, max_frames=6, threshold=0.12)
        self.assertTrue(any(abs(c.ts - 75.0) < 1e-6 for c in kf),
                        f"应检测到 75s 处的切换，实际: {[c.ts for c in kf]}")

    def test_always_includes_first(self):
        kf = frames.pick_keyframes(self._cands(), max_frames=4, threshold=0.5)
        self.assertEqual(kf[0].ts, 0.0)

    def test_respects_max_frames(self):
        for mx in (1, 3, 8):
            kf = frames.pick_keyframes(self._cands(60), max_frames=mx, threshold=0.01)
            self.assertLessEqual(len(kf), mx)

    def test_sorted_and_unique(self):
        kf = frames.pick_keyframes(self._cands(40), max_frames=10, threshold=0.05)
        ts = [c.ts for c in kf]
        self.assertEqual(ts, sorted(ts))
        self.assertEqual(len(ts), len(set(ts)))

    def test_empty(self):
        self.assertEqual(frames.pick_keyframes([], 5, 0.1), [])

    def test_static_video_still_fills(self):
        # 完全静止的片子：没有变化点，也应均匀补到 max_frames
        cands = [frames.Candidate(ts=float(i * 5), image=Image.new("RGB", (64, 36), (128, 128, 128)))
                 for i in range(40)]
        kf = frames.pick_keyframes(cands, max_frames=8, threshold=0.12)
        self.assertEqual(len(kf), 8)


class TestContactSheet(unittest.TestCase):
    def test_build_sheets_groups(self):
        shots = [(float(i * 10), Image.new("RGB", (320, 180), (i * 20, 40, 60))) for i in range(9)]
        sheets = frames.build_sheets(shots, cells=4, cell_w=320)
        self.assertEqual(len(sheets), 3)
        self.assertEqual([len(ts) for _, ts in sheets], [4, 4, 1])

    def test_sheet_label_bar_and_size(self):
        shots = [(0.0, Image.new("RGB", (320, 180))), (60.0, Image.new("RGB", (320, 180)))]
        sheet = frames.contact_sheet(shots, cells=2, cell_w=320)
        self.assertEqual(sheet.size, (640, 180 + 26))

    def test_jpeg_bytes_compresses(self):
        img = frames.contact_sheet(
            [(float(i), Image.new("RGB", (640, 360), (i * 30, 90, 120))) for i in range(4)],
            cells=4, cell_w=640)
        blob = frames.jpeg_bytes(img, quality=80)
        self.assertTrue(blob.startswith(b"\xff\xd8"))
        self.assertLess(len(blob), 400 * 1024)

    def test_jpeg_max_side(self):
        img = Image.new("RGB", (2000, 1000))
        small = Image.open(io.BytesIO(frames.jpeg_bytes(img, max_side=1000)))
        self.assertEqual(max(small.size), 1000)


class TestTimeline(unittest.TestCase):
    def test_align_merges_visual_and_speech(self):
        tr = Transcript([Segment(0, 10, "开场白"), Segment(70, 80, "讲第二个要点")])
        vn = VisualNote([FrameNote(0, "slides", "封面"), FrameNote(75, "code", "print(1)")],
                        gist="讲 print")
        text = summarize.align_timeline(tr, vn)
        self.assertIn("讲 print", text)
        self.assertIn("封面", text)
        self.assertIn("print(1)", text)
        self.assertIn("开场白", text)
        self.assertIn("讲第二个要点", text)

    def test_align_without_transcript(self):
        vn = VisualNote([FrameNote(0, "native", "模型看完了", True)], gist="一句话")
        text = summarize.align_timeline(None, vn)
        self.assertIn("模型看完了", text)

    def test_align_without_visual(self):
        tr = Transcript([Segment(0, 5, "只有声音")])
        self.assertIn("只有声音", summarize.align_timeline(tr, None))

    def test_align_both_empty(self):
        self.assertEqual(summarize.align_timeline(None, None), "")

    def test_uncovered_speech_kept(self):
        tr = Transcript([Segment(0, 5, "第一段"), Segment(500, 510, "画面没覆盖的段落")])
        vn = VisualNote([FrameNote(2, "slides", "开场画面")])
        text = summarize.align_timeline(tr, vn, window=10)
        self.assertIn("画面没覆盖的段落", text)


class TestVisionParsing(unittest.TestCase):
    def test_extract_plain_json(self):
        self.assertEqual(_extract_json('{"a":1}'), {"a": 1})

    def test_extract_fenced_json(self):
        self.assertEqual(_extract_json('```json\n{"a":2}\n```'), {"a": 2})

    def test_extract_with_chatter(self):
        self.assertEqual(_extract_json('好的，结果如下：\n{"a":3}\n希望有帮助'), {"a": 3})

    def test_extract_garbage(self):
        self.assertIsNone(_extract_json("完全不是 JSON"))

    def test_visual_note_markdown(self):
        vn = VisualNote([FrameNote(0, "slides", "封面"), FrameNote(90, "code", "x=1", True)],
                        gist="概要")
        md = vn.as_markdown()
        self.assertIn("概要", md)
        self.assertIn("★", md)
        self.assertIn("code", md)

    def test_merge_prefers_new_and_dedupes_gist(self):
        a = VisualNote([FrameNote(0, "slides", "旧")], gist="同一句")
        b = VisualNote([FrameNote(0, "code", "新")], gist="同一句")
        m = a.merge(b)
        self.assertEqual(len(m.items), 1)
        self.assertEqual(m.items[0].text, "新")
        self.assertEqual(m.gist, "同一句")


class TestVisionFallback(unittest.TestCase):
    """模型名写错时要能自动切备选模型（哪怕 retries=1）。"""

    def test_fallback_model_used_when_retries_is_one(self):
        import json as _json

        from mock_vlm import MockServer

        from aicbili.vision import VisionModel

        with MockServer() as srv:
            cfg = Config.load()
            cfg.retries = 1
            cfg.vlm.api_key = "k"
            cfg.vlm.base_url = srv.base_url
            cfg.vlm.model = "bad-model"
            cfg.vlm.fallback_model = "good-model"
            srv.state.fail_models = {"bad-model"}
            srv.state.reply_image = _json.dumps(
                {"frames": [{"i": 1, "kind": "slides", "text": "备选模型生效"}], "gist": "g"})
            note = VisionModel(cfg, log=lambda m: None).describe_sheet(
                b"\xff\xd8\xff\xd9", [0.0], title="t")
            self.assertEqual(note.items[0].text, "备选模型生效")
            self.assertEqual([q["body"]["model"] for q in srv.state.requests],
                             ["bad-model", "good-model"])

    def test_all_models_fail_raises(self):
        from mock_vlm import MockServer

        from aicbili.vision import VisionError, VisionModel

        with MockServer() as srv:
            cfg = Config.load()
            cfg.retries = 1
            cfg.vlm.api_key = "k"
            cfg.vlm.base_url = srv.base_url
            cfg.vlm.model = "m1"
            cfg.vlm.fallback_model = "m2"
            srv.state.fail_models = {"m1", "m2"}
            with self.assertRaises(VisionError):
                VisionModel(cfg, log=lambda m: None).describe_sheet(
                    b"\xff\xd8\xff\xd9", [0.0], title="t")


FAKE_WCPP = r"""#!/usr/bin/env python3
import json, sys
a = sys.argv[1:]
v = lambda f, d=None: a[a.index(f)+1] if f in a else d
prefix, wav, model = v("-of"), v("-f"), v("-m")
assert "-oj" in a, "-oj 必须存在"
assert wav and wav.endswith(".wav"), "必须传 16k wav"
assert model and model.endswith(".bin"), "必须传 ggml 模型"
json.dump({"result": {"language": "zh"}, "transcription": [
    {"offsets": {"from": 0, "to": 1500}, "text": " 大家好"},
    {"offsets": {"from": 1500, "to": 3000}, "text": " 讲 print 函数"}]},
    open(prefix + ".json", "w", encoding="utf-8"), ensure_ascii=False)
"""


@unittest.skipUnless(shutil.which("ffmpeg"), "需要 ffmpeg")
class TestWhisperCpp(unittest.TestCase):
    """本地 whisper.cpp 后端：不依赖真模型也能验证接口与解析。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="aicbili_wcpp_test_")
        cls.bin = os.path.join(cls.tmp, "fake-whisper-cli")
        Path(cls.bin).write_text(FAKE_WCPP, encoding="utf-8")
        os.chmod(cls.bin, 0o755)
        cls.model = os.path.join(cls.tmp, "ggml-base.bin")
        Path(cls.model).write_bytes(b"fake")
        cls.audio = os.path.join(cls.tmp, "a.mp3")
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "sine=frequency=440:duration=2", "-c:a", "libmp3lame", cls.audio],
                       check=True)

    @classmethod
    def tearDownClass(cls):
        import shutil as _sh
        _sh.rmtree(cls.tmp, ignore_errors=True)

    def _cfg(self):
        cfg = Config.load()
        cfg.whisper_cpp_bin = self.bin
        cfg.whisper_cpp_model = self.model
        return cfg

    def test_transcribes_with_offsets(self):
        from aicbili.asr import transcribe_whisper_cpp

        tr = transcribe_whisper_cpp(self.audio, self._cfg(), log=lambda m: None)
        self.assertEqual(tr.source, "whisper.cpp")
        self.assertEqual(len(tr.segments), 2)
        self.assertEqual(tr.segments[0].text, "大家好")
        self.assertAlmostEqual(tr.segments[0].start, 0.0)
        self.assertAlmostEqual(tr.segments[1].start, 1.5)   # 毫秒 → 秒
        self.assertEqual(tr.language, "zh")

    def test_missing_model_gives_actionable_error(self):
        from aicbili.asr import transcribe_whisper_cpp

        cfg = self._cfg()
        cfg.whisper_cpp_model = "/nonexistent/ggml-x.bin"
        with self.assertRaises(RuntimeError) as ctx:
            transcribe_whisper_cpp(self.audio, cfg, log=lambda m: None)
        self.assertIn("ggml", str(ctx.exception))

    def test_model_discovery_finds_ggml_and_skips_mmproj(self):
        from aicbili.asr import find_whisper_cpp_model

        d = Path(self.tmp) / "models"
        d.mkdir(exist_ok=True)
        (d / "mmproj-model.bin").write_bytes(b"x")
        (d / "ggml-base.bin").write_bytes(b"x")
        cwd = os.getcwd()
        try:
            os.chdir(self.tmp)
            found = find_whisper_cpp_model("./models")
            self.assertTrue(found and found.endswith("ggml-base.bin"))
        finally:
            os.chdir(cwd)


class TestTruncationRetry(unittest.TestCase):
    """输出被 max_tokens 截断时必须自动加大预算重试，而不是交出半份报告。"""

    def _srv(self):
        from mock_vlm import MockServer
        return MockServer()

    def test_vision_retries_with_bigger_budget(self):
        import json as _json

        from aicbili.vision import VisionModel

        with self._srv() as srv:
            cfg = Config.load()
            cfg.retries = 1          # 最容易踩坑：普通重试只有 1 次
            cfg.vlm.api_key = "k"
            cfg.vlm.base_url = srv.base_url
            cfg.vlm.model = "m"
            srv.state.reply_image = _json.dumps(
                {"frames": [{"i": 1, "kind": "x", "text": "ok"}], "gist": "g"})
            srv.state.truncate_below = 4000
            note = VisionModel(cfg, log=lambda m: None).describe_sheet(
                b"\xff\xd8\xff\xd9", [0.0], title="t")
            self.assertEqual(note.items[0].text, "ok")
            budgets = [q["body"].get("max_tokens") for q in srv.state.requests]
            self.assertEqual(budgets, [1800, 3600, 7200])

    def test_truncated_output_raises_when_budget_exhausted(self):
        from aicbili.vision import TruncatedOutput, VisionModel

        with self._srv() as srv:
            cfg = Config.load()
            cfg.retries = 1
            cfg.vlm.api_key = "k"
            cfg.vlm.base_url = srv.base_url
            cfg.vlm.model = "m"
            srv.state.truncate_below = 999999      # 永远截断
            with self.assertRaises(TruncatedOutput):
                VisionModel(cfg, log=lambda m: None).describe_sheet(
                    b"\xff\xd8\xff\xd9", [0.0], title="t")

    def test_finish_reason_of_helper(self):
        from aicbili.vision import finish_reason_of

        self.assertEqual(finish_reason_of(
            {"choices": [{"finish_reason": "length", "message": {}}]}), "length")
        self.assertEqual(finish_reason_of({}), "")


class TestMultimodalProbe(unittest.TestCase):
    """doctor 要能分辨「真能看图」和「纯文本模型」。"""

    def _probe(self, reply="OK 42", status=200):
        from mock_vlm import MockServer

        from aicbili.vision import probe_multimodal

        with MockServer() as srv:
            cfg = Config.load()
            cfg.retries = 1
            cfg.vlm.api_key = "k"
            cfg.vlm.base_url = srv.base_url
            cfg.vlm.model = "deepseek-flash"
            srv.state.reply_image = reply
            srv.state.status = status
            ok, msg = probe_multimodal(cfg)
            return ok, msg, srv.state.requests[-1]

    def test_vision_model_passes_and_sends_image(self):
        ok, msg, req = self._probe("OK 42")
        self.assertTrue(ok)
        self.assertTrue(req["has_media"], "图片必须真的随请求发出去")

    def test_text_only_model_detected(self):
        ok, _, _ = self._probe("抱歉，我无法查看图片。")
        self.assertFalse(ok)

    def test_bad_key_detected(self):
        ok, msg, _ = self._probe(status=401)
        self.assertFalse(ok)
        self.assertIn("401", msg)


class TestStallDetection(unittest.TestCase):
    """CDN 卡死时必须尽快放弃该节点，而不是干等到整体超时。"""

    def test_stalled_process_is_killed(self):
        import time

        from aicbili.media import FFmpegError, _run

        with tempfile.TemporaryDirectory() as d:
            watch = os.path.join(d, "out.mp3")
            Path(watch).write_bytes(b"x" * 10)   # 文件存在但不再增长
            t0 = time.time()
            with self.assertRaises(FFmpegError) as ctx:
                _run([sys.executable, "-c", "import time; time.sleep(30)"],
                     timeout=60, tag="测试", watch_path=watch, stall_secs=1.0)
            elapsed = time.time() - t0
            self.assertIn("卡住", str(ctx.exception))
            self.assertLess(elapsed, 10, "应在 ~1s 内放弃，而不是等到 60s 超时")

    def test_normal_process_not_affected(self):
        from aicbili.media import _run

        with tempfile.TemporaryDirectory() as d:
            watch = os.path.join(d, "out.mp3")
            _run([sys.executable, "-c", "open('%s','wb').write(b'ok')" % watch],
                 timeout=30, tag="测试", watch_path=watch, stall_secs=1.0)
            self.assertTrue(os.path.exists(watch))


class TestConfig(unittest.TestCase):
    def test_env_file_and_cookie(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / ".env"
            p.write_text("AICBILI_SESSDATA=abc123\nAICBILI_MAX_FRAMES=7\n"
                         "AICBILI_VLM_MODEL=test-vl\nAICBILI_RELAY=https://r.example/\n")
            env = load_env(str(p))
            self.assertEqual(env["AICBILI_SESSDATA"], "abc123")
        cfg = Config.load()
        cfg.sessdata = "abc123"
        self.assertTrue(cfg.logged_in)
        self.assertIn("SESSDATA=abc123", cfg.cookie_header)

    def test_overrides(self):
        cfg = Config.load(max_frames=3, **{"vlm.model": "m1"})
        self.assertEqual(cfg.max_frames, 3)
        self.assertEqual(cfg.vlm.model, "m1")

    def test_masked_hides_key(self):
        cfg = Config.load()
        cfg.vlm.api_key = "sk-1234567890abcdef"
        self.assertNotIn("1234567890abcdef", str(cfg.masked()))

    def test_relay_url(self):
        u = relay_url("https://r.dev/", "https://cdn.example/a.mp4?x=1")
        self.assertTrue(u.startswith("https://r.dev/"))
        self.assertIn("cdn.example", u)
        self.assertNotIn("a.mp4?x=1", u.split("/", 3)[-1][:20])


class TestHeaders(unittest.TestCase):
    def test_referer_injected_for_bilibili(self):
        h = http_header_arg("https://upos-sz-mirrorcos.bilivideo.com/x.mp4")
        self.assertIn("Referer: https://www.bilibili.com", h)

    def test_no_referer_for_other_hosts(self):
        self.assertIsNone(http_header_arg("https://example.com/a.mp4"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
