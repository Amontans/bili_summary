"""联网集成测试（默认跳过）。

运行：AICBILI_LIVE=1 python3 -m unittest tests.test_live -v
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LIVE = os.environ.get("AICBILI_LIVE") == "1"


@unittest.skipUnless(LIVE, "设置 AICBILI_LIVE=1 才跑联网测试")
class TestLiveBili(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from aicbili.bili import BiliClient

        cls.c = BiliClient()
        cls.v = cls.c.resolve("BV1GJ411x7h7")

    @classmethod
    def tearDownClass(cls):
        cls.c.close()

    def test_resolve(self):
        self.assertEqual(self.v.bvid, "BV1GJ411x7h7")
        self.assertGreater(self.v.duration, 0)

    def test_dash_streams(self):
        d = self.c.dash(self.v.bvid, self.v.cid)
        v, a = d.best_video(), d.best_audio()
        self.assertIsNotNone(v)
        self.assertGreaterEqual(v.height, 360)
        self.assertGreater(len(a.all_urls), 0)

    def test_referer_required(self):
        """核心结论：视频流无 Referer 会 403，带 Referer 可访问。"""
        from aicbili import http, media

        d = self.c.dash(self.v.bvid, self.v.cid)
        url = d.best_video().url
        with self.assertRaises(http.HttpError):
            http.get_bytes(url, headers={"User-Agent": "curl/8"}, retries=1, byte_range=(0, 63))
        body = http.get_bytes(url, headers={"User-Agent": "Mozilla/5.0",
                                            "Referer": "https://www.bilibili.com"},
                              retries=2, byte_range=(0, 63))
        self.assertGreater(len(body), 0)

    def test_grab_single_frame(self):
        import tempfile

        from aicbili import media

        d = self.c.dash(self.v.bvid, self.v.cid)
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "f.jpg")
            r = media.grab_frame(d.best_video(), 30, out, width=640)
            self.assertTrue(os.path.exists(out))
            self.assertGreater(r.bytes, 5000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
