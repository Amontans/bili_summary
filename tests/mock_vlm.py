"""一个假的 OpenAI 兼容服务，用来在没有 API Key 的情况下端到端验证调用链。

它会记录收到的请求体，方便断言「发出去的到底是什么内容」（图片/视频/文本）。
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List


class MockState:
    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []
        self.reply = "（mock 回复）"
        self.reply_image = ""     # 含图片/视频的请求用这条
        self.reply_text = ""      # 纯文本请求用这条
        self.status = 200
        self.fail_models: set = set()   # 这些模型名返回 404
        # 小于该 max_tokens 时返回 finish_reason=length（模拟输出被截断）
        self.truncate_below: int = 0
        # 按 prompt 关键词路由回复：[(关键词, 回复), ...]，先匹配先生效
        self.routes: List = []
        self.lock = threading.Lock()

    def reset(self) -> None:
        with self.lock:
            self.requests.clear()
            self.reply = "（mock 回复）"
            self.reply_image = self.reply_text = ""
            self.status = 200


def _prompt_text(payload: Any) -> str:
    """把请求里所有文本部分拼起来，供关键词路由用。"""
    out = []
    try:
        for m in payload.get("messages") or []:
            c = m.get("content")
            if isinstance(c, str):
                out.append(c)
            elif isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        out.append(str(part.get("text") or ""))
    except Exception:
        pass
    return "\n".join(out)


def _has_media(payload: Any) -> bool:
    """请求里是否带了图片/视频内容。"""
    try:
        for m in payload.get("messages") or []:
            c = m.get("content")
            if isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") in ("image_url", "video_url"):
                        return True
    except Exception:
        pass
    return False


def make_handler(state: MockState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # 静音
            return

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                payload = {"_raw_len": len(raw), "_content_type": self.headers.get("Content-Type")}
            has_media = _has_media(payload)
            with state.lock:
                state.requests.append({"path": self.path, "body": payload,
                                       "auth": self.headers.get("Authorization"),
                                       "has_media": has_media})
                status = state.status
                reply = (state.reply_image if has_media else state.reply_text) or state.reply
                prompt_txt = _prompt_text(payload)
                for kw, r in state.routes:
                    if kw in prompt_txt:
                        reply = r
                        break
                if isinstance(payload, dict) and payload.get("model") in state.fail_models:
                    status, reply = 404, '{"error":{"message":"model not found"}}' 
            mt = int((payload or {}).get("max_tokens") or 0) if isinstance(payload, dict) else 0
            finish = "length" if (state.truncate_below and mt < state.truncate_below) else "stop"
            body = json.dumps({
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "model": (payload.get("model") if isinstance(payload, dict) else "mock"),
                "choices": [{"index": 0, "finish_reason": finish,
                             "message": {"role": "assistant", "content": reply}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            body = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


class MockServer:
    """上下文管理器：with MockServer() as srv: srv.base_url / srv.state"""

    def __init__(self, port: int = 0) -> None:
        self.state = MockState()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(self.state))
        self.thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> "MockServer":
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *a) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


if __name__ == "__main__":
    with MockServer(port=8765) as srv:
        print("mock VLM on", srv.base_url)
        threading.Event().wait()
