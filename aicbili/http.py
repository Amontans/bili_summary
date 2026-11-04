"""极简 HTTP 客户端（仅用标准库）。

为什么不用 httpx/requests？
--------------------------
这个工具的核心路径是「调几个 JSON 接口 + 让 ffmpeg 拉流」，
用标准库足够，还能让用户 `pip install` 时零依赖（Pillow 只用于拼图，也可降级）。
同时这里集中处理超时、重试、UA 与 Referer，避免各处重复实现。
"""
from __future__ import annotations

import json
import mimetypes
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Mapping, Optional, Sequence, Tuple

DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


class HttpError(RuntimeError):
    def __init__(self, status: int, url: str, body: str = ""):
        self.status = status
        self.url = url
        self.body = body
        super().__init__(f"HTTP {status} {url[:120]}" + (f" :: {body[:200]}" if body else ""))


class Response:
    def __init__(self, status: int, headers: Mapping[str, str], body: bytes, url: str):
        self.status = status
        self.headers = headers
        self.body = body
        self.url = url

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8", errors="replace"))

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


def _build_opener() -> urllib.request.OpenerDirector:
    opener = urllib.request.build_opener()
    opener.addheaders = []
    return opener


def request(method: str, url: str, *, headers: Optional[Mapping[str, str]] = None,
            params: Optional[Mapping[str, Any]] = None, data: Optional[bytes] = None,
            timeout: float = 30.0, retries: int = 3, follow: bool = True) -> Response:
    if params:
        qs = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = url + ("&" if "?" in url else "?") + qs
    hdrs = {"User-Agent": DEFAULT_UA}
    if headers:
        hdrs.update({k: v for k, v in headers.items() if v is not None})
    last: Optional[Exception] = None
    for attempt in range(max(1, retries)):
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
                return Response(r.status, dict(r.headers), body, r.url if follow else url)
        except urllib.error.HTTPError as e:
            body = b""
            try:
                body = e.read()
            except Exception:
                pass
            # 4xx 基本没必要重试（412 是风控，值得退避后重试）
            if e.code in (429, 412, 500, 502, 503, 504) and attempt + 1 < retries:
                last = HttpError(e.code, url, body[:300].decode("utf-8", "replace"))
                time.sleep(1.0 * (attempt + 1))
                continue
            raise HttpError(e.code, url, body[:300].decode("utf-8", "replace")) from None
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
            last = e
            if attempt + 1 < retries:
                time.sleep(0.8 * (attempt + 1))
                continue
            raise HttpError(-1, url, str(e)) from None
    raise HttpError(-1, url, str(last))


def get_json(url: str, *, headers=None, params=None, timeout: float = 20.0,
             retries: int = 3) -> Any:
    return request("GET", url, headers=headers, params=params,
                   timeout=timeout, retries=retries).json()


def get_bytes(url: str, *, headers=None, timeout: float = 60.0, retries: int = 3,
              byte_range: Optional[Tuple[int, int]] = None) -> bytes:
    h = dict(headers or {})
    if byte_range:
        h["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    return request("GET", url, headers=h, timeout=timeout, retries=retries).body


def post_json(url: str, payload: Mapping[str, Any], *, headers=None,
              timeout: float = 180.0, retries: int = 2) -> Any:
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return request("POST", url, headers=h, data=body, timeout=timeout, retries=retries).json()


def post_multipart(url: str, fields: Mapping[str, str],
                   files: Sequence[Tuple[str, str, bytes, str]], *,
                   headers=None, timeout: float = 900.0, retries: int = 1) -> Any:
    """files: [(field_name, filename, content, mime), ...]"""
    boundary = "----aicbili" + uuid.uuid4().hex
    out = bytearray()
    for k, v in fields.items():
        out += f"--{boundary}\r\n".encode()
        out += f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode()
        out += str(v).encode("utf-8") + b"\r\n"
    for field, filename, content, mime in files:
        mime = mime or (mimetypes.guess_type(filename)[0] or "application/octet-stream")
        out += f"--{boundary}\r\n".encode()
        out += (f'Content-Disposition: form-data; name="{field}"; '
                f'filename="{os.path.basename(filename)}"\r\n').encode()
        out += f"Content-Type: {mime}\r\n\r\n".encode()
        out += content + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    h = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if headers:
        h.update(headers)
    return request("POST", url, headers=h, data=bytes(out), timeout=timeout,
                   retries=retries).json()


__all__ = ["DEFAULT_UA", "HttpError", "Response", "get_bytes", "get_json", "post_json",
           "post_multipart", "request"]
