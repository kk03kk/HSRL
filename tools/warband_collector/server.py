"""Loopback-only collector. No native extension, model, client control or cloud dependency."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit, unquote
from urllib.request import urlopen
import webbrowser
import zipfile

from .storage import Store, Conflict, MAX_BYTES, dumps, atomic, uid, number

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).parent / "web"
APP = "autotavern-warband-collector-v1"


def catalog(source, origin):
    meta = source.get("meta", {})
    version = source.get("version") or meta.get("client_version")
    cards = source.get("cards")
    if not isinstance(version, str) or not version.strip() or not isinstance(cards, list) or len(cards) > 100000:
        raise ValueError("卡表需要 version（或meta.client_version）和cards数组")
    rows, ids = [], set()
    for c in cards:
        if not isinstance(c, dict) or not isinstance(c.get("id"), str) or c["id"] in ids:
            raise ValueError("卡表卡牌ID缺失或重复")
        ids.add(c["id"])
        name = c.get("name_zh") or c.get("name") or c["id"]
        if not isinstance(name, str): raise ValueError("卡名必须为文本")
        for key in ("atk", "hp", "attack", "health", "tier"):
            number(c.get(key), f"卡表{key}")
        for key in ("name", "name_zh", "text", "text_zh", "kind", "type"):
            if key in c and not isinstance(c[key], str): raise ValueError(f"卡表{key}必须为文本")
        for key in ("keywords", "races"):
            if key in c and (not isinstance(c[key], list) or not all(isinstance(x, str) for x in c[key])):
                raise ValueError(f"卡表{key}必须为文本数组")
        rows.append({"id": c["id"], "name": name, "name_en": c.get("name", ""),
                     "kind": c.get("kind", c.get("type", "unknown").lower()), "attack": c.get("atk", c.get("attack")), "health": c.get("hp", c.get("health")),
                     "tier": c.get("tier"), "golden": bool(c.get("normal")),
                     "normal_id": c.get("normal") or c["id"],
                     "golden_id": c.get("golden") if isinstance(c.get("golden"),str) and c.get("golden") else (c["id"] if c.get("normal") else ""),
                     "pool_known": "pools" in c, "recruit_pool": "minion" in c.get("pools", []),
                     "associations": c.get("assoc", []),
                     "text": c.get("text_zh") or c.get("text") or "",
                     "keywords": c.get("keywords", []), "races": c.get("races", [])})
    return {"version": version.strip(), "origin": origin, "sha256": hashlib.sha256(dumps(source)).hexdigest(), "cards": rows}


class CollectorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, directory):
        self.store = Store(directory)
        self.catalog_dir = self.store.directory / "catalogs"
        self.catalog_dir.mkdir(exist_ok=True)
        data = json.loads((ROOT / "crates/taverngym/data/cards.json").read_text(encoding="utf-8"))
        self.builtin = catalog(data, "engine_reference_not_live_pool")
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def response(self, body, kind="application/json; charset=utf-8", status=200, download=None):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob: data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        if download: self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        self.wfile.write(body)

    def check_origin(self):
        expected = f"127.0.0.1:{self.server.server_port}"
        if self.headers.get("Host") != expected or self.headers.get("Origin", f"http://{expected}") != f"http://{expected}":
            raise ValueError("只接受本机采集器页面的请求")

    def body(self):
        size = int(self.headers.get("Content-Length", "0"))
        if size <= 0 or size > MAX_BYTES: raise ValueError("请求为空或超过40MB")
        return self.rfile.read(size)

    def do_GET(self):
        try:
            self.check_origin()
            path = urlsplit(self.path).path
            if path == "/api/health":
                self.response(dumps({"app": APP, "directory": str(self.server.store.directory), "project": str(ROOT)}))
            elif path == "/api/data": self.response(dumps(self.server.store.load()))
            elif path == "/api/profile": self.response((Path(__file__).parent / "current_profile.json").read_bytes())
            elif path == "/api/catalog":
                rows = [self.server.builtin]
                rows += [json.loads(p.read_text(encoding="utf-8")) for p in sorted(self.server.catalog_dir.glob("*.json"))]
                self.response(dumps(rows))
            elif path == "/api/backup.zip": self.response(self.server.store.backup(), "application/zip", download="warbands-backup.zip")
            elif path == "/api/export.jsonl": self.response(self.server.store.jsonl(), "application/x-ndjson; charset=utf-8", download="warbands-observations.jsonl")
            elif path.startswith("/attachments/"):
                name = path.removeprefix("/attachments/")
                if not re.fullmatch(r"[a-f0-9]{32}\.(png|jpg|jpeg|webp|mp4|webm)", name): raise ValueError("无效附件路径")
                p = self.server.store.attachments / name
                media = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".mp4": "video/mp4", ".webm": "video/webm"}
                self.response(p.read_bytes(), media[p.suffix])
            elif path in ("/", "/index.html", "/app.js", "/core.js", "/search.js", "/style.css"):
                name = "index.html" if path == "/" else path[1:]
                kind = "text/html; charset=utf-8" if name.endswith("html") else "text/css; charset=utf-8" if name.endswith("css") else "text/javascript; charset=utf-8"
                self.response((ASSETS / name).read_bytes(), kind)
            else: self.response(dumps({"error": "未找到"}), status=404)
        except (ValueError, OSError, TypeError, KeyError, zipfile.BadZipFile) as e:
            self.response(dumps({"error": str(e)}), status=400)

    def do_POST(self):
        try:
            self.check_origin()
            path = urlsplit(self.path).path
            content = self.body()
            if path == "/api/save":
                obj = json.loads(content)
                result = self.server.store.save(obj["data"], obj["revision"])
            elif path == "/api/attachment":
                result = {"filename": self.server.store.attach(unquote(self.headers.get("X-Filename", "")), content)}
            elif path == "/api/import":
                result = self.server.store.merge(content, int(self.headers["X-Revision"]), self.headers.get("Content-Type") == "application/zip")
            elif path == "/api/catalog":
                result = catalog(json.loads(content), "user_supplied_unverified")
                atomic(self.server.catalog_dir / (uid() + ".json"), dumps(result))
                result = {"version": result["version"], "count": len(result["cards"])}
            else: raise ValueError("未知写入操作")
            self.response(dumps(result))
        except Conflict as e:
            self.response(dumps({"error": str(e)}), status=409)
        except (ValueError, OSError, TypeError, KeyError, zipfile.BadZipFile) as e:
            self.response(dumps({"error": str(e)}), status=400)


def lock_directory(directory):
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / ".collector.lock").open("a+b")
    try:
        handle.seek(0)
        if handle.read(1) == b"": handle.write(b"0"); handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise ValueError("该数据目录已经被另一个采集器使用；请使用已打开的页面")
    return handle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--data-dir", default=str(ROOT / "runs/warband-collector"))
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args()
    directory = Path(a.data_dir).resolve()
    if not directory.is_relative_to(ROOT / "runs"): p.error("采集数据必须保存在本主项目 runs 内")
    if not 1024 <= a.port <= 65535: p.error("port 必须为1024–65535")
    url = f"http://127.0.0.1:{a.port}/"
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"): stream.reconfigure(encoding="utf-8")
    try:
        with urlopen(url + "api/health", timeout=1) as response:
            running = json.load(response)
    except (OSError, ValueError):
        running = None
    if running is not None:
        if (running.get("app") != APP or running.get("project") != str(ROOT)
                or running.get("directory") != str(directory)):
            p.exit(1, "该端口被其他服务/数据目录使用，请用 --port 指定另一端口。\n")
        print(f"采集器已在运行：{url}", flush=True)
        if not a.no_browser: webbrowser.open(url)
        return
    try:
        handle = lock_directory(directory)
        server = CollectorServer(("127.0.0.1", a.port), directory)
    except (ValueError, OSError) as e:
        p.exit(1, f"启动失败：{e}\n如果已启动，请打开 {url}；不要同时运行多个实例。\n")
    print(f"AutoTavern 战团采集器：{url}\n保存目录：{directory}\n此窗口保持运行；Ctrl+C 停止。", flush=True)
    if not a.no_browser: webbrowser.open(url)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close(); handle.close()


if __name__ == "__main__": main()
