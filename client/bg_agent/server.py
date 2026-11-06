from dataclasses import asdict
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from urllib.parse import urlsplit

from .policy import RulePolicy
from .sandbox import destination,is_drag


def serve(backend, port=8765, *, sandbox=False):
    """Loopback-only dashboard. Read-only backend has no mutation routes."""
    policy = RulePolicy()
    lock = threading.RLock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass
        def reply(self, value, status=200):
            data = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type","application/json; charset=utf-8")
            self.send_header("Cache-Control","no-store")
            self.send_header("X-Content-Type-Options","nosniff")
            self.end_headers()
            self.wfile.write(data)
        def do_GET(self):
            if self.path == "/":
                data = (Path(__file__).parent / "web" / "dashboard.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type","text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/api/state":
                with lock:
                    obs = backend.observe()
                    self.reply(dict(observation=obs.to_dict(), decision=asdict(policy.decide(obs)), sandbox=sandbox,
                                    action_keys=[a.key for a in obs.actions],
                                    destinations=[destination(a) for a in obs.actions],
                                    label=backend.rows[backend.index][0] if sandbox else "真实客户端 · 只读建议",
                                    done=backend.done if sandbox else False,
                                    stopped=backend.stopped if sandbox else False,
                                    error=getattr(backend,"error",None)))
            else:
                self.reply({"error":"not found"},404)
        def do_POST(self):
            # Reject cross-origin/form submissions and DNS-rebinding hosts.
            host = self.headers.get("Host","")
            origin = self.headers.get("Origin")
            if (host != f"127.0.0.1:{self.server.server_port}" or
                (origin and origin != f"http://{host}") or
                self.headers.get("Content-Type") != "application/json"):
                return self.reply({"error":"origin/content type rejected"},403)
            if not sandbox:
                return self.reply({"error":"read-only adapter"},403)
            try:
                size = int(self.headers.get("Content-Length","0"))
                if not 0 < size <= 8192:
                    raise ValueError("invalid body length")
                body = json.loads(self.rfile.read(size))
                with lock:
                    if self.path == "/api/stop":
                        backend.stopped = True
                        if backend.audit: backend.audit.write("ui_stop")
                    elif self.path == "/api/tick":
                        backend.tick()
                    elif self.path == "/api/action":
                        obs = backend.observe()
                        action = next((a for a in obs.actions if a.key == body.get("key")), None)
                        if action is None:
                            raise ValueError("unknown action")
                        if is_drag(action) and body.get("destination") != destination(action):
                            raise ValueError("wrong drop destination")
                        backend.perform(action, body.get("revision"))
                    else:
                        return self.reply({"error":"not found"},404)
                self.reply({"ok":True})
            except (ValueError, RuntimeError, TypeError) as exc:
                self.reply({"error":str(exc)},409)
    server = ThreadingHTTPServer(("127.0.0.1",port),Handler)
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    return server
