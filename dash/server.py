#!/usr/bin/env python3
"""basalt-dash: a standalone observer that keeps the basalt web UI up for the whole life of the
container, through wrapper restarts and llm-flip pauses.

The basalt web UI normally lives inside `python -m serve.server`, whose lifetime equals the
engine session's: the host flip tears the wrapper down and the UI's port stops answering. This
sidecar separates "the monitoring UI" from "the engine session":

  GET  /  /web/*  /fonts/*   static UI, served from --web (default: the repo's serve/web, so the
                             dashboard always matches the engine build checked out on this box)
  GET  /metrics   /slots     live relay of the wrapper's own endpoints; 503 {"error":...} when the
                             wrapper is down — the UI then renders its Offline pill, page stays up
  POST /v1/load              relayed verbatim (the UI's model hot-load); the dash adds no powers
  GET  /api/logs             THIS dash's ring, fed by tailing --log: one id stream that survives
                             every wrapper session (the wrapper's own panel resets on restart)
  GET  /health               this dash, with engine_up from the wrapper poller

A read-only observer apart from relaying the UI's own POST; it never spawns, signals, restarts or
reads secrets from the engine or wrapper. Vendored serve/util/loglevel.py and logbuf.py (import
rewritten) keep level classification and the ring byte-identical to the wrapper's own semantics.

Run:  python3 dash/server.py --listen 0.0.0.0:8099 --wrapper 127.0.0.1:8097 \
          --log /var/log/basalt-serve.log [--web /path/to/serve/web]
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from logbuf import LogBuffer, tail_file

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR_DEFAULT = os.path.normpath(os.path.join(ROOT, "..", "serve", "web"))
FONTS_DIR_NAME = "fonts"


class State:
    def __init__(self, log_path: str):
        self.ring = LogBuffer()
        self.log_path = log_path
        self.engine_up = False
        self.api_key = False
        self.model = ""
        self.web_dir = WEB_DIR_DEFAULT


def poll_wrapper(wrapper: str, state: State) -> None:
    """Liveness + public facts, the same way capy-dash watches nInfer: GET /health only."""
    while True:
        try:
            with urllib.request.urlopen(f"http://{wrapper}/health", timeout=1.5) as r:
                h = json.load(r)
            state.engine_up = True
            state.api_key = bool(h.get("api_key"))
            state.model = h.get("model") or state.model
        except Exception:
            state.engine_up = False
        time.sleep(2.0)


def seed(ring: LogBuffer, path: str, tail_lines: int) -> int:
    """Preload the ring with the newest lines of the wrapper log; timestamps are arrival times
    (the log file itself carries none — the wrapper stamps its own panel the same way)."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(0, size - 256 * 1024))
            chunk = f.read()
    except OSError:
        return 0
    lines = chunk.decode("utf-8", "replace").splitlines()[-tail_lines:]
    if chunk and size > 256 * 1024 and lines:
        lines = lines[1:] if len(chunk) >= 256 * 1024 else lines   # drop a cut first line
    ring.add("eng", "\n".join(lines))
    return len(lines)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "basalt-dash"
    state: State = None       # set by build_server
    args = None               

    def log_message(self, fmt, *args):      # keep the unit log to real events
        pass

    # ---------------------------------------------------------------- plumbing
    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, full: str, ctype: str) -> None:
        try:
            with open(full, "rb") as f:
                body = f.read()
        except OSError:
            return self._json(404, {"error": "not found"})
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _relay(self, path_with_query: str, method: str = "GET") -> None:
        """Forward one request to the wrapper and relay its answer verbatim (incl. 401/400).
        Connection-level failure means the session is gone: gate 503, the UI's Offline path."""
        data = None
        headers = {}
        if self.headers.get("Authorization"):
            headers["Authorization"] = self.headers["Authorization"]
        if method == "POST":
            try:
                data = self.rfile.read(int(self.headers.get("Content-Length", "0") or 0))
            except ValueError:
                return self._json(400, {"error": {"message": "bad Content-Length"}})
        req = urllib.request.Request(f"http://{self.args.wrapper}{path_with_query}",
                                     data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                body, code, ctype = r.read(), r.status, r.headers.get("Content-Type", "application/json")
            self.state.engine_up = True
        except urllib.error.HTTPError as e:            # the wrapper answered: relay its answer
            body, code, ctype = e.read(), e.code, e.headers.get("Content-Type", "application/json")
        except Exception:
            self.state.engine_up = False
            return self._json(503, {"error": {"message": "wrapper unavailable — engine session down"}})
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---------------------------------------------------------------- dashboard
    def do_GET(self):
        path = urlsplit(self.path).path.rstrip("/") or "/"
        query = urlsplit(self.path).query
        if path in ("/", "") or path.startswith("/web/"):
            return self._static(path)
        if path.startswith("/fonts/"):
            return self._fonts(path)
        if path in ("/health", "/api/health"):
            return self._json(200, {"status": "ok", "service": "basalt-dash",
                                    "engine_up": bool(self.state.engine_up),
                                    "api_key": bool(self.state.api_key),
                                    "model": self.state.model})
        if path in ("/metrics", "/slots"):
            return self._relay(path + (("?" + query) if query else ""))
        if path == "/api/logs":
            return self._logs(query)
        return self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        path = urlsplit(self.path).path.rstrip("/")
        if path == "/v1/load":                          # the UI's own hot-load control
            return self._relay(path, "POST")
        return self._json(405, {"error": {"message": "the observer relays /v1/load only"}})

    def _static(self, path: str):
        if path in ("", "/"):
            return self._file(os.path.join(self.state.web_dir, "index.html"), "text/html; charset=utf-8")
        name = path[len("/web/"):]                      # flat files only, no traversal
        if not name or "/" in name or name.startswith("."):
            return self._json(404, {"error": {"message": "not found"}})
        ctype = ("text/css" if name.endswith(".css") else
                 "text/javascript" if name.endswith(".js") else
                 "image/svg+xml" if name.endswith(".svg") else "application/octet-stream")
        return self._file(os.path.join(self.state.web_dir, name), ctype)

    def _fonts(self, path: str):
        name = path[len("/fonts/"):]                    # flat .woff2 only (wrapper handler parity)
        if not name or "/" in name or "\\" in name or not name.endswith(".woff2"):
            return self._json(404, {"error": {"message": "not found"}})
        return self._file(os.path.join(self.state.web_dir, FONTS_DIR_NAME, name), "font/woff2")

    def _logs(self, query: str):
        q = parse_qs(query)

        def number(name, default):
            try:
                return max(0, int(q.get(name, [default])[0]))
            except (ValueError, IndexError):
                return default
        after = number("after", 0)
        limit = min(number("limit", 500), 2000)
        level = (q.get("level", [""])[0]).lower()
        buf = self.state.ring
        if level in ("info", "warn", "err"):
            lines, last = buf.level_lines(level, after, limit)
        else:
            lines, last = buf.since(after, limit)
        return self._json(200, {"lines": lines, "last": last, "levels": buf.levels()})


class DashServer(ThreadingHTTPServer):
    daemon_threads = True
    state: "State | None" = None  # set by build_server


def build_server(args) -> "DashServer":
    """Builds an unstarted server; the poller/tailer threads start in main()."""
    state = State(args.log)
    state.web_dir = args.web
    handler = type("BoundHandler", (Handler,), {"state": state, "args": args})
    host, _, port = args.listen.rpartition(":")
    srv = DashServer((host or "0.0.0.0", int(port)), handler)
    srv.state = state
    return srv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="0.0.0.0:8099")
    ap.add_argument("--wrapper", default="127.0.0.1:8097", help="basalt serve wrapper host:port")
    ap.add_argument("--log", default="/var/log/basalt-serve.log", help="wrapper log to tail")
    ap.add_argument("--web", default=WEB_DIR_DEFAULT, help="serve/web directory to display")
    ap.add_argument("--seed", type=int, default=500, help="log lines to preload into the ring")
    args = ap.parse_args()

    srv = build_server(args)
    n = seed(srv.state.ring, args.log, args.seed)
    threading.Thread(target=poll_wrapper, args=(args.wrapper, srv.state), daemon=True).start()
    threading.Thread(target=tail_file, args=(srv.state.ring, args.log), daemon=True).start()
    print(f"[basalt-dash] listening on http://{args.listen} | wrapper http://{args.wrapper} "
          f"| log {args.log} ({n} lines seeded) | web {args.web}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
