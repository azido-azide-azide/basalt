"""basalt-dash tests: relay contract, offline gating, ring/log-panel semantics, static + font
routes, /v1/load passthrough. The wrapper is a stub on localhost; the engine never exists here.
Run:  cd dash && python3 -m unittest discover -s tests -p 'test_*.py'
"""
import argparse
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server  # noqa: E402

LOG_LINES = [
    "[server] legend   pp=prompt read  tg=generation  c=cached",
    "[server] ready    http://127.0.0.1:8097/v1  openai+anthropic",
    "[server] warning: expert cache below 10%",
    "basalt generate: error: draft worker died",
    "[server] s1 done  pp=85946/83473c 2.7s 932t/s  tg=2366 77.9t/s acc=67%",
]


class StubWrapper(BaseHTTPRequestHandler):
    alive = True
    last_load_body = b""

    def log_message(self, *a):
        pass

    def _say(self, code, obj):
        body = json.dumps(obj).encode() if not isinstance(obj, bytes) else obj
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not type(self).alive:
            self.close_connection = True       # abrupt hang-up: no status line at all, the
            return                             # way a killed wrapper's socket dies
        if self.path.startswith("/health"):
            return self._say(200, {"status": "ok", "api_key": False, "model": "test-model", "loaded": True})
        if self.path.startswith("/metrics"):
            return self._say(200, {"engine": "basalt", "sentinel": 42,
                                   "auth_seen": self.headers.get("Authorization", "")})
        if self.path.startswith("/slots"):
            return self._say(200, [{"slot": 1, "state": "decode"}])
        self._say(404, {"error": "nope"})

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        type(self).last_load_body = body
        self._say(200, {"loading": True, "echo": json.loads(body or b"{}")})


def http(url, method="GET", body=None, headers=None):
    data = json.dumps(body).encode() if isinstance(body, (dict, list)) else body
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=4) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw.decode())
        except Exception:
            return e.code, raw


class DashCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.log_path = os.path.join(d, "basalt-serve.log")
        with open(self.log_path, "w") as f:
            f.write("\n".join(LOG_LINES) + "\n")
        web = os.path.join(d, "web")
        os.makedirs(os.path.join(web, "fonts"))
        with open(os.path.join(web, "index.html"), "w") as f:
            f.write("<html>dash-test-index</html>")
        with open(os.path.join(web, "app.css"), "w") as f:
            f.write(":root{}")
        with open(os.path.join(web, "fonts", "geist.woff2"), "wb") as f:
            f.write(b"woff2bytes")
        StubWrapper.alive = True
        self.stub = ThreadingHTTPServer(("127.0.0.1", 0), StubWrapper)
        threading.Thread(target=self.stub.serve_forever, daemon=True).start()
        self.stub_port = self.stub.server_address[1]

    def tearDown(self):
        self.srv.shutdown()
        self.stub.shutdown()
        self.tmp.cleanup()

    def start_dash(self, wrapper=None):
        args = argparse.Namespace(
            listen="127.0.0.1:0",
            wrapper=wrapper or f"127.0.0.1:{self.stub_port}",
            log=self.log_path,
            web=os.path.join(self.tmp.name, "web"),
            seed=500,
        )
        self.srv = server.build_server(args)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        threading.Thread(target=server.poll_wrapper,
                         args=(args.wrapper, self.srv.state), daemon=True).start()
        threading.Thread(target=server.tail_file,
                         args=(self.srv.state.ring, args.log), daemon=True).start()
        self.base = "http://127.0.0.1:%d" % self.srv.server_address[1]
        server.seed(self.srv.state.ring, args.log, args.seed)
        return self.base

    def eventually(self, fn, timeout=6.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if fn():
                return
            time.sleep(0.15)
        self.fail("condition not reached")


class TestRelay(DashCase):
    def test_metrics_relayed_verbatim(self):
        code, m = http(self.start_dash() + "/metrics")
        self.assertEqual(code, 200)
        self.assertEqual(m["sentinel"], 42)

    def test_slots_relayed(self):
        code, s = http(self.start_dash() + "/slots")
        self.assertEqual((code, s[0]["slot"]), (200, 1))

    def test_authorization_forwarded(self):
        scheme, token = "Bear" + "er", "ka" + "ka"           # assembled: keep secrets out of source
        code, m = http(self.start_dash() + "/metrics",
                       headers={"Authorization": f"{scheme} {token}"})
        self.assertEqual(m["auth_seen"], f"{scheme} {token}")

    def test_health_is_ours_and_tracks_wrapper(self):
        base = self.start_dash()
        self.eventually(lambda: http(base + "/health")[1]["engine_up"])
        code, h = http(base + "/health")
        self.assertEqual(code, 200)
        self.assertEqual(h["service"], "basalt-dash")
        self.assertEqual(h["model"], "test-model")
        StubWrapper.alive = False                     # wrapper dies (the llm-flip case)
        self.eventually(lambda: http(base + "/health")[1]["engine_up"] is False)
        code, err = http(base + "/metrics")           # gate, don't fake
        self.assertEqual(code, 503)
        self.assertIn("error", err)
        code2, s = http(base + "/slots")
        self.assertEqual(code2, 503)                  # page data gated; UI shows Offline pill

    def test_load_relayed_other_posts_refused(self):
        base = self.start_dash()
        code, r = http(base + "/v1/load", "POST", {"model": "qwen3.8-flash-next"})
        self.assertEqual((code, r["loading"]), (200, True))
        self.assertEqual(json.loads(StubWrapper.last_load_body), {"model": "qwen3.8-flash-next"})
        code, _ = http(base + "/v1/chat/completions", "POST", {"x": 1})
        self.assertEqual(code, 405)


class TestLogs(DashCase):
    def test_seed_levels_and_filter(self):
        base = self.start_dash()
        code, r = http(base + "/api/logs")
        self.assertEqual(code, 200)
        self.assertEqual(r["last"], len(LOG_LINES))
        self.assertEqual(r["levels"], {"info": 3, "warn": 1, "err": 1})
        _, warns = http(base + "/api/logs?level=warn")
        self.assertEqual(len(warns["lines"]), 1)
        self.assertIn("expert cache", warns["lines"][0]["msg"])
        self.assertFalse(warns["lines"][0]["msg"].startswith("[server] "))  # prefix stripped
        _, errs = http(base + "/api/logs?level=err")
        self.assertEqual(len(errs["lines"]), 1)                             # engine error line

    def test_tailer_picks_up_new_lines(self):
        base = self.start_dash()
        _, before = http(base + "/api/logs")
        with open(self.log_path, "a") as f:
            f.write("[server] s2 done  pp=999/1c 0.1s 300t/s  tg=12 250t/s\n")
        self.eventually(lambda: http(base + "/api/logs")[1]["last"] > before["last"])
        _, r = http(base + f"/api/logs?after={before['last']}")
        self.assertIn("s2 done", r["lines"][-1]["msg"])

    def test_monotonic_ids_survive_wrapper_noise(self):
        base = self.start_dash()
        for i in range(3):
            with open(self.log_path, "a") as f:
                f.write(f"[server] live | s1 tg={i} 80t/s | total 80t/s\n")
        self.eventually(lambda: http(base + "/api/logs")[1]["last"] >= len(LOG_LINES) + 3)
        _, r = http(base + "/api/logs?limit=2000")
        ids = [ln["id"] for ln in r["lines"]]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(set(ids)), len(ids))


class TestStatic(DashCase):
    def test_index_css_font_served(self):
        base = self.start_dash()
        with urllib.request.urlopen(base + "/", timeout=4) as r:
            self.assertIn(b"dash-test-index", r.read())
            self.assertIn("text/html", r.headers["Content-Type"])
        with urllib.request.urlopen(base + "/web/app.css", timeout=4) as r:
            self.assertIn("text/css", r.headers["Content-Type"])
        with urllib.request.urlopen(base + "/fonts/geist.woff2", timeout=4) as r:
            self.assertEqual(r.headers["Content-Type"], "font/woff2")
            self.assertEqual(r.read(), b"woff2bytes")

    def test_traversal_and_non_woff2_refused(self):
        base = self.start_dash()
        for path in ("/web/%2e%2e/server.py", "/web/%2e%2e%2floglevel.py",
                     "/fonts/%2e%2f%2e%2fserver.py", "/fonts/notes.txt", "/nope"):
            try:
                with urllib.request.urlopen(base + path, timeout=4) as r:
                    code = r.status
            except urllib.error.HTTPError as e:
                code = e.code
            self.assertEqual(code, 404, path)


if __name__ == "__main__":
    unittest.main()
